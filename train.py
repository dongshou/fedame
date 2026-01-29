"""
FedAME 主训练脚本
联邦类增量学习实验
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from config import get_config, Config
from data import CIFAR10Federated, create_data_loaders
from models import (
    create_backbone,
    AnchorBasedRouter,
    ExpertPool,
    DistributionPool
)
from anchor import create_anchor_generator, LLMDecisionMaker
from federated import FedAMEClient, FedAMEServer


def print_routing_diagnosis(diagnosis: Dict, indent: str = "   "):
    """
    打印路由诊断信息
    
    Args:
        diagnosis: 路由诊断结果字典
        indent: 缩进字符串
    """
    class_routing_stats = diagnosis['class_routing_stats']
    class_expected_expert = diagnosis['class_expected_expert']
    class_total_samples = diagnosis['class_total_samples']
    class_routing_accuracy = diagnosis['class_routing_accuracy']
    class_names = diagnosis['class_names']
    
    for cls in sorted(class_routing_stats.keys()):
        expected_exp = class_expected_expert[cls]
        total = class_total_samples[cls]
        correct = class_routing_stats[cls].get(expected_exp, 0)
        accuracy = class_routing_accuracy[cls]
        
        # 构建实际路由分布
        routing_dist = []
        for exp_id in sorted(class_routing_stats[cls].keys()):
            count = class_routing_stats[cls][exp_id]
            routing_dist.append(f"E{exp_id}:{count}")
        
        # 判断是否正确
        is_correct = accuracy > 0.5
        status = "✓" if is_correct else "✗"
        
        # 打印
        class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"{indent}{status} {class_name:12s} (class {cls}) → "
              f"Expected: E{expected_exp}, "
              f"Actual: [{', '.join(routing_dist)}], "
              f"Correct: {correct}/{total} ({accuracy*100:.1f}%)")
    
    # 打印总体准确率
    print(f"{indent}📊 Overall Routing Accuracy: "
          f"{diagnosis['total_correct']}/{diagnosis['total_samples']} "
          f"({diagnosis['overall_accuracy']*100:.1f}%)")



def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_clients(
    num_clients: int,
    backbone: nn.Module,
    server: FedAMEServer,
    config: Config
) -> List[FedAMEClient]:
    """创建客户端"""
    clients = []
    
    for k in range(num_clients):
        # 创建路由层（从服务端复制初始参数）
        router = AnchorBasedRouter(
            input_dim=config.model.feature_dim,
            hidden_dim=config.model.router_hidden_dim,
            anchor_dim=config.model.anchor_dim,
            temperature=config.training.temperature_route
        )
        router.load_state_dict(server.global_router.state_dict(), strict=False)
        # 设置锚点
        router.set_class_anchors(server.class_anchors.clone())
        router.set_cluster_anchors(
            server.cluster_anchors.clone(),
            server.global_router.cluster_to_expert
        )
        
        # 创建专家池（从服务端复制初始参数）
        expert_pool = ExpertPool(
            input_dim=config.model.anchor_dim,
            hidden_dim=config.model.expert_hidden_dim,
            output_dim=config.model.expert_output_dim,
            num_initial_experts=server.global_expert_pool.num_experts
        )
        for exp_id in server.global_expert_pool.experts.keys():
            expert_pool.experts[exp_id].load_state_dict(
                server.global_expert_pool.experts[exp_id].state_dict()
            )
        
        # 创建分布池
        distribution_pool = DistributionPool(
            dim=config.model.feature_dim  # backbone 输出维度，512
        )
        
        # 创建客户端
        client = FedAMEClient(
            client_id=k,
            backbone=backbone,
            router=router,
            expert_pool=expert_pool,
            distribution_pool=distribution_pool,
            class_anchors=server.class_anchors,
            cluster_anchors=server.cluster_anchors,
            device=config.device,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        clients.append(client)
    
    return clients


def train_task(
    task_id: int,
    task_classes: List[int],
    server: FedAMEServer,
    clients: List[FedAMEClient],
    fed_data: CIFAR10Federated,
    config: Config
) -> Dict:
    """
    训练单个任务
    
    Args:
        task_id: 任务ID
        task_classes: 任务类别
        server: 服务端
        clients: 客户端列表
        fed_data: 联邦数据集
        config: 配置
    
    Returns:
        metrics: 训练指标
    """
    print(f"\n{'='*60}")
    print(f"Task {task_id + 1}: {[fed_data.class_names[c] for c in task_classes]}")
    print('='*60)
    
    # 1. 服务端准备任务
    task_info = server.prepare_task(task_classes)
    old_classes = task_info['old_classes']
    
    # 2. 获取客户端数据划分
    client_data = fed_data.get_client_task_data(task_classes)
    
    # 3. 为每个客户端配置
    for k, (indices, local_classes) in client_data.items():
        if len(local_classes) == 0:
            continue
        
        # 获取客户端配置
        client_config = server.get_client_config(k, local_classes)
        
        # 加载全局模型到客户端
        clients[k].load_global_model(
            client_config['router_state'],
            client_config['expert_states'],
            client_config['distribution_params']
        )
        
        # 设置本地数据信息
        clients[k].setup_local_data(
            local_classes=local_classes,
            local_experts=client_config['local_experts'],
            class_to_expert=client_config['class_to_expert'],
            expert_to_cluster=client_config['expert_to_cluster']
        )

        # 创建本地数据加载器（用于初始化分布）
        train_subset = Subset(fed_data.train_dataset, indices)
        init_loader = create_data_loaders(
            train_subset,
            batch_size=config.federated.local_batch_size,
            shuffle=False
        )

        # 用真实特征初始化分布
        clients[k].init_distributions_from_data(init_loader)

        # 保存旧模型（用于防遗忘）
        if len(old_classes) > 0:
            clients[k].save_old_model()
    
    # 4. 联邦训练
    all_metrics = []
    
    # 预先创建测试集
    test_classes = list(set(server.learned_classes + task_classes))
    test_dataset = fed_data.get_cumulative_test_data(test_classes)
    test_loader = create_data_loaders(
        test_dataset,
        batch_size=config.federated.local_batch_size * 2,
        shuffle=False
    )
    
    for round_idx in range(config.training.num_rounds):
        round_metrics = {'round': round_idx + 1}
        client_updates = {}
        client_distribution_params = {}
        
        # 4.1 选择参与的客户端
        all_active_clients = []
        for k, (indices, local_classes) in client_data.items():
            if len(local_classes) > 0 and len(indices) > 0:
                all_active_clients.append(k)
        
        if len(all_active_clients) == 0:
            print(f"Round {round_idx + 1}: No active clients")
            continue
        
        # 按participation_rate随机选择客户端
        num_selected = max(1, int(len(all_active_clients) * config.federated.participation_rate))
        active_clients = random.sample(all_active_clients, num_selected)
        
        # 4.2 客户端本地训练
        client_losses = []
        client_train_accs = []
        client_sample_counts = {}  # 客户端总样本数
        client_class_counts = {}   # 客户端每类样本数
        
        for k in active_clients:
            indices = client_data[k][0]
            local_classes = client_data[k][1]
            
            # 记录样本数量
            client_sample_counts[k] = len(indices)
            
            # 统计每类样本数
            train_subset = Subset(fed_data.train_dataset, indices)
            class_counts = {}
            for idx in indices:
                _, label = fed_data.train_dataset[idx]
                class_counts[label] = class_counts.get(label, 0) + 1
            client_class_counts[k] = class_counts
            
            # 创建本地数据加载器
            train_loader = create_data_loaders(
                train_subset,
                batch_size=config.federated.local_batch_size,
                shuffle=True
            )
            
            # 本地训练（3阶段）
            # 阶段1: 训练路由器
            for epoch in range(config.federated.local_epochs):
                router_metrics = clients[k].train_router_only(
                    train_loader,
                    num_pseudo_samples=50,
                    lambda_pseudo=0.2
                )
            print(k)
            print('router', router_metrics)
            # 阶段2: 训练专家
            for epoch in range(config.federated.local_epochs):
                expert_metrics = clients[k].train_expert_only(
                    train_loader,
                    old_classes
                )
            print('expert', expert_metrics)
            # 阶段3: 训练分布参数
            for epoch in range(config.federated.local_epochs):
                dist_metrics = clients[k].train_distribution_only(
                    num_iterations=100,
                    num_samples_per_class=16
                )
            print('Prob', dist_metrics)
            # 合并metrics
            metrics = {
                'loss': (router_metrics.get('loss', 0) + expert_metrics.get('loss', 0) + dist_metrics.get('loss', 0)) / 3,
                'accuracy': expert_metrics.get('accuracy', 0),
                'routing_accuracy': router_metrics.get('routing_accuracy', 0)
            }
            
            # 收集更新
            client_updates[k] = clients[k].get_model_updates()
            client_distribution_params[k] = clients[k].get_distribution_params()
            
            round_metrics[f'client_{k}_loss'] = metrics['loss']
            round_metrics[f'client_{k}_train_acc'] = metrics['accuracy']
            client_losses.append(metrics['loss'])
            client_train_accs.append(metrics['accuracy'])
        
        # 4.3 服务端聚合（传入样本数量用于加权）
        server.aggregate(
            client_updates, 
            client_distribution_params,
            client_sample_counts=client_sample_counts,
            client_class_counts=client_class_counts
        )
        
        # 4.4 分发更新后的全局模型
        for k in active_clients:
            local_classes = client_data[k][1]
            client_config = server.get_client_config(k, local_classes)
            clients[k].load_global_model(
                client_config['router_state'],
                client_config['expert_states'],
                client_config['distribution_params']
            )
        
        # 4.5 评估本地和全局模型
        # 本地评估（随机选一个客户端）
        sample_client_id = active_clients[0]
        local_eval_metrics = clients[sample_client_id].evaluate(test_loader)
        
        # 全局评估
        eval_metrics = server.evaluate(test_loader)
        
        round_metrics['local_test_acc'] = local_eval_metrics['accuracy']
        round_metrics['local_routing_acc'] = local_eval_metrics['routing_accuracy']
        round_metrics['local_expert_acc_gt'] = local_eval_metrics['expert_accuracy_with_gt_routing']
        
        round_metrics['global_test_acc'] = eval_metrics['accuracy']
        round_metrics['global_routing_acc'] = eval_metrics['routing_accuracy']
        round_metrics['global_expert_acc_gt'] = eval_metrics['expert_accuracy_with_gt_routing']
        
        # 计算平均值
        avg_loss = sum(client_losses) / len(client_losses)
        avg_train_acc = sum(client_train_accs) / len(client_train_accs)
        
        # 打印基本日志
        print(f"Round {round_idx + 1:3d}/{config.training.num_rounds} | "
              f"Clients: {len(active_clients)}/{len(all_active_clients)} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train: {avg_train_acc:.2f}% | "
              f"Test: {eval_metrics['accuracy']:.2f}% | "
              f"Route: {eval_metrics['routing_accuracy']:.1f}%", end="")
        
        # 定期打印详细信息
        if (round_idx + 1) % config.log_interval == 0:
            class_accs = []
            for cls in test_classes:
                key = f'class_{cls}_acc'
                if key in eval_metrics:
                    class_accs.append(f"{fed_data.class_names[cls]}:{eval_metrics[key]:.1f}%")
            print(f"\n         Per-class Acc: {', '.join(class_accs)}")
            
            # 打印本地性能（Client #{sample_client_id}）
            print(f"\n         📱 Local (Client #{sample_client_id}):")
            print(f"            Acc: {local_eval_metrics['accuracy']:.2f}% | "
                  f"Routing: {local_eval_metrics['routing_accuracy']:.2f}% | "
                  f"Expert (GT): {local_eval_metrics['expert_accuracy_with_gt_routing']:.2f}%")
            
            # 打印全局性能
            print(f"         🌍 Global:")
            print(f"            Acc: {eval_metrics['accuracy']:.2f}% | "
                  f"Routing: {eval_metrics['routing_accuracy']:.2f}% | "
                  f"Expert (GT): {eval_metrics['expert_accuracy_with_gt_routing']:.2f}%")
            
            # 路由诊断
            print("         🔀 Routing Diagnosis (class → expert):")
            diagnosis = server.diagnose_routing(test_loader, fed_data.class_names)
            print_routing_diagnosis(diagnosis, indent="            ")
        else:
            print()  # 换行
        
        all_metrics.append(round_metrics)
    
    # 5. 完成任务
    server.finish_task(task_classes)
    
    # 6. 最终评估
    test_dataset = fed_data.get_cumulative_test_data(server.learned_classes)
    test_loader = create_data_loaders(
        test_dataset,
        batch_size=config.federated.local_batch_size * 2,
        shuffle=False
    )
    
    final_metrics = server.evaluate(test_loader)
    
    print(f"\n{'─'*60}")
    print(f"Task {task_id + 1} Completed!")
    print(f"{'─'*60}")
    print(f"  Overall Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"  Learned classes: {[fed_data.class_names[c] for c in server.learned_classes]}")
    
    # 打印各任务类别的准确率
    if task_id > 0:
        # 计算旧任务的平均准确率（检测遗忘）
        old_task_accs = []
        for old_task_idx in range(task_id):
            old_task_classes = config.incremental.tasks[old_task_idx]
            task_acc = 0
            for cls in old_task_classes:
                key = f'class_{cls}_acc'
                if key in final_metrics:
                    task_acc += final_metrics[key]
            task_acc /= len(old_task_classes)
            old_task_accs.append(task_acc)
            print(f"  Task {old_task_idx + 1} Acc (old): {task_acc:.2f}%")
        
        # 当前任务准确率
        current_task_acc = 0
        for cls in task_classes:
            key = f'class_{cls}_acc'
            if key in final_metrics:
                current_task_acc += final_metrics[key]
        current_task_acc /= len(task_classes)
        print(f"  Task {task_id + 1} Acc (new): {current_task_acc:.2f}%")
        
        # 遗忘指标
        if old_task_accs:
            avg_old_acc = sum(old_task_accs) / len(old_task_accs)
            print(f"  Avg Old Tasks Acc: {avg_old_acc:.2f}%")
    
    print(f"{'─'*60}\n")
    
    return {
        'task_id': task_id,
        'task_classes': task_classes,
        'round_metrics': all_metrics,
        'final_metrics': final_metrics
    }


def main():
    """主函数"""
    # 获取配置
    config = get_config()
    
    # 设置设备
    if torch.cuda.is_available():
        config.device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        config.device = "cpu"
        print("Using CPU")
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("FedAME: Federated Anchor-guided MoE")
    print("for Class-Incremental Learning")
    print("="*60)
    
    # 创建联邦数据集
    print("\n[1] Loading CIFAR-10 dataset...")
    fed_data = CIFAR10Federated(
        data_root=config.data.data_root,
        num_clients=config.federated.num_clients,
        alpha=config.federated.alpha,
        seed=config.seed
    )
    
    # 创建服务端
    print("\n[2] Initializing server...")
    model_config = {
        'backbone': config.model.backbone,
        'backbone_pretrained': config.model.backbone_pretrained,
        'feature_dim': config.model.feature_dim,
        'router_hidden_dim': config.model.router_hidden_dim,
        'anchor_dim': config.model.anchor_dim,
        'expert_hidden_dim': config.model.expert_hidden_dim,
        'expert_output_dim': config.model.expert_output_dim,
        'temperature_route': config.training.temperature_route
    }
    
    server = FedAMEServer(
        num_classes=config.data.num_classes,
        class_names=config.data.class_names,
        cluster_config=config.data.semantic_clusters,
        model_config=model_config,
        device=config.device,
        use_clip=False,  # 使用简化版锚点生成器
        use_real_llm=config.anchor.use_real_llm
    )
    
    # 创建共享backbone
    print("\n[3] Creating shared backbone...")
    backbone = create_backbone(
        backbone_type=config.model.backbone,
        pretrained=config.model.backbone_pretrained,
        frozen=True
    ).to(config.device)
    
    # 创建客户端
    print("\n[4] Creating clients...")
    clients = create_clients(
        num_clients=config.federated.num_clients,
        backbone=backbone,
        server=server,
        config=config
    )
    print(f"Created {len(clients)} clients")
    
    # 训练所有任务
    print("\n[5] Starting incremental learning...")
    all_results = []
    
    for task_id, task_classes in enumerate(config.incremental.tasks):
        result = train_task(
            task_id=task_id,
            task_classes=task_classes,
            server=server,
            clients=clients,
            fed_data=fed_data,
            config=config
        )
        all_results.append(result)
    
    # 最终评估
    print("\n" + "="*60)
    print("Final Evaluation on All Classes")
    print("="*60)
    
    test_dataset = fed_data.get_cumulative_test_data(server.learned_classes)
    test_loader = create_data_loaders(
        test_dataset,
        batch_size=config.federated.local_batch_size * 2,
        shuffle=False
    )
    
    final_metrics = server.evaluate(test_loader)
    
    print(f"\n📊 Final Results:")
    print(f"   Overall Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"   Total classes learned: {len(server.learned_classes)}")
    
    # 打印每个任务的最终准确率
    print(f"\n📈 Per-Task Accuracy:")
    for task_idx, task_cls in enumerate(config.incremental.tasks):
        task_acc = 0
        class_details = []
        for cls in task_cls:
            key = f'class_{cls}_acc'
            if key in final_metrics:
                task_acc += final_metrics[key]
                class_details.append(f"{fed_data.class_names[cls]}:{final_metrics[key]:.1f}%")
        task_acc /= len(task_cls)
        print(f"   Task {task_idx + 1}: {task_acc:.2f}% ({', '.join(class_details)})")
    
    # 打印专家结构
    print(f"\n🔧 Final Expert Structure:")
    expert_info = server.global_expert_pool.get_expert_info()
    for exp_id, info in expert_info.items():
        class_names = [fed_data.class_names[c] for c in info['responsible_classes']]
        print(f"   Expert {exp_id}: {class_names}")
    
    # 保存结果
    results = {
        'config': {
            'num_clients': config.federated.num_clients,
            'num_tasks': config.incremental.num_tasks,
            'alpha': config.federated.alpha,
            'num_rounds': config.training.num_rounds,
            'local_epochs': config.federated.local_epochs
        },
        'tasks': all_results,
        'final_accuracy': final_metrics['accuracy'],
        'learned_classes': server.learned_classes
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    # 保存模型
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save(server.get_global_model_state(), model_path)
    print(f"Model saved to: {model_path}")
    
    return results


if __name__ == "__main__":
    results = main()