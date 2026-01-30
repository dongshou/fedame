"""
FedAME 解耦路由器训练脚本
使用N个独立的二分类Router
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    DecoupledRouterPool,
    ExpertPool
)
from federated import DecoupledClient, DecoupledServer


def print_routing_diagnosis(diagnosis: Dict, indent: str = "   "):
    """打印路由诊断信息"""
    class_routing_stats = diagnosis['class_routing_stats']
    class_total_samples = diagnosis['class_total_samples']
    class_routing_accuracy = diagnosis['class_routing_accuracy']
    class_names = diagnosis['class_names']
    
    for cls in sorted(class_routing_stats.keys()):
        total = class_total_samples[cls]
        correct = class_routing_stats[cls].get(cls, 0)
        accuracy = class_routing_accuracy[cls]
        
        # 构建实际路由分布
        routing_dist = []
        for routed_cls in sorted(class_routing_stats[cls].keys()):
            count = class_routing_stats[cls][routed_cls]
            if count > 0:
                routing_dist.append(f"C{routed_cls}:{count}")
        
        # 判断是否正确
        is_correct = accuracy > 0.5
        status = "✓" if is_correct else "✗"
        
        # 打印
        class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"{indent}{status} {class_name:12s} (class {cls}) → "
              f"Correct: {correct}/{total} ({accuracy*100:.1f}%) "
              f"[{', '.join(routing_dist[:5])}]")
    
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
    num_classes: int,
    backbone: nn.Module,
    server: DecoupledServer,
    config: Config
) -> List[DecoupledClient]:
    """创建客户端"""
    clients = []
    
    for k in range(num_clients):
        # 创建解耦路由器池
        router_pool = DecoupledRouterPool(
            num_classes=num_classes,
            input_dim=config.model.feature_dim,
            hidden_dim=config.model.router_hidden_dim,
            output_dim=config.model.anchor_dim,
            num_layers=config.model.router_num_layers,
            dropout=config.model.router_dropout
        )
        
        # 从服务端复制初始参数
        for class_id in range(num_classes):
            params = server.global_router_pool.get_router_params(class_id)
            router_pool.set_router_params(class_id, params)
        
        # 设置锚点
        router_pool.set_class_anchors(server.class_anchors.clone())
        
        # 创建专家池
        expert_pool = ExpertPool(
            input_dim=config.model.anchor_dim,
            hidden_dim=config.model.expert_hidden_dim,
            output_dim=config.model.expert_output_dim,
            num_initial_experts=num_classes
        )
        
        # 从服务端复制专家参数
        for exp_id in server.global_expert_pool.experts.keys():
            expert_pool.experts[exp_id].load_state_dict(
                server.global_expert_pool.experts[exp_id].state_dict()
            )
        
        # 创建客户端
        client = DecoupledClient(
            client_id=k,
            num_classes=num_classes,
            backbone=backbone,
            router_pool=router_pool,
            expert_pool=expert_pool,
            class_anchors=server.class_anchors,
            device=config.device,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            margin=1.0,
            hard_negative_k=config.model.hard_negative_k if hasattr(config.model, 'hard_negative_k') else 10
        )
        
        clients.append(client)
    
    return clients


def train_task(
    task_id: int,
    task_classes: List[int],
    server: DecoupledServer,
    clients: List[DecoupledClient],
    fed_data: CIFAR10Federated,
    config: Config
) -> Dict:
    """训练单个任务"""
    print(f"\n{'='*60}")
    print(f"Task {task_id + 1}: {[fed_data.class_names[c] for c in task_classes]}")
    print('='*60)
    
    # 1. 服务端准备任务
    task_info = server.prepare_task(task_classes)
    
    # 2. 获取客户端数据划分
    client_data = fed_data.get_client_task_data(task_classes)
    
    # 3. 为每个客户端配置
    for k, (indices, local_classes) in client_data.items():
        if len(local_classes) == 0:
            continue
        
        # 获取客户端配置
        client_config = server.get_client_config(k, local_classes)
        
        # 加载全局Router参数
        clients[k].load_global_routers(client_config['router_params'])
        
        # 加载全局专家参数
        clients[k].load_global_experts(client_config['expert_states'])
        
        # 设置全局视觉原型
        clients[k].set_global_prototypes(
            client_config['prototype_info']['prototypes'],
            client_config['prototype_info']['counts']
        )
        
        # 设置本地数据信息
        clients[k].setup_local_data(
            local_classes=local_classes,
            local_experts=client_config['local_experts'],
            class_to_expert=client_config['class_to_expert']
        )
        
        # 提取本地视觉原型
        train_subset = Subset(fed_data.train_dataset, indices)
        train_loader = create_data_loaders(
            train_subset,
            batch_size=config.federated.local_batch_size,
            shuffle=False
        )
        clients[k].extract_prototypes_from_data(train_loader)
    
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
        
        # 4.1 选择参与的客户端
        all_active_clients = []
        for k, (indices, local_classes) in client_data.items():
            if len(local_classes) > 0 and len(indices) > 0:
                all_active_clients.append(k)
        
        if len(all_active_clients) == 0:
            print(f"Round {round_idx + 1}: No active clients")
            continue
        
        # 按participation_rate随机选择客户端
        num_to_select = max(1, int(config.federated.num_clients * config.federated.participation_rate))
        num_selected = min(num_to_select, len(all_active_clients))
        active_clients = random.sample(all_active_clients, num_selected)
        
        if round_idx == 0:
            print(f"   📊 Client selection: {num_selected}/{len(all_active_clients)} active")
        
        # 4.2 客户端本地训练
        client_router_updates = {}
        client_prototype_updates = {}
        client_train_stats = {}
        client_losses = []
        
        for k in active_clients:
            indices = client_data[k][0]
            local_classes = client_data[k][1]
            
            # 加载最新的全局模型
            client_config = server.get_client_config(k, local_classes)
            clients[k].load_global_routers(client_config['router_params'])
            clients[k].set_global_prototypes(
                client_config['prototype_info']['prototypes'],
                client_config['prototype_info']['counts']
            )
            
            # 创建本地数据加载器
            train_subset = Subset(fed_data.train_dataset, indices)
            train_loader = create_data_loaders(
                train_subset,
                batch_size=config.federated.local_batch_size,
                shuffle=True
            )
            
            # 本地训练
            metrics = clients[k].train_routers(
                train_loader,
                num_epochs=config.federated.local_epochs
            )
            
            # 收集更新
            client_router_updates[k] = clients[k].get_router_updates()
            client_prototype_updates[k] = clients[k].get_prototype_updates()
            client_train_stats[k] = clients[k].get_train_stats()
            
            client_losses.append(metrics['loss'])
            round_metrics[f'client_{k}_loss'] = metrics['loss']
        
        # 4.3 服务端聚合
        server.aggregate(
            client_router_updates,
            client_prototype_updates,
            client_train_stats
        )
        
        # 4.4 评估
        eval_metrics = server.evaluate(test_loader)
        
        round_metrics['global_test_acc'] = eval_metrics['accuracy']
        round_metrics['global_routing_acc'] = eval_metrics['routing_accuracy']
        
        # 计算平均值
        avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0
        
        # 打印日志
        print(f"Round {round_idx + 1:3d}/{config.training.num_rounds} | "
              f"Clients: {len(active_clients)} | "
              f"Loss: {avg_loss:.4f} | "
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
            
            # 路由诊断
            print("         🔀 Routing Diagnosis:")
            diagnosis = server.diagnose_routing(test_loader, fed_data.class_names)
            print_routing_diagnosis(diagnosis, indent="            ")
        else:
            print()
        
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
    print(f"  Routing Accuracy: {final_metrics['routing_accuracy']:.2f}%")
    print(f"  Learned classes: {[fed_data.class_names[c] for c in server.learned_classes]}")
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
    output_dir = os.path.join(config.output_dir, f"decoupled_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("FedAME with Decoupled Routers")
    print("N independent binary routers for heterogeneous FL")
    print("="*60)
    
    # 打印关键配置
    print(f"\n📋 Configuration:")
    print(f"   Clients: {config.federated.num_clients}")
    print(f"   Participation rate: {config.federated.participation_rate}")
    print(f"   Dirichlet α: {config.federated.alpha}")
    print(f"   Local epochs: {config.federated.local_epochs}")
    print(f"   Rounds per task: {config.training.num_rounds}")
    print(f"   Router layers: {config.model.router_num_layers}")
    
    # 创建联邦数据集
    print("\n[1] Loading CIFAR-10 dataset...")
    fed_data = CIFAR10Federated(
        data_root=config.data.data_root,
        num_clients=config.federated.num_clients,
        alpha=config.federated.alpha,
        seed=config.seed
    )
    
    num_classes = config.data.num_classes
    
    # 创建服务端
    print("\n[2] Initializing server with decoupled routers...")
    model_config = {
        'backbone': config.model.backbone,
        'backbone_pretrained': config.model.backbone_pretrained,
        'feature_dim': config.model.feature_dim,
        'router_hidden_dim': config.model.router_hidden_dim,
        'router_num_layers': config.model.router_num_layers,
        'router_dropout': config.model.router_dropout,
        'anchor_dim': config.model.anchor_dim,
        'expert_hidden_dim': config.model.expert_hidden_dim,
        'expert_output_dim': config.model.expert_output_dim,
    }
    
    server = DecoupledServer(
        num_classes=num_classes,
        class_names=config.data.class_names,
        cluster_config=config.data.semantic_clusters,
        model_config=model_config,
        device=config.device,
        use_clip=False,
        use_real_llm=config.anchor.use_real_llm
    )
    
    # 打印Router参数量
    router_params = sum(p.numel() for p in server.global_router_pool.parameters())
    single_router_params = sum(p.numel() for p in server.global_router_pool.get_router(0).parameters())
    print(f"   Total router parameters: {router_params:,}")
    print(f"   Single router parameters: {single_router_params:,}")
    print(f"   Number of routers: {num_classes}")
    
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
        num_classes=num_classes,
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
    print(f"   Routing Accuracy: {final_metrics['routing_accuracy']:.2f}%")
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
    
    # 路由诊断
    print(f"\n🔀 Final Routing Diagnosis:")
    diagnosis = server.diagnose_routing(test_loader, fed_data.class_names)
    print_routing_diagnosis(diagnosis, indent="   ")
    
    # 保存结果
    results = {
        'config': {
            'num_clients': config.federated.num_clients,
            'num_tasks': config.incremental.num_tasks,
            'alpha': config.federated.alpha,
            'num_rounds': config.training.num_rounds,
            'local_epochs': config.federated.local_epochs,
            'router_type': 'decoupled'
        },
        'tasks': all_results,
        'final_accuracy': final_metrics['accuracy'],
        'final_routing_accuracy': final_metrics['routing_accuracy'],
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