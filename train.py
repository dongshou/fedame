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
            anchor_dim=config.model.anchor_dim
        )
        distribution_pool.set_anchors(server.class_anchors)
        
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
            class_to_expert=client_config['class_to_expert']
        )
        
        # 保存旧模型（用于防遗忘）
        if len(old_classes) > 0:
            clients[k].save_old_model()
    
    # 4. 联邦训练
    all_metrics = []
    
    for round_idx in range(config.training.num_rounds):
        round_metrics = {'round': round_idx + 1}
        client_updates = {}
        client_distribution_params = {}
        
        # 4.1 选择参与的客户端
        active_clients = []
        for k, (indices, local_classes) in client_data.items():
            if len(local_classes) > 0 and len(indices) > 0:
                active_clients.append(k)
        
        if len(active_clients) == 0:
            print(f"Round {round_idx + 1}: No active clients")
            continue
        
        # 4.2 客户端本地训练
        for k in active_clients:
            indices = client_data[k][0]
            
            # 创建本地数据加载器
            train_subset = Subset(fed_data.train_dataset, indices)
            train_loader = create_data_loaders(
                train_subset,
                batch_size=config.federated.local_batch_size,
                shuffle=True
            )
            
            # 本地训练
            for epoch in range(config.federated.local_epochs):
                metrics = clients[k].train_epoch(train_loader, old_classes)
            
            # 收集更新
            client_updates[k] = clients[k].get_model_updates()
            client_distribution_params[k] = clients[k].get_distribution_params()
            
            round_metrics[f'client_{k}_loss'] = metrics['loss']
            round_metrics[f'client_{k}_acc'] = metrics['accuracy']
        
        # 4.3 服务端聚合
        server.aggregate(client_updates, client_distribution_params)
        
        # 4.4 分发更新后的全局模型
        for k in active_clients:
            local_classes = client_data[k][1]
            client_config = server.get_client_config(k, local_classes)
            clients[k].load_global_model(
                client_config['router_state'],
                client_config['expert_states'],
                client_config['distribution_params']
            )
        
        # 4.5 评估
        if (round_idx + 1) % config.log_interval == 0:
            # 创建测试集
            test_dataset = fed_data.get_cumulative_test_data(
                server.learned_classes + task_classes
            )
            test_loader = create_data_loaders(
                test_dataset,
                batch_size=config.federated.local_batch_size * 2,
                shuffle=False
            )
            
            eval_metrics = server.evaluate(test_loader)
            round_metrics['test_acc'] = eval_metrics['accuracy']
            
            print(f"Round {round_idx + 1}/{config.training.num_rounds} | "
                  f"Test Acc: {eval_metrics['accuracy']:.2f}%")
        
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
    
    print(f"\nTask {task_id + 1} Final Results:")
    print(f"  Overall Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"  Classes learned so far: {server.learned_classes}")
    
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
    
    print(f"\nFinal Overall Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Total classes learned: {len(server.learned_classes)}")
    
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