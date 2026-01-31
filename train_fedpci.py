"""
FedPCI 训练脚本

双分支架构训练：
- g_common: 共性分支（选择性聚合）
- g_ind: 个性化分支（不聚合）
- 原型 (μ, σ): 选择性聚合
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
from models.fedpci_model import FedPCIModel
from models.backbone import create_backbone
from federated.fedpci_client import FedPCIClient
from federated.fedpci_server import FedPCIServer


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
    server: FedPCIServer,
    config: Config
) -> List[FedPCIClient]:
    """创建客户端"""
    clients = []
    
    for k in range(num_clients):
        # 创建 FedPCI 模型（从服务端复制）
        model = FedPCIModel(
            num_classes=num_classes,
            input_dim=config.model.feature_dim,
            hidden_dim=config.model.fedpci_hidden_dim if hasattr(config.model, 'fedpci_hidden_dim') else 256,
            output_dim=config.model.fedpci_output_dim if hasattr(config.model, 'fedpci_output_dim') else 128,
            num_layers=config.model.fedpci_num_layers if hasattr(config.model, 'fedpci_num_layers') else 3,
            dropout=config.model.router_dropout,
            sigma_min=config.model.distribution_min_std,
            sigma_max=config.model.distribution_max_std,
            lambda_ind=config.training.lambda_ind if hasattr(config.training, 'lambda_ind') else 0.5,
            temperature=config.training.temperature_cls
        )
        
        # 从服务端复制模型参数
        model.load_state_dict(server.global_model.state_dict())
        
        # 创建客户端
        client = FedPCIClient(
            client_id=k,
            num_classes=num_classes,
            backbone=backbone,
            model=model,
            device=config.device,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            lambda_ind=config.training.lambda_ind if hasattr(config.training, 'lambda_ind') else 0.5,
            temperature=config.training.temperature_cls,
            lambda_cls_common=config.training.lambda_cls if hasattr(config.training, 'lambda_cls') else 1.0,
            lambda_cls_full=config.training.lambda_cls_full if hasattr(config.training, 'lambda_cls_full') else 1.0,
            lambda_global=config.training.lambda_contrast,
            lambda_common=config.training.lambda_common if hasattr(config.training, 'lambda_common') else 0.3,
            lambda_sigma=config.training.lambda_sigma if hasattr(config.training, 'lambda_sigma') else 0.01,
            lambda_proto_align=getattr(config.training, 'lambda_proto_align', 0.1)
        )
        
        clients.append(client)
    
    return clients


def train_task(
    task_id: int,
    task_classes: List[int],
    server: FedPCIServer,
    clients: List[FedPCIClient],
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
        
        # 加载全局共性分支参数
        clients[k].load_common_params(client_config['common_params'])
        
        # 加载全局原型参数
        clients[k].load_prototype_params(client_config['prototype_params'])
        
        # 设置本地数据信息
        clients[k].setup_local_data(local_classes=local_classes)
        
        # 提取本地原型
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
        
        # 按 participation_rate 随机选择客户端
        num_to_select = max(1, int(config.federated.num_clients * config.federated.participation_rate))
        num_selected = min(num_to_select, len(all_active_clients))
        active_clients = random.sample(all_active_clients, num_selected)
        
        if round_idx == 0:
            print(f"   📊 Client selection: {num_selected}/{len(all_active_clients)} active")
        
        # 4.2 客户端本地训练
        client_common_updates = {}
        client_prototype_updates = {}
        client_losses = []
        client_detailed_metrics = []  # 收集详细指标
        
        for k in active_clients:
            indices = client_data[k][0]
            local_classes = client_data[k][1]
            
            # 加载最新的全局模型
            client_config = server.get_client_config(k, local_classes)
            clients[k].load_common_params(client_config['common_params'])
            clients[k].load_prototype_params(client_config['prototype_params'])
            
            # 创建数据加载器
            train_subset = Subset(fed_data.train_dataset, indices)
            train_loader = create_data_loaders(
                train_subset,
                batch_size=config.federated.local_batch_size,
                shuffle=True
            )
            
            # 本地训练
            metrics = clients[k].train(
                train_loader=train_loader,
                num_epochs=config.federated.local_epochs,
                use_global_loss=True
            )
            
            client_losses.append(metrics['loss'])
            client_detailed_metrics.append({
                'client_id': k,
                'local_classes': local_classes,
                'metrics': metrics
            })
            
            # 收集更新
            client_common_updates[k] = clients[k].get_common_updates()
            client_prototype_updates[k] = clients[k].get_prototype_updates()
        
        # 4.3 本地评估（验证 g_ind 效果）
        local_acc_common_list = []
        local_acc_full_list = []
        local_gains = []
        
        for k in active_clients:
            indices = client_data[k][0]
            local_classes = client_data[k][1]
            
            # 用本地数据的一部分做验证（这里用训练数据，实际应该用验证集）
            eval_subset = Subset(fed_data.train_dataset, indices)
            eval_loader = create_data_loaders(
                eval_subset,
                batch_size=config.federated.local_batch_size,
                shuffle=False
            )
            
            local_metrics = clients[k].evaluate(eval_loader, local_classes)
            local_acc_common_list.append(local_metrics['accuracy_common'])
            local_acc_full_list.append(local_metrics['accuracy_full'])
            local_gains.append(local_metrics['grpo_gain'])
        
        avg_local_acc_common = sum(local_acc_common_list) / len(local_acc_common_list)
        avg_local_acc_full = sum(local_acc_full_list) / len(local_acc_full_list)
        avg_local_gain = sum(local_gains) / len(local_gains)
        
        round_metrics['local_acc_common'] = avg_local_acc_common
        round_metrics['local_acc_full'] = avg_local_acc_full
        round_metrics['local_grpo_gain'] = avg_local_gain
        
        # 4.4 服务端聚合（每 log_interval 轮打印详细日志）
        verbose_aggregation = ((round_idx + 1) % config.log_interval == 0)
        server.aggregate(
            client_common_updates=client_common_updates,
            client_prototype_updates=client_prototype_updates,
            verbose=verbose_aggregation
        )
        
        # 4.5 全局评估（只看 g_common 的效果）
        eval_metrics = server.evaluate(test_loader, test_classes)
        round_metrics.update(eval_metrics)
        
        # 计算平均值
        avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0
        
        # 打印日志
        print(f"Round {round_idx + 1:3d}/{config.training.num_rounds} | "
              f"Clients: {len(active_clients)} | "
              f"Loss: {avg_loss:.4f} | "
              f"Global: {eval_metrics['accuracy_common']:.2f}% | "
              f"Local: {avg_local_acc_common:.2f}%→{avg_local_acc_full:.2f}% | "
              f"LocalGain: {avg_local_gain:+.2f}%", end="")
        
        # 定期打印详细信息
        if (round_idx + 1) % config.log_interval == 0:
            print()
            
            # 1. 客户端训练详细损失
            print(f"         📊 Client Training Losses (avg across {len(active_clients)} clients):")
            avg_losses = {
                'cls_common': 0, 'cls_full': 0, 'global': 0,
                'common_compact': 0, 'sigma_reg': 0, 'proto_align': 0, 'orth':0
            }
            for cm in client_detailed_metrics:
                m = cm['metrics']
                avg_losses['cls_common'] += m.get('cls_common_loss', 0)
                avg_losses['cls_full'] += m.get('cls_full_loss', 0)
                avg_losses['global'] += m.get('global_loss', 0)
                avg_losses['common_compact'] += m.get('common_compact_loss', 0)
                avg_losses['sigma_reg'] += m.get('sigma_reg_loss', 0)
                avg_losses['proto_align'] += m.get('proto_align_loss', 0)
                avg_losses['orth'] += m.get('orth_loss', 0)
            
            n = len(client_detailed_metrics)
            print(f"            L_cls_common:    {avg_losses['cls_common']/n:.4f}")
            print(f"            L_cls_full:      {avg_losses['cls_full']/n:.4f}")
            print(f"            L_global:        {avg_losses['global']/n:.4f}")
            print(f"            L_common_compact:{avg_losses['common_compact']/n:.4f}")
            print(f"            L_sigma_reg:     {avg_losses['sigma_reg']/n:.4f}")
            print(f"            L_proto_align:   {avg_losses['proto_align']/n:.4f}")
            print(f"            L_orth:          {avg_losses['orth']/n:.4f}")
            
            # 2. 客户端原型距离
            print(f"         📊 Client Prototype Distances (after training):")
            all_proto_dists = {}
            for cm in client_detailed_metrics:
                for c, d in cm['metrics'].get('proto_distances', {}).items():
                    if c not in all_proto_dists:
                        all_proto_dists[c] = []
                    all_proto_dists[c].append(d)
            
            for c in sorted(all_proto_dists.keys())[:5]:  # 只打印前5个类
                dists = all_proto_dists[c]
                print(f"            Class {c} ({fed_data.class_names[c]:10s}): "
                      f"n={len(dists)}, mean={sum(dists)/len(dists):.3f}, "
                      f"min={min(dists):.3f}, max={max(dists):.3f}")
            
            # 3. 全局评估（g_common only）
            print(f"         📊 Global Evaluation (g_common only):")
            for cls in test_classes:
                key_common = f'class_{cls}_acc_common'
                if key_common in eval_metrics:
                    print(f"            {fed_data.class_names[cls]:12s}: {eval_metrics[key_common]:.1f}%")
            
            # 4. 本地评估（g_common + g_ind）
            print(f"         📊 Local Evaluation (g_common + g_ind):")
            print(f"            Avg AccCommon: {avg_local_acc_common:.2f}%")
            print(f"            Avg AccFull:   {avg_local_acc_full:.2f}%")
            print(f"            Avg GRPO Gain: {avg_local_gain:+.2f}%")
            
            # 5. 每个客户端的本地 gain
            print(f"         📊 Per-Client Details:")
            for i, k in enumerate(active_clients[:5]):  # 只打印前5个
                cm = client_detailed_metrics[i]
                proto_dist_str = ", ".join([f"{c}:{d:.2f}" for c, d in 
                                           list(cm['metrics'].get('proto_distances', {}).items())[:3]])
                print(f"            Client {k}: classes={cm['local_classes'][:3]}..., "
                      f"Common={local_acc_common_list[i]:.1f}%, "
                      f"Full={local_acc_full_list[i]:.1f}%, "
                      f"Gain={local_gains[i]:+.1f}%, "
                      f"proto_dist=[{proto_dist_str}]")
            if len(active_clients) > 5:
                print(f"            ... and {len(active_clients) - 5} more clients")
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
    
    # 7. 最终本地评估汇总
    final_local_gains = []
    for k, (indices, local_classes) in client_data.items():
        if len(local_classes) > 0 and len(indices) > 0:
            eval_subset = Subset(fed_data.train_dataset, indices)
            eval_loader = create_data_loaders(
                eval_subset,
                batch_size=config.federated.local_batch_size,
                shuffle=False
            )
            local_metrics = clients[k].evaluate(eval_loader, local_classes)
            final_local_gains.append({
                'client_id': k,
                'acc_common': local_metrics['accuracy_common'],
                'acc_full': local_metrics['accuracy_full'],
                'gain': local_metrics['grpo_gain']
            })
    
    avg_final_local_gain = sum(g['gain'] for g in final_local_gains) / len(final_local_gains) if final_local_gains else 0
    
    print(f"\n{'─'*60}")
    print(f"Task {task_id + 1} Completed!")
    print(f"{'─'*60}")
    print(f"  Global Accuracy (g_common): {final_metrics['accuracy_common']:.2f}%")
    print(f"  ")
    print(f"  Local Evaluation Summary ({len(final_local_gains)} clients):")
    print(f"    Avg Local AccCommon:   {sum(g['acc_common'] for g in final_local_gains)/len(final_local_gains):.2f}%")
    print(f"    Avg Local AccFull:     {sum(g['acc_full'] for g in final_local_gains)/len(final_local_gains):.2f}%")
    print(f"    Avg Local GRPO Gain:   {avg_final_local_gain:+.2f}%")
    print(f"  ")
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
    output_dir = os.path.join(config.output_dir, f"fedpci_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("FedPCI: Federated Prototype-based Class-Incremental Learning")
    print("Dual-branch architecture with selective aggregation")
    print("="*60)
    
    # 打印关键配置
    print(f"\n📋 Configuration:")
    print(f"   Clients: {config.federated.num_clients}")
    print(f"   Participation rate: {config.federated.participation_rate}")
    print(f"   Dirichlet α: {config.federated.alpha}")
    print(f"   Local epochs: {config.federated.local_epochs}")
    print(f"   Rounds per task: {config.training.num_rounds}")
    
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
    print("\n[2] Initializing FedPCI server...")
    model_config = {
        'backbone': config.model.backbone,
        'backbone_pretrained': config.model.backbone_pretrained,
        'feature_dim': config.model.feature_dim,
        'hidden_dim': getattr(config.model, 'fedpci_hidden_dim', 256),
        'output_dim': getattr(config.model, 'fedpci_output_dim', 128),
        'num_layers': getattr(config.model, 'fedpci_num_layers', 3),
        'dropout': config.model.router_dropout,
        'sigma_min': config.model.distribution_min_std,
        'sigma_max': config.model.distribution_max_std,
        'lambda_ind': getattr(config.training, 'lambda_ind', 0.5),
        'temperature': config.training.temperature_cls
    }
    
    server = FedPCIServer(
        num_classes=num_classes,
        class_names=config.data.class_names,
        model_config=model_config,
        device=config.device,
        prototype_momentum=getattr(config.training, 'prototype_momentum', 0.9)
    )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in server.global_model.parameters())
    single_network_params = sum(p.numel() for p in server.global_model.get_class_network(0).parameters())
    print(f"   Total model parameters: {total_params:,}")
    print(f"   Single class network parameters: {single_network_params:,}")
    print(f"   Number of classes: {num_classes}")
    
    # 创建共享 backbone
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
    
    print(f"\n📊 Global Results (g_common only):")
    print(f"   Global Accuracy: {final_metrics['accuracy_common']:.2f}%")
    print(f"   Total classes learned: {len(server.learned_classes)}")
    
    # 最终本地评估
    print(f"\n📊 Local Results (g_common + g_ind):")
    
    # 获取所有客户端的本地评估
    all_local_results = []
    client_data = fed_data.get_client_task_data(server.learned_classes)
    
    for k, (indices, local_classes) in client_data.items():
        if len(local_classes) > 0 and len(indices) > 0:
            eval_subset = Subset(fed_data.train_dataset, indices)
            eval_loader = create_data_loaders(
                eval_subset,
                batch_size=config.federated.local_batch_size,
                shuffle=False
            )
            local_metrics = clients[k].evaluate(eval_loader, local_classes)
            all_local_results.append({
                'client_id': k,
                'num_classes': len(local_classes),
                'acc_common': local_metrics['accuracy_common'],
                'acc_full': local_metrics['accuracy_full'],
                'gain': local_metrics['grpo_gain']
            })
    
    if all_local_results:
        avg_common = sum(r['acc_common'] for r in all_local_results) / len(all_local_results)
        avg_full = sum(r['acc_full'] for r in all_local_results) / len(all_local_results)
        avg_gain = sum(r['gain'] for r in all_local_results) / len(all_local_results)
        
        print(f"   Total clients evaluated: {len(all_local_results)}")
        print(f"   Avg Local AccCommon:     {avg_common:.2f}%")
        print(f"   Avg Local AccFull:       {avg_full:.2f}%")
        print(f"   Avg Local GRPO Gain:     {avg_gain:+.2f}%")
        
        # 统计 gain 的分布
        positive_gains = [r for r in all_local_results if r['gain'] > 0]
        negative_gains = [r for r in all_local_results if r['gain'] < 0]
        zero_gains = [r for r in all_local_results if r['gain'] == 0]
        
        print(f"\n   GRPO Gain Distribution:")
        print(f"     Positive (g_ind helps):  {len(positive_gains)} clients ({100*len(positive_gains)/len(all_local_results):.1f}%)")
        print(f"     Negative (g_ind hurts):  {len(negative_gains)} clients ({100*len(negative_gains)/len(all_local_results):.1f}%)")
        print(f"     Zero:                    {len(zero_gains)} clients ({100*len(zero_gains)/len(all_local_results):.1f}%)")
        
        if positive_gains:
            print(f"     Max positive gain: {max(r['gain'] for r in positive_gains):+.2f}%")
        if negative_gains:
            print(f"     Max negative gain: {min(r['gain'] for r in negative_gains):+.2f}%")
        
        # 打印 top 5 gain 最高和最低的客户端
        sorted_results = sorted(all_local_results, key=lambda x: x['gain'], reverse=True)
        print(f"\n   Top 5 Clients with Highest Gain:")
        for r in sorted_results[:5]:
            print(f"     Client {r['client_id']:3d}: Common={r['acc_common']:.1f}% Full={r['acc_full']:.1f}% Gain={r['gain']:+.1f}%")
        
        print(f"\n   Top 5 Clients with Lowest Gain:")
        for r in sorted_results[-5:]:
            print(f"     Client {r['client_id']:3d}: Common={r['acc_common']:.1f}% Full={r['acc_full']:.1f}% Gain={r['gain']:+.1f}%")
    
    # 打印每类全局准确率
    print(f"\n📈 Per-Class Global Accuracy:")
    for cls in server.learned_classes:
        key_common = f'class_{cls}_acc_common'
        if key_common in final_metrics:
            print(f"   {fed_data.class_names[cls]:12s}: {final_metrics[key_common]:.1f}%")
    
    # 诊断
    print(f"\n🔍 Model Diagnosis:")
    diagnosis = server.diagnose(test_loader)
    
    # 打印容易混淆的类
    print("   Most confused pairs (global model):")
    confusion = diagnosis['confusion_common']  # 用 common 的混淆矩阵
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 10:
                print(f"      {fed_data.class_names[i]} → {fed_data.class_names[j]}: "
                      f"{int(confusion[i, j].item())} samples")
    
    # 保存结果
    results = {
        'config': {
            'num_clients': config.federated.num_clients,
            'num_tasks': len(config.incremental.tasks),
            'alpha': config.federated.alpha,
            'num_rounds': config.training.num_rounds,
            'local_epochs': config.federated.local_epochs,
            'architecture': 'FedPCI (dual-branch)'
        },
        'tasks': all_results,
        'global_accuracy': final_metrics['accuracy_common'],
        'local_avg_accuracy_common': avg_common if all_local_results else 0,
        'local_avg_accuracy_full': avg_full if all_local_results else 0,
        'local_avg_grpo_gain': avg_gain if all_local_results else 0,
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