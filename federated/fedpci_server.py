"""
FedPCI 联邦学习服务端

核心特点：
- 管理全局模型
- 聚合规则：
  - g_common[c]: 选择性聚合（仅拥有类c的客户端参与）
  - g_ind[c]: 不聚合（完全本地）
  - 原型 (μ, σ): 选择性聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import copy

from models.fedpci_model import FedPCIModel
from models.backbone import create_backbone


class FedPCIServer:
    """
    FedPCI 联邦服务端
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        model_config: Dict,
        device: str = "cuda",
        prototype_momentum: float = 0.9
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.prototype_momentum = prototype_momentum  # 原型动量系数
        
        # 初始化全局模型
        self._init_global_model(model_config)
        
        # 已学习的类别
        self.learned_classes: List[int] = []
        
        # 客户端信息
        self.client_info: Dict[int, Dict] = {}
    
    def _init_global_model(self, config: Dict):
        """初始化全局模型"""
        # Backbone（冻结）
        self.backbone = create_backbone(
            backbone_type=config.get('backbone', 'resnet18'),
            pretrained=config.get('backbone_pretrained', True),
            frozen=True
        ).to(self.device)
        
        # FedPCI 模型
        self.global_model = FedPCIModel(
            num_classes=self.num_classes,
            input_dim=config.get('feature_dim', 512),
            hidden_dim=config.get('hidden_dim', 256),
            output_dim=config.get('output_dim', 128),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
            sigma_min=config.get('sigma_min', 0.1),
            sigma_max=config.get('sigma_max', 2.0),
            lambda_ind=config.get('lambda_ind', 0.5),
            temperature=config.get('temperature', 0.1)
        ).to(self.device)
    
    def prepare_task(self, task_classes: List[int]) -> Dict:
        """
        准备新任务
        
        Args:
            task_classes: 任务包含的类别ID
        
        Returns:
            task_info: 任务信息
        """
        new_classes = [c for c in task_classes if c not in self.learned_classes]
        
        return {
            'task_classes': task_classes,
            'new_classes': new_classes,
            'old_classes': self.learned_classes.copy()
        }
    
    def get_client_config(
        self,
        client_id: int,
        client_classes: List[int]
    ) -> Dict:
        """
        获取客户端配置
        
        Args:
            client_id: 客户端ID
            client_classes: 客户端拥有的类别
        
        Returns:
            config: 客户端配置
        """
        # 保存客户端信息
        self.client_info[client_id] = {
            'classes': client_classes
        }
        
        # 获取所有类的共性分支参数
        global_common_params = self.global_model.get_all_common_params()
        
        # 获取所有类的原型参数
        global_prototype_params = self.global_model.get_all_prototype_params()
        
        return {
            'client_id': client_id,
            'local_classes': client_classes,
            'common_params': global_common_params,
            'prototype_params': global_prototype_params
        }
    
    def aggregate(
        self,
        client_common_updates: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
        client_prototype_updates: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
        verbose: bool = False
    ):
        """
        聚合客户端更新
        
        聚合规则：
        - g_common[c]: 选择性聚合，用原型距离加权 + 动量
        - g_ind[c]: 不聚合（客户端本地保留）
        - prototype[c]: 选择性聚合，样本数加权 + 动量
        
        Args:
            client_common_updates: {client_id: {class_id: {param_name: param_value}}}
            client_prototype_updates: {client_id: {class_id: {param_name: param_value}}}
            verbose: 是否打印详细日志
        """
        # 记录聚合前的状态
        if verbose:
            pre_state = self._get_model_state_summary()
        
        # 1. 聚合共性分支参数（使用原型距离加权 + 动量）
        proto_info = self._aggregate_prototype_params(client_prototype_updates, verbose)

        new_global_prototypes = {}
        for class_id in range(self.num_classes):
            proto_params = self.global_model.get_prototype_params(class_id)
            new_global_prototypes[class_id] = proto_params['mean'].cpu().float()
        # 2. 聚合原型参数（样本数加权 + 动量）
        agg_info = self._aggregate_common_params(
            client_common_updates,
            client_prototype_updates,
            new_global_prototypes,  # 传入新的全局原型
            verbose=verbose
        )
        
        # 记录聚合后的状态
        if verbose:
            post_state = self._get_model_state_summary()
            self._print_aggregation_summary(pre_state, post_state, agg_info, proto_info)
    
    def _get_model_state_summary(self) -> Dict:
        """获取模型状态摘要"""
        state = {
            'prototypes': {},
            'g_common_norms': {}
        }
        for c in range(self.num_classes):
            # 原型
            mu = self.global_model.get_prototype_mean(c)
            sigma = self.global_model.get_class_network(c).prototype.sigma
            state['prototypes'][c] = {
                'mu_norm': torch.norm(mu).item(),
                'mu_mean': mu.mean().item(),
                'sigma_mean': sigma.mean().item()
            }
            # g_common 参数范数
            params = self.global_model.get_common_params(c)
            total_norm = sum(torch.norm(p).item() for p in params.values())
            state['g_common_norms'][c] = total_norm
        return state
    
    def _print_aggregation_summary(self, pre_state, post_state, agg_info, proto_info):
        """打印聚合摘要"""
        print("\n         📊 Aggregation Summary:")
        print("         ┌─────────────────────────────────────────────────────────┐")
        
        # g_common 聚合信息
        if agg_info:
            print("         │ g_common aggregation (distance-weighted + momentum):   │")
            for c, info in sorted(agg_info.items()):
                if info['num_clients'] > 0:
                    weights_str = ", ".join([f"{w:.2f}" for w in info['weights'][:3]])
                    if len(info['weights']) > 3:
                        weights_str += "..."
                    print(f"         │   Class {c}: {info['num_clients']} clients, "
                          f"dists=[{', '.join([f'{d:.2f}' for d in info['distances'][:3]])}], "
                          f"weights=[{weights_str}]")
        
        # 原型聚合信息
        if proto_info:
            print("         │ Prototype aggregation (sample-weighted + momentum):    │")
            for c, info in sorted(proto_info.items()):
                if info['num_clients'] > 0:
                    print(f"         │   Class {c}: {info['num_clients']} clients, "
                          f"μ_change={info['mu_change']:.4f}, "
                          f"σ_change={info['sigma_change']:.4f}")
        
        # 状态变化
        print("         │ State changes:                                          │")
        for c in range(min(5, self.num_classes)):  # 只打印前5个类
            pre_mu = pre_state['prototypes'][c]['mu_norm']
            post_mu = post_state['prototypes'][c]['mu_norm']
            pre_g = pre_state['g_common_norms'][c]
            post_g = post_state['g_common_norms'][c]
            print(f"         │   Class {c}: μ_norm {pre_mu:.2f}→{post_mu:.2f}, "
                  f"g_norm {pre_g:.1f}→{post_g:.1f}")
        
        print("         └─────────────────────────────────────────────────────────┘")
    
    def _aggregate_common_params(
        self,
        client_updates: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
        client_prototype_updates: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
        new_global_prototypes: Dict[int, torch.Tensor],  # ← 新增参数
        verbose: bool = False
    ) -> Dict[int, Dict]:
        """
        聚合共性分支参数
        
        使用新聚合的全局原型计算距离权重
        
        Args:
            client_updates: 客户端共性参数更新
            client_prototype_updates: 客户端原型更新（用于获取本地原型）
            new_global_prototypes: 新聚合的全局原型 {class_id: mean_tensor}
        """
        agg_info = {}
        
        # 收集每个类的更新
        class_updates: Dict[int, List[Tuple[int, Dict[str, torch.Tensor]]]] = {
            c: [] for c in range(self.num_classes)
        }
        
        for client_id, updates in client_updates.items():
            for class_id, params in updates.items():
                class_updates[class_id].append((client_id, params))
        
        # 对每个类进行聚合
        for class_id in range(self.num_classes):
            updates = class_updates[class_id]
            
            agg_info[class_id] = {
                'num_clients': len(updates),
                'distances': [],
                'weights': [],
                'client_ids': []
            }
            
            if len(updates) == 0:
                continue
            
            # ========== 关键修改：使用新的全局原型 ==========
            global_proto = new_global_prototypes[class_id]
            # ================================================
            
            # 计算每个客户端的距离和权重
            distances = []
            client_ids = []
            for client_id, params in updates:
                client_ids.append(client_id)
                if client_id in client_prototype_updates and class_id in client_prototype_updates[client_id]:
                    local_proto = client_prototype_updates[client_id][class_id]['mean'].cpu().float()
                else:
                    local_proto = global_proto.clone()
                
                distance = torch.norm(local_proto - global_proto).item()
                distances.append(distance)
            
            # 计算 softmax 权重
            distances_tensor = torch.tensor(distances)
            
            if distances_tensor.max() < 1e-8:
                weights = torch.ones(len(distances)) / len(distances)
            else:
                weights = torch.softmax(-distances_tensor, dim=0)
            
            agg_info[class_id]['distances'] = distances
            agg_info[class_id]['weights'] = weights.tolist()
            agg_info[class_id]['client_ids'] = client_ids
            
            # 加权聚合参数
            aggregated_params = {}
            first_params = updates[0][1]
            
            for param_name in first_params.keys():
                param_sum = torch.zeros_like(first_params[param_name].cpu().float())
                for i, (client_id, params) in enumerate(updates):
                    param_sum += params[param_name].cpu().float() * weights[i].item()
                aggregated_params[param_name] = param_sum
            
            # 动量更新
            old_params = self.global_model.get_common_params(class_id)
            momentum = self.prototype_momentum
            for param_name in aggregated_params.keys():
                if param_name in old_params:
                    old_param = old_params[param_name].cpu().float()
                    aggregated_params[param_name] = (
                        momentum * old_param + (1 - momentum) * aggregated_params[param_name]
                    )
            
            self.global_model.set_common_params(class_id, aggregated_params)
    
        return agg_info
    def _aggregate_prototype_params(
        self,
        client_updates: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
        verbose: bool = False
    ) -> Dict[int, Dict]:
        """
        聚合原型参数（使用动量更新）
        
        对于每个类 c，只有拥有类 c 的客户端参与聚合
        使用动量聚合：μ_new = momentum * μ_old + (1 - momentum) * avg(μ_clients)
        
        Returns:
            proto_info: 每个类的原型聚合信息
        """
        proto_info = {}
        
        # 收集每个类的更新
        class_updates: Dict[int, List[Dict[str, torch.Tensor]]] = {
            c: [] for c in range(self.num_classes)
        }
        
        for client_id, updates in client_updates.items():
            for class_id, params in updates.items():
                class_updates[class_id].append(params)
        
        # 对每个类进行聚合
        for class_id in range(self.num_classes):
            updates = class_updates[class_id]
            
            proto_info[class_id] = {
                'num_clients': len(updates),
                'mu_change': 0.0,
                'sigma_change': 0.0
            }
            
            if len(updates) == 0:
                continue
            
            # 获取旧的全局原型（用于计算变化）
            old_params = self.global_model.get_prototype_params(class_id)
            old_mean = old_params['mean'].cpu().float()
            old_log_sigma = old_params['log_sigma'].cpu().float()
            
            # 计算总样本数（转换为 float 避免溢出）
            total_count = 0.0
            for p in updates:
                if 'sample_count' in p:
                    cnt = p['sample_count']
                    if isinstance(cnt, torch.Tensor):
                        total_count += float(cnt.item())
                    else:
                        total_count += float(cnt)
                else:
                    total_count += 1.0
            
            if total_count == 0:
                total_count = float(len(updates))  # 均等权重
            
            # 聚合 mean 和 log_sigma（客户端平均）
            dim = updates[0]['mean'].shape[0]
            aggregated_mean = torch.zeros(dim)
            aggregated_log_sigma = torch.zeros(dim)
            
            for params in updates:
                if 'sample_count' in params:
                    cnt = params['sample_count']
                    if isinstance(cnt, torch.Tensor):
                        count = float(cnt.item())
                    else:
                        count = float(cnt)
                else:
                    count = 1.0
                weight = count / total_count if total_count > 0 else 1.0 / len(updates)
                
                aggregated_mean += params['mean'].cpu().float() * weight
                aggregated_log_sigma += params['log_sigma'].cpu().float() * weight
            
            # 动量更新：μ_new = momentum * μ_old + (1 - momentum) * aggregated
            momentum = self.prototype_momentum
            new_mean = momentum * old_mean + (1 - momentum) * aggregated_mean
            new_log_sigma = momentum * old_log_sigma + (1 - momentum) * aggregated_log_sigma
            
            # 记录变化量
            proto_info[class_id]['mu_change'] = torch.norm(new_mean - old_mean).item()
            proto_info[class_id]['sigma_change'] = torch.norm(new_log_sigma - old_log_sigma).item()
            
            # 设置聚合后的参数
            self.global_model.set_prototype_params(class_id, {
                'mean': new_mean,
                'log_sigma': new_log_sigma,
                'sample_count': torch.tensor(min(total_count, 1e6))
            })
        
        return proto_info
    
    def finish_task(self, task_classes: List[int]):
        """完成任务，更新已学习类别"""
        for cls in task_classes:
            if cls not in self.learned_classes:
                self.learned_classes.append(cls)
    
    def evaluate(
        self,
        test_loader: DataLoader,
        classes_to_eval: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        评估全局模型
        
        Args:
            test_loader: 测试数据加载器
            classes_to_eval: 要评估的类别
        
        Returns:
            metrics: 评估指标
        """
        self.backbone.eval()
        self.global_model.eval()
        
        total_correct_common = 0
        total_correct_full = 0
        total_samples = 0
        
        class_correct_common = {}
        class_correct_full = {}
        class_total = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                features = self.backbone(images)
                
                # 计算距离
                d_total, d_common, d_ind = self.global_model(features)
                
                # 预测
                pred_common = torch.argmin(d_common, dim=-1)
                pred_full = torch.argmin(d_total, dim=-1)
                
                # 统计
                for i in range(len(labels)):
                    label = labels[i].item()
                    
                    if classes_to_eval and label not in classes_to_eval:
                        continue
                    
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct_common[label] = 0
                        class_correct_full[label] = 0
                    
                    class_total[label] += 1
                    total_samples += 1
                    
                    if pred_common[i].item() == label:
                        class_correct_common[label] += 1
                        total_correct_common += 1
                    
                    if pred_full[i].item() == label:
                        class_correct_full[label] += 1
                        total_correct_full += 1
        
        metrics = {
            'accuracy_common': total_correct_common / max(total_samples, 1) * 100,
            'accuracy_full': total_correct_full / max(total_samples, 1) * 100,
            'total_samples': total_samples
        }
        
        for cls in class_total:
            metrics[f'class_{cls}_acc_common'] = (
                class_correct_common[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
            metrics[f'class_{cls}_acc_full'] = (
                class_correct_full[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
        
        # GRPO Gain
        metrics['grpo_gain'] = metrics['accuracy_full'] - metrics['accuracy_common']
        
        return metrics
    
    def diagnose(
        self,
        test_loader: DataLoader
    ) -> Dict:
        """
        诊断模型性能
        
        分析每个类的预测情况
        """
        self.backbone.eval()
        self.global_model.eval()
        
        # 混淆矩阵
        confusion_common = torch.zeros(self.num_classes, self.num_classes)
        confusion_full = torch.zeros(self.num_classes, self.num_classes)
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                features = self.backbone(images)
                d_total, d_common, _ = self.global_model(features)
                
                pred_common = torch.argmin(d_common, dim=-1)
                pred_full = torch.argmin(d_total, dim=-1)
                
                for i in range(len(labels)):
                    true_cls = labels[i].item()
                    confusion_common[true_cls, pred_common[i].item()] += 1
                    confusion_full[true_cls, pred_full[i].item()] += 1
        
        # 计算每类准确率
        class_acc_common = {}
        class_acc_full = {}
        
        for cls in range(self.num_classes):
            total = confusion_common[cls].sum().item()
            if total > 0:
                class_acc_common[cls] = confusion_common[cls, cls].item() / total
                class_acc_full[cls] = confusion_full[cls, cls].item() / total
            else:
                class_acc_common[cls] = 0
                class_acc_full[cls] = 0
        
        return {
            'confusion_common': confusion_common,
            'confusion_full': confusion_full,
            'class_acc_common': class_acc_common,
            'class_acc_full': class_acc_full,
            'class_names': self.class_names
        }
    
    def get_global_model_state(self) -> Dict:
        """获取全局模型状态"""
        return {
            'model_state': self.global_model.state_dict(),
            'learned_classes': self.learned_classes.copy()
        }
    
    def load_global_model_state(self, state: Dict):
        """加载全局模型状态"""
        self.global_model.load_state_dict(state['model_state'])
        self.learned_classes = state['learned_classes']


# 测试
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    model_config = {
        'backbone': 'resnet18',
        'backbone_pretrained': True,
        'feature_dim': 512,
        'hidden_dim': 256,
        'output_dim': 128,
        'num_layers': 3,
        'dropout': 0.1,
        'sigma_min': 0.1,
        'sigma_max': 2.0,
        'lambda_ind': 0.5,
        'temperature': 0.1
    }
    
    server = FedPCIServer(
        num_classes=10,
        class_names=class_names,
        model_config=model_config,
        device=device
    )
    
    print(f"Server created on {device}")
    print(f"Number of classes: {server.num_classes}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in server.global_model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # 测试获取客户端配置
    config = server.get_client_config(client_id=0, client_classes=[0, 1, 2])
    print(f"\nClient config keys: {list(config.keys())}")
    print(f"Common params classes: {list(config['common_params'].keys())}")