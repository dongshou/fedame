"""
FedPCI 联邦学习服务端 (重构版)

核心特点：
- 管理全局模型
- 聚合规则：
  - g_common: 聚合
  - g_ind: 不聚合（客户端本地保留）
  - classifier_common: 聚合
  - classifier_full: 不聚合
  - prototypes: 选择性聚合（仅拥有该类的客户端参与）
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
    FedPCI 联邦服务端 (重构版)
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        model_config: Dict,
        device: str = "cuda",
        momentum: float = 0.5
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.momentum = momentum  # 动量系数
        
        # 初始化全局模型
        self._init_global_model(model_config)
        
        # 已学习的类别
        self.learned_classes: List[int] = []
        
        # 记录每个类有多少客户端拥有
        self.class_client_count: Dict[int, int] = {c: 0 for c in range(num_classes)}
    
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
            dropout=config.get('dropout', 0.1)
        ).to(self.device)
    
    def get_global_params(self) -> Dict[str, any]:
        """
        获取全局参数（发送给客户端）
        
        Returns:
            dict containing:
                - g_common: 共性分支参数
                - classifier_common: 共性分类头参数
                - prototypes: 原型参数
        """
        return self.global_model.get_aggregatable_params()
    
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
    
    def aggregate(
        self,
        client_updates: List[Dict[str, any]],
        verbose: bool = False
    ):
        """
        聚合客户端更新
        
        聚合规则：
        - g_common: FedAvg + 动量
        - classifier_common: FedAvg + 动量
        - prototypes: 选择性聚合（仅拥有该类的客户端参与）+ 动量
        
        Args:
            client_updates: 客户端更新列表，每个元素包含:
                - g_common: 共性分支参数
                - classifier_common: 共性分类头参数
                - prototypes: 原型参数
                - local_classes: 本地拥有的类别
            verbose: 是否打印详细日志
        """
        if len(client_updates) == 0:
            return
        
        num_clients = len(client_updates)
        
        # ============ 1. 聚合 g_common ============
        old_g_common = self.global_model.get_common_branch_params()
        new_g_common = self._aggregate_params(
            [u['g_common'] for u in client_updates],
            old_g_common
        )
        self.global_model.set_common_branch_params(new_g_common)
        
        # ============ 2. 聚合 classifier_common ============
        old_classifier = self.global_model.get_classifier_common_params()
        new_classifier = self._aggregate_params(
            [u['classifier_common'] for u in client_updates],
            old_classifier
        )
        self.global_model.set_classifier_common_params(new_classifier)
        
        # ============ 3. 选择性聚合 prototypes ============
        old_prototypes = self.global_model.get_prototype_params()  # [num_classes, d]
        new_prototypes = old_prototypes.clone()
        
        # 统计每个类有哪些客户端
        class_updates: Dict[int, List[torch.Tensor]] = {c: [] for c in range(self.num_classes)}
        
        for update in client_updates:
            local_classes = update['local_classes']
            client_protos = update['prototypes']  # [num_classes, d]
            
            for c in local_classes:
                class_updates[c].append(client_protos[c])
        
        # 对每个类聚合原型
        aggregation_info = {}
        for c in range(self.num_classes):
            if len(class_updates[c]) == 0:
                continue
            
            # 简单平均
            stacked = torch.stack(class_updates[c], dim=0)  # [n, d]
            avg_proto = stacked.mean(dim=0)  # [d]
            
            # 动量更新
            new_prototypes[c] = self.momentum * old_prototypes[c] + (1 - self.momentum) * avg_proto
            
            self.class_client_count[c] = len(class_updates[c])
            aggregation_info[c] = len(class_updates[c])
        
        self.global_model.set_prototype_params(new_prototypes)
        
        if verbose:
            self._print_aggregation_info(num_clients, aggregation_info)
    
    def _aggregate_params(
        self,
        client_params_list: List[Dict[str, torch.Tensor]],
        old_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        聚合参数（FedAvg + 动量）
        
        Args:
            client_params_list: 客户端参数列表
            old_params: 旧的全局参数
        
        Returns:
            聚合后的参数
        """
        num_clients = len(client_params_list)
        
        # FedAvg
        aggregated = {}
        for key in old_params.keys():
            stacked = torch.stack([p[key] for p in client_params_list], dim=0)
            aggregated[key] = stacked.mean(dim=0)
        
        # 动量更新
        new_params = {}
        for key in old_params.keys():
            new_params[key] = self.momentum * old_params[key] + (1 - self.momentum) * aggregated[key]
        
        return new_params
    
    def _print_aggregation_info(self, num_clients: int, aggregation_info: Dict[int, int]):
        """打印聚合信息"""
        print(f"\n         📊 Aggregation Summary:")
        print(f"         ├─ Total clients: {num_clients}")
        print(f"         ├─ Prototype aggregation (selective):")
        for c, count in sorted(aggregation_info.items()):
            if count > 0:
                print(f"         │  Class {c} ({self.class_names[c]:10s}): {count} clients")
        print(f"         └─ Momentum: {self.momentum}")
    
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
                
                # 前向传播
                output = self.global_model(features)
                
                # 预测
                pred_common = torch.argmax(output['logits_common'], dim=-1)
                pred_full = torch.argmax(output['logits_full'], dim=-1)
                
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
        
        # 每类准确率
        for cls in class_total:
            metrics[f'class_{cls}_acc_common'] = (
                class_correct_common[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
            metrics[f'class_{cls}_acc_full'] = (
                class_correct_full[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
        
        return metrics
    
    def diagnose(self, test_loader: DataLoader) -> Dict:
        """
        诊断模型性能
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
                output = self.global_model(features)
                
                pred_common = torch.argmax(output['logits_common'], dim=-1)
                pred_full = torch.argmax(output['logits_full'], dim=-1)
                
                for i in range(len(labels)):
                    true_cls = labels[i].item()
                    confusion_common[true_cls, pred_common[i].item()] += 1
                    confusion_full[true_cls, pred_full[i].item()] += 1
        
        return {
            'confusion_common': confusion_common,
            'confusion_full': confusion_full,
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
        'dropout': 0.1
    }
    
    server = FedPCIServer(
        num_classes=10,
        class_names=class_names,
        model_config=model_config,
        device=device,
        momentum=0.9
    )
    
    print(f"Server created on {device}")
    print(f"Number of classes: {server.num_classes}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in server.global_model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # 测试获取全局参数
    global_params = server.get_global_params()
    print(f"\nGlobal params keys: {list(global_params.keys())}")
    print(f"g_common: {len(global_params['g_common'])} tensors")
    print(f"classifier_common: {list(global_params['classifier_common'].keys())}")
    print(f"prototypes shape: {global_params['prototypes'].shape}")
    
    # 模拟聚合
    print("\n--- 模拟聚合测试 ---")
    
    # 模拟3个客户端的更新
    client_updates = []
    for i in range(3):
        local_classes = [i, i+1, i+2]  # 每个客户端3个类
        update = {
            'g_common': {k: v + torch.randn_like(v) * 0.1 for k, v in global_params['g_common'].items()},
            'classifier_common': {k: v + torch.randn_like(v) * 0.1 for k, v in global_params['classifier_common'].items()},
            'prototypes': global_params['prototypes'] + torch.randn_like(global_params['prototypes']) * 0.1,
            'local_classes': local_classes
        }
        client_updates.append(update)
    
    server.aggregate(client_updates, verbose=True)
    print("\nAggregation completed!")