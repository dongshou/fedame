"""
FedPCI 联邦学习客户端 (重构版)

核心特点：
- 双分支架构：g_common (聚合) + g_ind (不聚合)
- 两个分类头：classifier_common (聚合) + classifier_full (不聚合)
- 可学习原型：选择性聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import copy

from models.fedpci_model import FedPCIModel
from losses_fedpci import FedPCILoss


class FedPCIClient:
    """
    FedPCI 联邦客户端 (重构版)
    """
    
    def __init__(
        self,
        client_id: int,
        num_classes: int,
        backbone: nn.Module,
        model: FedPCIModel,
        device: str = "cuda",
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        # 损失权重
        lambda_local_align: float = 0.5,
        lambda_global_align: float = 0.3,
        lambda_proto_contrast: float = 0.5,
        temperature: float = 0.1
    ):
        self.client_id = client_id
        self.num_classes = num_classes
        self.device = device
        
        # 模型组件
        self.backbone = backbone.to(device)
        self.backbone.eval()  # backbone 始终冻结
        
        self.model = model.to(device)
        
        # 本地数据信息
        self.local_classes: List[int] = []
        
        # 保存全局原型（用于对齐和对比损失）
        self.global_prototypes: Optional[torch.Tensor] = None
        
        # 损失函数
        self.criterion = FedPCILoss(
            lambda_local_align=lambda_local_align,
            lambda_global_align=lambda_global_align,
            lambda_proto_contrast=lambda_proto_contrast,
            temperature=temperature
        )
        
        # 优化器参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = None
    
    def setup_local_data(self, local_classes: List[int]):
        """设置本地拥有的类别"""
        self.local_classes = local_classes
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 聚合的参数：g_common, classifier_common, prototypes
        # 不聚合的参数：g_ind, classifier_full
        # 全部参与本地训练
        params = list(self.model.parameters())
        
        self.optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
    
    def load_global_params(self, global_params: Dict[str, any]):
        """
        加载全局参数
        
        Args:
            global_params: 包含 g_common, classifier_common, prototypes
        """
        self.model.set_aggregatable_params(global_params)
        
        # 保存全局原型副本（用于损失计算）
        if 'prototypes' in global_params:
            self.global_prototypes = global_params['prototypes'].clone().to(self.device)
    
    def get_update_params(self) -> Dict[str, any]:
        """
        获取需要上传的参数更新
        
        Returns:
            dict containing:
                - g_common: 共性分支参数
                - classifier_common: 共性分类头参数
                - prototypes: 本地原型参数
                - local_classes: 本地拥有的类别
        """
        return {
            'g_common': self.model.get_common_branch_params(),
            'classifier_common': self.model.get_classifier_common_params(),
            'prototypes': self.model.get_prototype_params(),
            'local_classes': self.local_classes.copy()
        }
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 5
    ) -> Dict[str, float]:
        """
        本地训练
        
        Args:
            train_loader: 训练数据加载器
            num_epochs: 本地训练轮数
        
        Returns:
            metrics: 训练指标
        """
        self._init_optimizer()
        self.model.train()
        
        total_loss = 0.0
        total_cls_common = 0.0
        total_cls_full = 0.0
        total_local_align = 0.0
        total_global_align = 0.0
        total_proto_contrast = 0.0
        total_batches = 0
        
        for epoch in range(num_epochs):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 1. 提取 backbone 特征
                with torch.no_grad():
                    features = self.backbone(images)
                
                # 2. 前向传播
                output = self.model(features, return_features=True)
                
                # 3. 获取原型
                local_prototypes = self.model.get_prototypes()  # [num_classes, d]
                
                # 如果没有全局原型，用本地原型代替（第一轮）
                if self.global_prototypes is None:
                    global_prototypes = local_prototypes.detach()
                else:
                    global_prototypes = self.global_prototypes
                
                # 4. 计算损失
                losses = self.criterion(
                    logits_common=output['logits_common'],
                    logits_full=output['logits_full'],
                    targets=labels,
                    z_common=output['z_common'],
                    local_prototypes=local_prototypes,
                    global_prototypes=global_prototypes,
                    local_classes=self.local_classes
                )
                
                # 5. 反向传播
                self.optimizer.zero_grad()
                losses['total'].backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 记录损失
                total_loss += losses['total'].item()
                total_cls_common += losses['cls_common'].item()
                total_cls_full += losses['cls_full'].item()
                total_local_align += losses['local_align'].item()
                total_global_align += losses['global_align'].item()
                total_proto_contrast += losses['proto_contrast'].item()
                total_batches += 1
        
        # 计算平均损失
        n = max(total_batches, 1)
        return {
            'loss': total_loss / n,
            'cls_common_loss': total_cls_common / n,
            'cls_full_loss': total_cls_full / n,
            'local_align_loss': total_local_align / n,
            'global_align_loss': total_global_align / n,
            'proto_contrast_loss': total_proto_contrast / n,
            'num_epochs': num_epochs,
            'num_batches': total_batches
        }
    
    def evaluate(
        self,
        test_loader: DataLoader,
        classes_to_eval: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            classes_to_eval: 要评估的类别
        
        Returns:
            metrics: 评估指标
        """
        self.backbone.eval()
        self.model.eval()
        
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
                output = self.model(features)
                
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
        
        # 个性化增益 = Full - Common
        metrics['personalization_gain'] = metrics['accuracy_full'] - metrics['accuracy_common']
        
        return metrics


if __name__ == "__main__":
    from models.backbone import create_backbone
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10
    
    # 创建组件
    backbone = create_backbone("resnet18", pretrained=True, frozen=True)
    model = FedPCIModel(
        num_classes=num_classes,
        input_dim=512,
        hidden_dim=256,
        output_dim=128
    )
    
    # 创建客户端
    client = FedPCIClient(
        client_id=0,
        num_classes=num_classes,
        backbone=backbone,
        model=model,
        device=device
    )
    
    # 设置本地数据信息
    client.setup_local_data(local_classes=[0, 1, 2])
    
    print(f"Client {client.client_id} created")
    print(f"Local classes: {client.local_classes}")
    print(f"Device: {device}")
    
    # 测试参数获取
    update_params = client.get_update_params()
    print(f"\nUpdate params keys: {list(update_params.keys())}")
    print(f"g_common params: {len(update_params['g_common'])} tensors")
    print(f"classifier_common params: {list(update_params['classifier_common'].keys())}")
    print(f"prototypes shape: {update_params['prototypes'].shape}")