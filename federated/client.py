"""
联邦学习客户端模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Tuple
import copy

from ..models import (
    create_backbone,
    AnchorBasedRouter,
    ExpertPool,
    DistributionPool
)
from ..losses import FedAMELoss


class FedAMEClient:
    """
    FedAME客户端
    持有部分专家，但拥有完整的全局锚点
    """
    
    def __init__(
        self,
        client_id: int,
        backbone: nn.Module,
        router: AnchorBasedRouter,
        expert_pool: ExpertPool,
        distribution_pool: DistributionPool,
        class_anchors: torch.Tensor,
        cluster_anchors: torch.Tensor,
        device: str = "cuda",
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4
    ):
        self.client_id = client_id
        self.device = device
        
        # 模型组件
        self.backbone = backbone.to(device)
        self.router = router.to(device)
        self.expert_pool = expert_pool.to(device)
        self.distribution_pool = distribution_pool.to(device)
        
        # 锚点
        self.class_anchors = class_anchors.to(device)
        self.cluster_anchors = cluster_anchors.to(device)
        
        # 设置锚点到路由器
        self.router.set_class_anchors(self.class_anchors)
        
        # 本地数据信息
        self.local_classes: List[int] = []
        self.local_experts: List[int] = []
        
        # 损失函数
        self.criterion = FedAMELoss()
        
        # 优化器（稍后初始化）
        self.optimizer = None
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 旧模型（用于防遗忘）
        self.old_router = None
        self.old_expert_pool = None
    
    def setup_local_data(
        self,
        local_classes: List[int],
        local_experts: List[int],
        class_to_expert: Dict[int, int]
    ):
        """
        设置本地数据信息
        
        Args:
            local_classes: 本地拥有的类别
            local_experts: 本地需要的专家
            class_to_expert: 类别到专家的映射
        """
        self.local_classes = local_classes
        self.local_experts = local_experts
        
        # 更新专家池的映射
        for cls, exp in class_to_expert.items():
            self.expert_pool.assign_class_to_expert(cls, exp)
        
        # 为本地类添加分布
        for cls in local_classes:
            if not self.distribution_pool.has_class(cls):
                self.distribution_pool.add_class(cls)
        
        # 初始化优化器
        self._init_optimizer()
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 收集可训练参数
        params = []
        
        # 路由层参数
        params.extend(self.router.parameters())
        
        # 本地专家参数
        for exp_id in self.local_experts:
            expert = self.expert_pool.get_expert(exp_id)
            params.extend(expert.parameters())
        
        # 分布参数
        for cls in self.local_classes:
            if self.distribution_pool.has_class(cls):
                dist = self.distribution_pool.get_distribution(cls)
                params.extend(dist.parameters())
        
        self.optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
    
    def save_old_model(self):
        """保存旧模型（用于防遗忘）"""
        self.old_router = copy.deepcopy(self.router)
        self.old_expert_pool = copy.deepcopy(self.expert_pool)
        
        # 冻结旧模型
        for param in self.old_router.parameters():
            param.requires_grad = False
        for param in self.old_expert_pool.parameters():
            param.requires_grad = False
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        old_classes: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            old_classes: 旧类别列表（用于防遗忘）
        
        Returns:
            metrics: 训练指标
        """
        self.router.train()
        self.expert_pool.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loss_components = {
            'cls': 0.0, 'route': 0.0, 'contrast': 0.0,
            'forget': 0.0, 'dist': 0.0
        }
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            # 1. Backbone特征提取
            with torch.no_grad():
                backbone_features = self.backbone(images)
            
            # 2. 路由
            expert_ids, routing_probs, projected = self.router(backbone_features)
            
            # 3. 获取目标专家ID（基于标签）
            target_experts = torch.tensor(
                [self.expert_pool.get_expert_for_class(l.item()) for l in labels],
                device=self.device
            )
            
            # 4. 专家处理
            cls_logits, expert_features = self.expert_pool(
                projected, target_experts, self.class_anchors
            )
            
            # 5. 分布采样（可选，用于增强）
            # ...
            
            # 6. 计算防遗忘损失
            old_logits = None
            new_logits_for_old = None
            
            if old_classes and len(old_classes) > 0 and self.old_router is not None:
                # 从旧类分布采样
                old_samples, old_labels = self.distribution_pool.sample_batch(
                    old_classes, num_samples_per_class=2
                )
                
                if old_samples is not None:
                    old_samples = old_samples.to(self.device)
                    
                    # 旧模型输出
                    with torch.no_grad():
                        _, _, old_projected = self.old_router(
                            old_samples, return_projected=True
                        )
                        # 简化：直接使用路由层输出的类logits
                        old_logits = self.old_router.compute_class_logits(old_projected)
                    
                    # 新模型输出
                    _, _, new_projected = self.router(old_samples, return_projected=True)
                    new_logits_for_old = self.router.compute_class_logits(new_projected)
            
            # 7. 收集分布残差
            residuals = []
            for cls in self.local_classes:
                if self.distribution_pool.has_class(cls):
                    dist = self.distribution_pool.get_distribution(cls)
                    residuals.append(dist.residual)
            
            # 8. 计算损失
            losses = self.criterion(
                cls_logits=cls_logits,
                targets=labels,
                routing_probs=routing_probs,
                target_experts=target_experts,
                features=projected,
                all_anchors=self.class_anchors,
                valid_classes=self.local_classes,
                old_logits=old_logits,
                new_logits_for_old=new_logits_for_old,
                old_classes=old_classes,
                residuals=residuals
            )
            
            # 9. 反向传播
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 10. 更新分布的样本计数
            for cls in labels.unique():
                cls = cls.item()
                if self.distribution_pool.has_class(cls):
                    count = (labels == cls).sum().item()
                    self.distribution_pool.get_distribution(cls).update_sample_count(count)
            
            # 统计
            total_loss += losses['total'].item()
            
            # 计算准确率
            valid_mask = torch.tensor(
                [l.item() in self.local_classes for l in labels],
                device=self.device
            )
            if valid_mask.any():
                valid_logits = cls_logits[valid_mask]
                valid_labels = labels[valid_mask]
                _, predicted = valid_logits.max(1)
                total_correct += predicted.eq(valid_labels).sum().item()
                total_samples += valid_mask.sum().item()
            
            for key in loss_components:
                loss_components[key] += losses[key].item()
        
        # 平均
        num_batches = len(train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / max(total_samples, 1) * 100,
        }
        for key in loss_components:
            metrics[f'loss_{key}'] = loss_components[key] / num_batches
        
        return metrics
    
    def get_model_updates(self) -> Dict[str, torch.Tensor]:
        """
        获取模型更新（用于联邦聚合）
        
        Returns:
            updates: {参数名: 参数值}
        """
        updates = {}
        
        # 路由层参数
        for name, param in self.router.named_parameters():
            updates[f'router.{name}'] = param.data.clone()
        
        # 本地专家参数
        for exp_id in self.local_experts:
            expert = self.expert_pool.get_expert(exp_id)
            for name, param in expert.named_parameters():
                updates[f'expert.{exp_id}.{name}'] = param.data.clone()
        
        return updates
    
    def get_distribution_params(self) -> Dict[int, Dict]:
        """获取分布参数"""
        return self.distribution_pool.get_all_params()
    
    def load_global_model(
        self,
        global_router_state: Dict[str, torch.Tensor],
        global_expert_states: Dict[int, Dict[str, torch.Tensor]],
        global_distribution_params: Optional[Dict[int, Dict]] = None
    ):
        """
        加载全局模型
        
        Args:
            global_router_state: 全局路由层参数
            global_expert_states: 全局专家参数 {expert_id: state_dict}
            global_distribution_params: 全局分布参数
        """
        # 加载路由层
        router_state = {}
        for key, value in global_router_state.items():
            if key.startswith('router.'):
                router_state[key[7:]] = value
            else:
                router_state[key] = value
        self.router.load_state_dict(router_state, strict=False)
        
        # 加载专家
        for exp_id in self.local_experts:
            if exp_id in global_expert_states:
                self.expert_pool.get_expert(exp_id).load_state_dict(
                    global_expert_states[exp_id]
                )
        
        # 加载分布参数
        if global_distribution_params:
            for cls in self.local_classes:
                if cls in global_distribution_params:
                    self.distribution_pool.set_class_params(
                        cls, global_distribution_params[cls]
                    )
    
    def evaluate(
        self,
        test_loader: DataLoader,
        classes_to_eval: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            classes_to_eval: 要评估的类别（None表示全部）
        
        Returns:
            metrics: 评估指标
        """
        self.router.eval()
        self.expert_pool.eval()
        
        total_correct = 0
        total_samples = 0
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                backbone_features = self.backbone(images)
                expert_ids, routing_probs, projected = self.router(backbone_features)
                
                # 使用路由选择的专家
                cls_logits, _ = self.expert_pool(
                    projected, expert_ids, self.class_anchors
                )
                
                # 预测
                _, predicted = cls_logits.max(1)
                
                # 统计
                for i in range(len(labels)):
                    label = labels[i].item()
                    
                    if classes_to_eval and label not in classes_to_eval:
                        continue
                    
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1
                        total_correct += 1
                    total_samples += 1
        
        # 计算指标
        metrics = {
            'accuracy': total_correct / max(total_samples, 1) * 100,
            'total_samples': total_samples
        }
        
        # 每个类的准确率
        for cls in class_total:
            metrics[f'class_{cls}_acc'] = (
                class_correct[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
        
        return metrics
