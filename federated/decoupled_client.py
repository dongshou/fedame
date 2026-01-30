"""
联邦学习客户端模块（解耦路由器版本）
使用N个独立的二分类Router
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import copy

from models import (
    create_backbone,
    DecoupledRouterPool,
    ContrastiveLoss,
    ExpertPool
)


class DecoupledClient:
    """
    使用解耦路由器的联邦客户端
    
    核心特点：
    - 持有N个独立的BinaryRouter（每个类一个）
    - 使用对比学习 + Hard Negative Mining训练
    - 使用视觉原型补偿缺失的正样本
    """
    
    def __init__(
        self,
        client_id: int,
        num_classes: int,
        backbone: nn.Module,
        router_pool: DecoupledRouterPool,
        expert_pool: ExpertPool,
        class_anchors: torch.Tensor,
        device: str = "cuda",
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        margin: float = 1.0,
        hard_negative_k: int = 10
    ):
        self.client_id = client_id
        self.num_classes = num_classes
        self.device = device
        
        # 模型组件
        self.backbone = backbone.to(device)
        self.backbone.eval()  # backbone始终冻结
        
        self.router_pool = router_pool.to(device)
        self.expert_pool = expert_pool.to(device)
        
        # 锚点
        self.class_anchors = class_anchors.to(device)
        self.router_pool.set_class_anchors(self.class_anchors)
        
        # 本地数据信息
        self.local_classes: List[int] = []
        self.local_experts: List[int] = []
        self.class_to_expert: Dict[int, int] = {}
        
        # 本地视觉原型（从本地数据提取）
        self.local_prototypes: Dict[int, torch.Tensor] = {}
        self.local_prototype_counts: Dict[int, int] = {}
        
        # 全局视觉原型（从服务端获取）
        self.global_prototypes: Optional[torch.Tensor] = None
        self.global_prototype_counts: Dict[int, int] = {}
        
        # 损失函数
        self.contrastive_loss = ContrastiveLoss(
            margin=margin,
            hard_negative_k=hard_negative_k
        )
        
        # 优化器参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = None
        
        # 训练统计
        self.train_stats: Dict[int, Dict] = {}  # {class_id: {pos_count, neg_count}}
    
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
        self.class_to_expert = class_to_expert
        
        # 更新专家池的映射
        for cls, exp in class_to_expert.items():
            if str(exp) not in self.expert_pool.experts:
                self.expert_pool.add_expert(exp)
                self.expert_pool.experts[str(exp)].to(self.device)
            self.expert_pool.assign_class_to_expert(cls, exp)
    
    def extract_prototypes_from_data(self, train_loader: DataLoader):
        """
        从本地数据中提取视觉原型
        
        Args:
            train_loader: 训练数据加载器
        """
        self.backbone.eval()
        
        # 收集每个类的特征
        class_features: Dict[int, List[torch.Tensor]] = {
            cls: [] for cls in self.local_classes
        }
        
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                features = self.backbone(images)
                
                for i, label in enumerate(labels):
                    cls = label.item()
                    if cls in class_features:
                        class_features[cls].append(features[i].cpu())
        
        # 计算每个类的原型（均值）
        for cls in self.local_classes:
            if len(class_features[cls]) > 0:
                feats = torch.stack(class_features[cls], dim=0)
                prototype = feats.mean(dim=0).to(self.device)
                
                self.local_prototypes[cls] = prototype
                self.local_prototype_counts[cls] = len(class_features[cls])
    
    def set_global_prototypes(
        self, 
        prototypes: torch.Tensor,
        counts: Dict[int, int]
    ):
        """
        设置全局视觉原型（从服务端获取）
        
        Args:
            prototypes: 全局原型 [num_classes, feature_dim]
            counts: 每个类的样本数
        """
        self.global_prototypes = prototypes.to(self.device)
        self.global_prototype_counts = counts
        
        # 同步到router_pool
        self.router_pool.set_visual_prototypes(prototypes, counts)
    
    def _init_optimizer(self):
        """初始化优化器"""
        params = list(self.router_pool.parameters())
        
        # 也可以加入专家参数
        for exp_id in self.local_experts:
            expert = self.expert_pool.get_expert(exp_id)
            params.extend(expert.parameters())
        
        self.optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
    
    def _collect_features_by_class(
        self, 
        train_loader: DataLoader
    ) -> Dict[int, torch.Tensor]:
        """
        收集并按类别组织特征
        
        Returns:
            class_features: {class_id: features [N, dim]}
        """
        self.backbone.eval()
        
        class_features: Dict[int, List[torch.Tensor]] = {
            cls: [] for cls in range(self.num_classes)
        }
        
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                features = self.backbone(images)
                
                for i, label in enumerate(labels):
                    cls = label.item()
                    class_features[cls].append(features[i])
        
        # 转换为tensor
        result = {}
        for cls, feats in class_features.items():
            if len(feats) > 0:
                result[cls] = torch.stack(feats, dim=0)
        
        return result
    
    def train_routers(
        self,
        train_loader: DataLoader,
        num_epochs: int = 5,
        expert_loss_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        训练所有Router和Expert
        
        核心逻辑：
        - Router训练：对比学习
          - 正样本：本地类i的真实特征（如果有），否则用全局原型
          - 负样本：本地所有非i类的特征
          - 使用Hard Negative Mining
        - Expert训练：分类损失
          - 用真实标签对应的Expert处理特征
          - 计算交叉熵损失
        
        Args:
            train_loader: 训练数据加载器
            num_epochs: 本地训练轮数
            expert_loss_weight: Expert分类损失的权重
        
        Returns:
            metrics: 训练指标
        """
        self._init_optimizer()
        self.router_pool.train()
        self.expert_pool.train()  # Expert也需要训练
        
        total_loss = 0.0
        total_router_loss = 0.0
        total_expert_loss = 0.0
        total_batches = 0
        
        # 初始化训练统计
        self.train_stats = {i: {'pos_count': 0, 'neg_count': 0} for i in range(self.num_classes)}
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 1. 提取backbone特征
                with torch.no_grad():
                    features = self.backbone(images)
                
                # 2. 按类别组织特征
                batch_class_features: Dict[int, torch.Tensor] = {}
                for cls in range(self.num_classes):
                    mask = (labels == cls)
                    if mask.sum() > 0:
                        batch_class_features[cls] = features[mask]
                
                # 3. 训练每个Router（对比损失）
                router_loss = torch.tensor(0.0, device=self.device)
                
                for class_id in range(self.num_classes):
                    router = self.router_pool.get_router(class_id)
                    anchor = self.class_anchors[class_id]
                    
                    # 正样本
                    if class_id in batch_class_features:
                        pos_features = batch_class_features[class_id]
                        pos_prototype = None
                        self.train_stats[class_id]['pos_count'] += len(pos_features)
                    else:
                        pos_features = None
                        # 使用全局原型作为伪正样本
                        if (self.global_prototypes is not None and 
                            self.global_prototype_counts.get(class_id, 0) > 0):
                            pos_prototype = self.global_prototypes[class_id]
                        else:
                            pos_prototype = None
                    
                    # 负样本：所有非该类的特征
                    neg_features_list = []
                    for other_cls, other_feats in batch_class_features.items():
                        if other_cls != class_id:
                            neg_features_list.append(other_feats)
                    
                    if len(neg_features_list) > 0:
                        neg_features = torch.cat(neg_features_list, dim=0)
                        self.train_stats[class_id]['neg_count'] += len(neg_features)
                    else:
                        neg_features = None
                    
                    # 计算Router对比损失
                    loss, _ = self.contrastive_loss(
                        router=router,
                        anchor=anchor,
                        pos_features=pos_features,
                        neg_features=neg_features,
                        pos_prototype=pos_prototype
                    )
                    
                    router_loss = router_loss + loss
                
                # 4. 计算Expert分类损失
                # 关键：用真实标签对应的Expert，而不是Router预测的Expert
                # 这样Expert可以正确学习，不受Router错误的影响
                true_expert_ids = torch.tensor(
                    [self.class_to_expert.get(l.item(), 0) for l in labels],
                    device=self.device
                )
                
                # Expert处理特征并计算分类logits
                cls_logits, _ = self.expert_pool(
                    features, true_expert_ids, self.class_anchors
                )
                expert_loss = F.cross_entropy(cls_logits, labels)
                
                # 5. 总损失
                batch_loss = router_loss + expert_loss_weight * expert_loss
                
                # 6. 反向传播
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # 对Router和Expert都进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.router_pool.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.expert_pool.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
                total_router_loss += router_loss.item()
                total_expert_loss += expert_loss.item()
                total_batches += 1
            
            total_loss += epoch_loss
        
        # 计算平均损失
        avg_loss = total_loss / max(total_batches, 1)
        avg_router_loss = total_router_loss / max(total_batches, 1)
        avg_expert_loss = total_expert_loss / max(total_batches, 1)
        
        return {
            'loss': avg_loss,
            'router_loss': avg_router_loss,
            'expert_loss': avg_expert_loss,
            'num_epochs': num_epochs,
            'num_batches': total_batches
        }
    
    def get_router_updates(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        获取所有Router的参数更新
        
        Returns:
            updates: {class_id: {param_name: param_value}}
        """
        updates = {}
        for class_id in range(self.num_classes):
            updates[class_id] = self.router_pool.get_router_params(class_id)
        return updates
    
    def get_prototype_updates(self) -> Dict[int, Dict]:
        """
        获取本地原型更新
        
        Returns:
            updates: {class_id: {'prototype': tensor, 'count': int}}
        """
        updates = {}
        for cls, proto in self.local_prototypes.items():
            updates[cls] = {
                'prototype': proto.cpu(),
                'count': self.local_prototype_counts.get(cls, 0)
            }
        return updates
    
    def get_train_stats(self) -> Dict[int, Dict]:
        """获取训练统计信息"""
        return self.train_stats
    
    def get_expert_updates(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        获取本地Expert的参数更新
        
        Returns:
            updates: {expert_id: state_dict}
        """
        updates = {}
        for exp_id in self.local_experts:
            expert = self.expert_pool.get_expert(exp_id)
            updates[exp_id] = {
                name: param.data.clone().cpu()
                for name, param in expert.named_parameters()
            }
        return updates
    
    def load_global_routers(
        self,
        global_router_params: Dict[int, Dict[str, torch.Tensor]]
    ):
        """
        加载全局Router参数
        
        Args:
            global_router_params: {class_id: {param_name: param_value}}
        """
        for class_id, params in global_router_params.items():
            self.router_pool.set_router_params(class_id, params)
    
    def load_global_experts(
        self,
        global_expert_states: Dict[int, Dict[str, torch.Tensor]]
    ):
        """
        加载全局专家参数
        
        Args:
            global_expert_states: {expert_id: state_dict}
        """
        for exp_id, state in global_expert_states.items():
            if str(exp_id) not in self.expert_pool.experts:
                self.expert_pool.add_expert(exp_id)
                self.expert_pool.experts[str(exp_id)].to(self.device)
            self.expert_pool.get_expert(exp_id).load_state_dict(state)
    
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
        self.backbone.eval()
        self.router_pool.eval()
        self.expert_pool.eval()
        
        total_correct = 0
        total_samples = 0
        routing_correct = 0
        
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                features = self.backbone(images)
                
                # 路由决策
                routed_classes, distances, similarities = self.router_pool(features)
                
                # 获取目标专家
                target_experts = torch.tensor(
                    [self.class_to_expert.get(l.item(), 0) for l in labels],
                    device=self.device
                )
                
                # 路由准确率
                routing_match = (routed_classes == labels)
                routing_correct += routing_match.sum().item()
                
                # 使用路由的专家进行分类
                cls_logits, _ = self.expert_pool(
                    features, routed_classes, self.class_anchors
                )
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
                    total_samples += 1
                    
                    if predicted[i].item() == label:
                        class_correct[label] += 1
                        total_correct += 1
        
        metrics = {
            'accuracy': total_correct / max(total_samples, 1) * 100,
            'routing_accuracy': routing_correct / max(total_samples, 1) * 100,
            'total_samples': total_samples
        }
        
        for cls in class_total:
            metrics[f'class_{cls}_acc'] = (
                class_correct[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
        
        return metrics


# 测试
if __name__ == "__main__":
    from models import create_backbone
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10
    
    # 创建组件
    backbone = create_backbone("resnet18", pretrained=True, frozen=True)
    router_pool = DecoupledRouterPool(
        num_classes=num_classes,
        input_dim=512,
        hidden_dim=256,
        output_dim=512
    )
    expert_pool = ExpertPool(
        input_dim=512,
        hidden_dim=256,
        output_dim=512,
        num_initial_experts=num_classes
    )
    class_anchors = F.normalize(torch.randn(num_classes, 512), dim=-1)
    
    # 创建客户端
    client = DecoupledClient(
        client_id=0,
        num_classes=num_classes,
        backbone=backbone,
        router_pool=router_pool,
        expert_pool=expert_pool,
        class_anchors=class_anchors,
        device=device
    )
    
    # 设置本地数据信息
    client.setup_local_data(
        local_classes=[0, 1, 2],
        local_experts=[0, 1, 2],
        class_to_expert={i: i for i in range(num_classes)}
    )
    
    print(f"Client {client.client_id} created")
    print(f"Local classes: {client.local_classes}")
    print(f"Device: {device}")
    
    # 统计参数量
    router_params = sum(p.numel() for p in client.router_pool.parameters())
    print(f"Router pool parameters: {router_params:,}")