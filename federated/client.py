"""
联邦学习客户端模块
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
    AnchorBasedRouter,
    ExpertPool,
    DistributionPool
)
from losses import FedAMELoss


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
        
        # 专家到簇的映射（用于路由损失）
        self.expert_to_cluster: Dict[int, int] = {}
    
    def setup_local_data(
        self,
        local_classes: List[int],
        local_experts: List[int],
        class_to_expert: Dict[int, int],
        expert_to_cluster: Optional[Dict[int, int]] = None
    ):
        """
        设置本地数据信息
        
        Args:
            local_classes: 本地拥有的类别
            local_experts: 本地需要的专家
            class_to_expert: 类别到专家的映射
            expert_to_cluster: 专家到簇的映射
        """
        self.local_classes = local_classes
        self.local_experts = local_experts
        
        # 保存专家到簇的映射
        if expert_to_cluster is not None:
            self.expert_to_cluster = expert_to_cluster
        
        # 更新专家池的映射（确保专家存在）
        for cls, exp in class_to_expert.items():
            if str(exp) not in self.expert_pool.experts:
                self.expert_pool.add_expert(exp)
                self.expert_pool.experts[str(exp)].to(self.device)
            self.expert_pool.assign_class_to_expert(cls, exp)
        
        # 注意：分布的初始化移到 init_distributions_from_data
        
    
    def init_distributions_from_data(self, train_loader: DataLoader):
        """
        用真实 backbone 特征均值初始化分布
        
        Args:
            train_loader: 训练数据加载器
        """
        self.backbone.eval()
        
        # 收集每个类的特征
        class_features = {cls: [] for cls in self.local_classes}
        
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                features = self.backbone(images)  # [B, 512]
                
                for i, label in enumerate(labels):
                    cls = label.item()
                    if cls in class_features:
                        class_features[cls].append(features[i].cpu())
        
        # 计算均值并初始化分布
        for cls in self.local_classes:
            if len(class_features[cls]) > 0:
                feats = torch.stack(class_features[cls], dim=0)  # [N, 512]
                mean = feats.mean(dim=0).to(self.device)  # [512]
                
                self.distribution_pool.add_class(cls, init_mean=mean)
                self.distribution_pool.get_distribution(cls).update_sample_count(len(feats))
        # 重新初始化优化器（包含新的分布参数）
        self._init_optimizer()
    
    def _init_optimizer(self):
        """初始化优化器"""
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
        
        for param in self.old_router.parameters():
            param.requires_grad = False
        for param in self.old_expert_pool.parameters():
            param.requires_grad = False
    
    def train_router_only(
        self,
        train_loader: DataLoader,
        num_pseudo_samples: int = 50,
        lambda_pseudo: float = 0.2
    ) -> Dict[str, float]:
        """
        阶段1: 只训练路由器（冻结专家）
        """
        self.router.train()
        self.expert_pool.eval()
        
        total_loss = 0.0
        total_routing_correct = 0
        total_routing_samples = 0
        
        loss_components = {'route': 0.0, 'contrast': 0.0, 'pseudo': 0.0}
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            for param in self.expert_pool.parameters():
                param.requires_grad = False
            for param in self.router.parameters():
                param.requires_grad = True
            
            with torch.no_grad():
                backbone_features = self.backbone(images)
            
            expert_ids, routing_probs, projected = self.router(backbone_features)
            
            target_experts = torch.tensor(
                [self.expert_pool.get_expert_for_class(l.item()) for l in labels],
                device=self.device
            )
            target_clusters = torch.tensor(
                [self.expert_to_cluster.get(exp.item(), 0) for exp in target_experts],
                device=self.device
            )
            
            # 路由损失
            log_probs = torch.log(routing_probs + 1e-10)
            L_route = F.nll_loss(log_probs, target_clusters)
            
            # 对比损失
            L_contrast = self.criterion.contrast_loss(
                projected, labels, self.class_anchors
            )
            
            self.optimizer.zero_grad()
            router_loss = L_route + 0.3 * L_contrast
            router_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += router_loss.item()
            loss_components['route'] += L_route.item()
            loss_components['contrast'] += L_contrast.item()
            
            routed_clusters = torch.argmax(routing_probs, dim=-1)
            routing_correct = (routed_clusters == target_clusters).sum().item()
            total_routing_correct += routing_correct
            total_routing_samples += len(labels)
        
        # 伪样本训练路由器
        non_local_classes = [
            c for c in range(len(self.class_anchors))
            if c not in self.local_classes and self.distribution_pool.has_class(c)
        ]
        
        if len(non_local_classes) > 0 and num_pseudo_samples > 0:
            for cls in non_local_classes:
                try:
                    dist = self.distribution_pool.get_distribution(cls)
                    if dist.sample_count < 10:
                        continue
                    
                    # 采样（backbone 空间）
                    z_pseudo = dist.sample(num_pseudo_samples)
                    
                    if torch.isnan(z_pseudo).any() or torch.isinf(z_pseudo).any():
                        continue
                    
                    # 通过 Router 前向传播
                    _, routing_probs, _ = self.router(z_pseudo)
                    
                    target_expert = self.expert_pool.get_expert_for_class(cls)
                    if target_expert not in self.expert_to_cluster:
                        continue
                    target_cluster = self.expert_to_cluster[target_expert]
                    target_clusters = torch.full(
                        (num_pseudo_samples,), target_cluster,
                        dtype=torch.long, device=self.device
                    )
                    
                    log_probs = torch.log(routing_probs + 1e-10)
                    L_pseudo_route = F.nll_loss(log_probs, target_clusters)
                    
                    if torch.isnan(L_pseudo_route) or torch.isinf(L_pseudo_route):
                        continue
                    
                    self.optimizer.zero_grad()
                    (lambda_pseudo * L_pseudo_route).backward()
                    torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    loss_components['pseudo'] += L_pseudo_route.item()
                    
                except Exception as e:
                    continue
        
        num_batches = len(train_loader)
        pseudo_batches = len(non_local_classes) if len(non_local_classes) > 0 else 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'routing_accuracy': total_routing_correct / max(total_routing_samples, 1) * 100,
        }
        for key in loss_components:
            if key == 'pseudo':
                metrics[f'loss_{key}'] = loss_components[key] / pseudo_batches
            else:
                metrics[f'loss_{key}'] = loss_components[key] / num_batches
        
        return metrics
    
    def train_expert_only(
        self,
        train_loader: DataLoader,
        old_classes: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        阶段2: 只训练专家（冻结路由器）
        """
        self.router.eval()
        self.expert_pool.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        loss_components = {'cls': 0.0, 'forget': 0.0}
        for param in self.router.parameters():
            param.requires_grad = False
        for param in self.expert_pool.parameters():
            param.requires_grad = True
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                backbone_features = self.backbone(images)
                _, _, projected = self.router(backbone_features)
                
            projected = projected.detach().requires_grad_(True)
            target_experts = torch.tensor(
                [self.expert_pool.get_expert_for_class(l.item()) for l in labels],
                device=self.device
            )
            
            cls_logits, expert_features = self.expert_pool(
                projected, target_experts, self.class_anchors
            )
            
            # 分类损失
            L_cls = F.cross_entropy(cls_logits, labels)
            
            # 防遗忘损失
            L_forget = torch.tensor(0.0, device=self.device)
            if old_classes and len(old_classes) > 0 and self.old_router is not None:
                old_samples, old_labels = self.distribution_pool.sample_all(
                    old_classes, num_samples_per_class=2
                )
                if old_samples is not None:
                    with torch.no_grad():
                        _, _, old_proj = self.old_router(old_samples)
                        old_logits = self.old_router.compute_class_logits(old_proj)
                    
                    _, _, new_proj = self.router(old_samples)
                    new_logits = self.router.compute_class_logits(new_proj)
                    
                    L_forget = self.criterion.forget_loss(
                        old_logits, new_logits, old_classes
                    )
            
            self.optimizer.zero_grad()
            expert_loss = L_cls + 0.5 * L_forget
            
            expert_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.expert_pool.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += expert_loss.item()
            loss_components['cls'] += L_cls.item()
            loss_components['forget'] += L_forget.item()
            
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
        
        num_batches = len(train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / max(total_samples, 1) * 100,
        }
        for key in loss_components:
            metrics[f'loss_{key}'] = loss_components[key] / num_batches
        
        return metrics
    
    def train_distribution_only(
        self,
        num_iterations: int = 100,
        num_samples_per_class: int = 16
    ) -> Dict[str, float]:
        """
        阶段3: 训练分布参数（Prompt Tuning 思路）
        
        流程：
        1. 从所有本地类批量采样（backbone 空间）
        2. 通过冻结的 Router + Expert 前向传播
        3. 使用分类损失 + 路由损失
        4. 梯度只更新分布参数 (μ, σ)
        """
        # 冻结 Router 和 Expert
        self.router.eval()
        self.expert_pool.eval()
        for param in self.router.parameters():
            param.requires_grad = False
        for param in self.expert_pool.parameters():
            param.requires_grad = False
        
        # 只训练分布参数
        for param in self.distribution_pool.parameters():
            param.requires_grad = True
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        loss_components = {'cls': 0.0, 'route': 0.0}
        
        for iteration in range(num_iterations):
            # 1. 批量采样（backbone 空间）
            features, labels = self.distribution_pool.sample_all(
                self.local_classes, num_samples_per_class
            )
            
            if features is None:
                continue
            
            # 数值检查
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            
            # 2. Router 前向传播
            expert_ids, routing_probs, projected = self.router(features)
            
            # 目标路由
            target_experts = torch.tensor(
                [self.expert_pool.get_expert_for_class(l.item()) for l in labels],
                device=self.device
            )
            target_clusters = torch.tensor(
                [self.expert_to_cluster.get(exp.item(), 0) for exp in target_experts],
                device=self.device
            )
            
            # 路由损失
            L_route = F.cross_entropy(
                torch.log(routing_probs + 1e-10), target_clusters
            )
            
            # 3. Expert 前向传播
            cls_logits, _ = self.expert_pool(projected, target_experts, self.class_anchors)
            
            # 分类损失
            L_cls = F.cross_entropy(cls_logits, labels)
            
            # 4. 总损失，更新分布参数
            loss = L_cls + 0.5 * L_route
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.distribution_pool.parameters(),
                max_norm=1.0
            )
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            loss_components['cls'] += L_cls.item()
            loss_components['route'] += L_route.item()
            
            _, predicted = cls_logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += len(labels)
        
        metrics = {
            'loss': total_loss / max(num_iterations, 1),
            'accuracy': total_correct / max(total_samples, 1) * 100,
            'loss_cls': loss_components['cls'] / max(num_iterations, 1),
            'loss_route': loss_components['route'] / max(num_iterations, 1),
        }
        
        return metrics
    
    def get_model_updates(self) -> Dict[str, torch.Tensor]:
        """获取模型更新（用于联邦聚合）"""
        updates = {}
        
        for name, param in self.router.named_parameters():
            updates[f'router.{name}'] = param.data.clone()
        
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
        """加载全局模型"""
        # 加载路由层
        router_state = {}
        for key, value in global_router_state.items():
            if key.startswith('router.'):
                router_state[key[7:]] = value
            else:
                router_state[key] = value
        self.router.load_state_dict(router_state, strict=False)
        
        # 加载专家
        for exp_id, exp_state in global_expert_states.items():
            if str(exp_id) not in self.expert_pool.experts:
                self.expert_pool.add_expert(exp_id)
                self.expert_pool.experts[str(exp_id)].to(self.device)
            self.expert_pool.get_expert(exp_id).load_state_dict(exp_state)
        
        # 加载分布参数
        if global_distribution_params:
            for cls, params in global_distribution_params.items():
                if not self.distribution_pool.has_class(cls):
                    # 用全局参数的 mean 初始化
                    self.distribution_pool.add_class(cls, init_mean=params['mean'].to(self.device))
                self.distribution_pool.set_class_params(cls, params)
    
    def evaluate(
        self,
        test_loader: DataLoader,
        classes_to_eval: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """评估模型"""
        self.router.eval()
        self.expert_pool.eval()
        
        total_correct = 0
        total_samples = 0
        routing_correct = 0
        routing_samples = 0
        expert_correct_with_gt_routing = 0
        expert_samples = 0
        
        class_correct = {}
        class_total = {}
        class_routing_correct = {}
        class_routing_total = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                backbone_features = self.backbone(images)
                routed_expert_ids, routing_probs, projected = self.router(backbone_features)
                
                target_experts = torch.tensor(
                    [self.expert_pool.get_expert_for_class(l.item()) for l in labels],
                    device=self.device
                )
                target_clusters = torch.tensor(
                    [self.expert_to_cluster.get(exp.item(), 0) for exp in target_experts],
                    device=self.device
                )
                routed_clusters = torch.argmax(routing_probs, dim=-1)
                
                routing_match = (routed_clusters == target_clusters)
                routing_correct += routing_match.sum().item()
                routing_samples += len(labels)
                
                cls_logits_routed, _ = self.expert_pool(
                    projected, routed_expert_ids, self.class_anchors
                )
                _, predicted_routed = cls_logits_routed.max(1)
                
                cls_logits_gt, _ = self.expert_pool(
                    projected, target_experts, self.class_anchors
                )
                _, predicted_gt = cls_logits_gt.max(1)
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    
                    if classes_to_eval and label not in classes_to_eval:
                        continue
                    
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                        class_routing_total[label] = 0
                        class_routing_correct[label] = 0
                    
                    class_total[label] += 1
                    class_routing_total[label] += 1
                    
                    if predicted_routed[i].item() == label:
                        class_correct[label] += 1
                        total_correct += 1
                    
                    if routing_match[i].item():
                        class_routing_correct[label] += 1
                    
                    if predicted_gt[i].item() == label:
                        expert_correct_with_gt_routing += 1
                    
                    total_samples += 1
                    expert_samples += 1
        
        metrics = {
            'accuracy': total_correct / max(total_samples, 1) * 100,
            'routing_accuracy': routing_correct / max(routing_samples, 1) * 100,
            'expert_accuracy_with_gt_routing': expert_correct_with_gt_routing / max(expert_samples, 1) * 100,
            'total_samples': total_samples
        }
        
        for cls in class_total:
            metrics[f'class_{cls}_acc'] = (
                class_correct[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
            metrics[f'class_{cls}_routing_acc'] = (
                class_routing_correct[cls] / class_routing_total[cls] * 100
                if class_routing_total[cls] > 0 else 0
            )
        
        return metrics