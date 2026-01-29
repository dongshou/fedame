"""
损失函数模块
包含分类损失、路由损失、对比损失、防遗忘损失等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class ClassificationLoss(nn.Module):
    """分类损失"""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_classes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes]
            targets: [B]
            valid_classes: 有效类别列表（忽略其他类的logits）
        """
        if valid_classes is not None:
            # 创建mask，只保留有效类别
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for cls in valid_classes:
                mask[:, cls] = True
            
            # 将无效类别的logits设为很小的值
            logits = logits.clone()
            logits[~mask] = float('-inf')
        
        return self.criterion(logits, targets)


class RoutingLoss(nn.Module):
    """路由损失：确保样本被路由到正确的专家"""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        routing_probs: torch.Tensor,
        target_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            routing_probs: [B, num_experts] 路由概率
            target_experts: [B] 目标专家ID
        """
        # routing_probs已经是softmax后的概率，需要转换为logits
        # 或者直接使用NLLLoss
        log_probs = torch.log(routing_probs + 1e-10)
        return F.nll_loss(log_probs, target_experts)


class GlobalContrastiveLoss(nn.Module):
    """
    全局对比损失
    确保特征在全局锚点空间中正确对齐
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        all_anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: [B, dim] 投影后的特征
            targets: [B] 类别标签
            all_anchors: [num_classes, dim] 所有类的锚点
        """
        # 归一化
        features = F.normalize(features, p=2, dim=-1)
        all_anchors = F.normalize(all_anchors, p=2, dim=-1)
        
        # 计算与所有锚点的相似度
        logits = torch.mm(features, all_anchors.t()) / self.temperature
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, targets)
        
        return loss


class ForgettingLoss(nn.Module):
    """
    防遗忘损失
    基于分布回放的知识蒸馏
    """
    
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        old_logits: torch.Tensor,
        new_logits: torch.Tensor,
        old_classes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Args:
            old_logits: [B, num_classes] 旧模型的logits
            new_logits: [B, num_classes] 新模型的logits
            old_classes: 旧类别列表（只在这些类上计算KL）
        """
        if old_classes is not None and len(old_classes) > 0:
            old_logits = old_logits[:, old_classes]
            new_logits = new_logits[:, old_classes]
        
        # 软化的概率分布
        old_probs = F.softmax(old_logits / self.temperature, dim=-1)
        new_log_probs = F.log_softmax(new_logits / self.temperature, dim=-1)
        
        # KL散度
        loss = F.kl_div(
            new_log_probs,
            old_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return loss


class DistributionRegularizationLoss(nn.Module):
    """
    分布正则化损失
    约束分布残差的范数
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        residuals: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            residuals: 各类别的分布残差列表
        """
        if len(residuals) == 0:
            return torch.tensor(0.0)
        
        loss = 0.0
        for residual in residuals:
            loss = loss + torch.sum(residual ** 2)
        
        return loss / len(residuals)


class FedAMELoss(nn.Module):
    """
    FedAME总损失
    整合所有损失项
    """
    
    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_route: float = 0.5,
        lambda_contrast: float = 0.3,
        lambda_forget: float = 0.5,
        lambda_dist: float = 0.1,
        temperature_contrast: float = 0.1,
        temperature_forget: float = 2.0
    ):
        super().__init__()
        
        self.lambda_cls = lambda_cls
        self.lambda_route = lambda_route
        self.lambda_contrast = lambda_contrast
        self.lambda_forget = lambda_forget
        self.lambda_dist = lambda_dist
        
        self.cls_loss = ClassificationLoss()
        self.route_loss = RoutingLoss()
        self.contrast_loss = GlobalContrastiveLoss(temperature_contrast)
        self.forget_loss = ForgettingLoss(temperature_forget)
        self.dist_loss = DistributionRegularizationLoss()
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        targets: torch.Tensor,
        routing_probs: torch.Tensor,
        target_experts: torch.Tensor,
        features: torch.Tensor,
        all_anchors: torch.Tensor,
        valid_classes: Optional[List[int]] = None,
        old_logits: Optional[torch.Tensor] = None,
        new_logits_for_old: Optional[torch.Tensor] = None,
        old_classes: Optional[List[int]] = None,
        residuals: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Returns:
            loss_dict: 包含各项损失的字典
        """
        losses = {}
        
        # 分类损失
        losses['cls'] = self.cls_loss(cls_logits, targets, valid_classes)
        
        # 路由损失
        losses['route'] = self.route_loss(routing_probs, target_experts)
        
        # 对比损失
        losses['contrast'] = self.contrast_loss(features, targets, all_anchors)
        
        # 防遗忘损失
        if old_logits is not None and new_logits_for_old is not None:
            losses['forget'] = self.forget_loss(
                old_logits, new_logits_for_old, old_classes
            )
        else:
            losses['forget'] = torch.tensor(0.0, device=cls_logits.device)
        
        # 分布正则化损失
        if residuals is not None and len(residuals) > 0:
            losses['dist'] = self.dist_loss(residuals)
        else:
            losses['dist'] = torch.tensor(0.0, device=cls_logits.device)
        
        # 总损失
        losses['total'] = (
            self.lambda_cls * losses['cls'] +
            self.lambda_route * losses['route'] +
            self.lambda_contrast * losses['contrast'] +
            self.lambda_forget * losses['forget'] +
            self.lambda_dist * losses['dist']
        )
        
        return losses


# 测试
if __name__ == "__main__":
    # 测试各个损失函数
    batch_size = 8
    num_classes = 10
    num_experts = 2
    dim = 512
    
    # 模拟数据
    cls_logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    routing_probs = F.softmax(torch.randn(batch_size, num_experts), dim=-1)
    target_experts = torch.randint(0, num_experts, (batch_size,))
    features = F.normalize(torch.randn(batch_size, dim), dim=-1)
    all_anchors = F.normalize(torch.randn(num_classes, dim), dim=-1)
    
    # 创建总损失
    criterion = FedAMELoss()
    
    # 计算损失
    losses = criterion(
        cls_logits=cls_logits,
        targets=targets,
        routing_probs=routing_probs,
        target_experts=target_experts,
        features=features,
        all_anchors=all_anchors,
        valid_classes=[0, 1, 2, 3, 4],
        old_logits=None,
        new_logits_for_old=None,
        old_classes=None,
        residuals=[torch.randn(dim), torch.randn(dim)]
    )
    
    print("Loss values:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
