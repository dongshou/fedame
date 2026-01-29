"""
概率分布提示模块
每个类别维护一个概率分布，用于知识压缩和防遗忘回放
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class ClassDistribution(nn.Module):
    """
    单个类别的概率分布
    均值 = 锚点 + 可学习残差
    协方差 = 可学习对角矩阵
    """
    
    def __init__(
        self,
        class_id: int,
        anchor: torch.Tensor,
        dim: int = 512,
        max_residual_norm: float = 0.5,
        init_std: float = 0.1
    ):
        super().__init__()
        
        self.class_id = class_id
        self.dim = dim
        self.max_residual_norm = max_residual_norm
        
        # 固定的锚点
        self.register_buffer('anchor', anchor.clone())
        
        # 可学习的残差（有界）
        self.residual = nn.Parameter(torch.zeros(dim))
        
        # 可学习的对数标准差（使用对角协方差）
        self.log_std = nn.Parameter(torch.full((dim,), math.log(init_std)))
        
        # 样本计数（用于聚合时的加权）
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    @property
    def mean(self) -> torch.Tensor:
        """分布均值 = 锚点 + 有界残差"""
        # 约束残差范数
        residual_norm = torch.norm(self.residual)
        if residual_norm > self.max_residual_norm:
            bounded_residual = self.residual * (self.max_residual_norm / residual_norm)
        else:
            bounded_residual = self.residual
        return self.anchor + bounded_residual
    
    @property
    def std(self) -> torch.Tensor:
        """分布标准差"""
        return torch.exp(self.log_std).clamp(min=1e-6, max=1.0)
    
    @property
    def variance(self) -> torch.Tensor:
        """分布方差"""
        return self.std ** 2
    
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        从分布中采样
        
        Args:
            num_samples: 采样数量
        
        Returns:
            samples: [num_samples, dim]
        """
        mean = self.mean
        std = self.std
        
        # 重参数化采样
        eps = torch.randn(num_samples, self.dim, device=mean.device)
        samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        
        return samples
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算对数概率密度
        
        Args:
            x: 输入 [B, dim]
        
        Returns:
            log_prob: [B]
        """
        mean = self.mean
        var = self.variance
        
        # 高斯对数概率
        log_prob = -0.5 * (
            self.dim * math.log(2 * math.pi) +
            torch.sum(torch.log(var)) +
            torch.sum((x - mean.unsqueeze(0)) ** 2 / var.unsqueeze(0), dim=-1)
        )
        
        return log_prob
    
    def kl_divergence(self, other: 'ClassDistribution') -> torch.Tensor:
        """
        计算与另一个分布的KL散度
        KL(self || other)
        """
        mean_self = self.mean
        var_self = self.variance
        mean_other = other.mean
        var_other = other.variance
        
        kl = 0.5 * torch.sum(
            torch.log(var_other / var_self) +
            (var_self + (mean_self - mean_other) ** 2) / var_other - 1
        )
        
        return kl
    
    def update_sample_count(self, count: int):
        """更新样本计数"""
        self.sample_count = self.sample_count + count
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """获取分布参数（用于聚合）"""
        return {
            'residual': self.residual.data.clone(),
            'log_std': self.log_std.data.clone(),
            'sample_count': self.sample_count.clone()
        }
    
    def set_params(self, params: Dict[str, torch.Tensor]):
        """设置分布参数"""
        self.residual.data = params['residual']
        self.log_std.data = params['log_std']
        self.sample_count = params['sample_count']


class DistributionPool(nn.Module):
    """
    分布池
    管理所有类别的分布
    """
    
    def __init__(
        self,
        anchor_dim: int = 512,
        max_residual_norm: float = 0.5,
        init_std: float = 0.1
    ):
        super().__init__()
        
        self.anchor_dim = anchor_dim
        self.max_residual_norm = max_residual_norm
        self.init_std = init_std
        
        # 类别分布字典
        self.distributions: Dict[int, ClassDistribution] = nn.ModuleDict()
        
        # 全局锚点引用 - 使用register_buffer初始化为None
        self.register_buffer('class_anchors', None)
    
    def set_anchors(self, anchors: torch.Tensor):
        """设置类锚点"""
        self.class_anchors = anchors
    
    def add_class(self, class_id: int, anchor: Optional[torch.Tensor] = None):
        """
        添加新类别的分布
        
        Args:
            class_id: 类别ID
            anchor: 类锚点（如果为None，从class_anchors获取）
        """
        if str(class_id) in self.distributions:
            return
        
        if anchor is None:
            if self.class_anchors is None:
                raise ValueError("No anchor provided and class_anchors not set")
            anchor = self.class_anchors[class_id]
        
        dist = ClassDistribution(
            class_id=class_id,
            anchor=anchor,
            dim=self.anchor_dim,
            max_residual_norm=self.max_residual_norm,
            init_std=self.init_std
        )
        
        self.distributions[str(class_id)] = dist
    
    def get_distribution(self, class_id: int) -> ClassDistribution:
        """获取类别分布"""
        return self.distributions[str(class_id)]
    
    def has_class(self, class_id: int) -> bool:
        """检查是否有某类别的分布"""
        return str(class_id) in self.distributions
    
    def sample(
        self,
        class_id: int,
        num_samples: int = 1
    ) -> torch.Tensor:
        """从指定类别的分布采样"""
        return self.get_distribution(class_id).sample(num_samples)
    
    def sample_batch(
        self,
        class_ids: List[int],
        num_samples_per_class: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量采样多个类别
        
        Returns:
            samples: [total_samples, dim]
            labels: [total_samples]
        """
        all_samples = []
        all_labels = []
        
        for cls in class_ids:
            if self.has_class(cls):
                samples = self.sample(cls, num_samples_per_class)
                labels = torch.full((num_samples_per_class,), cls)
                all_samples.append(samples)
                all_labels.append(labels)
        
        if len(all_samples) == 0:
            return None, None
        
        samples = torch.cat(all_samples, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return samples, labels
    
    def compute_residual_loss(self, class_ids: Optional[List[int]] = None) -> torch.Tensor:
        """
        计算残差正则化损失
        
        Args:
            class_ids: 计算哪些类别（None表示全部）
        """
        if class_ids is None:
            class_ids = [int(k) for k in self.distributions.keys()]
        
        loss = 0.0
        count = 0
        
        for cls in class_ids:
            if self.has_class(cls):
                dist = self.get_distribution(cls)
                loss = loss + torch.sum(dist.residual ** 2)
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss
    
    def get_all_params(self) -> Dict[int, Dict]:
        """获取所有分布的参数"""
        params = {}
        for cls_str, dist in self.distributions.items():
            params[int(cls_str)] = dist.get_params()
        return params
    
    def set_class_params(self, class_id: int, params: Dict[str, torch.Tensor]):
        """设置指定类别的分布参数"""
        if self.has_class(class_id):
            self.get_distribution(class_id).set_params(params)
    
    @property
    def num_classes(self) -> int:
        return len(self.distributions)
    
    @property
    def class_list(self) -> List[int]:
        return [int(k) for k in self.distributions.keys()]


def aggregate_distributions(
    local_params_list: List[Dict[int, Dict]],
    global_anchors: torch.Tensor,
    anchor_dim: int = 512
) -> Dict[int, Dict]:
    """
    聚合多个客户端的分布参数
    
    Args:
        local_params_list: 各客户端的分布参数列表
        global_anchors: 全局锚点
        anchor_dim: 锚点维度
    
    Returns:
        global_params: 聚合后的全局分布参数
    """
    # 收集所有类别
    all_classes = set()
    for params in local_params_list:
        all_classes.update(params.keys())
    
    global_params = {}
    
    for cls in all_classes:
        # 收集该类别在各客户端的参数
        cls_params_list = []
        total_count = 0.0
        
        for params in local_params_list:
            if cls in params:
                cls_params_list.append(params[cls])
                total_count += params[cls]['sample_count'].item()
        
        if len(cls_params_list) == 0:
            continue
        
        if total_count == 0:
            total_count = len(cls_params_list)  # 均匀权重
        
        # 加权聚合残差
        weighted_residual = torch.zeros(anchor_dim)
        weighted_log_std = torch.zeros(anchor_dim)
        
        for p in cls_params_list:
            weight = p['sample_count'].item() / total_count if total_count > 0 else 1.0 / len(cls_params_list)
            weighted_residual += weight * p['residual'].cpu()
            weighted_log_std += weight * p['log_std'].cpu()
        
        global_params[cls] = {
            'residual': weighted_residual,
            'log_std': weighted_log_std,
            'sample_count': torch.tensor(total_count)
        }
    
    return global_params


# 测试
if __name__ == "__main__":
    # 创建分布池
    pool = DistributionPool(
        anchor_dim=512,
        max_residual_norm=0.5
    )
    
    # 设置锚点
    anchors = F.normalize(torch.randn(10, 512), dim=-1)
    pool.set_anchors(anchors)
    
    # 添加一些类别
    for cls in [0, 1, 2, 3]:
        pool.add_class(cls)
    
    print(f"Number of classes: {pool.num_classes}")
    print(f"Class list: {pool.class_list}")
    
    # 测试采样
    samples = pool.sample(class_id=0, num_samples=5)
    print(f"\nSamples from class 0: {samples.shape}")
    
    # 批量采样
    batch_samples, batch_labels = pool.sample_batch([0, 1, 2], num_samples_per_class=3)
    print(f"Batch samples: {batch_samples.shape}")
    print(f"Batch labels: {batch_labels}")
    
    # 计算残差损失
    residual_loss = pool.compute_residual_loss()
    print(f"\nResidual loss: {residual_loss.item():.6f}")
    
    # 测试聚合
    print("\n--- Testing Aggregation ---")
    
    # 模拟两个客户端的参数
    local_params_1 = pool.get_all_params()
    
    # 修改一些参数模拟另一个客户端
    pool2 = DistributionPool(anchor_dim=512)
    pool2.set_anchors(anchors)
    for cls in [0, 1, 2, 3]:
        pool2.add_class(cls)
    
    # 更新样本计数
    for cls in [0, 1]:
        pool.get_distribution(cls).update_sample_count(100)
        pool2.get_distribution(cls).update_sample_count(50)
    
    local_params_2 = pool2.get_all_params()
    
    # 聚合
    global_params = aggregate_distributions(
        [local_params_1, local_params_2],
        anchors
    )
    
    print(f"Aggregated classes: {list(global_params.keys())}")
