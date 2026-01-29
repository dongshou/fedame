"""
概率分布提示模块
每个类别维护一个概率分布（backbone 空间）
用于知识压缩和防遗忘回放
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


class ClassDistribution(nn.Module):
    """
    单个类别的概率分布（backbone 空间）
    均值 μ: 用真实 backbone 特征均值初始化
    标准差 σ: 随机初始化
    """
    
    def __init__(
        self,
        class_id: int,
        dim: int = 512,
        init_mean: Optional[torch.Tensor] = None,
        init_std: float = 0.1
    ):
        super().__init__()
        
        self.class_id = class_id
        self.dim = dim
        
        # 可学习的均值（用真实特征均值初始化）
        if init_mean is not None:
            self.mean = nn.Parameter(init_mean.clone())
        else:
            self.mean = nn.Parameter(torch.zeros(dim))
        
        # 可学习的对数标准差（随机初始化）
        self.log_std = nn.Parameter(torch.randn(dim) * 0.01 + math.log(init_std))
        
        # 样本计数（用于聚合时的加权）
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    @property
    def std(self) -> torch.Tensor:
        """分布标准差"""
        return torch.exp(self.log_std).clamp(min=1e-6, max=2.0)
    
    @property
    def variance(self) -> torch.Tensor:
        """分布方差"""
        return self.std ** 2
    
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        重参数化采样（梯度可传）
        
        Args:
            num_samples: 采样数量
        
        Returns:
            samples: [num_samples, dim]
        """
        eps = torch.randn(num_samples, self.dim, device=self.mean.device)
        samples = self.mean.unsqueeze(0) + eps * self.std.unsqueeze(0)
        return samples
    
    def update_sample_count(self, count: int):
        """更新样本计数"""
        self.sample_count = self.sample_count + count
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """获取分布参数（用于联邦聚合）"""
        return {
            'mean': self.mean.data.clone(),
            'log_std': self.log_std.data.clone(),
            'sample_count': self.sample_count.clone()
        }
    
    def set_params(self, params: Dict[str, torch.Tensor]):
        """设置分布参数"""
        self.mean.data = params['mean'].to(self.mean.device)
        self.log_std.data = params['log_std'].to(self.log_std.device)
        self.sample_count = params['sample_count'].to(self.sample_count.device)


class DistributionPool(nn.Module):
    """
    分布池
    管理所有类别的分布
    """
    
    def __init__(
        self,
        dim: int = 512,
        init_std: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.init_std = init_std
        
        # 类别分布字典
        self.distributions = nn.ModuleDict()
    
    def add_class(
        self,
        class_id: int,
        init_mean: Optional[torch.Tensor] = None
    ):
        """
        添加新类别的分布
        
        Args:
            class_id: 类别ID
            init_mean: 初始均值（用真实特征均值）
        """
        if str(class_id) in self.distributions:
            return
        
        dist = ClassDistribution(
            class_id=class_id,
            dim=self.dim,
            init_mean=init_mean,
            init_std=self.init_std
        )
        
        # 移动到正确设备
        if init_mean is not None:
            dist = dist.to(init_mean.device)
        
        self.distributions[str(class_id)] = dist
    
    def get_distribution(self, class_id: int) -> ClassDistribution:
        """获取类别分布"""
        return self.distributions[str(class_id)]
    
    def has_class(self, class_id: int) -> bool:
        """检查是否有某类别的分布"""
        return str(class_id) in self.distributions
    
    def sample(self, class_id: int, num_samples: int = 1) -> torch.Tensor:
        """从指定类别的分布采样"""
        return self.get_distribution(class_id).sample(num_samples)
    
    def sample_all(
        self,
        class_ids: List[int],
        num_samples_per_class: int = 1
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        批量采样所有指定类别
        
        Args:
            class_ids: 要采样的类别列表
            num_samples_per_class: 每个类采样数量
        
        Returns:
            samples: [total_samples, dim]
            labels: [total_samples]
        """
        all_samples = []
        all_labels = []
        
        for cls in class_ids:
            if self.has_class(cls):
                samples = self.sample(cls, num_samples_per_class)
                labels = torch.full(
                    (num_samples_per_class,), cls,
                    dtype=torch.long, device=samples.device
                )
                all_samples.append(samples)
                all_labels.append(labels)
        
        if len(all_samples) == 0:
            return None, None
        
        return torch.cat(all_samples, dim=0), torch.cat(all_labels, dim=0)
    
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
    dim: int = 512
) -> Dict[int, Dict]:
    """
    聚合多个客户端的分布参数
    
    Args:
        local_params_list: 各客户端的分布参数列表
        dim: 特征维度
    
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
        
        # 加权聚合
        weighted_mean = torch.zeros(dim)
        weighted_log_std = torch.zeros(dim)
        
        for p in cls_params_list:
            count = p['sample_count'].item()
            weight = count / total_count if total_count > 0 else 1.0 / len(cls_params_list)
            weighted_mean += weight * p['mean'].cpu()
            weighted_log_std += weight * p['log_std'].cpu()
        
        global_params[cls] = {
            'mean': weighted_mean,
            'log_std': weighted_log_std,
            'sample_count': torch.tensor(total_count)
        }
    
    return global_params


# 测试
if __name__ == "__main__":
    # 创建分布池
    pool = DistributionPool(dim=512, init_std=0.1)
    
    # 模拟用真实特征均值初始化
    for cls in [0, 1, 2, 3]:
        fake_mean = torch.randn(512)  # 假设这是真实特征的均值
        pool.add_class(cls, init_mean=fake_mean)
    
    print(f"Number of classes: {pool.num_classes}")
    print(f"Class list: {pool.class_list}")
    
    # 测试采样
    samples = pool.sample(class_id=0, num_samples=5)
    print(f"\nSamples from class 0: {samples.shape}")
    
    # 批量采样
    batch_samples, batch_labels = pool.sample_all([0, 1, 2], num_samples_per_class=3)
    print(f"Batch samples: {batch_samples.shape}")
    print(f"Batch labels: {batch_labels}")
    
    # 测试聚合
    print("\n--- Testing Aggregation ---")
    
    local_params_1 = pool.get_all_params()
    
    # 模拟另一个客户端
    pool2 = DistributionPool(dim=512)
    for cls in [0, 1, 2, 3]:
        pool2.add_class(cls, init_mean=torch.randn(512))
    
    # 更新样本计数
    for cls in [0, 1]:
        pool.get_distribution(cls).update_sample_count(100)
        pool2.get_distribution(cls).update_sample_count(50)
    
    local_params_2 = pool2.get_all_params()
    
    # 聚合
    global_params = aggregate_distributions([local_params_1, local_params_2], dim=512)
    
    print(f"Aggregated classes: {list(global_params.keys())}")