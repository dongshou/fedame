"""
FedPCI 模型模块 (重构版)

架构：
- 单一双分支网络：g_common + g_ind
- 两个分类头：classifier_common (聚合) + classifier_full (不聚合)
- 可学习原型：μ_local

特征流：
- z_common = g_common(f) → classifier_common → logits_common
- z_full = z_common + z_ind → classifier_full → logits_full
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FeatureBranch(nn.Module):
    """
    特征分支网络
    
    结构: Linear -> LayerNorm -> GELU -> Dropout -> ... -> Linear
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        layers = []
        
        # 输入层
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
            normalize: 是否L2归一化
        
        Returns:
            z: [B, output_dim]
        """
        z = self.network(x)
        if normalize:
            z = F.normalize(z, p=2, dim=-1)
        return z


class LearnablePrototypes(nn.Module):
    """
    可学习原型
    
    每个类别一个原型向量 μ
    """
    
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        
        self.num_classes = num_classes
        self.dim = dim
        
        # 可学习原型 [num_classes, dim]
        self.prototypes = nn.Parameter(torch.zeros(num_classes, dim))
        nn.init.xavier_uniform_(self.prototypes)
    
    def forward(self, class_ids: Optional[List[int]] = None) -> torch.Tensor:
        """
        获取原型
        
        Args:
            class_ids: 指定类别，None表示所有类别
        
        Returns:
            prototypes: [len(class_ids), dim] 或 [num_classes, dim]
        """
        if class_ids is None:
            return self.prototypes
        else:
            return self.prototypes[class_ids]
    
    def get_prototype(self, class_id: int) -> torch.Tensor:
        """获取单个类别的原型"""
        return self.prototypes[class_id]
    
    def set_prototype(self, class_id: int, value: torch.Tensor):
        """设置单个类别的原型"""
        self.prototypes.data[class_id] = value.to(self.prototypes.device)
    
    def set_all_prototypes(self, values: torch.Tensor):
        """设置所有原型"""
        self.prototypes.data = values.to(self.prototypes.device)


class FedPCIModel(nn.Module):
    """
    FedPCI 模型 (重构版)
    
    架构：
    - g_common: 共性分支，提取共享特征
    - g_ind: 个性化分支，提取个性化特征
    - classifier_common: 共性分类头 (聚合)
    - classifier_full: 完整分类头 (不聚合)
    - prototypes: 可学习原型
    
    聚合策略：
    - g_common: ✅ 聚合
    - g_ind: ❌ 不聚合
    - classifier_common: ✅ 聚合
    - classifier_full: ❌ 不聚合
    - prototypes: ✅ 选择性聚合
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 共性分支
        self.g_common = FeatureBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 个性化分支
        self.g_ind = FeatureBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 共性分类头 (聚合)
        self.classifier_common = nn.Linear(output_dim, num_classes)
        
        # 完整分类头 (不聚合)
        self.classifier_full = nn.Linear(output_dim, num_classes)
        
        # 可学习原型
        self.prototypes = LearnablePrototypes(num_classes, output_dim)
        
        self._init_classifiers()
    
    def _init_classifiers(self):
        nn.init.xavier_uniform_(self.classifier_common.weight)
        nn.init.zeros_(self.classifier_common.bias)
        nn.init.xavier_uniform_(self.classifier_full.weight)
        nn.init.zeros_(self.classifier_full.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: backbone特征 [B, input_dim]
            return_features: 是否返回中间特征
        
        Returns:
            dict containing:
                - logits_common: [B, num_classes]
                - logits_full: [B, num_classes]
                - z_common: [B, output_dim] (if return_features)
                - z_ind: [B, output_dim] (if return_features)
                - z_full: [B, output_dim] (if return_features)
        """
        # 共性特征
        z_common = self.g_common(x)
        
        # 个性化特征
        z_ind = self.g_ind(x)
        
        # 完整特征 (残差连接)
        z_full = z_common + z_ind
        
        # 分类
        logits_common = self.classifier_common(z_common)
        logits_full = self.classifier_full(z_full)
        
        result = {
            'logits_common': logits_common,
            'logits_full': logits_full
        }
        
        if return_features:
            result.update({
                'z_common': z_common,
                'z_ind': z_ind,
                'z_full': z_full
            })
        
        return result
    
    def get_prototypes(self, class_ids: Optional[List[int]] = None) -> torch.Tensor:
        """获取原型"""
        return self.prototypes(class_ids)
    
    # ============ 参数获取方法 (用于聚合) ============
    
    def get_common_branch_params(self) -> Dict[str, torch.Tensor]:
        """获取共性分支参数"""
        return {k: v.cpu().clone() for k, v in self.g_common.state_dict().items()}
    
    def set_common_branch_params(self, params: Dict[str, torch.Tensor]):
        """设置共性分支参数"""
        device = next(self.g_common.parameters()).device
        state_dict = {k: v.to(device) for k, v in params.items()}
        self.g_common.load_state_dict(state_dict)
    
    def get_ind_branch_params(self) -> Dict[str, torch.Tensor]:
        """获取个性化分支参数"""
        return {k: v.cpu().clone() for k, v in self.g_ind.state_dict().items()}
    
    def set_ind_branch_params(self, params: Dict[str, torch.Tensor]):
        """设置个性化分支参数"""
        device = next(self.g_ind.parameters()).device
        state_dict = {k: v.to(device) for k, v in params.items()}
        self.g_ind.load_state_dict(state_dict)
    
    def get_classifier_common_params(self) -> Dict[str, torch.Tensor]:
        """获取共性分类头参数"""
        return {
            'weight': self.classifier_common.weight.cpu().clone(),
            'bias': self.classifier_common.bias.cpu().clone()
        }
    
    def set_classifier_common_params(self, params: Dict[str, torch.Tensor]):
        """设置共性分类头参数"""
        device = self.classifier_common.weight.device
        self.classifier_common.weight.data = params['weight'].to(device)
        self.classifier_common.bias.data = params['bias'].to(device)
    
    def get_classifier_full_params(self) -> Dict[str, torch.Tensor]:
        """获取完整分类头参数"""
        return {
            'weight': self.classifier_full.weight.cpu().clone(),
            'bias': self.classifier_full.bias.cpu().clone()
        }
    
    def set_classifier_full_params(self, params: Dict[str, torch.Tensor]):
        """设置完整分类头参数"""
        device = self.classifier_full.weight.device
        self.classifier_full.weight.data = params['weight'].to(device)
        self.classifier_full.bias.data = params['bias'].to(device)
    
    def get_prototype_params(self) -> torch.Tensor:
        """获取原型参数"""
        return self.prototypes.prototypes.cpu().clone()
    
    def set_prototype_params(self, params: torch.Tensor):
        """设置原型参数"""
        self.prototypes.set_all_prototypes(params)
    
    def get_prototype_for_class(self, class_id: int) -> torch.Tensor:
        """获取单个类别的原型"""
        return self.prototypes.get_prototype(class_id).cpu().clone()
    
    def set_prototype_for_class(self, class_id: int, value: torch.Tensor):
        """设置单个类别的原型"""
        self.prototypes.set_prototype(class_id, value)
    
    # ============ 聚合相关的便捷方法 ============
    
    def get_aggregatable_params(self) -> Dict[str, any]:
        """
        获取所有需要聚合的参数
        
        Returns:
            dict containing:
                - g_common: 共性分支参数
                - classifier_common: 共性分类头参数
                - prototypes: 原型参数
        """
        return {
            'g_common': self.get_common_branch_params(),
            'classifier_common': self.get_classifier_common_params(),
            'prototypes': self.get_prototype_params()
        }
    
    def set_aggregatable_params(self, params: Dict[str, any]):
        """设置所有聚合参数"""
        if 'g_common' in params:
            self.set_common_branch_params(params['g_common'])
        if 'classifier_common' in params:
            self.set_classifier_common_params(params['classifier_common'])
        if 'prototypes' in params:
            self.set_prototype_params(params['prototypes'])
    
    def get_local_params(self) -> Dict[str, any]:
        """
        获取本地参数 (不聚合)
        
        Returns:
            dict containing:
                - g_ind: 个性化分支参数
                - classifier_full: 完整分类头参数
        """
        return {
            'g_ind': self.get_ind_branch_params(),
            'classifier_full': self.get_classifier_full_params()
        }
    
    def set_local_params(self, params: Dict[str, any]):
        """设置本地参数"""
        if 'g_ind' in params:
            self.set_ind_branch_params(params['g_ind'])
        if 'classifier_full' in params:
            self.set_classifier_full_params(params['classifier_full'])


if __name__ == "__main__":
    # 测试
    model = FedPCIModel(
        num_classes=10,
        input_dim=512,
        hidden_dim=256,
        output_dim=128,
        num_layers=3
    )
    
    x = torch.randn(4, 512)
    output = model(x, return_features=True)
    
    print("Model Architecture Test:")
    print(f"  Input: {x.shape}")
    print(f"  logits_common: {output['logits_common'].shape}")
    print(f"  logits_full: {output['logits_full'].shape}")
    print(f"  z_common: {output['z_common'].shape}")
    print(f"  z_ind: {output['z_ind'].shape}")
    print(f"  z_full: {output['z_full'].shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    g_common_params = sum(p.numel() for p in model.g_common.parameters())
    g_ind_params = sum(p.numel() for p in model.g_ind.parameters())
    cls_common_params = sum(p.numel() for p in model.classifier_common.parameters())
    cls_full_params = sum(p.numel() for p in model.classifier_full.parameters())
    proto_params = model.prototypes.prototypes.numel()
    
    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  g_common: {g_common_params:,}")
    print(f"  g_ind: {g_ind_params:,}")
    print(f"  classifier_common: {cls_common_params:,}")
    print(f"  classifier_full: {cls_full_params:,}")
    print(f"  prototypes: {proto_params:,}")
    
    # 测试参数获取
    agg_params = model.get_aggregatable_params()
    print(f"\nAggregatable params keys: {list(agg_params.keys())}")
    
    local_params = model.get_local_params()
    print(f"Local params keys: {list(local_params.keys())}")