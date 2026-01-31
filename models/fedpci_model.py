"""
FedPCI 模型模块
双分支架构：共性分支 g_common + 个性化分支 g_ind
可学习原型：μ (均值) + σ (维度级标准差)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class FeatureBranch(nn.Module):
    """
    特征分支网络（共性分支或个性化分支共用结构）
    
    结构: Linear -> LayerNorm -> GELU -> Dropout -> ... -> Linear
    输出: L2 归一化的特征向量
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_classes=10,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 构建MLP网络
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
        
        self.classifer = nn.Sequential(
            nn.Linear(output_dim, num_classes),
            nn.Softmax(dim=-1)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, normalize: bool = True, commond=None) -> torch.Tensor:
        """
        特征变换
        
        Args:
            x: 输入特征 [B, input_dim]
            normalize: 是否L2归一化
        
        Returns:
            transformed: 变换后的特征 [B, output_dim]
        """
        transformed = self.network(x)
        if commond!= None:
            transformed = transformed + commond
        if normalize:
            transformed = F.normalize(transformed, p=2, dim=-1)
        result = self.classifer(transformed)
        return transformed, result


class ClassPrototype(nn.Module):
    """
    类原型
    
    包含：
    - μ (mean): 类中心，维度 [d]
    - log_σ (log_std): 对数标准差，维度 [d]，用于归一化个性化特征
    
    σ 的作用：
    - 维度级别的容忍度
    - σ[i] 大：该维度变化大，放松约束
    - σ[i] 小：该维度变化小，严格约束
    """
    
    def __init__(
        self,
        class_id: int,
        dim: int = 128,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
        init_mean: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.class_id = class_id
        self.dim = dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # 可学习的均值 μ
        if init_mean is not None:
            self.mean = nn.Parameter(init_mean.clone())
        else:
            self.mean = nn.Parameter(torch.zeros(dim))
        
        # 可学习的对数标准差 log_σ（初始化为0，即σ=1）
        self.log_sigma = nn.Parameter(torch.zeros(dim))
        
        # 样本计数（用于聚合时加权）
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    @property
    def sigma(self) -> torch.Tensor:
        """获取标准差（带上下界约束）"""
        return torch.exp(self.log_sigma).clamp(min=self.sigma_min, max=self.sigma_max)
    
    def update_mean(self, new_mean: torch.Tensor, count: int = 1):
        """更新均值（用于从数据中提取）"""
        self.mean.data = new_mean.to(self.mean.device)
        # 直接设置 count，而不是累加（每轮重新统计）
        self.sample_count.fill_(float(count))
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """获取原型参数（用于联邦聚合）"""
        return {
            'mean': self.mean.data.clone().cpu(),
            'log_sigma': self.log_sigma.data.clone().cpu(),
            'sample_count': self.sample_count.clone().cpu()
        }
    
    def set_params(self, params: Dict[str, torch.Tensor]):
        """设置原型参数"""
        if 'mean' in params:
            self.mean.data = params['mean'].to(self.mean.device)
        if 'log_sigma' in params:
            self.log_sigma.data = params['log_sigma'].to(self.log_sigma.device)
        if 'sample_count' in params:
            self.sample_count = params['sample_count'].to(self.sample_count.device)


class DualBranchNetwork(nn.Module):
    """
    单个类的双分支网络
    
    包含：
    - g_common: 共性分支，学习"类之所以为类"的本质特征
    - g_ind: 个性化分支，学习"样本相对于原型的偏移"
    - prototype: 类原型 (μ, σ)
    """
    
    def __init__(
        self,
        class_id: int,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
        num_classes: int = 10
    ):
        super().__init__()
        
        self.class_id = class_id
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
        
        # 类原型
        self.prototype = ClassPrototype(
            class_id=class_id,
            dim=output_dim,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: backbone特征 [B, input_dim]
        
        Returns:
            z_common: 共性特征 [B, output_dim]
            z_ind: 个性化特征 [B, output_dim]
        """
        z_common,commond_result = self.g_common(x)
        z_ind,ind_result = self.g_ind(x,commond=z_common)
        return z_common, z_ind, commond_result, ind_result
    
    def compute_distance(
        self, 
        x: torch.Tensor, 
        lambda_ind: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算样本与该类的距离
        
        Args:
            x: backbone特征 [B, input_dim]
            lambda_ind: 个性化距离的权重
        
        Returns:
            d_total: 总距离 [B]
            d_common: 共性距离 [B]
            d_ind: 个性化距离 [B]
        """
        z_common, z_ind, commond_result, ind_result = self.forward(x)
        
        # 共性距离：||z_common - μ||²
        d_common = torch.sum((z_common - self.prototype.mean.unsqueeze(0)) ** 2, dim=-1)
        
        # 个性化距离：||z_ind / σ||²
        sigma = self.prototype.sigma.unsqueeze(0)  # [1, d]
        d_ind = torch.sum((z_ind / sigma) ** 2, dim=-1)
        
        # 总距离
        d_total = d_common + lambda_ind * d_ind
        
        return d_total, d_common, d_ind, commond_result, ind_result
    
    def get_common_params(self) -> Dict[str, torch.Tensor]:
        """获取共性分支参数"""
        return {name: param.data.clone() for name, param in self.g_common.named_parameters()}
    
    def set_common_params(self, params: Dict[str, torch.Tensor]):
        """设置共性分支参数"""
        state_dict = self.g_common.state_dict()
        for name, value in params.items():
            if name in state_dict:
                state_dict[name] = value.to(next(self.g_common.parameters()).device)
        self.g_common.load_state_dict(state_dict)
    
    def get_ind_params(self) -> Dict[str, torch.Tensor]:
        """获取个性化分支参数"""
        return {name: param.data.clone() for name, param in self.g_ind.named_parameters()}
    
    def set_ind_params(self, params: Dict[str, torch.Tensor]):
        """设置个性化分支参数"""
        state_dict = self.g_ind.state_dict()
        for name, value in params.items():
            if name in state_dict:
                state_dict[name] = value.to(next(self.g_ind.parameters()).device)
        self.g_ind.load_state_dict(state_dict)
    
    def get_prototype_params(self) -> Dict[str, torch.Tensor]:
        """获取原型参数"""
        return self.prototype.get_params()
    
    def set_prototype_params(self, params: Dict[str, torch.Tensor]):
        """设置原型参数"""
        self.prototype.set_params(params)


class FedPCIModel(nn.Module):
    """
    FedPCI 完整模型
    
    管理所有类的双分支网络
    
    架构：
    - 10个类，每个类有独立的双分支网络
    - 每个类有独立的原型 (μ, σ)
    
    聚合规则：
    - g_common[c]: 选择性聚合（仅拥有类c的客户端参与）
    - g_ind[c]: 不聚合（完全本地）
    - prototype[c]: 选择性聚合
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
        lambda_ind: float = 0.5,
        temperature: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_ind = lambda_ind
        self.temperature = temperature
        
        # 创建每个类的双分支网络
        self.class_networks = nn.ModuleDict({
            str(c): DualBranchNetwork(
                class_id=c,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                sigma_min=sigma_min,
                sigma_max=sigma_max
            )
            for c in range(num_classes)
        })
    
    def get_class_network(self, class_id: int) -> DualBranchNetwork:
        """获取指定类的网络"""
        return self.class_networks[str(class_id)]
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算与所有类的距离
        
        Args:
            x: backbone特征 [B, input_dim]
        
        Returns:
            d_total: 总距离矩阵 [B, num_classes]
            d_common: 共性距离矩阵 [B, num_classes]
            d_ind: 个性化距离矩阵 [B, num_classes]
        """
        batch_size = x.size(0)
        device = x.device
        
        d_total_list = []
        d_common_list = []
        d_ind_list = []
        commond_result_list = []
        ind_result_list = []
        
        for c in range(self.num_classes):
            network = self.get_class_network(c)
            d_total, d_common, d_ind, commond_result, ind_result = network.compute_distance(x, self.lambda_ind)
            d_total_list.append(d_total)
            d_common_list.append(d_common)
            d_ind_list.append(d_ind)
            commond_result_list.append(commond_result)
            ind_result_list.append(ind_result)
        
        d_total = torch.stack(d_total_list, dim=-1)  # [B, num_classes]
        d_common = torch.stack(d_common_list, dim=-1)
        d_ind = torch.stack(d_ind_list, dim=-1)
        commond_logits = torch.stack(commond_result_list, dim=-1)
        ind_logits = torch.stack(ind_result_list, dim=-1)
        
        return d_total, d_common, d_ind, commond_logits, ind_logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测类别
        
        Args:
            x: backbone特征 [B, input_dim]
        
        Returns:
            pred: 预测类别 [B]
            probs: 类别概率 [B, num_classes]
        """
        d_total, _, _ = self.forward(x)
        
        # 距离越小越好，转换为logits
        logits = -d_total / self.temperature
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmin(d_total, dim=-1)
        
        return pred, probs
    
    def compute_logits_common(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅用共性距离计算logits
        
        Args:
            x: backbone特征 [B, input_dim]
        
        Returns:
            logits: [B, num_classes]
        """
        _, d_common, _ = self.forward(x)
        return -d_common / self.temperature
    
    def compute_logits_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        用完整距离计算logits
        
        Args:
            x: backbone特征 [B, input_dim]
        
        Returns:
            logits: [B, num_classes]
        """
        d_total, _, _ = self.forward(x)
        return -d_total / self.temperature
    
    def get_features_for_class(
        self, 
        x: torch.Tensor, 
        class_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定类的特征
        
        Args:
            x: backbone特征 [B, input_dim]
            class_id: 类别ID
        
        Returns:
            z_common: 共性特征 [B, output_dim]
            z_ind: 个性化特征 [B, output_dim]
        """
        network = self.get_class_network(class_id)
        return network.forward(x)
    
    # ============ 参数获取/设置方法（用于联邦聚合）============
    
    def get_all_common_params(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """获取所有类的共性分支参数"""
        return {c: self.get_class_network(c).get_common_params() 
                for c in range(self.num_classes)}
    
    def get_common_params(self, class_id: int) -> Dict[str, torch.Tensor]:
        """获取指定类的共性分支参数"""
        return self.get_class_network(class_id).get_common_params()
    
    def set_common_params(self, class_id: int, params: Dict[str, torch.Tensor]):
        """设置指定类的共性分支参数"""
        self.get_class_network(class_id).set_common_params(params)
    
    def get_all_ind_params(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """获取所有类的个性化分支参数"""
        return {c: self.get_class_network(c).get_ind_params() 
                for c in range(self.num_classes)}
    
    def get_ind_params(self, class_id: int) -> Dict[str, torch.Tensor]:
        """获取指定类的个性化分支参数"""
        return self.get_class_network(class_id).get_ind_params()
    
    def set_ind_params(self, class_id: int, params: Dict[str, torch.Tensor]):
        """设置指定类的个性化分支参数"""
        self.get_class_network(class_id).set_ind_params(params)
    
    def get_all_prototype_params(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """获取所有类的原型参数"""
        return {c: self.get_class_network(c).get_prototype_params() 
                for c in range(self.num_classes)}
    
    def get_prototype_params(self, class_id: int) -> Dict[str, torch.Tensor]:
        """获取指定类的原型参数"""
        return self.get_class_network(class_id).get_prototype_params()
    
    def set_prototype_params(self, class_id: int, params: Dict[str, torch.Tensor]):
        """设置指定类的原型参数"""
        self.get_class_network(class_id).set_prototype_params(params)
    
    def get_prototype_mean(self, class_id: int) -> torch.Tensor:
        """获取指定类的原型均值"""
        return self.get_class_network(class_id).prototype.mean
    
    def get_prototype_sigma(self, class_id: int) -> torch.Tensor:
        """获取指定类的原型标准差"""
        return self.get_class_network(class_id).prototype.sigma


# 测试
if __name__ == "__main__":
    # 创建模型
    model = FedPCIModel(
        num_classes=10,
        input_dim=512,
        hidden_dim=256,
        output_dim=128,
        num_layers=3
    )
    
    # 测试前向传播
    x = torch.randn(4, 512)
    d_total, d_common, d_ind = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"d_total shape: {d_total.shape}")
    print(f"d_common shape: {d_common.shape}")
    print(f"d_ind shape: {d_ind.shape}")
    
    # 测试预测
    pred, probs = model.predict(x)
    print(f"\nPrediction: {pred}")
    print(f"Probs shape: {probs.shape}")
    
    # 测试参数获取
    common_params = model.get_common_params(0)
    print(f"\nCommon params keys: {list(common_params.keys())}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    single_network_params = sum(p.numel() for p in model.get_class_network(0).parameters())
    common_params_count = sum(p.numel() for p in model.get_class_network(0).g_common.parameters())
    ind_params_count = sum(p.numel() for p in model.get_class_network(0).g_ind.parameters())
    prototype_params_count = sum(p.numel() for p in model.get_class_network(0).prototype.parameters())
    
    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Single class network: {single_network_params:,}")
    print(f"    - g_common: {common_params_count:,}")
    print(f"    - g_ind: {ind_params_count:,}")
    print(f"    - prototype: {prototype_params_count:,}")