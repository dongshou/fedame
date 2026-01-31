"""
FedPCI 损失函数模块 (重构版)

损失函数：
1. L_cls_common: 共性特征分类损失
2. L_cls_full: 完整特征分类损失
3. L_local_align: z_common ↔ μ_local 对齐损失
4. L_global_align: μ_local ↔ μ_global 对齐损失
5. L_proto_contrast: 原型对比损失 (本地原型 vs 全局所有原型)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class ClassificationLoss(nn.Module):
    """
    分类损失 (交叉熵)
    
    支持局部分类（只在本地类别上计算softmax）
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        local_classes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes]
            targets: [B]
            local_classes: 本地类别列表，如果为None则使用所有类别
        
        Returns:
            loss: 分类损失
        """
        if local_classes is not None:
            # 局部分类：只在本地类别上计算
            local_logits = logits[:, local_classes]
            # 将全局标签映射到局部索引
            local_targets = torch.tensor(
                [local_classes.index(t.item()) for t in targets],
                device=targets.device
            )
            return F.cross_entropy(local_logits, local_targets)
        else:
            return F.cross_entropy(logits, targets)


class LocalAlignmentLoss(nn.Module):
    """
    本地对齐损失
    
    约束共性特征向对应类别的本地原型靠拢
    L_local_align = (1/B) Σ_i ||z_common_i - μ_local_{y_i}||²
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        z_common: torch.Tensor,
        prototypes: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_common: 共性特征 [B, d]
            prototypes: 本地原型 [num_classes, d]
            targets: 标签 [B]
        
        Returns:
            loss: 对齐损失
        """
        # 获取目标类别的原型
        target_prototypes = prototypes[targets]  # [B, d]
        
        # 计算L2距离
        loss = F.mse_loss(z_common, target_prototypes)
        
        return loss


class GlobalAlignmentLoss(nn.Module):
    """
    全局对齐损失
    
    约束本地原型向全局原型对齐（同类）
    L_global_align = (1/C_local) Σ_c ||μ_local_c - μ_global_c||²
    
    注意：全局原型不参与梯度计算
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        local_prototypes: torch.Tensor,
        global_prototypes: torch.Tensor,
        local_classes: List[int]
    ) -> torch.Tensor:
        """
        Args:
            local_prototypes: 本地原型 [num_classes, d] (全部)
            global_prototypes: 全局原型 [num_classes, d] (全部，不参与梯度)
            local_classes: 本地拥有的类别列表
        
        Returns:
            loss: 对齐损失
        """
        if len(local_classes) == 0:
            return torch.tensor(0.0, device=local_prototypes.device)
        
        total_loss = 0.0
        for c in local_classes:
            local_p = local_prototypes[c]
            global_p = global_prototypes[c].detach()  # 不参与梯度
            total_loss += F.mse_loss(local_p, global_p)
        
        return total_loss / len(local_classes)


class PrototypeContrastLoss(nn.Module):
    """
    原型对比损失
    
    将本地原型与全局所有原型进行对比学习：
    - 拉近同类全局原型
    - 推远其他类全局原型
    
    L_contrast = (1/C_local) Σ_c -log(
        exp(μ_local_c · μ_global_c / τ) / 
        Σ_k exp(μ_local_c · μ_global_k / τ)
    )
    
    作用：即使本地没有某些类别的数据，也能通过全局原型学习到
    本地原型应该与那些类保持距离
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        local_prototypes: torch.Tensor,
        global_prototypes: torch.Tensor,
        local_classes: List[int]
    ) -> torch.Tensor:
        """
        Args:
            local_prototypes: 本地原型 [num_classes, d]
            global_prototypes: 全局原型 [num_classes, d] (不参与梯度)
            local_classes: 本地拥有的类别列表
        
        Returns:
            loss: 对比损失
        """
        if len(local_classes) == 0:
            return torch.tensor(0.0, device=local_prototypes.device)
        
        # 归一化原型 (对比学习通常需要归一化)
        local_normed = F.normalize(local_prototypes, p=2, dim=-1)
        global_normed = F.normalize(global_prototypes.detach(), p=2, dim=-1)
        
        total_loss = 0.0
        for c in local_classes:
            # 本地原型作为 anchor
            anchor = local_normed[c]  # [d]
            
            # 相似度：与所有全局原型
            sim = torch.matmul(anchor, global_normed.T) / self.temperature  # [num_classes]
            
            # 正样本是同类全局原型
            labels = torch.tensor(c, device=sim.device)
            
            total_loss += F.cross_entropy(sim.unsqueeze(0), labels.unsqueeze(0))
        
        return total_loss / len(local_classes)


class FedPCILoss(nn.Module):
    """
    FedPCI 总损失 (重构版)
    
    L = L_cls_common + L_cls_full + λ1·L_local_align + λ2·L_global_align + λ3·L_proto_contrast
    
    各损失的作用：
    - L_cls_common: 共性特征的分类能力
    - L_cls_full: 完整特征的分类能力（个性化增益）
    - L_local_align: z_common 紧凑性（向本地原型靠拢）
    - L_global_align: 本地原型的联邦一致性
    - L_proto_contrast: 全局类间区分度（即使没有某类数据）
    """
    
    def __init__(
        self,
        lambda_local_align: float = 0.5,
        lambda_global_align: float = 0.3,
        lambda_proto_contrast: float = 0.5,
        temperature: float = 0.1
    ):
        super().__init__()
        
        # 权重
        self.lambda_local_align = lambda_local_align
        self.lambda_global_align = lambda_global_align
        self.lambda_proto_contrast = lambda_proto_contrast
        
        # 损失模块
        self.cls_loss = ClassificationLoss()
        self.local_align_loss = LocalAlignmentLoss()
        self.global_align_loss = GlobalAlignmentLoss()
        self.proto_contrast_loss = PrototypeContrastLoss(temperature)
    
    def forward(
        self,
        logits_common: torch.Tensor,
        logits_full: torch.Tensor,
        targets: torch.Tensor,
        z_common: torch.Tensor,
        local_prototypes: torch.Tensor,
        global_prototypes: torch.Tensor,
        local_classes: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            logits_common: 共性分类logits [B, num_classes]
            logits_full: 完整分类logits [B, num_classes]
            targets: 标签 [B]
            z_common: 共性特征 [B, d]
            local_prototypes: 本地原型 [num_classes, d]
            global_prototypes: 全局原型 [num_classes, d]
            local_classes: 本地拥有的类别列表
        
        Returns:
            loss_dict: 包含各项损失的字典
        """
        losses = {}
        
        # 1. 共性分类损失 (局部)
        losses['cls_common'] = self.cls_loss(logits_common, targets, local_classes)
        
        # 2. 完整分类损失 (局部)
        losses['cls_full'] = self.cls_loss(logits_full, targets, local_classes)
        
        # 3. 本地对齐损失 (z_common ↔ μ_local)
        losses['local_align'] = self.local_align_loss(z_common, local_prototypes, targets)
        
        # 4. 全局对齐损失 (μ_local ↔ μ_global)
        losses['global_align'] = self.global_align_loss(
            local_prototypes, global_prototypes, local_classes
        )
        
        # 5. 原型对比损失
        losses['proto_contrast'] = self.proto_contrast_loss(
            local_prototypes, global_prototypes, local_classes
        )
        
        # 总损失
        losses['total'] = (
            losses['cls_common'] +
            losses['cls_full'] +
            self.lambda_local_align * losses['local_align'] +
            self.lambda_global_align * losses['global_align'] +
            self.lambda_proto_contrast * losses['proto_contrast']
        )
        
        return losses
    
    def get_loss_weights(self) -> Dict[str, float]:
        """返回当前的损失权重配置"""
        return {
            'cls_common': 1.0,
            'cls_full': 1.0,
            'local_align': self.lambda_local_align,
            'global_align': self.lambda_global_align,
            'proto_contrast': self.lambda_proto_contrast
        }


if __name__ == "__main__":
    print("=" * 60)
    print("FedPCI 损失函数测试 (重构版)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 8
    num_classes = 10
    dim = 128
    local_classes = [0, 1, 2, 3, 4]
    
    # 模拟数据
    logits_common = torch.randn(batch_size, num_classes)
    logits_full = torch.randn(batch_size, num_classes)
    targets = torch.tensor([local_classes[i % len(local_classes)] for i in range(batch_size)])
    z_common = torch.randn(batch_size, dim)
    local_prototypes = torch.randn(num_classes, dim)
    global_prototypes = torch.randn(num_classes, dim)
    
    # 测试单个损失
    print("\n--- 单个损失测试 ---")
    
    cls_loss = ClassificationLoss()
    loss_cls = cls_loss(logits_common, targets, local_classes)
    print(f"分类损失 (局部): {loss_cls.item():.4f}")
    
    local_align = LocalAlignmentLoss()
    loss_local = local_align(z_common, local_prototypes, targets)
    print(f"本地对齐损失: {loss_local.item():.4f}")
    
    global_align = GlobalAlignmentLoss()
    loss_global = global_align(local_prototypes, global_prototypes, local_classes)
    print(f"全局对齐损失: {loss_global.item():.4f}")
    
    proto_contrast = PrototypeContrastLoss(temperature=0.1)
    loss_contrast = proto_contrast(local_prototypes, global_prototypes, local_classes)
    print(f"原型对比损失: {loss_contrast.item():.4f}")
    
    # 测试完整损失
    print("\n--- 完整损失测试 ---")
    
    criterion = FedPCILoss(
        lambda_local_align=0.5,
        lambda_global_align=0.3,
        lambda_proto_contrast=0.5,
        temperature=0.1
    )
    
    losses = criterion(
        logits_common=logits_common,
        logits_full=logits_full,
        targets=targets,
        z_common=z_common,
        local_prototypes=local_prototypes,
        global_prototypes=global_prototypes,
        local_classes=local_classes
    )
    
    print("\nFedPCI 各项损失:")
    print("-" * 40)
    weights = criterion.get_loss_weights()
    for name, value in losses.items():
        if name != 'total':
            weight = weights.get(name, 1.0)
            print(f"  {name:15s}: {value.item():8.4f} (权重: {weight})")
    print("-" * 40)
    print(f"  {'total':15s}: {losses['total'].item():8.4f}")
    
    # 测试原型对比损失的特性
    print("\n--- 原型对比损失特性测试 ---")
    
    # 场景1: 本地原型与全局原型相同
    same_protos = global_prototypes.clone()
    loss_same = proto_contrast(same_protos, global_prototypes, local_classes)
    print(f"本地=全局时的对比损失: {loss_same.item():.4f}")
    
    # 场景2: 本地原型完全随机
    random_protos = torch.randn(num_classes, dim)
    loss_random = proto_contrast(random_protos, global_prototypes, local_classes)
    print(f"本地随机时的对比损失: {loss_random.item():.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)