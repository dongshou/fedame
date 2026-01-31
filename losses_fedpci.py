"""
FedPCI 损失函数模块

包含以下损失：
1. CommonClassificationLoss: 共性特征分类损失
2. FullClassificationLoss: 完整特征分类损失
3. GlobalContrastLoss: 全局对比损失
4. CommonCompactLoss: 共性紧凑损失
5. SigmaRegularizationLoss: σ正则化损失
6. PrototypeAlignmentLoss: 原型对齐损失
7. CovarianceOrthogonalLoss: 协方差正交损失（新增）
8. FedPCILoss: 总损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class CommonClassificationLoss(nn.Module):
    """
    共性特征分类损失
    
    使用共性特征到原型的距离进行分类
    L_cls_common = -log(exp(-d_common_y/τ) / Σ_c exp(-d_common_c/τ))
    
    支持局部分类（只在本地类别上计算softmax）
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        d_common: torch.Tensor,
        targets: torch.Tensor,
        local_classes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Args:
            d_common: 共性距离矩阵 [B, num_classes]
            targets: 标签 [B]
            local_classes: 本地类别列表，如果为None则使用所有类别
        
        Returns:
            loss: 分类损失
        """
        # 转换为相似度（负距离）
        logits = -d_common / self.temperature
        
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


class FullClassificationLoss(nn.Module):
    """
    完整特征分类损失
    
    使用完整特征（共性+个性）到原型的距离进行分类
    L_cls_full = -log(exp(-d_total_y/τ) / Σ_c exp(-d_total_c/τ))
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        d_total: torch.Tensor,
        targets: torch.Tensor,
        local_classes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Args:
            d_total: 总距离矩阵 [B, num_classes]
            targets: 标签 [B]
            local_classes: 本地类别列表
        
        Returns:
            loss: 分类损失
        """
        logits = -d_total / self.temperature
        
        if local_classes is not None:
            local_logits = logits[:, local_classes]
            local_targets = torch.tensor(
                [local_classes.index(t.item()) for t in targets],
                device=targets.device
            )
            return F.cross_entropy(local_logits, local_targets)
        else:
            return F.cross_entropy(logits, targets)


class GlobalContrastLoss(nn.Module):
    """
    全局对比损失
    
    在所有类别上进行对比学习，增强类间区分度
    L_global = -log(exp(-d_y/τ) / Σ_c exp(-d_c/τ))
    
    与分类损失的区别：这里使用所有类别，不限于本地类别
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        d_total: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            d_total: 总距离矩阵 [B, num_classes]
            targets: 标签 [B]
        
        Returns:
            loss: 全局对比损失
        """
        logits = -d_total / self.temperature
        return F.cross_entropy(logits, targets)


class CommonCompactLoss(nn.Module):
    """
    共性紧凑损失
    
    约束共性特征向对应类别的原型靠拢
    L_common = (1/B) Σ_i ||z_common_i - μ_{y_i}||²
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
            prototypes: 原型均值 [num_classes, d]
            targets: 标签 [B]
        
        Returns:
            loss: 紧凑损失
        """
        # 获取目标类别的原型
        target_prototypes = prototypes[targets]  # [B, d]
        
        # 计算L2距离
        loss = F.mse_loss(z_common, target_prototypes)
        
        return loss


class SigmaRegularizationLoss(nn.Module):
    """
    σ 正则化损失
    
    防止σ过大或过小，保持合理的不确定性估计
    L_sigma = (1/C) Σ_c ||log(σ_c)||²
    
    这个损失鼓励 σ 接近 1（即 log(σ) 接近 0）
    """
    
    def __init__(self, target_log_sigma: float = 0.0):
        super().__init__()
        self.target_log_sigma = target_log_sigma
    
    def forward(self, log_sigmas: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            log_sigmas: 各类的 log_σ 列表，每个元素 [d]
        
        Returns:
            loss: σ正则化损失
        """
        total_loss = 0.0
        for log_sigma in log_sigmas:
            total_loss += torch.mean((log_sigma - self.target_log_sigma) ** 2)
        
        return total_loss / len(log_sigmas)


class PrototypeAlignmentLoss(nn.Module):
    """
    原型对齐损失
    
    约束本地原型向全局原型对齐
    L_align = (1/C_local) Σ_c ||μ_c^local - μ_c^global||²
    
    注意：全局原型不参与梯度计算
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        local_prototypes: List[torch.Tensor],
        global_prototypes: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            local_prototypes: 本地原型列表，每个元素 [d]
            global_prototypes: 全局原型列表，每个元素 [d]（不参与梯度）
        
        Returns:
            loss: 对齐损失
        """
        if len(local_prototypes) == 0:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        for local_p, global_p in zip(local_prototypes, global_prototypes):
            # 全局原型不参与梯度
            total_loss += F.mse_loss(local_p, global_p.detach())
        
        return total_loss / len(local_prototypes)


class CovarianceOrthogonalLoss(nn.Module):
    """
    协方差正交损失
    
    约束共性特征和个性化特征统计独立（线性不相关）
    L_orth = (1/(d1*d2)) * ||Z_common^T @ Z_individual||_F^2
    
    其中 Z_common 和 Z_individual 是中心化后的特征矩阵
    
    理论依据：
    - 协方差为零意味着两个随机变量线性不相关
    - 这是实现特征解耦的必要条件
    - 相比互信息最小化，计算更简单且稳定
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        z_common: torch.Tensor,
        z_individual: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_common: 共性特征 [B, d1]
            z_individual: 个性化特征 [B, d2]
        
        Returns:
            loss: 协方差正交损失
        """
        batch_size = z_common.size(0)
        
        # 批次大小为1时无法计算协方差
        if batch_size <= 1:
            return torch.tensor(0.0, device=z_common.device)
        
        # 中心化：减去均值
        z_common_centered = z_common - z_common.mean(dim=0, keepdim=True)
        z_individual_centered = z_individual - z_individual.mean(dim=0, keepdim=True)
        
        # 计算协方差矩阵 [d1, d2]
        # Cov(Z_c, Z_i) = (1/(B-1)) * Z_c^T @ Z_i
        cov_matrix = torch.mm(
            z_common_centered.t(), 
            z_individual_centered
        ) / (batch_size - 1)
        
        # Frobenius范数的平方，并归一化
        d1, d2 = z_common.size(1), z_individual.size(1)
        loss = torch.sum(cov_matrix ** 2) / (d1 * d2)
        
        return loss


class FedPCILoss(nn.Module):
    """
    FedPCI 总损失
    
    L = λ1·L_cls_common + λ2·L_cls_full + λ3·L_global + λ4·L_common 
        + λ5·L_sigma + λ6·L_proto_align + λ7·L_orth
    
    各损失的作用：
    - L_cls_common: 共性特征的分类能力
    - L_cls_full: 完整特征的分类能力
    - L_global: 全局类间区分度
    - L_common: 共性特征的类内紧凑性
    - L_sigma: 不确定性估计的正则化
    - L_proto_align: 本地-全局原型对齐
    - L_orth: 共性-个性特征解耦
    
    权重建议：
    - L_cls_common: 1.0（主分类损失-共性）
    - L_cls_full: 1.0（主分类损失-完整）
    - L_global: 0.5（全局对比）
    - L_common: 0.3（共性紧凑）
    - L_sigma: 0.01（σ正则化）
    - L_proto_align: 0.1（原型对齐）
    - L_orth: 0.1（正交约束）
    """
    
    def __init__(
        self,
        lambda_cls_common: float = 1.0,
        lambda_cls_full: float = 1.0,
        lambda_global: float = 0.5,
        lambda_common: float = 0.3,
        lambda_sigma: float = 0.01,
        lambda_proto_align: float = 0.1,
        lambda_orth: float = 0.1,
        temperature: float = 0.1
    ):
        super().__init__()
        
        # 保存权重
        self.lambda_cls_common = lambda_cls_common
        self.lambda_cls_full = lambda_cls_full
        self.lambda_global = lambda_global
        self.lambda_common = lambda_common
        self.lambda_sigma = lambda_sigma
        self.lambda_proto_align = lambda_proto_align
        self.lambda_orth = lambda_orth
        
        # 初始化各损失模块
        self.cls_common_loss = CommonClassificationLoss(temperature)
        self.cls_full_loss = FullClassificationLoss(temperature)
        self.global_loss = GlobalContrastLoss(temperature)
        self.common_compact_loss = CommonCompactLoss()
        self.sigma_reg_loss = SigmaRegularizationLoss()
        self.proto_align_loss = PrototypeAlignmentLoss()
        self.orth_loss = CovarianceOrthogonalLoss()
        self.ces_loss = torch.nn.CrossEntropyLoss
    
    def forward(
        self,
        d_total: torch.Tensor,
        d_common: torch.Tensor,
        targets: torch.Tensor,
        z_common_target: torch.Tensor,
        prototypes: torch.Tensor,
        log_sigmas: List[torch.Tensor],
        local_classes: Optional[List[int]] = None,
        use_global_loss: bool = True,
        local_prototypes: Optional[List[torch.Tensor]] = None,
        global_prototypes: Optional[List[torch.Tensor]] = None,
        z_common: Optional[torch.Tensor] = None,
        z_individual: Optional[torch.Tensor] = None,
        comm_logits: Optional[torch.Tensor] = None,
        ind_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            d_total: 总距离矩阵 [B, num_classes]
            d_common: 共性距离矩阵 [B, num_classes]
            targets: 标签 [B]
            z_common_target: 目标类的共性特征 [B, d]
            prototypes: 原型均值 [num_classes, d]
            log_sigmas: 各类的 log_σ 列表
            local_classes: 本地类别列表（用于局部分类损失）
            use_global_loss: 是否使用全局对比损失
            local_prototypes: 本地原型列表（用于原型对齐）
            global_prototypes: 全局原型列表（用于原型对齐，不参与梯度）
            z_common: 共性特征 [B, d]（用于正交约束）
            z_individual: 个性化特征 [B, d]（用于正交约束）
        
        Returns:
            loss_dict: 包含各项损失的字典
                - 'cls_common': 共性分类损失
                - 'cls_full': 完整分类损失
                - 'global': 全局对比损失
                - 'common_compact': 共性紧凑损失
                - 'sigma_reg': σ正则化损失
                - 'proto_align': 原型对齐损失
                - 'orth': 正交损失
                - 'total': 总损失
        """
        losses = {}
        device = d_total.device
        
        if comm_logits is not None:
            print(f"comm_logits: {comm_logits.shape}")
            print(targets.shape)
            losses['cls_common_cla'] = self.ces_loss(comm_logits, targets)
            
        if ind_logits is not None:
            losses['cls_ind_cla'] = self.ces_loss(ind_logits, targets)
        
        # 1. 共性分类损失（局部）
        losses['cls_common'] = self.cls_common_loss(d_common, targets, local_classes)
        
        # 2. 完整分类损失（局部）
        losses['cls_full'] = self.cls_full_loss(d_total, targets, local_classes)
        
        # 3. 全局对比损失
        if use_global_loss:
            losses['global'] = self.global_loss(d_total, targets)
        else:
            losses['global'] = torch.tensor(0.0, device=device)
        
        # 4. 共性紧凑损失
        losses['common_compact'] = self.common_compact_loss(
            z_common_target, prototypes, targets
        )
        
        # 5. σ 正则化损失
        losses['sigma_reg'] = self.sigma_reg_loss(log_sigmas)
        
        # 6. 原型对齐损失
        if local_prototypes is not None and global_prototypes is not None:
            losses['proto_align'] = self.proto_align_loss(local_prototypes, global_prototypes)
        else:
            losses['proto_align'] = torch.tensor(0.0, device=device)
        
        # 7. 协方差正交损失
        if z_common is not None and z_individual is not None:
            losses['orth'] = self.orth_loss(z_common, z_individual)
        else:
            losses['orth'] = torch.tensor(0.0, device=device)
        
        # 计算总损失
        losses['total'] = (
            self.lambda_cls_common * losses['cls_common'] +
            self.lambda_cls_full * losses['cls_full'] +
            self.lambda_global * losses['global'] +
            self.lambda_common * losses['common_compact'] +
            self.lambda_sigma * losses['sigma_reg'] +
            self.lambda_proto_align * losses['proto_align'] +
            self.lambda_orth * losses['orth'] +
            losses['cls_common_cla'] +
            losses['cls_ind_cla']
        )
        
        return losses
    
    def get_loss_weights(self) -> Dict[str, float]:
        """返回当前的损失权重配置"""
        return {
            'cls_common': self.lambda_cls_common,
            'cls_full': self.lambda_cls_full,
            'global': self.lambda_global,
            'common_compact': self.lambda_common,
            'sigma_reg': self.lambda_sigma,
            'proto_align': self.lambda_proto_align,
            'orth': self.lambda_orth
        }
    
    def set_loss_weight(self, name: str, value: float):
        """动态设置损失权重"""
        weight_map = {
            'cls_common': 'lambda_cls_common',
            'cls_full': 'lambda_cls_full',
            'global': 'lambda_global',
            'common_compact': 'lambda_common',
            'sigma_reg': 'lambda_sigma',
            'proto_align': 'lambda_proto_align',
            'orth': 'lambda_orth'
        }
        if name in weight_map:
            setattr(self, weight_map[name], value)
        else:
            raise ValueError(f"Unknown loss name: {name}")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("FedPCI 损失函数测试")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 8
    num_classes = 10
    dim = 128
    
    # 模拟数据
    d_total = torch.randn(batch_size, num_classes)
    d_common = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    z_common_target = torch.randn(batch_size, dim)
    prototypes = torch.randn(num_classes, dim)
    log_sigmas = [torch.randn(dim) for _ in range(num_classes)]
    local_classes = [0, 1, 2, 3, 4]
    
    # 确保targets在local_classes中
    targets = torch.tensor([local_classes[i % len(local_classes)] for i in range(batch_size)])
    
    # 模拟本地和全局原型
    local_prototypes = [torch.randn(dim) for _ in local_classes]
    global_prototypes = [torch.randn(dim) for _ in local_classes]
    
    # 模拟共性和个性化特征
    z_common = torch.randn(batch_size, dim)
    z_individual = torch.randn(batch_size, dim)
    
    # ==================== 测试单个损失 ====================
    print("\n--- 单个损失测试 ---")
    
    # 1. 共性分类损失
    cls_common = CommonClassificationLoss()
    loss_cls_common = cls_common(d_common, targets, local_classes)
    print(f"共性分类损失: {loss_cls_common.item():.4f}")
    
    # 2. 完整分类损失
    cls_full = FullClassificationLoss()
    loss_cls_full = cls_full(d_total, targets, local_classes)
    print(f"完整分类损失: {loss_cls_full.item():.4f}")
    
    # 3. 全局对比损失
    global_contrast = GlobalContrastLoss()
    loss_global = global_contrast(d_total, targets)
    print(f"全局对比损失: {loss_global.item():.4f}")
    
    # 4. 共性紧凑损失
    common_compact = CommonCompactLoss()
    loss_compact = common_compact(z_common_target, prototypes, targets)
    print(f"共性紧凑损失: {loss_compact.item():.4f}")
    
    # 5. σ正则化损失
    sigma_reg = SigmaRegularizationLoss()
    loss_sigma = sigma_reg(log_sigmas)
    print(f"σ正则化损失: {loss_sigma.item():.4f}")
    
    # 6. 原型对齐损失
    proto_align = PrototypeAlignmentLoss()
    loss_align = proto_align(local_prototypes, global_prototypes)
    print(f"原型对齐损失: {loss_align.item():.4f}")
    
    # 7. 协方差正交损失
    orth = CovarianceOrthogonalLoss()
    loss_orth = orth(z_common, z_individual)
    print(f"协方差正交损失: {loss_orth.item():.4f}")
    
    # ==================== 测试正交损失的特性 ====================
    print("\n--- 正交损失特性测试 ---")
    
    # 测试1: 随机特征
    z1_rand = torch.randn(32, 64)
    z2_rand = torch.randn(32, 64)
    loss_rand = orth(z1_rand, z2_rand)
    print(f"随机特征的正交损失: {loss_rand.item():.6f}")
    
    # 测试2: 相同特征（应该有较大损失）
    z_same = torch.randn(32, 64)
    loss_same = orth(z_same, z_same)
    print(f"相同特征的正交损失: {loss_same.item():.6f}")
    
    # 测试3: 构造正交特征
    z1_orth = torch.randn(32, 32)
    z2_orth = torch.randn(32, 32)
    # 中心化
    z1_centered = z1_orth - z1_orth.mean(dim=0, keepdim=True)
    z2_centered = z2_orth - z2_orth.mean(dim=0, keepdim=True)
    # 使用SVD构造正交特征
    U, S, V = torch.svd(z1_centered)
    z2_orth_constructed = U @ torch.diag(S) @ V.t()  # 重构
    # 构造与z1正交的z2
    z2_perpendicular = z2_centered - z1_centered @ (z1_centered.t() @ z2_centered) / (torch.norm(z1_centered) ** 2 + 1e-8)
    loss_perpendicular = orth(z1_centered, z2_perpendicular)
    print(f"正交化特征的正交损失: {loss_perpendicular.item():.6f}")
    
    # ==================== 测试完整损失 ====================
    print("\n--- 完整损失测试 ---")
    
    criterion = FedPCILoss(
        lambda_cls_common=1.0,
        lambda_cls_full=1.0,
        lambda_global=0.5,
        lambda_common=0.3,
        lambda_sigma=0.01,
        lambda_proto_align=0.1,
        lambda_orth=0.1
    )
    
    losses = criterion(
        d_total=d_total,
        d_common=d_common,
        targets=targets,
        z_common_target=z_common_target,
        prototypes=prototypes,
        log_sigmas=log_sigmas,
        local_classes=local_classes,
        use_global_loss=True,
        local_prototypes=local_prototypes,
        global_prototypes=global_prototypes,
        z_common=z_common,
        z_individual=z_individual
    )
    
    print("\nFedPCI 各项损失:")
    print("-" * 40)
    for name, value in losses.items():
        if name != 'total':
            weight = criterion.get_loss_weights().get(name, 1.0)
            print(f"  {name:15s}: {value.item():8.4f} (权重: {weight})")
    print("-" * 40)
    print(f"  {'total':15s}: {losses['total'].item():8.4f}")
    
    # ==================== 测试梯度 ====================
    print("\n--- 梯度测试 ---")
    
    # 创建需要梯度的张量
    z_common_grad = torch.randn(batch_size, dim, requires_grad=True)
    z_individual_grad = torch.randn(batch_size, dim, requires_grad=True)
    
    loss_orth_grad = orth(z_common_grad, z_individual_grad)
    loss_orth_grad.backward()
    
    print(f"z_common 梯度范数: {z_common_grad.grad.norm().item():.4f}")
    print(f"z_individual 梯度范数: {z_individual_grad.grad.norm().item():.4f}")
    
    # ==================== 测试边界情况 ====================
    print("\n--- 边界情况测试 ---")
    
    # 批次大小为1
    z1_single = torch.randn(1, 64)
    z2_single = torch.randn(1, 64)
    loss_single = orth(z1_single, z2_single)
    print(f"批次大小为1时的正交损失: {loss_single.item():.4f}")
    
    # 不同维度的特征
    z1_diff = torch.randn(16, 64)
    z2_diff = torch.randn(16, 128)
    loss_diff = orth(z1_diff, z2_diff)
    print(f"不同维度特征的正交损失: {loss_diff.item():.4f}")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)