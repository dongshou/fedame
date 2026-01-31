"""
FedPCI 联邦学习客户端

核心特点：
- 双分支架构：g_common（共性） + g_ind（个性化）
- g_common：选择性聚合（仅拥有该类的客户端参与）
- g_ind：不聚合（完全本地）
- 原型 (μ, σ)：选择性聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import copy

from models.fedpci_model import FedPCIModel
from models.backbone import create_backbone
from losses_fedpci import FedPCILoss


class FedPCIClient:
    """
    FedPCI 联邦客户端
    """
    
    def __init__(
        self,
        client_id: int,
        num_classes: int,
        backbone: nn.Module,
        model: FedPCIModel,
        device: str = "cuda",
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        lambda_ind: float = 0.5,
        temperature: float = 0.1,
        # 损失权重
        lambda_cls_common: float = 1.0,
        lambda_cls_full: float = 1.0,
        lambda_global: float = 0.5,
        lambda_common: float = 0.3,
        lambda_sigma: float = 0.01,
        lambda_proto_align: float = 0.1
    ):
        self.client_id = client_id
        self.num_classes = num_classes
        self.device = device
        self.lambda_ind = lambda_ind
        
        # 模型组件
        self.backbone = backbone.to(device)
        self.backbone.eval()  # backbone 始终冻结
        
        self.model = model.to(device)
        
        # 本地数据信息
        self.local_classes: List[int] = []
        
        # 保存全局原型（用于原型对齐损失）
        self.global_prototypes: Dict[int, torch.Tensor] = {}
        
        # 损失函数
        self.criterion = FedPCILoss(
            lambda_cls_common=lambda_cls_common,
            lambda_cls_full=lambda_cls_full,
            lambda_global=lambda_global,
            lambda_common=lambda_common,
            lambda_sigma=lambda_sigma,
            lambda_proto_align=lambda_proto_align,
            temperature=temperature
        )
        
        # 优化器参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = None
    
    def setup_local_data(self, local_classes: List[int]):
        """设置本地拥有的类别"""
        self.local_classes = local_classes
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 只优化本地拥有类别的参数
        params = []
        
        for cls in self.local_classes:
            network = self.model.get_class_network(cls)
            # g_common 参与优化（后续会聚合）
            params.extend(network.g_common.parameters())
            # g_ind 参与优化（不聚合，完全本地）
            params.extend(network.g_ind.parameters())
            # 原型参与优化（后续会聚合）
            params.extend(network.prototype.parameters())
        
        self.optimizer = optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
    
    def extract_prototypes_from_data(self, train_loader: DataLoader):
        """
        从本地数据中提取原型均值
        
        用 backbone 特征的均值初始化 g_common 输出的原型
        """
        self.backbone.eval()
        self.model.eval()
        
        # 收集每个类的共性特征
        class_features: Dict[int, List[torch.Tensor]] = {
            cls: [] for cls in self.local_classes
        }
        class_counts: Dict[int, int] = {cls: 0 for cls in self.local_classes}
        
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                features = self.backbone(images)
                
                for i, label in enumerate(labels):
                    cls = label.item()
                    if cls in class_features:
                        # 提取该类的共性特征
                        network = self.model.get_class_network(cls)
                        z_common,_ = network.g_common(features[i:i+1])
                        class_features[cls].append(z_common.squeeze(0).cpu())
                        class_counts[cls] += 1
        
        # 计算每个类的原型均值
        for cls in self.local_classes:
            if len(class_features[cls]) > 0:
                feats = torch.stack(class_features[cls], dim=0)
                prototype_mean = feats.mean(dim=0).to(self.device)
                
                # 更新原型
                network = self.model.get_class_network(cls)
                network.prototype.update_mean(prototype_mean, class_counts[cls])
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 5,
        use_global_loss: bool = True
    ) -> Dict[str, float]:
        """
        本地训练
        
        Args:
            train_loader: 训练数据加载器
            num_epochs: 本地训练轮数
            use_global_loss: 是否使用全局对比损失
        
        Returns:
            metrics: 训练指标（包含各项损失）
        """
        self._init_optimizer()
        self.model.train()
        
        total_loss = 0.0
        total_cls_common = 0.0
        total_cls_full = 0.0
        total_global = 0.0
        total_common_compact = 0.0
        total_sigma_reg = 0.0
        total_proto_align = 0.0
        total_batches = 0
        
        for epoch in range(num_epochs):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 1. 提取 backbone 特征
                with torch.no_grad():
                    features = self.backbone(images)
                
                # 2. 前向传播：计算所有类的距离
                d_total, d_common, d_ind,comm_logit, ind_logit = self.model(features)
                
                # 3. 获取目标类的共性特征（用于紧凑损失）
                batch_size = features.size(0)
                z_common_target_list = []
                for i in range(batch_size):
                    cls = labels[i].item()
                    network = self.model.get_class_network(cls)
                    z_common, _, _, _ = network(features[i:i+1])
                    z_common_target_list.append(z_common.squeeze(0))
                z_common_target = torch.stack(z_common_target_list, dim=0)
                
                # 4. 收集原型和 log_sigma
                prototypes = torch.stack([
                    self.model.get_prototype_mean(c) 
                    for c in range(self.num_classes)
                ], dim=0)
                
                log_sigmas = [
                    self.model.get_class_network(c).prototype.log_sigma
                    for c in self.local_classes  # 只对本地类计算 sigma 正则化
                ]
                
                # 5. 收集本地原型和全局原型（用于对齐损失）
                local_prototypes = [
                    self.model.get_prototype_mean(c)
                    for c in self.local_classes
                ]
                global_prototypes = [
                    self.global_prototypes.get(c, self.model.get_prototype_mean(c))
                    for c in self.local_classes
                ]
                
                # 6. 计算损失
                losses = self.criterion(
                    d_total=d_total,
                    d_common=d_common,
                    targets=labels,
                    z_common_target=z_common_target,
                    prototypes=prototypes,
                    log_sigmas=log_sigmas,
                    local_classes=self.local_classes,
                    use_global_loss=use_global_loss,
                    local_prototypes=local_prototypes,
                    global_prototypes=global_prototypes,
                    comm_logits=comm_logit,
                    ind_logits=ind_logit
                )
                
                # 7. 反向传播
                self.optimizer.zero_grad()
                losses['total'].backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += losses['total'].item()
                total_cls_common += losses['cls_common'].item()
                total_cls_full += losses['cls_full'].item()
                total_global += losses['global'].item()
                total_common_compact += losses['common_compact'].item()
                total_sigma_reg += losses['sigma_reg'].item()
                total_proto_align += losses['proto_align'].item()
                total_batches += 1
        
        avg_loss = total_loss / max(total_batches, 1)
        avg_cls_common = total_cls_common / max(total_batches, 1)
        avg_cls_full = total_cls_full / max(total_batches, 1)
        avg_global = total_global / max(total_batches, 1)
        avg_common_compact = total_common_compact / max(total_batches, 1)
        avg_sigma_reg = total_sigma_reg / max(total_batches, 1)
        avg_proto_align = total_proto_align / max(total_batches, 1)
        
        # 计算训练后本地原型与全局原型的距离
        proto_distances = {}
        for c in self.local_classes:
            local_mu = self.model.get_prototype_mean(c)
            global_mu = self.global_prototypes.get(c, local_mu)
            dist = torch.norm(local_mu - global_mu).item()
            proto_distances[c] = dist
        
        return {
            'loss': avg_loss,
            'cls_common_loss': avg_cls_common,
            'cls_full_loss': avg_cls_full,
            'global_loss': avg_global,
            'common_compact_loss': avg_common_compact,
            'sigma_reg_loss': avg_sigma_reg,
            'proto_align_loss': avg_proto_align,
            'num_epochs': num_epochs,
            'num_batches': total_batches,
            'proto_distances': proto_distances  # 各类原型距离
        }
    
    # ============ 参数获取方法（用于上传到服务端）============
    
    def get_common_updates(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        获取共性分支参数更新
        
        只返回本地拥有类别的共性分支参数
        """
        updates = {}
        for cls in self.local_classes:
            updates[cls] = self.model.get_common_params(cls)
            # 转移到 CPU
            updates[cls] = {k: v.cpu() for k, v in updates[cls].items()}
        return updates
    
    def get_prototype_updates(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        获取原型参数更新
        
        只返回本地拥有类别的原型参数
        """
        updates = {}
        for cls in self.local_classes:
            updates[cls] = self.model.get_prototype_params(cls)
        return updates
    
    # ============ 参数加载方法（从服务端下载）============
    
    def load_common_params(self, global_common_params: Dict[int, Dict[str, torch.Tensor]]):
        """
        加载全局共性分支参数
        
        对于所有类都加载（包括本地没有的类）
        """
        for cls, params in global_common_params.items():
            self.model.set_common_params(cls, params)
    
    def load_prototype_params(self, global_prototype_params: Dict[int, Dict[str, torch.Tensor]]):
        """
        加载全局原型参数
        
        对于所有类都加载（包括本地没有的类）
        同时保存一份副本用于原型对齐损失
        """
        for cls, params in global_prototype_params.items():
            self.model.set_prototype_params(cls, params)
            # 保存全局原型副本（用于对齐损失）
            if 'mean' in params:
                self.global_prototypes[cls] = params['mean'].clone().to(self.device)
    
    # ============ 评估方法 ============
    
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
        self.model.eval()
        
        total_correct_common = 0  # 仅用共性距离
        total_correct_full = 0    # 用完整距离
        total_samples = 0
        
        class_correct_common = {}
        class_correct_full = {}
        class_total = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                features = self.backbone(images)
                
                # 计算距离
                d_total, d_common, d_ind, com_logit, ind_logit = self.model(features)
                
                # 预测（仅用共性）
                pred_common = torch.argmin(com_logit, dim=-1)
                
                # 预测（完整）
                pred_full = torch.argmin(ind_logit, dim=-1)
                
                # 统计
                for i in range(len(labels)):
                    label = labels[i].item()
                    
                    if classes_to_eval and label not in classes_to_eval:
                        continue
                    
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct_common[label] = 0
                        class_correct_full[label] = 0
                    
                    class_total[label] += 1
                    total_samples += 1
                    
                    if pred_common[i].item() == label:
                        class_correct_common[label] += 1
                        total_correct_common += 1
                    
                    if pred_full[i].item() == label:
                        class_correct_full[label] += 1
                        total_correct_full += 1
        
        metrics = {
            'accuracy_common': total_correct_common / max(total_samples, 1) * 100,
            'accuracy_full': total_correct_full / max(total_samples, 1) * 100,
            'total_samples': total_samples
        }
        
        # 每类准确率
        for cls in class_total:
            metrics[f'class_{cls}_acc_common'] = (
                class_correct_common[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
            metrics[f'class_{cls}_acc_full'] = (
                class_correct_full[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
        
        # GRPO Gain = Full - Common
        metrics['grpo_gain'] = metrics['accuracy_full'] - metrics['accuracy_common']
        
        return metrics


# 测试
if __name__ == "__main__":
    from models.backbone import create_backbone
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10
    
    # 创建组件
    backbone = create_backbone("resnet18", pretrained=True, frozen=True)
    model = FedPCIModel(
        num_classes=num_classes,
        input_dim=512,
        hidden_dim=256,
        output_dim=128
    )
    
    # 创建客户端
    client = FedPCIClient(
        client_id=0,
        num_classes=num_classes,
        backbone=backbone,
        model=model,
        device=device
    )
    
    # 设置本地数据信息
    client.setup_local_data(local_classes=[0, 1, 2])
    
    print(f"Client {client.client_id} created")
    print(f"Local classes: {client.local_classes}")
    print(f"Device: {device}")
    
    # 测试参数获取
    common_updates = client.get_common_updates()
    print(f"\nCommon updates classes: {list(common_updates.keys())}")
    
    prototype_updates = client.get_prototype_updates()
    print(f"Prototype updates classes: {list(prototype_updates.keys())}")