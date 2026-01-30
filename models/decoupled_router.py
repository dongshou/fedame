"""
解耦路由器模块
N个独立的二分类网络，每个判断"是否属于类i"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class BinaryRouter(nn.Module):
    """
    单个类的二分类路由器
    
    学习一个特征变换W，使得：
    - 属于该类的特征变换后靠近对应锚点
    - 不属于该类的特征变换后远离对应锚点
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
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
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征变换
        
        Args:
            x: 输入特征 [B, input_dim]
        
        Returns:
            transformed: 变换后的特征 [B, output_dim]，已L2归一化
        """
        transformed = self.network(x)
        transformed = F.normalize(transformed, p=2, dim=-1)
        return transformed
    
    def compute_distance(
        self, 
        x: torch.Tensor, 
        anchor: torch.Tensor
    ) -> torch.Tensor:
        """
        计算变换后特征与锚点的距离
        
        Args:
            x: 输入特征 [B, input_dim]
            anchor: 类锚点 [output_dim]
        
        Returns:
            distance: 距离 [B]
        """
        transformed = self.forward(x)
        anchor = F.normalize(anchor.unsqueeze(0), p=2, dim=-1)  # [1, output_dim]
        
        # 欧氏距离
        distance = torch.norm(transformed - anchor, p=2, dim=-1)
        return distance
    
    def compute_similarity(
        self, 
        x: torch.Tensor, 
        anchor: torch.Tensor
    ) -> torch.Tensor:
        """
        计算变换后特征与锚点的余弦相似度
        
        Args:
            x: 输入特征 [B, input_dim]
            anchor: 类锚点 [output_dim]
        
        Returns:
            similarity: 相似度 [B]，范围[-1, 1]
        """
        transformed = self.forward(x)
        anchor = F.normalize(anchor.unsqueeze(0), p=2, dim=-1)  # [1, output_dim]
        
        # 余弦相似度
        similarity = torch.sum(transformed * anchor, dim=-1)
        return similarity


class DecoupledRouterPool(nn.Module):
    """
    解耦路由器池
    
    管理N个独立的BinaryRouter，每个负责一个类
    """
    
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 创建N个独立的Router
        self.routers = nn.ModuleDict({
            str(i): BinaryRouter(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout
            )
            for i in range(num_classes)
        })
        
        # 类锚点（CLIP生成，固定）
        self.register_buffer('class_anchors', None)
        
        # 视觉原型（从数据中提取，可更新）
        self.register_buffer('visual_prototypes', None)
        self.prototype_counts = {}  # 记录每个类的样本数
    
    def set_class_anchors(self, anchors: torch.Tensor):
        """
        设置类锚点（CLIP语义锚点）
        
        Args:
            anchors: 类锚点 [num_classes, output_dim]
        """
        self.class_anchors = F.normalize(anchors, p=2, dim=-1)
    
    def set_visual_prototypes(
        self, 
        prototypes: torch.Tensor,
        counts: Optional[Dict[int, int]] = None
    ):
        """
        设置视觉原型
        
        Args:
            prototypes: 视觉原型 [num_classes, input_dim]
            counts: 每个类的样本数
        """
        self.visual_prototypes = prototypes
        if counts is not None:
            self.prototype_counts = counts
    
    def update_visual_prototype(
        self, 
        class_id: int, 
        prototype: torch.Tensor,
        count: int
    ):
        """
        更新单个类的视觉原型
        
        Args:
            class_id: 类ID
            prototype: 原型 [input_dim]
            count: 样本数
        """
        if self.visual_prototypes is None:
            self.visual_prototypes = torch.zeros(
                self.num_classes, self.input_dim, 
                device=prototype.device
            )
        
        self.visual_prototypes[class_id] = prototype
        self.prototype_counts[class_id] = count
    
    def get_router(self, class_id: int) -> BinaryRouter:
        """获取指定类的Router"""
        return self.routers[str(class_id)]
    
    def compute_all_distances(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        计算特征与所有类锚点的距离
        
        Args:
            x: 输入特征 [B, input_dim]
        
        Returns:
            distances: 距离矩阵 [B, num_classes]
        """
        if self.class_anchors is None:
            raise ValueError("Class anchors not set. Call set_class_anchors first.")
        
        distances = []
        for i in range(self.num_classes):
            router = self.get_router(i)
            anchor = self.class_anchors[i]
            dist = router.compute_distance(x, anchor)
            distances.append(dist)
        
        return torch.stack(distances, dim=-1)  # [B, num_classes]
    
    def compute_all_similarities(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        计算特征与所有类锚点的相似度
        
        Args:
            x: 输入特征 [B, input_dim]
        
        Returns:
            similarities: 相似度矩阵 [B, num_classes]
        """
        if self.class_anchors is None:
            raise ValueError("Class anchors not set. Call set_class_anchors first.")
        
        similarities = []
        for i in range(self.num_classes):
            router = self.get_router(i)
            anchor = self.class_anchors[i]
            sim = router.compute_similarity(x, anchor)
            similarities.append(sim)
        
        return torch.stack(similarities, dim=-1)  # [B, num_classes]
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        路由决策
        
        Args:
            x: 输入特征 [B, input_dim]
        
        Returns:
            routed_classes: 路由到的类别 [B]
            distances: 距离矩阵 [B, num_classes]
            similarities: 相似度矩阵 [B, num_classes]
        """
        distances = self.compute_all_distances(x)
        similarities = self.compute_all_similarities(x)
        
        # 路由决策：选择距离最小（相似度最大）的类
        routed_classes = torch.argmin(distances, dim=-1)
        
        return routed_classes, distances, similarities
    
    def get_router_params(self, class_id: int) -> Dict[str, torch.Tensor]:
        """获取指定Router的参数"""
        router = self.get_router(class_id)
        return {name: param.data.clone() for name, param in router.named_parameters()}
    
    def set_router_params(self, class_id: int, params: Dict[str, torch.Tensor]):
        """设置指定Router的参数"""
        router = self.get_router(class_id)
        state_dict = router.state_dict()
        for name, value in params.items():
            if name in state_dict:
                state_dict[name] = value
        router.load_state_dict(state_dict)


class ContrastiveLoss(nn.Module):
    """
    对比学习损失
    
    用于训练解耦Router
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.1,
        hard_negative_k: int = 5
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.hard_negative_k = hard_negative_k
    
    def forward(
        self,
        router: BinaryRouter,
        anchor: torch.Tensor,
        pos_features: Optional[torch.Tensor],
        neg_features: Optional[torch.Tensor],
        pos_prototype: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对比损失
        
        Args:
            router: 二分类路由器
            anchor: 类锚点 [output_dim]
            pos_features: 正样本特征 [N_pos, input_dim]，可为None
            neg_features: 负样本特征 [N_neg, input_dim]，可为None
            pos_prototype: 正样本原型（当pos_features为None时使用）[input_dim]
        
        Returns:
            loss: 总损失
            metrics: 损失组成
        """
        loss = torch.tensor(0.0, device=anchor.device)
        metrics = {'loss_pos': 0.0, 'loss_neg': 0.0, 'loss_proto': 0.0}
        
        # 1. 正样本损失：拉近正样本与锚点
        if pos_features is not None and len(pos_features) > 0:
            pos_distances = router.compute_distance(pos_features, anchor)
            loss_pos = pos_distances.mean()
            loss = loss + loss_pos
            metrics['loss_pos'] = loss_pos.item()
        
        # 2. 原型损失：当没有正样本时，用原型作为伪正样本
        if (pos_features is None or len(pos_features) == 0) and pos_prototype is not None:
            proto_distance = router.compute_distance(
                pos_prototype.unsqueeze(0), anchor
            )
            loss_proto = proto_distance.mean()
            loss = loss + loss_proto
            metrics['loss_proto'] = loss_proto.item()
        
        # 3. 负样本损失：推远负样本与锚点（使用Hard Negative Mining）
        if neg_features is not None and len(neg_features) > 0:
            neg_distances = router.compute_distance(neg_features, anchor)
            
            # Hard Negative Mining：只选择最近的K个负样本
            if len(neg_features) > self.hard_negative_k:
                hard_neg_distances, _ = torch.topk(
                    neg_distances, 
                    self.hard_negative_k, 
                    largest=False
                )
            else:
                hard_neg_distances = neg_distances
            
            # Hinge Loss：margin - distance，希望距离大于margin
            loss_neg = F.relu(self.margin - hard_neg_distances).mean()
            loss = loss + loss_neg
            metrics['loss_neg'] = loss_neg.item()
        
        return loss, metrics


# 测试
if __name__ == "__main__":
    # 创建解耦路由器池
    num_classes = 10
    router_pool = DecoupledRouterPool(
        num_classes=num_classes,
        input_dim=512,
        hidden_dim=256,
        output_dim=512,
        num_layers=3
    )
    
    # 设置类锚点
    class_anchors = F.normalize(torch.randn(num_classes, 512), dim=-1)
    router_pool.set_class_anchors(class_anchors)
    
    # 测试前向传播
    x = torch.randn(4, 512)
    routed_classes, distances, similarities = router_pool(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Routed classes: {routed_classes}")
    print(f"Distances shape: {distances.shape}")
    print(f"Similarities shape: {similarities.shape}")
    
    # 测试对比损失
    loss_fn = ContrastiveLoss(margin=1.0, hard_negative_k=3)
    
    router_0 = router_pool.get_router(0)
    anchor_0 = class_anchors[0]
    
    pos_features = torch.randn(5, 512)  # 5个正样本
    neg_features = torch.randn(20, 512)  # 20个负样本
    
    loss, metrics = loss_fn(router_0, anchor_0, pos_features, neg_features)
    print(f"\nContrastive Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # 测试无正样本时使用原型
    pos_prototype = torch.randn(512)
    loss_with_proto, metrics_proto = loss_fn(
        router_0, anchor_0, 
        pos_features=None, 
        neg_features=neg_features,
        pos_prototype=pos_prototype
    )
    print(f"\nLoss with prototype: {loss_with_proto.item():.4f}")
    print(f"Metrics: {metrics_proto}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in router_pool.parameters())
    single_router_params = sum(p.numel() for p in router_pool.get_router(0).parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Single router parameters: {single_router_params:,}")
    print(f"Number of routers: {num_classes}")