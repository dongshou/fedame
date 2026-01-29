"""
路由层模块
可学习的投影层，用于将Backbone特征映射到锚点空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Router(nn.Module):
    """
    路由层
    将Backbone特征投影到与锚点对齐的空间
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
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
        Args:
            x: Backbone特征 [B, input_dim]
        
        Returns:
            projected: 投影后的特征 [B, output_dim]
        """
        projected = self.projector(x)
        # L2归一化，便于计算余弦相似度
        projected = F.normalize(projected, p=2, dim=-1)
        return projected


class AnchorBasedRouter(nn.Module):
    """
    基于锚点的路由器
    结合路由层和锚点进行路由决策
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        anchor_dim: int = 512,
        temperature: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.router = Router(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=anchor_dim,
            dropout=dropout
        )
        
        self.temperature = temperature
        self.anchor_dim = anchor_dim
        
        # 簇锚点（用于路由）
        self.cluster_anchors: Optional[torch.Tensor] = None
        # 类锚点（用于分类）
        self.class_anchors: Optional[torch.Tensor] = None
        # 簇到专家的映射
        self.cluster_to_expert: Optional[dict] = None
    
    def set_cluster_anchors(
        self,
        anchors: torch.Tensor,
        cluster_to_expert: dict
    ):
        """
        设置簇锚点
        
        Args:
            anchors: 簇锚点 [num_clusters, anchor_dim]
            cluster_to_expert: 簇ID到专家ID的映射
        """
        self.register_buffer('cluster_anchors', anchors)
        self.cluster_to_expert = cluster_to_expert
    
    def set_class_anchors(self, anchors: torch.Tensor):
        """
        设置类锚点
        
        Args:
            anchors: 类锚点 [num_classes, anchor_dim]
        """
        self.register_buffer('class_anchors', anchors)
    
    def compute_routing_logits(
        self,
        features: torch.Tensor,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        计算路由logits
        
        Args:
            features: 投影后的特征 [B, anchor_dim]
            anchors: 锚点 [N, anchor_dim]
        
        Returns:
            logits: 路由logits [B, N]
        """
        # 归一化
        features = F.normalize(features, p=2, dim=-1)
        anchors = F.normalize(anchors, p=2, dim=-1)
        
        # 余弦相似度
        logits = torch.mm(features, anchors.t()) / self.temperature
        return logits
    
    def forward(
        self,
        x: torch.Tensor,
        return_projected: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Backbone特征 [B, input_dim]
            return_projected: 是否返回投影特征
        
        Returns:
            expert_ids: 选择的专家ID [B]
            routing_probs: 路由概率 [B, num_clusters]
            projected: 投影后的特征 [B, anchor_dim]（如果return_projected=True）
        """
        # 投影
        projected = self.router(x)
        
        if self.cluster_anchors is None:
            raise ValueError("Cluster anchors not set. Call set_cluster_anchors first.")
        
        # 计算与簇锚点的相似度
        routing_logits = self.compute_routing_logits(projected, self.cluster_anchors)
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Top-1 路由
        expert_indices = torch.argmax(routing_probs, dim=-1)
        
        # 映射到专家ID
        if self.cluster_to_expert is not None:
            expert_ids = torch.tensor(
                [self.cluster_to_expert[idx.item()] for idx in expert_indices],
                device=x.device
            )
        else:
            expert_ids = expert_indices
        
        if return_projected:
            return expert_ids, routing_probs, projected
        else:
            return expert_ids, routing_probs, None
    
    def compute_class_logits(
        self,
        features: torch.Tensor,
        class_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算类别logits（用于分类损失和对比损失）
        
        Args:
            features: 投影后的特征 [B, anchor_dim]
            class_indices: 如果指定，只计算这些类的logits
        
        Returns:
            logits: 类别logits [B, num_classes]
        """
        if self.class_anchors is None:
            raise ValueError("Class anchors not set. Call set_class_anchors first.")
        
        if class_indices is not None:
            anchors = self.class_anchors[class_indices]
        else:
            anchors = self.class_anchors
        
        logits = self.compute_routing_logits(features, anchors)
        return logits


# 测试
if __name__ == "__main__":
    # 创建路由器
    router = AnchorBasedRouter(
        input_dim=512,
        hidden_dim=256,
        anchor_dim=512,
        temperature=0.1
    )
    
    # 模拟簇锚点（2个簇）
    cluster_anchors = F.normalize(torch.randn(2, 512), dim=-1)
    router.set_cluster_anchors(
        cluster_anchors,
        cluster_to_expert={0: 0, 1: 1}
    )
    
    # 模拟类锚点（10个类）
    class_anchors = F.normalize(torch.randn(10, 512), dim=-1)
    router.set_class_anchors(class_anchors)
    
    # 测试前向传播
    x = torch.randn(4, 512)
    expert_ids, routing_probs, projected = router(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Expert IDs: {expert_ids}")
    print(f"Routing probs shape: {routing_probs.shape}")
    print(f"Projected shape: {projected.shape}")
    
    # 测试类别logits
    class_logits = router.compute_class_logits(projected)
    print(f"Class logits shape: {class_logits.shape}")
