"""
Backbone 特征提取器
使用预训练的 ResNet-18，冻结参数
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNetBackbone(nn.Module):
    """
    ResNet-18 Backbone
    移除最后的全连接层，输出特征向量
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 512,
        frozen: bool = True
    ):
        super().__init__()
        
        # 加载预训练的ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = feature_dim
        
        # 冻结参数
        if frozen:
            self.freeze()
    
    def freeze(self):
        """冻结所有参数"""
        for param in self.features.parameters():
            param.requires_grad = False
        self.eval()
    
    def unfreeze(self):
        """解冻所有参数"""
        for param in self.features.parameters():
            param.requires_grad = True
        self.train()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            features: 特征向量 [B, feature_dim]
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features
    
    def train(self, mode: bool = True):
        """重写train方法，保持冻结状态"""
        # 始终保持eval模式（因为使用BatchNorm）
        super().train(False)
        return self


class BackboneWithAdapter(nn.Module):
    """
    带有轻量级适配层的Backbone
    Backbone冻结，只训练适配层
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 512,
        adapter_dim: int = 128,
        frozen_backbone: bool = True
    ):
        super().__init__()
        
        self.backbone = ResNetBackbone(
            pretrained=pretrained,
            feature_dim=feature_dim,
            frozen=frozen_backbone
        )
        
        # 轻量级适配层
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, adapter_dim),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_dim, feature_dim)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            features: 适配后的特征 [B, feature_dim]
        """
        with torch.no_grad():
            features = self.backbone(x)
        
        # 残差适配
        adapted = features + self.adapter(features)
        return adapted


def create_backbone(
    backbone_type: str = "resnet18",
    pretrained: bool = True,
    feature_dim: int = 512,
    frozen: bool = True,
    use_adapter: bool = False,
    adapter_dim: int = 128
) -> nn.Module:
    """
    创建Backbone
    
    Args:
        backbone_type: backbone类型
        pretrained: 是否使用预训练权重
        feature_dim: 特征维度
        frozen: 是否冻结
        use_adapter: 是否使用适配层
        adapter_dim: 适配层维度
    """
    if backbone_type == "resnet18":
        if use_adapter:
            return BackboneWithAdapter(
                pretrained=pretrained,
                feature_dim=feature_dim,
                adapter_dim=adapter_dim,
                frozen_backbone=frozen
            )
        else:
            return ResNetBackbone(
                pretrained=pretrained,
                feature_dim=feature_dim,
                frozen=frozen
            )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


# 测试
if __name__ == "__main__":
    # 测试Backbone
    backbone = create_backbone(
        backbone_type="resnet18",
        pretrained=True,
        frozen=True
    )
    
    # 测试输入
    x = torch.randn(4, 3, 32, 32)
    features = backbone(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Backbone frozen: {not any(p.requires_grad for p in backbone.parameters())}")
