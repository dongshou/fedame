"""
专家网络模块
动态专家池，每个专家负责一个语义簇的细粒度分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy


class Expert(nn.Module):
    """
    单个专家网络
    负责特定语义簇的特征精炼和细粒度分类
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 512,
        expert_id: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.expert_id = expert_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 特征精炼网络
        self.refine_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 该专家负责的类别列表
        self.responsible_classes: List[int] = []
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def add_class(self, class_id: int):
        """添加负责的类别"""
        if class_id not in self.responsible_classes:
            self.responsible_classes.append(class_id)
    
    def remove_class(self, class_id: int):
        """移除负责的类别"""
        if class_id in self.responsible_classes:
            self.responsible_classes.remove(class_id)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, input_dim]
        
        Returns:
            refined: 精炼后的特征 [B, output_dim]
        """
        refined = self.refine_net(x)
        # L2归一化
        refined = F.normalize(refined, p=2, dim=-1)
        return refined


class ExpertPool(nn.Module):
    """
    动态专家池
    管理多个专家，支持动态添加、拆分
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_initial_experts: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 专家字典
        self.experts = nn.ModuleDict()
        
        # 初始化专家
        for i in range(num_initial_experts):
            self.add_expert(i)
        
        # 类别到专家的映射
        self.class_to_expert: Dict[int, int] = {}
    
    def add_expert(self, expert_id: Optional[int] = None) -> int:
        """
        添加新专家
        
        Returns:
            expert_id: 新专家的ID
        """
        if expert_id is None:
            expert_id = len(self.experts)
        
        expert = Expert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            expert_id=expert_id,
            dropout=self.dropout
        )
        
        self.experts[str(expert_id)] = expert
        return expert_id
    
    def get_expert(self, expert_id: int) -> Expert:
        """获取专家"""
        return self.experts[str(expert_id)]
    
    def assign_class_to_expert(self, class_id: int, expert_id: int):
        """将类别分配给专家"""
        self.class_to_expert[class_id] = expert_id
        self.experts[str(expert_id)].add_class(class_id)
    
    def get_expert_for_class(self, class_id: int) -> int:
        """获取负责某类别的专家ID"""
        return self.class_to_expert.get(class_id, 0)
    
    def split_expert(
        self,
        expert_id: int,
        class_groups: List[List[int]]
    ) -> List[int]:
        """
        拆分专家
        
        Args:
            expert_id: 要拆分的专家ID
            class_groups: 拆分后各新专家负责的类别
        
        Returns:
            new_expert_ids: 新专家的ID列表
        """
        old_expert = self.get_expert(expert_id)
        new_expert_ids = []
        
        for i, classes in enumerate(class_groups):
            if i == 0:
                # 第一组继续使用原专家
                new_id = expert_id
                old_expert.responsible_classes = classes
            else:
                # 创建新专家，继承原专家参数
                new_id = self.add_expert()
                new_expert = self.get_expert(new_id)
                new_expert.load_state_dict(old_expert.state_dict())
                new_expert.responsible_classes = classes
            
            # 更新类别到专家的映射
            for cls in classes:
                self.class_to_expert[cls] = new_id
            
            new_expert_ids.append(new_id)
        
        return new_expert_ids
    
    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        class_anchors: torch.Tensor,
        temperature: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        num_classes = class_anchors.size(0)
        device = x.device
        num_experts = len(self.experts)
        expert_id_list = [int(k) for k in self.experts.keys()]
        
        class_anchors_norm = F.normalize(class_anchors, p=2, dim=-1)
        
        all_expert_features = []
        all_expert_logits = []
        
        for exp_id_str, expert in self.experts.items():
            expert_feat = expert(x)  # [B, output_dim]
            all_expert_features.append(expert_feat)
            
            sim = torch.mm(expert_feat, class_anchors_norm.t()) / temperature
            
            responsible_classes = expert.responsible_classes
            mask = torch.zeros(num_classes, device=device)
            if len(responsible_classes) > 0:
                for cls in responsible_classes:
                    mask[cls] = 1.0
            
            masked_sim = sim + (1 - mask) * (-1e9)
            all_expert_logits.append(masked_sim)
        
        # 堆叠: [E, B, ...]
        stacked_features = torch.stack(all_expert_features, dim=0)  # [E, B, output_dim]
        stacked_logits = torch.stack(all_expert_logits, dim=0)      # [E, B, num_classes]
        
        # 构建选择索引
        expert_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, eid in enumerate(expert_id_list):
            expert_idx[expert_ids == eid] = i
        
        # 从 [E, B, output_dim] 中选择每个样本对应的专家输出
        stacked_features = stacked_features.permute(1, 0, 2)  # [B, E, output_dim]
        stacked_logits = stacked_logits.permute(1, 0, 2)      # [B, E, num_classes]
        
        # 用 gather 选择
        idx_feat = expert_idx.view(batch_size, 1, 1).expand(-1, -1, self.output_dim)  # [B, 1, output_dim]
        all_features = stacked_features.gather(1, idx_feat).squeeze(1)  # [B, output_dim]
        
        idx_logit = expert_idx.view(batch_size, 1, 1).expand(-1, -1, num_classes)  # [B, 1, num_classes]
        all_logits = stacked_logits.gather(1, idx_logit).squeeze(1)  # [B, num_classes]
        
        return all_logits, all_features
    
    def forward_single_expert(
        self,
        x: torch.Tensor,
        expert_id: int
    ) -> torch.Tensor:
        """
        使用单个专家处理所有输入
        """
        expert = self.get_expert(expert_id)
        return expert(x)
    
    @property
    def num_experts(self) -> int:
        return len(self.experts)
    
    def get_expert_info(self) -> Dict:
        """获取所有专家的信息"""
        info = {}
        for exp_id, expert in self.experts.items():
            info[int(exp_id)] = {
                'responsible_classes': expert.responsible_classes.copy(),
                'num_classes': len(expert.responsible_classes)
            }
        return info


# 测试
if __name__ == "__main__":
    # 创建专家池
    expert_pool = ExpertPool(
        input_dim=512,
        hidden_dim=256,
        output_dim=512,
        num_initial_experts=2
    )
    
    # 分配类别
    # 专家0负责动物：bird(2), cat(3), deer(4), dog(5), frog(6), horse(7)
    for cls in [2, 3, 4, 5, 6, 7]:
        expert_pool.assign_class_to_expert(cls, 0)
    
    # 专家1负责交通工具：airplane(0), automobile(1), ship(8), truck(9)
    for cls in [0, 1, 8, 9]:
        expert_pool.assign_class_to_expert(cls, 1)
    
    print("Expert Pool Info:")
    print(expert_pool.get_expert_info())
    
    # 测试前向传播
    x = torch.randn(8, 512)
    expert_ids = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    class_anchors = F.normalize(torch.randn(10, 512), dim=-1)
    
    logits, features = expert_pool(x, expert_ids, class_anchors)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    
    # 测试专家拆分
    print("\n--- Splitting Expert 0 ---")
    new_ids = expert_pool.split_expert(
        expert_id=0,
        class_groups=[[2, 3, 4], [5, 6, 7]]  # 拆分成两组
    )
    print(f"New expert IDs: {new_ids}")
    print("Expert Pool Info after split:")
    print(expert_pool.get_expert_info())
