"""
联邦学习服务端模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy

from models import (
    create_backbone,
    AnchorBasedRouter,
    ExpertPool,
    DistributionPool,
    aggregate_distributions
)
from anchor import (
    create_anchor_generator,
    LLMDecisionMaker,
    ExpertManager
)


class FedAMEServer:
    """
    FedAME服务端
    管理全局模型、锚点、专家分配
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        cluster_config: Dict[str, List[str]],
        model_config: Dict,
        device: str = "cuda",
        use_clip: bool = False,
        use_real_llm: bool = False
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.cluster_config = cluster_config
        self.cluster_names = list(cluster_config.keys())
        self.device = device
        
        # 初始化锚点生成器
        self.anchor_generator = create_anchor_generator(
            use_clip=use_clip,
            device=device
        )
        
        # 生成锚点
        self._generate_anchors()
        
        # 初始化全局模型组件
        self._init_global_model(model_config)
        
        # 初始化LLM决策器和专家管理器
        self.llm_decision_maker = LLMDecisionMaker(use_real_llm=use_real_llm)
        self.expert_manager = ExpertManager(
            self.llm_decision_maker,
            class_names,
            self.cluster_names
        )
        self.expert_manager.initialize_experts(cluster_config)
        
        # 同步专家池的类别分配
        self._sync_expert_assignments()
        
        # 已学习的类别
        self.learned_classes: List[int] = []
        
        # 客户端信息
        self.client_info: Dict[int, Dict] = {}
    
    def _generate_anchors(self):
        """生成全局锚点"""
        # 生成类锚点
        self.class_anchors = self.anchor_generator.generate_anchors(
            self.class_names
        ).to(self.device)
        
        # 生成簇锚点
        self.cluster_anchors = self.anchor_generator.generate_anchors(
            self.cluster_names
        ).to(self.device)
        
        print(f"Generated class anchors: {self.class_anchors.shape}")
        print(f"Generated cluster anchors: {self.cluster_anchors.shape}")
    
    def _init_global_model(self, config: Dict):
        """初始化全局模型"""
        # Backbone（冻结）
        self.backbone = create_backbone(
            backbone_type=config.get('backbone', 'resnet18'),
            pretrained=config.get('backbone_pretrained', True),
            frozen=True
        ).to(self.device)
        
        # 路由层
        self.global_router = AnchorBasedRouter(
            input_dim=config.get('feature_dim', 512),
            hidden_dim=config.get('router_hidden_dim', 256),
            anchor_dim=config.get('anchor_dim', 512),
            temperature=config.get('temperature_route', 0.1)
        ).to(self.device)
        
        # 设置锚点
        self.global_router.set_class_anchors(self.class_anchors)
        self.global_router.set_cluster_anchors(
            self.cluster_anchors,
            {i: i for i in range(len(self.cluster_names))}  # 初始映射
        )
        
        # 专家池
        self.global_expert_pool = ExpertPool(
            input_dim=config.get('anchor_dim', 512),
            hidden_dim=config.get('expert_hidden_dim', 256),
            output_dim=config.get('expert_output_dim', 512),
            num_initial_experts=len(self.cluster_names)
        ).to(self.device)
        
        # 全局分布池
        self.global_distribution_pool = DistributionPool(
            anchor_dim=config.get('anchor_dim', 512)
        )
        self.global_distribution_pool.set_anchors(self.class_anchors)
    
    def _sync_expert_assignments(self):
        """同步专家分配到专家池"""
        for exp_id, info in self.expert_manager.expert_info.items():
            for cls in info['responsible_classes']:
                self.global_expert_pool.assign_class_to_expert(cls, exp_id)
    
    def prepare_task(
        self,
        task_classes: List[int]
    ) -> Dict:
        """
        准备新任务
        
        Args:
            task_classes: 任务包含的类别ID
        
        Returns:
            task_info: 任务信息，包括专家分配等
        """
        new_classes = [c for c in task_classes if c not in self.learned_classes]
        
        print(f"\n{'='*50}")
        print(f"Preparing task with classes: {[self.class_names[c] for c in task_classes]}")
        print(f"New classes: {[self.class_names[c] for c in new_classes]}")
        
        # 为新类分配专家
        for cls in new_classes:
            class_name = self.class_names[cls]
            expert_id, cluster = self.expert_manager.assign_new_class(
                class_name,
                self.class_anchors,
                self.cluster_anchors
            )
            
            # 同步到专家池
            self.global_expert_pool.assign_class_to_expert(cls, expert_id)
            
            # 添加到分布池
            if not self.global_distribution_pool.has_class(cls):
                self.global_distribution_pool.add_class(cls)
        
        # 检查是否需要拆分专家
        for exp_id in list(self.expert_manager.expert_info.keys()):
            split_result = self.expert_manager.check_and_split_expert(
                exp_id,
                self.class_anchors,
                max_classes=6
            )
            
            if split_result:
                # 在专家池中执行拆分
                groups = []
                for new_exp_id in split_result:
                    groups.append(
                        self.expert_manager.expert_info[new_exp_id]['responsible_classes']
                    )
                
                if len(groups) > 1:
                    self.global_expert_pool.split_expert(exp_id, groups)
        
        # 构建任务信息
        task_info = {
            'task_classes': task_classes,
            'new_classes': new_classes,
            'old_classes': self.learned_classes.copy(),
            'expert_assignments': self.global_expert_pool.class_to_expert.copy(),
            'expert_info': self.global_expert_pool.get_expert_info()
        }
        
        print(f"Expert assignments: {task_info['expert_assignments']}")
        print(f"Expert info: {task_info['expert_info']}")
        print('='*50)
        
        return task_info
    
    def get_client_config(
        self,
        client_id: int,
        client_classes: List[int]
    ) -> Dict:
        """
        获取客户端配置
        
        Args:
            client_id: 客户端ID
            client_classes: 客户端拥有的类别
        
        Returns:
            config: 客户端配置
        """
        # 确定客户端需要的专家
        needed_experts = set()
        for cls in client_classes:
            exp_id = self.global_expert_pool.get_expert_for_class(cls)
            needed_experts.add(exp_id)
        
        # 保存客户端信息
        self.client_info[client_id] = {
            'classes': client_classes,
            'experts': list(needed_experts)
        }
        
        # 获取相关专家的参数
        expert_states = {}
        for exp_id in needed_experts:
            expert = self.global_expert_pool.get_expert(exp_id)
            expert_states[exp_id] = expert.state_dict()
        
        # 获取相关类的分布参数
        distribution_params = {}
        for cls in client_classes:
            if self.global_distribution_pool.has_class(cls):
                distribution_params[cls] = (
                    self.global_distribution_pool.get_distribution(cls).get_params()
                )
        
        return {
            'client_id': client_id,
            'local_classes': client_classes,
            'local_experts': list(needed_experts),
            'class_to_expert': {
                cls: self.global_expert_pool.get_expert_for_class(cls)
                for cls in client_classes
            },
            'router_state': self.global_router.state_dict(),
            'expert_states': expert_states,
            'distribution_params': distribution_params,
            'class_anchors': self.class_anchors.clone(),
            'cluster_anchors': self.cluster_anchors.clone()
        }
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict],
        client_distribution_params: Dict[int, Dict[int, Dict]]
    ):
        """
        聚合客户端更新
        
        Args:
            client_updates: {client_id: {param_name: param_value}}
            client_distribution_params: {client_id: {class_id: params}}
        """
        print("\nAggregating client updates...")
        
        # 1. 聚合路由层（所有客户端参与）
        self._aggregate_router(client_updates)
        
        # 2. 聚合专家（按专家分组）
        self._aggregate_experts(client_updates)
        
        # 3. 聚合分布参数
        self._aggregate_distributions(client_distribution_params)
        
        print("Aggregation complete.")
    
    def _aggregate_router(self, client_updates: Dict[int, Dict]):
        """聚合路由层"""
        if len(client_updates) == 0:
            return
        
        # 收集路由层参数
        router_params = {}
        
        for client_id, updates in client_updates.items():
            client_weight = 1.0 / len(client_updates)  # 简单平均
            
            for name, value in updates.items():
                if name.startswith('router.'):
                    param_name = name[7:]  # 移除 'router.' 前缀
                    if param_name not in router_params:
                        router_params[param_name] = torch.zeros_like(value)
                    router_params[param_name] += client_weight * value
        
        # 更新全局路由层
        global_state = self.global_router.state_dict()
        for name, value in router_params.items():
            if name in global_state:
                global_state[name] = value
        
        self.global_router.load_state_dict(global_state)
    
    def _aggregate_experts(self, client_updates: Dict[int, Dict]):
        """聚合专家（按专家分组）"""
        # 收集每个专家的更新
        expert_updates: Dict[int, List[Dict]] = {}
        
        for client_id, updates in client_updates.items():
            for name, value in updates.items():
                if name.startswith('expert.'):
                    parts = name.split('.')
                    exp_id = int(parts[1])
                    param_name = '.'.join(parts[2:])
                    
                    if exp_id not in expert_updates:
                        expert_updates[exp_id] = []
                    
                    # 查找或创建该客户端对该专家的更新
                    found = False
                    for upd in expert_updates[exp_id]:
                        if upd['client_id'] == client_id:
                            upd['params'][param_name] = value
                            found = True
                            break
                    
                    if not found:
                        expert_updates[exp_id].append({
                            'client_id': client_id,
                            'params': {param_name: value}
                        })
        
        # 聚合每个专家
        for exp_id, updates_list in expert_updates.items():
            if len(updates_list) == 0:
                continue
            
            expert = self.global_expert_pool.get_expert(exp_id)
            global_state = expert.state_dict()
            
            # 简单平均
            weight = 1.0 / len(updates_list)
            
            aggregated = {}
            for upd in updates_list:
                for name, value in upd['params'].items():
                    if name not in aggregated:
                        aggregated[name] = torch.zeros_like(value)
                    aggregated[name] += weight * value
            
            for name, value in aggregated.items():
                if name in global_state:
                    global_state[name] = value
            
            expert.load_state_dict(global_state)
    
    def _aggregate_distributions(
        self,
        client_distribution_params: Dict[int, Dict[int, Dict]]
    ):
        """聚合分布参数"""
        if len(client_distribution_params) == 0:
            return
        
        # 转换格式
        params_list = list(client_distribution_params.values())
        
        # 聚合
        global_params = aggregate_distributions(
            params_list,
            self.class_anchors.cpu(),
            anchor_dim=self.class_anchors.size(1)
        )
        
        # 更新全局分布
        for cls, params in global_params.items():
            if self.global_distribution_pool.has_class(cls):
                self.global_distribution_pool.set_class_params(cls, params)
    
    def finish_task(self, task_classes: List[int]):
        """
        完成任务，更新已学习类别
        """
        for cls in task_classes:
            if cls not in self.learned_classes:
                self.learned_classes.append(cls)
        
        print(f"Task finished. Learned classes: {self.learned_classes}")
    
    def evaluate(
        self,
        test_loader,
        classes_to_eval: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        评估全局模型
        """
        self.backbone.eval()
        self.global_router.eval()
        self.global_expert_pool.eval()
        
        total_correct = 0
        total_samples = 0
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                backbone_features = self.backbone(images)
                expert_ids, _, projected = self.global_router(backbone_features)
                cls_logits, _ = self.global_expert_pool(
                    projected, expert_ids, self.class_anchors
                )
                
                _, predicted = cls_logits.max(1)
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    
                    if classes_to_eval and label not in classes_to_eval:
                        continue
                    
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1
                        total_correct += 1
                    total_samples += 1
        
        metrics = {
            'accuracy': total_correct / max(total_samples, 1) * 100,
            'total_samples': total_samples
        }
        
        for cls in class_total:
            metrics[f'class_{cls}_acc'] = (
                class_correct[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
        
        return metrics
    
    def get_global_model_state(self) -> Dict:
        """获取全局模型状态"""
        return {
            'router': self.global_router.state_dict(),
            'experts': {
                int(k): v.state_dict()
                for k, v in self.global_expert_pool.experts.items()
            },
            'distributions': self.global_distribution_pool.get_all_params(),
            'learned_classes': self.learned_classes.copy(),
            'expert_assignments': self.global_expert_pool.class_to_expert.copy()
        }
    
    def load_global_model_state(self, state: Dict):
        """加载全局模型状态"""
        self.global_router.load_state_dict(state['router'])
        
        for exp_id, exp_state in state['experts'].items():
            self.global_expert_pool.get_expert(exp_id).load_state_dict(exp_state)
        
        for cls, params in state['distributions'].items():
            if self.global_distribution_pool.has_class(cls):
                self.global_distribution_pool.set_class_params(cls, params)
        
        self.learned_classes = state['learned_classes']
