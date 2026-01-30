"""
联邦学习服务端模块（解耦路由器版本）
管理N个独立Router的聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import copy

from models import (
    create_backbone,
    DecoupledRouterPool,
    ExpertPool
)
from anchor import create_anchor_generator, LLMDecisionMaker, ExpertManager


class DecoupledServer:
    """
    使用解耦路由器的联邦服务端
    
    核心特点：
    - 管理N个独立的Router，分别聚合
    - 管理全局视觉原型
    - 按正负样本数量加权聚合
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
        
        # 生成CLIP语义锚点
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
        
        # 注意：global_visual_prototypes 和 global_prototype_counts 
        # 已在 _init_global_model 中初始化，不要在这里覆盖
        
        # 已学习的类别
        self.learned_classes: List[int] = []
        
        # 客户端信息
        self.client_info: Dict[int, Dict] = {}
    
    def _generate_anchors(self):
        """生成全局CLIP语义锚点"""
        # 生成类锚点
        self.class_anchors = self.anchor_generator.generate_anchors(
            self.class_names
        ).to(self.device)
        
        # 生成簇锚点
        self.cluster_anchors = self.anchor_generator.generate_anchors(
            self.cluster_names
        ).to(self.device)
        
        print(f"   📐 Generated {len(self.class_anchors)} class anchors")
    
    def _init_global_model(self, config: Dict):
        """初始化全局模型"""
        # Backbone（冻结）
        self.backbone = create_backbone(
            backbone_type=config.get('backbone', 'resnet18'),
            pretrained=config.get('backbone_pretrained', True),
            frozen=True
        ).to(self.device)
        
        # 解耦路由器池（N个独立的Router）
        self.global_router_pool = DecoupledRouterPool(
            num_classes=self.num_classes,
            input_dim=config.get('feature_dim', 512),
            hidden_dim=config.get('router_hidden_dim', 256),
            output_dim=config.get('anchor_dim', 512),
            num_layers=config.get('router_num_layers', 3),
            dropout=config.get('router_dropout', 0.1)
        ).to(self.device)
        
        # 设置CLIP锚点
        self.global_router_pool.set_class_anchors(self.class_anchors)
        
        # 专家池
        self.global_expert_pool = ExpertPool(
            input_dim=config.get('anchor_dim', 512),
            hidden_dim=config.get('expert_hidden_dim', 256),
            output_dim=config.get('expert_output_dim', 512),
            num_initial_experts=self.num_classes
        ).to(self.device)
        
        # 初始化全局视觉原型为零
        self.global_visual_prototypes = torch.zeros(
            self.num_classes, 
            config.get('feature_dim', 512),
            device=self.device
        )
        self.global_prototype_counts: Dict[int, int] = {}
    
    def _sync_expert_assignments(self):
        """同步专家分配到专家池"""
        for exp_id, info in self.expert_manager.expert_info.items():
            for cls in info['responsible_classes']:
                self.global_expert_pool.assign_class_to_expert(cls, exp_id)
    
    def prepare_task(self, task_classes: List[int]) -> Dict:
        """
        准备新任务
        
        Args:
            task_classes: 任务包含的类别ID
        
        Returns:
            task_info: 任务信息
        """
        new_classes = [c for c in task_classes if c not in self.learned_classes]
        
        # 为新类分配专家
        for cls in new_classes:
            class_name = self.class_names[cls]
            expert_id, cluster = self.expert_manager.assign_new_class(
                class_name,
                self.class_anchors,
                self.cluster_anchors
            )
            self.global_expert_pool.assign_class_to_expert(cls, expert_id)
        
        # 构建任务信息
        task_info = {
            'task_classes': task_classes,
            'new_classes': new_classes,
            'old_classes': self.learned_classes.copy(),
            'expert_assignments': self.global_expert_pool.class_to_expert.copy(),
            'expert_info': self.global_expert_pool.get_expert_info()
        }
        
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
        
        # 获取所有Router的参数
        router_params = {}
        for i in range(self.num_classes):
            router_params[i] = self.global_router_pool.get_router_params(i)
        
        # 获取相关专家的参数
        expert_states = {}
        for exp_id in needed_experts:
            expert = self.global_expert_pool.get_expert(exp_id)
            expert_states[exp_id] = expert.state_dict()
        
        # 获取全局视觉原型
        prototype_info = {
            'prototypes': self.global_visual_prototypes.cpu(),
            'counts': self.global_prototype_counts.copy()
        }
        
        return {
            'client_id': client_id,
            'local_classes': client_classes,
            'local_experts': list(needed_experts),
            'class_to_expert': self.global_expert_pool.class_to_expert.copy(),
            'router_params': router_params,
            'expert_states': expert_states,
            'prototype_info': prototype_info,
            'class_anchors': self.class_anchors.cpu()
        }
    
    def aggregate(
        self,
        client_router_updates: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
        client_prototype_updates: Dict[int, Dict[int, Dict]],
        client_train_stats: Dict[int, Dict[int, Dict]]
    ):
        """
        聚合客户端更新
        
        Args:
            client_router_updates: {client_id: {class_id: {param_name: param_value}}}
            client_prototype_updates: {client_id: {class_id: {'prototype': tensor, 'count': int}}}
            client_train_stats: {client_id: {class_id: {'pos_count': int, 'neg_count': int}}}
        """
        # 1. 聚合每个Router（按正负样本数量加权）
        self._aggregate_routers(client_router_updates, client_train_stats)
        
        # 2. 聚合视觉原型
        self._aggregate_prototypes(client_prototype_updates)
        
        # 3. 聚合专家（如果需要的话）
        # 这里暂时不聚合专家，因为专家是按类别分配的
    
    def _aggregate_routers(
        self,
        client_router_updates: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
        client_train_stats: Dict[int, Dict[int, Dict]]
    ):
        """
        聚合所有Router
        
        对于每个Router_i：
        - 收集所有客户端的Router_i参数
        - 按 α*正样本数 + β*负样本数 加权
        - α > β，给正样本更高权重
        """
        if len(client_router_updates) == 0:
            return
        
        alpha = 2.0  # 正样本权重
        beta = 1.0   # 负样本权重
        
        # 对每个类的Router分别聚合
        for class_id in range(self.num_classes):
            # 收集该类Router的所有更新
            class_updates = []
            class_weights = []
            
            for client_id, router_updates in client_router_updates.items():
                if class_id in router_updates:
                    params = router_updates[class_id]
                    
                    # 获取该客户端对该类的训练统计
                    stats = client_train_stats.get(client_id, {}).get(class_id, {})
                    pos_count = stats.get('pos_count', 0)
                    neg_count = stats.get('neg_count', 0)
                    
                    # 计算权重
                    weight = alpha * pos_count + beta * neg_count
                    
                    # 只有有训练数据的才参与聚合
                    if weight > 0:
                        class_updates.append(params)
                        class_weights.append(weight)
            
            # 如果有更新，进行加权聚合
            if len(class_updates) > 0:
                total_weight = sum(class_weights)
                normalized_weights = [w / total_weight for w in class_weights]
                
                # 聚合参数
                aggregated_params = {}
                for param_name in class_updates[0].keys():
                    aggregated_params[param_name] = sum(
                        w * upd[param_name].to(self.device)
                        for w, upd in zip(normalized_weights, class_updates)
                    )
                
                # 更新全局Router
                self.global_router_pool.set_router_params(class_id, aggregated_params)
    
    def _aggregate_prototypes(
        self,
        client_prototype_updates: Dict[int, Dict[int, Dict]]
    ):
        """
        聚合视觉原型
        
        按样本数量加权平均
        """
        if len(client_prototype_updates) == 0:
            return
        
        # 对每个类分别聚合
        for class_id in range(self.num_classes):
            prototypes = []
            counts = []
            
            for client_id, proto_updates in client_prototype_updates.items():
                if class_id in proto_updates:
                    proto_info = proto_updates[class_id]
                    prototypes.append(proto_info['prototype'])
                    counts.append(proto_info['count'])
            
            # 如果有原型更新
            if len(prototypes) > 0:
                total_count = sum(counts)
                if total_count > 0:
                    # 加权平均
                    weights = [c / total_count for c in counts]
                    aggregated_proto = sum(
                        w * p.to(self.device)
                        for w, p in zip(weights, prototypes)
                    )
                    
                    self.global_visual_prototypes[class_id] = aggregated_proto
                    self.global_prototype_counts[class_id] = total_count
    
    def finish_task(self, task_classes: List[int]):
        """完成任务，更新已学习类别"""
        for cls in task_classes:
            if cls not in self.learned_classes:
                self.learned_classes.append(cls)
    
    def evaluate(
        self,
        test_loader: DataLoader,
        classes_to_eval: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        评估全局模型
        
        Args:
            test_loader: 测试数据加载器
            classes_to_eval: 要评估的类别
        
        Returns:
            metrics: 评估指标
        """
        self.backbone.eval()
        self.global_router_pool.eval()
        self.global_expert_pool.eval()
        
        total_correct = 0
        total_samples = 0
        routing_correct = 0
        
        class_correct = {}
        class_total = {}
        class_routing_correct = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                features = self.backbone(images)
                
                # 路由决策
                routed_classes, distances, similarities = self.global_router_pool(features)
                
                # 目标专家
                target_experts = torch.tensor(
                    [self.global_expert_pool.get_expert_for_class(l.item()) for l in labels],
                    device=self.device
                )
                
                # 路由准确率
                routing_match = (routed_classes == labels)
                routing_correct += routing_match.sum().item()
                
                # 使用路由的专家进行分类
                cls_logits, _ = self.global_expert_pool(
                    features, routed_classes, self.class_anchors
                )
                _, predicted = cls_logits.max(1)
                
                # 统计
                for i in range(len(labels)):
                    label = labels[i].item()
                    
                    if classes_to_eval and label not in classes_to_eval:
                        continue
                    
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                        class_routing_correct[label] = 0
                    
                    class_total[label] += 1
                    total_samples += 1
                    
                    if predicted[i].item() == label:
                        class_correct[label] += 1
                        total_correct += 1
                    
                    if routing_match[i].item():
                        class_routing_correct[label] += 1
        
        metrics = {
            'accuracy': total_correct / max(total_samples, 1) * 100,
            'routing_accuracy': routing_correct / max(total_samples, 1) * 100,
            'total_samples': total_samples
        }
        
        for cls in class_total:
            metrics[f'class_{cls}_acc'] = (
                class_correct[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
            metrics[f'class_{cls}_routing_acc'] = (
                class_routing_correct[cls] / class_total[cls] * 100
                if class_total.get(cls, 0) > 0 else 0
            )
        
        return metrics
    
    def diagnose_routing(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        诊断路由情况
        
        Args:
            test_loader: 测试数据加载器
            class_names: 类别名称列表
        
        Returns:
            diagnosis: 路由诊断结果
        """
        self.backbone.eval()
        self.global_router_pool.eval()
        
        if class_names is None:
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        
        # 统计每个类被路由到哪些类
        class_routing_stats = {}  # {true_class: {routed_class: count}}
        class_total_samples = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                features = self.backbone(images)
                routed_classes, _, _ = self.global_router_pool(features)
                
                for i in range(len(labels)):
                    true_cls = labels[i].item()
                    routed_cls = routed_classes[i].item()
                    
                    if true_cls not in class_routing_stats:
                        class_routing_stats[true_cls] = {}
                        class_total_samples[true_cls] = 0
                    
                    if routed_cls not in class_routing_stats[true_cls]:
                        class_routing_stats[true_cls][routed_cls] = 0
                    
                    class_routing_stats[true_cls][routed_cls] += 1
                    class_total_samples[true_cls] += 1
        
        # 计算路由准确率
        class_routing_accuracy = {}
        total_correct = 0
        total_samples = 0
        
        for cls in class_routing_stats:
            correct = class_routing_stats[cls].get(cls, 0)
            total = class_total_samples[cls]
            class_routing_accuracy[cls] = correct / total if total > 0 else 0
            total_correct += correct
            total_samples += total
        
        return {
            'class_routing_stats': class_routing_stats,
            'class_total_samples': class_total_samples,
            'class_routing_accuracy': class_routing_accuracy,
            'overall_accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'total_correct': total_correct,
            'total_samples': total_samples,
            'class_names': class_names
        }
    
    def get_global_model_state(self) -> Dict:
        """获取全局模型状态"""
        router_params = {}
        for i in range(self.num_classes):
            router_params[i] = self.global_router_pool.get_router_params(i)
        
        return {
            'router_params': router_params,
            'experts': {
                int(k): v.state_dict()
                for k, v in self.global_expert_pool.experts.items()
            },
            'visual_prototypes': self.global_visual_prototypes.cpu(),
            'prototype_counts': self.global_prototype_counts.copy(),
            'learned_classes': self.learned_classes.copy(),
            'expert_assignments': self.global_expert_pool.class_to_expert.copy()
        }
    
    def load_global_model_state(self, state: Dict):
        """加载全局模型状态"""
        # 加载Router参数
        for class_id, params in state['router_params'].items():
            self.global_router_pool.set_router_params(class_id, params)
        
        # 加载专家参数
        for exp_id, exp_state in state['experts'].items():
            self.global_expert_pool.get_expert(exp_id).load_state_dict(exp_state)
        
        # 加载视觉原型
        self.global_visual_prototypes = state['visual_prototypes'].to(self.device)
        self.global_prototype_counts = state['prototype_counts']
        
        # 加载其他信息
        self.learned_classes = state['learned_classes']


# 测试
if __name__ == "__main__":
    # 简单测试
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cluster_config = {
        'animals': ['cat', 'dog', 'bird', 'deer', 'frog', 'horse'],
        'vehicles': ['airplane', 'automobile', 'ship', 'truck']
    }
    
    model_config = {
        'backbone': 'resnet18',
        'backbone_pretrained': True,
        'feature_dim': 512,
        'router_hidden_dim': 256,
        'anchor_dim': 512,
        'router_num_layers': 3,
        'router_dropout': 0.1,
        'expert_hidden_dim': 256,
        'expert_output_dim': 512
    }
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    server = DecoupledServer(
        num_classes=10,
        class_names=class_names,
        cluster_config=cluster_config,
        model_config=model_config,
        device=device,
        use_clip=False,
        use_real_llm=False
    )
    
    print(f"Server created on {device}")
    print(f"Number of classes: {server.num_classes}")
    print(f"Router pool parameters: {sum(p.numel() for p in server.global_router_pool.parameters()):,}")