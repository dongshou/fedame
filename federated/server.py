"""
联邦学习服务端模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
            dim=config.get('feature_dim', 512)  # backbone 输出维度
        )
        self.global_distribution_pool = DistributionPool(
            dim=config.get('feature_dim', 512)  # backbone 输出维度
        )
    
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
        
        return task_info
    
    def get_expert_to_cluster(self) -> Dict[int, int]:
        """获取专家到簇的映射"""
        expert_to_cluster = {}
        for exp_id, info in self.expert_manager.expert_info.items():
            cluster_name = info.get('cluster', self.cluster_names[0])
            cluster_idx = self.cluster_names.index(cluster_name) if cluster_name in self.cluster_names else 0
            expert_to_cluster[exp_id] = cluster_idx
        return expert_to_cluster
    
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
        
        # 获取所有已学习类的分布参数（用于伪样本训练）
        distribution_params = {}
        for cls in self.learned_classes:
            if self.global_distribution_pool.has_class(cls):
                distribution_params[cls] = (
                    self.global_distribution_pool.get_distribution(cls).get_params()
                )
        
        return {
            'client_id': client_id,
            'local_classes': client_classes,
            'local_experts': list(needed_experts),
            # 'class_to_expert': {
            #     cls: self.global_expert_pool.get_expert_for_class(cls)
            #     for cls in self.learned_classes  # 所有已学习的类
            # },
            'class_to_expert': self.global_expert_pool.class_to_expert.copy(),
            'expert_to_cluster': self.get_expert_to_cluster(),
            'router_state': self.global_router.state_dict(),
            'expert_states': expert_states,
            'distribution_params': distribution_params,
            'class_anchors': self.class_anchors.clone(),
            'cluster_anchors': self.cluster_anchors.clone()
        }
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict],
        client_distribution_params: Dict[int, Dict[int, Dict]],
        client_sample_counts: Dict[int, int] = None,
        client_class_counts: Dict[int, Dict[int, int]] = None
    ):
        """
        聚合客户端更新
        
        Args:
            client_updates: {client_id: {param_name: param_value}}
            client_distribution_params: {client_id: {class_id: params}}
            client_sample_counts: {client_id: num_samples} 客户端总样本数
            client_class_counts: {client_id: {class_id: count}} 客户端每类样本数
        """
        # 1. 聚合路由层（按客户端数据量加权）
        self._aggregate_router(client_updates, client_sample_counts)
        
        # 2. 聚合专家（按对应类的数据量加权）
        self._aggregate_experts(client_updates, client_class_counts)
        
        # 3. 聚合分布参数
        self._aggregate_distributions(client_distribution_params)
    
    def _aggregate_router(
        self, 
        client_updates: Dict[int, Dict],
        client_sample_counts: Dict[int, int] = None
    ):
        """聚合路由层 - 按数据量加权"""
        if len(client_updates) == 0:
            return
        
        # 计算权重
        if client_sample_counts:
            total_samples = sum(client_sample_counts.get(cid, 1) for cid in client_updates.keys())
            weights = {cid: client_sample_counts.get(cid, 1) / total_samples 
                      for cid in client_updates.keys()}
        else:
            # 简单平均
            weights = {cid: 1.0 / len(client_updates) for cid in client_updates.keys()}
        
        # 收集路由层参数
        router_params = {}
        
        for client_id, updates in client_updates.items():
            client_weight = weights[client_id]
            
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
    
    def _aggregate_experts(
        self, 
        client_updates: Dict[int, Dict],
        client_class_counts: Dict[int, Dict[int, int]] = None
    ):
        """聚合专家 - 只有拥有对应类数据的客户端参与，按数据量加权"""
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
            
            # 获取该专家负责的类
            responsible_classes = expert.responsible_classes
            
            # 计算每个客户端对该专家的权重（基于对应类的样本数）
            if client_class_counts and responsible_classes:
                client_weights = {}
                total_weight = 0
                for upd in updates_list:
                    cid = upd['client_id']
                    # 统计该客户端在该专家负责类上的样本数
                    weight = sum(
                        client_class_counts.get(cid, {}).get(cls, 0) 
                        for cls in responsible_classes
                    )
                    weight = max(weight, 1)  # 至少为1
                    client_weights[cid] = weight
                    total_weight += weight
                
                # 归一化
                for cid in client_weights:
                    client_weights[cid] /= total_weight
            else:
                # 简单平均
                client_weights = {upd['client_id']: 1.0 / len(updates_list) for upd in updates_list}
            
            aggregated = {}
            for upd in updates_list:
                weight = client_weights[upd['client_id']]
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
            dim=self.global_distribution_pool.dim
        )
        
        # 更新全局分布
        for cls, params in global_params.items():
            if not self.global_distribution_pool.has_class(cls):
                # 用聚合后的 mean 初始化
                self.global_distribution_pool.add_class(cls, init_mean=params['mean'])
            self.global_distribution_pool.set_class_params(cls, params)
    
    def finish_task(self, task_classes: List[int]):
        """
        完成任务，更新已学习类别
        """
        for cls in task_classes:
            if cls not in self.learned_classes:
                self.learned_classes.append(cls)
    
    def diagnose_routing(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        诊断路由情况 - 分析每个类别被路由到哪些专家
        
        Args:
            test_loader: 测试数据加载器
            class_names: 类别名称列表
        
        Returns:
            diagnosis: 路由诊断结果
        """
        self.backbone.eval()
        self.global_router.eval()
        self.global_expert_pool.eval()
        
        if class_names is None:
            class_names = [f"class_{i}" for i in range(len(self.class_anchors))]
        
        # 统计每个类被路由到哪些专家
        class_routing_stats = {}  # {class_id: {expert_id: count}}
        class_expected_expert = {}  # {class_id: expected_expert_id}
        class_total_samples = {}  # {class_id: total_count}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                backbone_features = self.backbone(images)
                routed_expert_ids, routing_probs, projected = self.global_router(backbone_features)
                
                # 统计路由结果
                for i in range(len(labels)):
                    cls = labels[i].item()
                    routed_exp = routed_expert_ids[i].item()
                    
                    # 初始化统计
                    if cls not in class_routing_stats:
                        class_routing_stats[cls] = {}
                        class_total_samples[cls] = 0
                        class_expected_expert[cls] = self.global_expert_pool.get_expert_for_class(cls)
                    
                    # 统计路由到的专家
                    if routed_exp not in class_routing_stats[cls]:
                        class_routing_stats[cls][routed_exp] = 0
                    class_routing_stats[cls][routed_exp] += 1
                    class_total_samples[cls] += 1
        
        # 计算每个类的路由准确率
        class_routing_accuracy = {}
        total_correct = 0
        total_samples = 0
        
        for cls in class_routing_stats:
            expected_exp = class_expected_expert[cls]
            correct = class_routing_stats[cls].get(expected_exp, 0)
            total = class_total_samples[cls]
            
            class_routing_accuracy[cls] = correct / total if total > 0 else 0
            total_correct += correct
            total_samples += total
        
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'class_routing_stats': class_routing_stats,
            'class_expected_expert': class_expected_expert,
            'class_total_samples': class_total_samples,
            'class_routing_accuracy': class_routing_accuracy,
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_samples': total_samples,
            'class_names': class_names
        }
    
    def evaluate(
        self,
        test_loader,
        classes_to_eval: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        评估全局模型（分离评估路由和专家）
        """
        self.backbone.eval()
        self.global_router.eval()
        self.global_expert_pool.eval()
        
        total_correct = 0
        total_samples = 0
        
        # 路由评估
        routing_correct = 0
        routing_samples = 0
        
        # 专家评估（假设路由正确）
        expert_correct_with_gt_routing = 0
        expert_samples = 0
        
        class_correct = {}
        class_total = {}
        class_routing_correct = {}
        class_routing_total = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                backbone_features = self.backbone(images)
                routed_expert_ids, routing_probs, projected = self.global_router(backbone_features)
                
                # ===== 评估路由准确率 =====
                target_experts = torch.tensor(
                    [self.global_expert_pool.get_expert_for_class(l.item()) for l in labels],
                    device=self.device
                )
                expert_to_cluster = self.get_expert_to_cluster()
                target_clusters = torch.tensor(
                    [expert_to_cluster.get(exp.item(), 0) for exp in target_experts],
                    device=self.device
                )
                routed_clusters = torch.argmax(routing_probs, dim=-1)
                
                routing_match = (routed_clusters == target_clusters)
                routing_correct += routing_match.sum().item()
                routing_samples += len(labels)
                
                # ===== 评估专家准确率（使用路由的专家） =====
                cls_logits_routed, _ = self.global_expert_pool(
                    projected, routed_expert_ids, self.class_anchors
                )
                _, predicted_routed = cls_logits_routed.max(1)
                
                # ===== 评估专家准确率（使用ground truth专家） =====
                cls_logits_gt, _ = self.global_expert_pool(
                    projected, target_experts, self.class_anchors
                )
                _, predicted_gt = cls_logits_gt.max(1)
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    
                    if classes_to_eval and label not in classes_to_eval:
                        continue
                    
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                        class_routing_total[label] = 0
                        class_routing_correct[label] = 0
                    
                    class_total[label] += 1
                    class_routing_total[label] += 1
                    
                    # 总体准确率（路由的专家）
                    if predicted_routed[i].item() == label:
                        class_correct[label] += 1
                        total_correct += 1
                    
                    # 路由准确率
                    if routing_match[i].item():
                        class_routing_correct[label] += 1
                    
                    # 专家准确率（ground truth路由）
                    if predicted_gt[i].item() == label:
                        expert_correct_with_gt_routing += 1
                    
                    total_samples += 1
                    expert_samples += 1
        
        metrics = {
            'accuracy': total_correct / max(total_samples, 1) * 100,
            'routing_accuracy': routing_correct / max(routing_samples, 1) * 100,
            'expert_accuracy_with_gt_routing': expert_correct_with_gt_routing / max(expert_samples, 1) * 100,
            'total_samples': total_samples
        }
        
        for cls in class_total:
            metrics[f'class_{cls}_acc'] = (
                class_correct[cls] / class_total[cls] * 100
                if class_total[cls] > 0 else 0
            )
            metrics[f'class_{cls}_routing_acc'] = (
                class_routing_correct[cls] / class_routing_total[cls] * 100
                if class_routing_total[cls] > 0 else 0
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