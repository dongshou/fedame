"""
LLM决策模块
使用DeepSeek API进行语义决策（专家分配、拆分等）
支持真实API调用和规则回退
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json
import os


class LLMDecisionMaker:
    """
    LLM决策器
    使用DeepSeek API进行语义决策
    """
    
    def __init__(
        self,
        api_base: str = "https://api.deepseek.com/v1",
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        use_real_llm: bool = False
    ):
        self.api_base = api_base or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        self.api_url = f"{self.api_base}/chat/completions"
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.use_real_llm = use_real_llm and self.api_key is not None
        
        if self.use_real_llm:
            try:
                import requests
                self.requests = requests
            except ImportError:
                self.use_real_llm = False
    
    def _call_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        if not self.use_real_llm:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的机器学习专家，帮助进行语义分类决策。请用JSON格式回复。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            response = self.requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API call failed: {e}")
            return None
    
    def decide_expert_assignment(
        self,
        new_class: str,
        expert_info: Dict[int, Dict],
        class_anchors: torch.Tensor,
        cluster_anchors: torch.Tensor,
        class_names: List[str],
        cluster_names: List[str],
        threshold: float = 0.6
    ) -> Dict:
        """
        决定新类应该分配给哪个专家
        
        Args:
            new_class: 新类名称
            expert_info: 现有专家信息 {expert_id: {responsible_classes: [...], ...}}
            class_anchors: 类锚点
            cluster_anchors: 簇锚点
            class_names: 类名列表
            cluster_names: 簇名列表
            threshold: 分配阈值
        
        Returns:
            decision: {
                'action': 'assign' | 'new',
                'expert_id': int (如果assign),
                'cluster': str,
                'reason': str
            }
        """
        new_class_idx = class_names.index(new_class) if new_class in class_names else -1
        
        if self.use_real_llm:
            # 构建prompt
            expert_desc = []
            for exp_id, info in expert_info.items():
                classes = info.get('responsible_classes', [])
                class_names_list = [class_names[c] for c in classes if c < len(class_names)]
                expert_desc.append(f"Expert_{exp_id}: {class_names_list}")
            
            prompt = f"""
任务：为新类别分配专家

当前专家结构：
{chr(10).join(expert_desc)}

语义簇：
- animals (动物): bird, cat, deer, dog, frog, horse
- vehicles (交通工具): airplane, automobile, ship, truck

新类别：{new_class}

请判断：
1. 该类别属于哪个语义簇？
2. 应该分配给现有哪个专家？还是需要创建新专家？

请用以下JSON格式回复：
{{"action": "assign"或"new", "expert_id": 专家ID(如果assign), "cluster": "animals"或"vehicles", "reason": "简短理由"}}
"""
            
            response = self._call_api(prompt)
            if response:
                try:
                    # 提取JSON
                    import re
                    json_match = re.search(r'\{[^}]+\}', response)
                    if json_match:
                        return json.loads(json_match.group())
                except:
                    pass
        
        # 规则回退：基于锚点相似度
        return self._rule_based_assignment(
            new_class, new_class_idx, expert_info,
            class_anchors, cluster_anchors,
            class_names, cluster_names, threshold
        )
    
    def _rule_based_assignment(
        self,
        new_class: str,
        new_class_idx: int,
        expert_info: Dict[int, Dict],
        class_anchors: torch.Tensor,
        cluster_anchors: torch.Tensor,
        class_names: List[str],
        cluster_names: List[str],
        threshold: float
    ) -> Dict:
        """基于规则的专家分配"""
        # 每个类对应自己的簇
        cluster = new_class if new_class in cluster_names else cluster_names[0]
        cluster_idx = cluster_names.index(cluster) if cluster in cluster_names else 0
        
        # 找到负责该簇的专家
        target_expert = None
        for exp_id, info in expert_info.items():
            if info.get('cluster') == cluster:
                target_expert = exp_id
                break
        
        if target_expert is not None:
            return {
                'action': 'assign',
                'expert_id': target_expert,
                'cluster': cluster,
                'reason': f'{new_class} assigned to Expert_{target_expert}'
            }
        else:
            return {
                'action': 'new',
                'cluster': cluster,
                'reason': f'Create new expert for {new_class}'
            }
    
    def decide_expert_split(
        self,
        expert_id: int,
        responsible_classes: List[int],
        class_anchors: torch.Tensor,
        class_names: List[str],
        max_classes: int = 6,
        split_threshold: float = 0.8
    ) -> Dict:
        """
        决定是否需要拆分专家
        
        Args:
            expert_id: 专家ID
            responsible_classes: 负责的类别列表
            class_anchors: 类锚点
            class_names: 类名列表
            max_classes: 最大类别数
            split_threshold: 拆分阈值（类内最大距离）
        
        Returns:
            decision: {
                'split': bool,
                'groups': [[class_ids], [class_ids]] (如果split=True),
                'group_names': [str, str] (如果split=True),
                'reason': str
            }
        """
        if len(responsible_classes) <= 2:
            return {
                'split': False,
                'reason': 'Too few classes to split'
            }
        
        # 计算类内最大距离
        relevant_anchors = class_anchors[responsible_classes]
        relevant_anchors = F.normalize(relevant_anchors, p=2, dim=-1)
        sim_matrix = torch.mm(relevant_anchors, relevant_anchors.t())
        
        # 找到最不相似的两个类
        min_sim = sim_matrix.min().item()
        max_dist = 1 - min_sim
        
        need_split = len(responsible_classes) > max_classes or max_dist > split_threshold
        
        if not need_split:
            return {
                'split': False,
                'reason': f'No need to split (max_dist={max_dist:.3f}, num_classes={len(responsible_classes)})'
            }
        
        # 简单聚类：基于与簇中心的距离分成两组
        # 使用第一个和最不相似的类作为两个中心
        min_idx = torch.argmin(sim_matrix.sum(dim=1)).item()
        
        group1 = []
        group2 = []
        
        center1 = relevant_anchors[0]
        center2 = relevant_anchors[min_idx]
        
        for i, cls in enumerate(responsible_classes):
            anchor = relevant_anchors[i]
            sim1 = F.cosine_similarity(anchor.unsqueeze(0), center1.unsqueeze(0)).item()
            sim2 = F.cosine_similarity(anchor.unsqueeze(0), center2.unsqueeze(0)).item()
            
            if sim1 >= sim2:
                group1.append(cls)
            else:
                group2.append(cls)
        
        # 确保两组都不为空
        if len(group1) == 0:
            group1.append(group2.pop())
        if len(group2) == 0:
            group2.append(group1.pop())
        
        group1_names = [class_names[c] for c in group1 if c < len(class_names)]
        group2_names = [class_names[c] for c in group2 if c < len(class_names)]
        
        return {
            'split': True,
            'groups': [group1, group2],
            'group_names': [str(group1_names), str(group2_names)],
            'reason': f'Split due to max_dist={max_dist:.3f} or num_classes={len(responsible_classes)}'
        }
    
    def suggest_cluster_assignment(
        self,
        class_name: str,
        available_clusters: List[str]
    ) -> str:
        """建议类别应该归属的簇"""
        # 每个类对应自己的簇
        if class_name in available_clusters:
            return class_name
        return available_clusters[0] if available_clusters else class_name


class ExpertManager:
    """
    专家管理器
    整合LLM决策和专家池操作
    """
    
    def __init__(
        self,
        llm_decision_maker: LLMDecisionMaker,
        class_names: List[str],
        cluster_names: List[str]
    ):
        self.llm = llm_decision_maker
        self.class_names = class_names
        self.cluster_names = cluster_names
        
        # 簇到专家的映射
        self.cluster_to_expert: Dict[str, int] = {}
        
        # 专家信息
        self.expert_info: Dict[int, Dict] = {}
    
    def initialize_experts(
        self,
        initial_clusters: Dict[str, List[str]]
    ):
        """
        初始化专家结构
        
        Args:
            initial_clusters: {簇名: [类别列表]}
        """
        for i, (cluster_name, classes) in enumerate(initial_clusters.items()):
            self.cluster_to_expert[cluster_name] = i
            class_ids = [self.class_names.index(c) for c in classes if c in self.class_names]
            self.expert_info[i] = {
                'responsible_classes': class_ids,
                'cluster': cluster_name
            }
    
    def assign_new_class(
        self,
        new_class: str,
        class_anchors: torch.Tensor,
        cluster_anchors: torch.Tensor
    ) -> Tuple[int, str]:
        """
        为新类分配专家
        
        Returns:
            (expert_id, cluster)
        """
        decision = self.llm.decide_expert_assignment(
            new_class=new_class,
            expert_info=self.expert_info,
            class_anchors=class_anchors,
            cluster_anchors=cluster_anchors,
            class_names=self.class_names,
            cluster_names=self.cluster_names
        )
        
        if decision['action'] == 'assign':
            expert_id = decision['expert_id']
        else:
            # 创建新专家
            expert_id = max(self.expert_info.keys()) + 1 if self.expert_info else 0
            self.expert_info[expert_id] = {
                'responsible_classes': [],
                'cluster': decision['cluster']
            }
            self.cluster_to_expert[decision['cluster']] = expert_id
        
        # 更新专家负责的类别
        class_id = self.class_names.index(new_class) if new_class in self.class_names else -1
        if class_id >= 0 and class_id not in self.expert_info[expert_id]['responsible_classes']:
            self.expert_info[expert_id]['responsible_classes'].append(class_id)
        
        return expert_id, decision['cluster']
    
    def check_and_split_expert(
        self,
        expert_id: int,
        class_anchors: torch.Tensor,
        max_classes: int = 6
    ) -> Optional[List[int]]:
        """
        检查并拆分专家
        
        Returns:
            new_expert_ids (如果拆分) 或 None
        """
        if expert_id not in self.expert_info:
            return None
        
        responsible_classes = self.expert_info[expert_id]['responsible_classes']
        
        decision = self.llm.decide_expert_split(
            expert_id=expert_id,
            responsible_classes=responsible_classes,
            class_anchors=class_anchors,
            class_names=self.class_names,
            max_classes=max_classes
        )
        
        if not decision['split']:
            return None
        
        # 执行拆分
        groups = decision['groups']
        
        # 第一组继续使用原专家
        self.expert_info[expert_id]['responsible_classes'] = groups[0]
        
        # 创建新专家
        new_expert_ids = [expert_id]
        for group in groups[1:]:
            new_id = max(self.expert_info.keys()) + 1
            self.expert_info[new_id] = {
                'responsible_classes': group,
                'cluster': self.expert_info[expert_id]['cluster']
            }
            new_expert_ids.append(new_id)
        
        return new_expert_ids


# 测试
if __name__ == "__main__":
    # 创建LLM决策器
    llm = LLMDecisionMaker(use_real_llm=False)
    
    # CIFAR-10类别
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    cluster_names = ["animals", "vehicles"]
    
    # 模拟锚点
    class_anchors = F.normalize(torch.randn(10, 512), dim=-1)
    cluster_anchors = F.normalize(torch.randn(2, 512), dim=-1)
    
    # 创建专家管理器
    manager = ExpertManager(llm, class_names, cluster_names)
    
    # 初始化
    initial_clusters = {
        "animals": ["bird", "cat"],
        "vehicles": ["airplane", "automobile"]
    }
    manager.initialize_experts(initial_clusters)
    
    print("Initial expert structure:")
    print(manager.expert_info)
    
    # 测试新类分配
    print("\n--- Testing new class assignment ---")
    for new_class in ["dog", "ship", "deer"]:
        expert_id, cluster = manager.assign_new_class(
            new_class, class_anchors, cluster_anchors
        )
        print(f"'{new_class}' -> Expert_{expert_id} ({cluster})")
    
    print("\nUpdated expert structure:")
    print(manager.expert_info)
    
    # 测试专家拆分
    print("\n--- Testing expert split ---")
    result = manager.check_and_split_expert(
        expert_id=0,
        class_anchors=class_anchors,
        max_classes=3
    )
    
    if result:
        print(f"Expert split into: {result}")
        print("Final expert structure:")
        print(manager.expert_info)