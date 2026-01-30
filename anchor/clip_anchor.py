"""
锚点生成模块
使用CLIP文本编码器生成语义锚点
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
import warnings


class AnchorGenerator:
    """
    锚点生成器
    使用CLIP文本编码器为类别生成语义锚点
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        use_cache: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.use_cache = use_cache
        
        # 锚点缓存
        self.anchor_cache: Dict[str, torch.Tensor] = {}
        
        # 延迟加载模型
        self._model = None
        self._processor = None
        self._tokenizer = None
    
    def _load_model(self):
        """延迟加载CLIP模型"""
        if self._model is not None:
            return
        
        try:
            from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
            
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            
            self._model = self._model.to(self.device)
            self._model.eval()
            
        except ImportError:
            warnings.warn(
                "transformers library not available. Using random anchors."
            )
            self._model = "random"
        except Exception as e:
            warnings.warn(f"Failed to load CLIP model: {e}. Using random anchors.")
            self._model = "random"
    
    def generate_anchor(
        self,
        text: str,
        template: str = "a photo of a {}"
    ) -> torch.Tensor:
        """
        为单个文本生成锚点
        
        Args:
            text: 类别名称
            template: 提示模板
        
        Returns:
            anchor: [dim] 锚点向量
        """
        # 检查缓存
        cache_key = f"{template}_{text}"
        if self.use_cache and cache_key in self.anchor_cache:
            return self.anchor_cache[cache_key]
        
        self._load_model()
        
        if self._model == "random":
            # 使用随机向量作为锚点
            anchor = torch.randn(512)
            anchor = F.normalize(anchor, p=2, dim=-1)
        else:
            prompt = template.format(text)
            
            with torch.no_grad():
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                text_features = self._model.get_text_features(**inputs)
                anchor = F.normalize(text_features, p=2, dim=-1).squeeze(0)
        
        # 缓存
        if self.use_cache:
            self.anchor_cache[cache_key] = anchor.cpu()
        
        return anchor.cpu()
    
    def generate_anchors(
        self,
        texts: List[str],
        template: str = "a photo of a {}"
    ) -> torch.Tensor:
        """
        为多个文本生成锚点
        
        Args:
            texts: 类别名称列表
            template: 提示模板
        
        Returns:
            anchors: [num_texts, dim] 锚点矩阵
        """
        anchors = []
        for text in texts:
            anchor = self.generate_anchor(text, template)
            anchors.append(anchor)
        
        return torch.stack(anchors, dim=0)
    
    def generate_class_anchors(
        self,
        class_names: List[str],
        templates: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        为类别生成锚点（支持多模板集成）
        
        Args:
            class_names: 类别名称列表
            templates: 提示模板列表（如果提供多个，取平均）
        
        Returns:
            anchors: [num_classes, dim]
        """
        if templates is None:
            templates = ["a photo of a {}"]
        
        all_anchors = []
        
        for template in templates:
            anchors = self.generate_anchors(class_names, template)
            all_anchors.append(anchors)
        
        # 多模板取平均
        final_anchors = torch.stack(all_anchors, dim=0).mean(dim=0)
        final_anchors = F.normalize(final_anchors, p=2, dim=-1)
        
        return final_anchors
    
    def generate_cluster_anchors(
        self,
        clusters: Dict[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        为语义簇生成锚点
        
        Args:
            clusters: {簇名称: [类别列表]}
        
        Returns:
            cluster_anchors: {簇名称: 锚点向量}
        """
        cluster_anchors = {}
        
        for cluster_name, classes in clusters.items():
            # 簇锚点 = 簇名称的锚点
            anchor = self.generate_anchor(cluster_name, template="a photo of {}")
            cluster_anchors[cluster_name] = anchor
        
        return cluster_anchors
    
    def compute_similarity(
        self,
        anchor1: torch.Tensor,
        anchor2: torch.Tensor
    ) -> float:
        """计算两个锚点的相似度"""
        anchor1 = F.normalize(anchor1.unsqueeze(0), p=2, dim=-1)
        anchor2 = F.normalize(anchor2.unsqueeze(0), p=2, dim=-1)
        return torch.mm(anchor1, anchor2.t()).item()
    
    def find_nearest_cluster(
        self,
        class_anchor: torch.Tensor,
        cluster_anchors: Dict[str, torch.Tensor]
    ) -> str:
        """
        找到与类锚点最近的簇
        
        Args:
            class_anchor: 类锚点
            cluster_anchors: 簇锚点字典
        
        Returns:
            nearest_cluster: 最近的簇名称
        """
        max_sim = -1
        nearest_cluster = None
        
        for cluster_name, cluster_anchor in cluster_anchors.items():
            sim = self.compute_similarity(class_anchor, cluster_anchor)
            if sim > max_sim:
                max_sim = sim
                nearest_cluster = cluster_name
        
        return nearest_cluster
    
    @property
    def anchor_dim(self) -> int:
        """锚点维度"""
        return 512


class SimpleAnchorGenerator:
    """
    简化版锚点生成器
    支持正交初始化，确保类别锚点之间正交（相似度=0）
    """
    
    def __init__(
        self,
        dim: int = 512,
        seed: int = 42,
        orthogonal: bool = True  # 是否使用正交初始化
    ):
        self.dim = dim
        self.seed = seed
        self.orthogonal = orthogonal
        self.anchor_cache = {}
        self._orthogonal_basis = None  # 缓存正交基
        self._basis_names = []  # 记录使用正交基的名称顺序
    
    def _generate_orthogonal_basis(self, num_vectors: int) -> torch.Tensor:
        """
        生成正交基
        使用QR分解确保向量两两正交
        """
        torch.manual_seed(self.seed)
        
        # 生成随机矩阵
        if num_vectors <= self.dim:
            # 可以生成完全正交的向量
            random_matrix = torch.randn(self.dim, num_vectors)
            q, r = torch.linalg.qr(random_matrix)
            basis = q[:, :num_vectors].t()  # [num_vectors, dim]
        else:
            # 向量数超过维度，无法完全正交，使用随机向量
            basis = torch.randn(num_vectors, self.dim)
            basis = F.normalize(basis, p=2, dim=-1)
        
        return basis
    
    def _generate_single_anchor(self, name: str) -> torch.Tensor:
        """为单个名称生成锚点（使用名称的hash确保确定性）"""
        if name in self.anchor_cache:
            return self.anchor_cache[name]
        
        # 使用名称的hash作为种子，确保相同名称总是生成相同向量
        name_seed = hash(name) % (2**32)
        gen = torch.Generator()
        gen.manual_seed(name_seed)
        
        anchor = torch.randn(self.dim, generator=gen)
        anchor = F.normalize(anchor, p=2, dim=-1)
        
        self.anchor_cache[name] = anchor
        return anchor
    
    def generate_anchors(
        self,
        class_names: List[str],
        force_orthogonal: bool = None
    ) -> torch.Tensor:
        """
        为类别生成锚点
        
        Args:
            class_names: 类别名称列表
            force_orthogonal: 是否强制正交（None则使用默认设置）
        
        Returns:
            anchors: [num_classes, dim] 锚点矩阵
        """
        use_orthogonal = force_orthogonal if force_orthogonal is not None else self.orthogonal
        
        if use_orthogonal:
            # 检查是否所有名称都已在正交基中
            all_cached = all(name in self.anchor_cache for name in class_names)
            
            if not all_cached:
                # 生成新的正交基
                num_classes = len(class_names)
                basis = self._generate_orthogonal_basis(num_classes)
                
                # 缓存
                for i, name in enumerate(class_names):
                    if name not in self.anchor_cache:
                        self.anchor_cache[name] = basis[i]
                        self._basis_names.append(name)
            
            # 从缓存获取
            anchors = torch.stack([self.anchor_cache[name] for name in class_names], dim=0)
        else:
            # 非正交模式：每个名称独立生成
            anchors = torch.stack([self._generate_single_anchor(name) for name in class_names], dim=0)
        
        return anchors
    
    def generate_cluster_anchors(
        self,
        cluster_names: List[str]
    ) -> torch.Tensor:
        """为簇生成锚点（复用类锚点如果名称相同）"""
        return self.generate_anchors(cluster_names)
    
    def get_anchor(self, name: str) -> torch.Tensor:
        """获取缓存的锚点"""
        return self.anchor_cache.get(name)
    
    def verify_orthogonality(self) -> Dict[str, float]:
        """验证锚点的正交性"""
        if len(self.anchor_cache) < 2:
            return {'max_similarity': 0.0, 'mean_similarity': 0.0}
        
        anchors = torch.stack(list(self.anchor_cache.values()), dim=0)
        anchors = F.normalize(anchors, p=2, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(anchors, anchors.t())
        
        # 去掉对角线
        mask = ~torch.eye(len(anchors), dtype=torch.bool)
        off_diag = sim_matrix[mask]
        
        return {
            'max_similarity': off_diag.abs().max().item(),
            'mean_similarity': off_diag.abs().mean().item(),
            'num_anchors': len(anchors)
        }
    
    @property
    def anchor_dim(self) -> int:
        return self.dim


def create_anchor_generator(
    use_clip: bool = True,
    clip_model: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    anchor_dim: int = 512,
    seed: int = 42,
    orthogonal: bool = True  # 默认使用正交初始化
):
    """
    创建锚点生成器
    
    Args:
        use_clip: 是否使用CLIP
        clip_model: CLIP模型名称
        device: 设备
        anchor_dim: 锚点维度（仅用于简化版）
        seed: 随机种子（仅用于简化版）
        orthogonal: 是否使用正交初始化（仅用于简化版）
    """
    if use_clip:
        return AnchorGenerator(
            model_name=clip_model,
            device=device
        )
    else:
        return SimpleAnchorGenerator(
            dim=anchor_dim,
            seed=seed,
            orthogonal=orthogonal
        )


# 测试
if __name__ == "__main__":
    # CIFAR-10类别
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    print("=" * 60)
    print("Testing Orthogonal Anchor Generator")
    print("=" * 60)
    
    # 测试正交初始化
    print("\n[1] Orthogonal initialization (default):")
    generator = create_anchor_generator(use_clip=False, anchor_dim=512, orthogonal=True)
    class_anchors = generator.generate_anchors(class_names)
    
    # 验证正交性
    ortho_info = generator.verify_orthogonality()
    print(f"    Max similarity: {ortho_info['max_similarity']:.6f} (should be ~0)")
    print(f"    Mean similarity: {ortho_info['mean_similarity']:.6f} (should be ~0)")
    
    # 打印相似度矩阵
    sim_matrix = torch.mm(class_anchors, class_anchors.t())
    print(f"\n    Similarity matrix (diagonal should be 1, others ~0):")
    print(f"    Diagonal values: {sim_matrix.diag()}")
    print(f"    Off-diagonal max: {(sim_matrix - torch.eye(10)).abs().max():.6f}")
    
    # 测试非正交初始化
    print("\n[2] Random initialization (non-orthogonal):")
    generator_random = create_anchor_generator(use_clip=False, anchor_dim=512, orthogonal=False)
    class_anchors_random = generator_random.generate_anchors(class_names)
    
    ortho_info_random = generator_random.verify_orthogonality()
    print(f"    Max similarity: {ortho_info_random['max_similarity']:.6f}")
    print(f"    Mean similarity: {ortho_info_random['mean_similarity']:.6f}")
    
    # 测试簇锚点复用
    print("\n[3] Cluster anchors (same name = same anchor):")
    cluster_names = class_names  # 每个类一个簇
    cluster_anchors = generator.generate_anchors(cluster_names)
    
    # 验证类锚点和簇锚点一致
    diff = (class_anchors - cluster_anchors).abs().max()
    print(f"    Class anchors == Cluster anchors: {diff < 1e-6}")
    
    print("\n" + "=" * 60)
    print("✅ Orthogonal anchors ready for easier routing!")
    print("=" * 60)