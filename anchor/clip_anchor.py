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
            
            print(f"Loading CLIP model: {self.model_name}")
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            
            self._model = self._model.to(self.device)
            self._model.eval()
            
            print("CLIP model loaded successfully")
            
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
    使用预定义的随机向量，确保类别间有区分度
    """
    
    def __init__(
        self,
        dim: int = 512,
        seed: int = 42
    ):
        self.dim = dim
        self.seed = seed
        self.anchor_cache = {}
        
        # 设置随机种子保证可复现
        torch.manual_seed(seed)
    
    def generate_anchors(
        self,
        class_names: List[str]
    ) -> torch.Tensor:
        """
        为类别生成正交化的锚点
        """
        num_classes = len(class_names)
        
        # 生成随机矩阵
        anchors = torch.randn(num_classes, self.dim)
        
        # QR分解获得正交基（如果维度足够）
        if num_classes <= self.dim:
            q, r = torch.linalg.qr(anchors.t())
            anchors = q.t()[:num_classes]
        
        # L2归一化
        anchors = F.normalize(anchors, p=2, dim=-1)
        
        # 缓存
        for i, name in enumerate(class_names):
            self.anchor_cache[name] = anchors[i]
        
        return anchors
    
    def generate_cluster_anchors(
        self,
        cluster_names: List[str]
    ) -> torch.Tensor:
        """为簇生成锚点"""
        return self.generate_anchors(cluster_names)
    
    def get_anchor(self, name: str) -> torch.Tensor:
        """获取缓存的锚点"""
        return self.anchor_cache.get(name)
    
    @property
    def anchor_dim(self) -> int:
        return self.dim


def create_anchor_generator(
    use_clip: bool = True,
    clip_model: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    anchor_dim: int = 512,
    seed: int = 42
):
    """
    创建锚点生成器
    
    Args:
        use_clip: 是否使用CLIP
        clip_model: CLIP模型名称
        device: 设备
        anchor_dim: 锚点维度（仅用于简化版）
        seed: 随机种子（仅用于简化版）
    """
    if use_clip:
        return AnchorGenerator(
            model_name=clip_model,
            device=device
        )
    else:
        return SimpleAnchorGenerator(
            dim=anchor_dim,
            seed=seed
        )


# 测试
if __name__ == "__main__":
    # CIFAR-10类别
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    # 语义簇
    clusters = {
        "animals": ["bird", "cat", "deer", "dog", "frog", "horse"],
        "vehicles": ["airplane", "automobile", "ship", "truck"]
    }
    
    print("=" * 50)
    print("Testing Anchor Generator")
    print("=" * 50)
    
    # 使用简化版生成器（不需要下载CLIP）
    generator = create_anchor_generator(use_clip=False, anchor_dim=512)
    
    # 生成类锚点
    class_anchors = generator.generate_anchors(class_names)
    print(f"\nClass anchors shape: {class_anchors.shape}")
    
    # 生成簇锚点
    cluster_anchors = generator.generate_cluster_anchors(list(clusters.keys()))
    print(f"Cluster anchors shape: {cluster_anchors.shape}")
    
    # 计算类别间相似度
    print("\nClass similarity matrix (first 5 classes):")
    sim_matrix = torch.mm(class_anchors[:5], class_anchors[:5].t())
    print(sim_matrix)
    
    # 测试CLIP版本（如果可用）
    print("\n" + "=" * 50)
    print("Testing CLIP Anchor Generator (if available)")
    print("=" * 50)
    
    try:
        clip_generator = create_anchor_generator(use_clip=True, device="cpu")
        clip_class_anchors = clip_generator.generate_class_anchors(class_names[:3])
        print(f"CLIP class anchors shape: {clip_class_anchors.shape}")
    except Exception as e:
        print(f"CLIP not available: {e}")
