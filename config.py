"""
FedAME Configuration
Federated Anchor-guided Mixture-of-Experts for Class-Incremental Learning
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

def load_env():
    """从.env文件加载环境变量"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

# 加载.env文件
load_env()

@dataclass
class DataConfig:
    """数据配置"""
    dataset: str = "cifar10"
    data_root: str = "./data"
    num_classes: int = 10
    image_size: int = 32
    
    # CIFAR-10 类别名称
    class_names: List[str] = field(default_factory=lambda: [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ])
    
    # 语义簇定义
    semantic_clusters: Dict[str, List[str]] = field(default_factory=lambda: {
        "animals": ["bird", "cat", "deer", "dog", "frog", "horse"],
        "vehicles": ["airplane", "automobile", "ship", "truck"]
    })

@dataclass
class ModelConfig:
    """模型配置"""
    # Backbone
    backbone: str = "resnet18"
    backbone_pretrained: bool = True
    feature_dim: int = 512
    
    # Router
    router_hidden_dim: int = 256
    router_output_dim: int = 128
    
    # Expert
    expert_hidden_dim: int = 256
    expert_output_dim: int = 128
    num_initial_experts: int = 2
    max_classes_per_expert: int = 6
    
    # Anchor
    anchor_dim: int = 512  # CLIP embedding dimension
    
    # Distribution Prompt
    distribution_dim: int = 64
    max_residual_norm: float = 0.5

@dataclass
class FederatedConfig:
    """联邦学习配置"""
    num_clients: int = 5
    participation_rate: float = 1.0  # 每轮参与的客户端比例
    local_epochs: int = 5
    local_batch_size: int = 32
    
    # 数据异构性
    alpha: float = 0.5  # Dirichlet分布参数，越小异构性越强

@dataclass
class IncrementalConfig:
    """增量学习配置"""
    # 任务划分 (CIFAR-10 分成多个任务)
    tasks: List[List[int]] = field(default_factory=lambda: [
        [0, 1, 2, 3],    # Task 1: airplane, automobile, bird, cat
        [4, 5, 6, 7],    # Task 2: deer, dog, frog, horse
        [8, 9]           # Task 3: ship, truck
    ])
    
    num_tasks: int = 3
    
    # 防遗忘
    num_replay_samples: int = 10  # 每个旧类回放的样本数

@dataclass
class TrainingConfig:
    """训练配置"""
    num_rounds: int = 100  # 每个任务的联邦轮数
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    
    # 损失权重
    lambda_cls: float = 1.0
    lambda_route: float = 0.5
    lambda_contrast: float = 0.3
    lambda_forget: float = 0.5
    lambda_dist: float = 0.1
    
    # 温度系数
    temperature_route: float = 0.1
    temperature_cls: float = 0.1
    temperature_contrast: float = 0.1
    
    # 决策阈值
    assign_threshold: float = 0.6
    split_threshold: float = 0.8

@dataclass
class AnchorConfig:
    """锚点生成配置"""
    # CLIP模型
    clip_model: str = "openai/clip-vit-base-patch32"
    
    # LLM API配置 (DeepSeek) - 从.env文件读取
    llm_api_base: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"))
    llm_api_url: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1") + "/chat/completions")
    llm_model: str = "deepseek-chat"
    llm_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY"))
    
    # 是否使用真实LLM（否则使用规则决策）
    use_real_llm: bool = False

@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    incremental: IncrementalConfig = field(default_factory=IncrementalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    
    # 其他
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs"
    log_interval: int = 10
    save_interval: int = 20

def get_config() -> Config:
    """获取配置"""
    config = Config()
    return config
