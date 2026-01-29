# FedAME: Federated Anchor-guided Mixture-of-Experts for Class-Incremental Learning

## 概述

FedAME是一个融合大模型语义知识与轻量级专家网络的联邦类增量学习框架。

### 核心特点

1. **大小模型融合**
   - 大模型（CLIP）：生成语义锚点，提供统一的特征空间坐标系
   - 大模型（DeepSeek）：通过API进行语义决策（专家分配、拆分）
   - 小模型（ResNet-18）：冻结的特征提取器
   - 小模型（Router + Experts）：可学习的路由和专家网络

2. **全局锚点系统**
   - 簇级锚点用于路由
   - 类级锚点用于分类
   - 全局对比损失确保特征空间一致

3. **动态专家网络**
   - 客户端只持有部分相关专家
   - 服务端维护完整专家池
   - 支持专家拆分和扩展

4. **概率分布提示**
   - 压缩存储类别知识
   - 支持无数据回放防遗忘
   - 基于锚点+残差的设计保证可聚合性

## 项目结构

```
fedame/
├── config.py              # 配置文件
├── losses.py              # 损失函数
├── train.py               # 主训练脚本
├── requirements.txt       # 依赖
├── data/
│   ├── __init__.py
│   └── dataset.py         # CIFAR-10联邦数据集
├── models/
│   ├── __init__.py
│   ├── backbone.py        # ResNet-18 Backbone
│   ├── router.py          # 路由层
│   ├── expert.py          # 专家网络
│   └── distribution.py    # 概率分布提示
├── anchor/
│   ├── __init__.py
│   ├── clip_anchor.py     # CLIP锚点生成
│   └── llm_decision.py    # LLM决策模块
└── federated/
    ├── __init__.py
    ├── client.py          # 联邦客户端
    └── server.py          # 联邦服务端
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
cd fedame
python train.py
```

## 配置说明

### 数据配置
- `num_clients`: 客户端数量（默认5）
- `alpha`: Dirichlet分布参数，控制异构程度（默认0.5，越小越异构）

### 增量任务配置
默认将CIFAR-10分成3个任务：
- Task 1: airplane, automobile, bird, cat
- Task 2: deer, dog, frog, horse  
- Task 3: ship, truck

### 模型配置
- `backbone`: 特征提取器（默认ResNet-18）
- `num_initial_experts`: 初始专家数量（默认2，对应2个语义簇）

## 实验设置

### CIFAR-10 语义簇划分
- **动物 (animals)**: bird, cat, deer, dog, frog, horse
- **交通工具 (vehicles)**: airplane, automobile, ship, truck

### 联邦学习设置
- 5个客户端
- 每轮100次通信
- 本地训练5个epoch
- Dirichlet α=0.5 模拟Non-IID

## 关键组件说明

### 1. 锚点生成 (anchor/clip_anchor.py)
```python
# 使用CLIP生成语义锚点
generator = AnchorGenerator(model_name="openai/clip-vit-base-patch32")
class_anchors = generator.generate_class_anchors(class_names)
```

### 2. LLM决策 (anchor/llm_decision.py)
```python
# 使用DeepSeek进行专家分配决策
llm = LLMDecisionMaker(use_real_llm=True)
decision = llm.decide_expert_assignment(new_class, expert_info, ...)
```

### 3. 路由层 (models/router.py)
```python
# 基于锚点的路由
router = AnchorBasedRouter(input_dim=512, anchor_dim=512)
expert_ids, routing_probs, features = router(backbone_features)
```

### 4. 分布提示 (models/distribution.py)
```python
# 概率分布：均值 = 锚点 + 有界残差
dist = ClassDistribution(class_id=0, anchor=anchor, max_residual_norm=0.5)
samples = dist.sample(num_samples=10)
```

## 损失函数

总损失：
$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_1 \mathcal{L}_{route} + \lambda_2 \mathcal{L}_{contrast} + \lambda_3 \mathcal{L}_{forget} + \lambda_4 \mathcal{L}_{dist}$$

- $\mathcal{L}_{cls}$: 分类损失
- $\mathcal{L}_{route}$: 路由损失
- $\mathcal{L}_{contrast}$: 全局对比损失
- $\mathcal{L}_{forget}$: 防遗忘损失（KL散度）
- $\mathcal{L}_{dist}$: 分布正则化损失

## 引用

如果您使用了本代码，请引用：

```bibtex
@article{fedame2024,
  title={FedAME: Federated Anchor-guided Mixture-of-Experts for Class-Incremental Learning},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License
