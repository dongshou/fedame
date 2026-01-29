# 方法

## 1. 问题定义

考虑包含 $K$ 个客户端和 $T$ 个顺序任务的联邦类增量学习场景。在任务 $t$ 时，客户端 $k$ 拥有本地数据集 $\mathcal{D}_k^t = \{(x_i, y_i)\}_{i=1}^{n_k^t}$，其中 $y_i \in \mathcal{C}_k^t$。不同客户端的类别集合存在异构性，即 $\mathcal{C}_i^t \neq \mathcal{C}_j^t$。全局类别集合随时间扩展：

$$\mathcal{C}^{1:t} = \bigcup_{\tau=1}^{t} \bigcup_{k=1}^{K} \mathcal{C}_k^\tau$$

我们的目标是学习一个全局模型，能够：(1) 准确分类所有已观测类别；(2) 在不访问旧数据的情况下保持旧类知识；(3) 处理客户端间的数据异构性。

## 2. 方法概述

我们提出 **FedAME**（Federated Anchor-guided Mixture-of-Experts），一个融合大模型语义知识与轻量级专家网络的联邦持续学习框架。核心思想是利用预训练大模型构建全局统一的语义锚点空间，指导动态专家网络的路由与学习，同时通过概率分布提示实现无数据回放的知识保持。

整体框架包含四个核心组件：
- **全局锚点系统**：大模型生成的语义锚点，提供统一的特征空间坐标系
- **动态专家网络**：按语义簇组织的专家池，支持动态扩展
- **概率分布提示**：压缩存储类别知识，支持无数据回放
- **大模型决策中心**：统筹专家分配、拆分与演化

## 3. 全局锚点系统

### 3.1 锚点生成

我们利用预训练视觉-语言大模型 $\mathcal{M}$（如CLIP）生成两级语义锚点：

**簇级锚点**（用于路由）：
$$\mathbf{A}_s = \mathcal{M}_{\text{text}}(\text{``a photo of } s\text{''}) \in \mathbb{R}^d, \quad s \in \mathcal{S}$$

其中 $\mathcal{S} = \{s_1, s_2, ..., s_M\}$ 为语义簇集合（如"动物"、"交通工具"等）。

**类级锚点**（用于分类）：
$$\mathbf{A}_c = \mathcal{M}_{\text{text}}(\text{``a photo of } c\text{''}) \in \mathbb{R}^d, \quad c \in \mathcal{C}$$

每个类级锚点 $\mathbf{A}_c$ 归属于某个语义簇 $s(c)$，该归属关系由大模型根据语义相似度决定。

### 3.2 锚点的作用

全局锚点在框架中承担三重职责：

1. **路由依据**：决定输入样本应由哪个专家处理
2. **对齐目标**：约束所有客户端的特征空间全局一致
3. **分类基准**：专家内细粒度分类的参照

关键设计：所有客户端持有**完整的全局锚点库**，确保在统一的语义坐标系下学习。

## 4. 网络架构

### 4.1 特征提取与路由

给定输入 $x$，特征提取与路由过程如下：

$$\mathbf{f} = \mathcal{F}_{\text{backbone}}(x) \in \mathbb{R}^{d_0}$$

$$\mathbf{f}' = \mathcal{R}_\theta(\mathbf{f}) \in \mathbb{R}^d$$

其中 $\mathcal{F}_{\text{backbone}}$ 为冻结的预训练视觉编码器，$\mathcal{R}_\theta$ 为可学习的路由投影层。

路由决策基于特征与簇级锚点的相似度：

$$p(E_i | x) = \frac{\exp(\text{sim}(\mathbf{f}', \mathbf{A}_{s_i}) / \tau_r)}{\sum_{j=1}^{M} \exp(\text{sim}(\mathbf{f}', \mathbf{A}_{s_j}) / \tau_r)}$$

$$i^* = \arg\max_i \, p(E_i | x)$$

其中 $\text{sim}(\cdot, \cdot)$ 为余弦相似度，$\tau_r$ 为温度系数，$E_i$ 为负责语义簇 $s_i$ 的专家。

### 4.2 专家网络

每个专家 $E_i$ 负责一个语义簇内的细粒度分类：

$$\mathbf{f}_{\text{expert}} = E_i(\mathbf{f}') = \text{MLP}_i(\mathbf{f}') \in \mathbb{R}^d$$

分类在专家负责的类别子集上进行：

$$p(c | x, E_i) = \frac{\exp(\text{sim}(\mathbf{f}_{\text{expert}}, \mathbf{A}_c) / \tau_c)}{\sum_{c' \in \mathcal{C}(E_i)} \exp(\text{sim}(\mathbf{f}_{\text{expert}}, \mathbf{A}_{c'}) / \tau_c)}$$

其中 $\mathcal{C}(E_i) \subset \mathcal{C}$ 为专家 $E_i$ 负责的类别集合。

### 4.3 客户端-服务端专家解耦

服务端维护完整的专家池 $\{E_1, E_2, ..., E_N\}$，而每个客户端仅持有与其本地数据相关的专家子集：

$$\mathcal{E}_k = \{E_i : \mathcal{C}_k \cap \mathcal{C}(E_i) \neq \emptyset\}$$

这一设计显著降低了客户端的计算和存储开销。

## 5. 概率分布提示

### 5.1 分布定义

为每个类别 $c$ 维护一个概率分布 $P(\mathbf{z}|c)$，作为该类知识的压缩表示：

$$P(\mathbf{z}|c) = \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$$

为保证跨客户端可聚合性，将分布均值分解为全局锚点与可学习残差：

$$\boldsymbol{\mu}_c = \mathbf{A}_c + \boldsymbol{\delta}_c, \quad \|\boldsymbol{\delta}_c\| \leq \epsilon$$

其中 $\mathbf{A}_c$ 为固定的类锚点，$\boldsymbol{\delta}_c$ 为有界的可学习残差。

### 5.2 分布的作用

**训练时增强**：从分布采样，与特征拼接后送入专家：

$$\mathbf{z} \sim P(\mathbf{z}|y), \quad \mathbf{f}_{\text{aug}} = [\mathbf{f}'; \mathbf{z}]$$

**防遗忘回放**：学习新类时，从旧类分布采样进行知识蒸馏：

$$\mathbf{z}_{\text{old}} \sim P(\mathbf{z}|c_{\text{old}}), \quad c_{\text{old}} \in \mathcal{C}^{1:t-1}$$

### 5.3 分布聚合

客户端 $k$ 上传分布参数 $\{\boldsymbol{\delta}_k^c, \boldsymbol{\Sigma}_k^c, n_k^c\}$，服务端按样本数加权聚合：

$$\boldsymbol{\delta}_{\text{global}}^c = \frac{\sum_{k \in \mathcal{K}_c} n_k^c \cdot \boldsymbol{\delta}_k^c}{\sum_{k \in \mathcal{K}_c} n_k^c}$$

$$\boldsymbol{\Sigma}_{\text{global}}^c = \frac{\sum_{k \in \mathcal{K}_c} n_k^c \cdot (\boldsymbol{\Sigma}_k^c + \boldsymbol{\delta}_k^c {\boldsymbol{\delta}_k^c}^\top)}{\sum_{k \in \mathcal{K}_c} n_k^c} - \boldsymbol{\delta}_{\text{global}}^c {\boldsymbol{\delta}_{\text{global}}^c}^\top$$

其中 $\mathcal{K}_c = \{k : c \in \mathcal{C}_k\}$ 为拥有类别 $c$ 数据的客户端集合。

## 6. 大模型决策机制

### 6.1 专家分配

当新类 $c_{\text{new}}$ 到来时，大模型计算其与现有簇的语义相似度：

$$s^* = \arg\max_{s \in \mathcal{S}} \text{sim}(\mathbf{A}_{c_{\text{new}}}, \mathbf{A}_s)$$

若 $\max_s \text{sim}(\cdot) > \tau_{\text{assign}}$，将新类分配至专家 $E_{s^*}$；否则触发新专家创建。

### 6.2 专家拆分

当专家 $E_i$ 满足以下条件时触发拆分：

$$|\mathcal{C}(E_i)| > \kappa \quad \text{or} \quad \max_{c, c' \in \mathcal{C}(E_i)} \|\mathbf{A}_c - \mathbf{A}_{c'}\| > \tau_{\text{split}}$$

大模型对 $\mathcal{C}(E_i)$ 进行语义聚类，将专家拆分为 $E_{i_1}, E_{i_2}$，参数继承自原专家。

## 7. 训练目标

### 7.1 总体损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_1 \mathcal{L}_{\text{route}} + \lambda_2 \mathcal{L}_{\text{contrast}} + \lambda_3 \mathcal{L}_{\text{forget}} + \lambda_4 \mathcal{L}_{\text{dist}}$$

### 7.2 各项损失

**分类损失**：专家内细粒度分类
$$\mathcal{L}_{\text{cls}} = -\log p(y | x, E_{i^*})$$

**路由损失**：确保正确的专家选择
$$\mathcal{L}_{\text{route}} = -\log p(E_{s(y)} | x)$$

**全局对比损失**：保证特征空间全局对齐
$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(\mathbf{f}', \mathbf{A}_y) / \tau)}{\sum_{c \in \mathcal{C}^{1:t}} \exp(\text{sim}(\mathbf{f}', \mathbf{A}_c) / \tau)}$$

**防遗忘损失**：基于分布回放的知识蒸馏
$$\mathcal{L}_{\text{forget}} = \sum_{c \in \mathcal{C}^{1:t-1}} \mathbb{E}_{\mathbf{z} \sim P(\mathbf{z}|c)} \left[ D_{\text{KL}}(q_{\text{old}}(\cdot|\mathbf{z}) \| q_{\text{new}}(\cdot|\mathbf{z})) \right]$$

**分布正则化**：约束分布围绕锚点
$$\mathcal{L}_{\text{dist}} = \sum_{c \in \mathcal{C}_k^t} \|\boldsymbol{\delta}_c\|^2$$

## 8. 联邦聚合策略

### 8.1 路由层聚合

路由层全局共享，所有客户端参与标准联邦聚合：

$$\theta_{\text{router}} = \sum_{k=1}^{K} \frac{n_k}{N} \theta_{\text{router}}^k$$

### 8.2 专家分组聚合

专家按语义簇分组，仅相关客户端参与聚合：

$$\phi_i = \sum_{k \in \mathcal{K}_i} \frac{n_k^i}{N_i} \phi_i^k, \quad \mathcal{K}_i = \{k : E_i \in \mathcal{E}_k\}$$

### 8.3 分布聚合

按类别聚合分布参数（见5.3节）。

## 9. 算法流程

**算法1**：FedAME 联邦类增量学习

---

**输入**：客户端集合 $\{1,...,K\}$，任务序列 $\{1,...,T\}$，预训练大模型 $\mathcal{M}$

**初始化**：全局锚点库 $\mathcal{A}$，专家池 $\mathcal{E}$，路由层 $\mathcal{R}_\theta$，分布库 $\mathcal{P}$

---

**For** 任务 $t = 1, ..., T$ **do**

$\quad$ **[阶段1: 类别上报与专家分配]**
  
$\quad$ **For** 客户端 $k = 1, ..., K$ **并行 do**
  
$\quad\quad$ 上报新类列表 $\mathcal{C}_k^{t,\text{new}}$ 至服务端
  
$\quad$ **End For**
  
$\quad$ 服务端：大模型为新类生成锚点并决定专家分配/拆分
  
$\quad$ 服务端：下发更新后的锚点库、专家分配表、相关专家参数

---

$\quad$ **[阶段2: 联邦训练]**
  
$\quad$ **For** 通信轮次 $r = 1, ..., R$ **do**
  
$\quad\quad$ **For** 客户端 $k = 1, ..., K$ **并行 do**
  
$\quad\quad\quad$ 本地训练：优化 $\mathcal{L}_{\text{total}}$，更新 $\{\mathcal{R}_\theta, \mathcal{E}_k, \mathcal{P}_k\}$
  
$\quad\quad\quad$ 上传：路由层更新、相关专家更新、分布参数
  
$\quad\quad$ **End For**
  
$\quad\quad$ 服务端：执行分组聚合
  
$\quad\quad$ 服务端：下发更新后的全局参数
  
$\quad$ **End For**

**End For**

---

**输出**：全局模型 $\{\mathcal{R}_\theta, \mathcal{E}, \mathcal{A}, \mathcal{P}\}$

## 10. 大小模型融合分析

本框架通过以下方式实现大模型与小模型的有效融合：

| 组件 | 模型类型 | 作用 | 位置 |
|:---:|:---:|:---:|:---:|
| 视觉编码器 | 大模型（冻结） | 提供稳定的特征表示 | 服务端/客户端 |
| 文本编码器 | 大模型（冻结） | 生成语义锚点 | 仅服务端 |
| 语义决策 | 大模型（推理） | 专家分配与拆分决策 | 仅服务端 |
| 路由层 | 小模型（可学习） | 特征投影与路由 | 服务端/客户端 |
| 专家网络 | 小模型（可学习） | 细粒度分类 | 按需分发 |
| 分布参数 | 小模型（可学习） | 知识压缩存储 | 按需分发 |

**融合优势**：

1. **知识互补**：大模型提供全局语义结构，小模型学习任务特定知识
2. **计算高效**：大模型仅用于推理（锚点生成、决策），不参与梯度更新
3. **通信高效**：只需传输轻量级小模型参数
4. **扩展灵活**：大模型决策使专家结构可动态演化
