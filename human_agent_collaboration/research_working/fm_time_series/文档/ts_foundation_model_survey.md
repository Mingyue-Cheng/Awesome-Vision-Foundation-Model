# 时间序列基础模型（Time Series Foundation Models）研究调研报告

**生成日期**：2026-04-07
**关键词**：Time Series Foundation Model, Zero-shot Forecasting, Pretrained Model, LLM for Time Series, Patch-based Transformer, Mixture of Experts, Benchmark
**信息来源数量**：35+ 篇文献 / 报告 / 技术博客

---

## 1. 背景综述

### 1.1 研究背景与动机

时间序列预测是工业界最普遍的数据挖掘任务之一，广泛应用于能源调度、金融风控、气象预警、交通规划和医疗健康等关键领域。然而，在基础模型兴起之前，该领域长期面临一个根本性困境：**每个任务、每个数据集几乎都需要从头训练一个专门模型**。

传统方法体系分为两代：

**第一代：经典统计方法**（ARIMA、ETS、TBATS、Prophet）
- 优势：可解释性强，适合平稳序列和低频数据
- 局限：需要逐系列建模，无法跨域泛化；参数假设强（线性、平稳性）；难以捕获复杂非线性模式

**第二代：深度学习方法**（DeepAR、N-BEATS、Informer、PatchTST、iTransformer）
- 优势：强大的非线性表达能力，可建模长程依赖
- 局限：仍需针对目标数据集训练，数据饥渴（data-hungry），不同域/频率间迁移性差；缺乏通用性

**第三代：时间序列基础模型（Time Series Foundation Models, TSFMs）**
- 目标：训练一个具有强大零样本/少样本泛化能力的预训练模型，能够在不见过的数据集上直接进行高质量预测
- 动机来源：NLP 中 GPT/BERT 的成功，以及 CV 中 ViT 的普适性表征学习经验

### 1.2 核心问题定义

时间序列基础模型面临的核心挑战与 NLP 有本质差异：

| 维度 | NLP 基础模型 | 时间序列基础模型 |
|------|------------|----------------|
| 数据模态 | 离散 token，语义明确 | 连续值，语义隐性 |
| 跨域统一性 | 文本模态天然统一 | 频率、粒度、量纲高度异质 |
| 词汇表 | 固定词汇表（3 万~10 万） | 连续实数，无天然词汇 |
| 上下文长度 | 自然语言段落/文档 | 从几十点到数万点不等 |
| 预训练规模 | 万亿 token | 数十亿至数千亿时间点 |

### 1.3 研究重要性

1. **效率**：无需为每个新数据集重新训练，显著降低部署成本
2. **数据效率**：在数据稀缺场景（冷启动）下依然可用
3. **通用性**：单一模型服务数十、数百个预测任务
4. **科学意义**：探索时间序列的通用规律和表征空间

### 1.4 报告结构

本报告依次覆盖：研究现状概览（§2）→ 核心方法分类（§3，含预训练范式、架构设计、零样本泛化）→ 重点模型深度解读（§4）→ 横向技术对比（§5）→ Benchmark 与评测体系（§6）→ 2024-2025 最新进展（§7）→ 核心挑战与未来方向（§8）→ 参考文献（§9）。

---

## 2. 研究现状概览

### 2.1 主要研究方向分类

时间序列基础模型可沿两条主线划分：

**主线 A：从零预训练时间序列专用基础模型**
> 在大规模、多域时间序列数据集上预训练专用模型，不依赖 NLP/CV 预训练权重
- 代表：TimesFM（Google）、Moirai（Salesforce）、Lag-Llama、Chronos（Amazon）、Timer（Tsinghua）、MOMENT、Time-MoE、TTM（IBM）

**主线 B：利用大型语言模型（LLM）适配时间序列**
> 将预训练 LLM（GPT-2、LLaMA 等）通过重编程（reprogramming）、提示（prompting）或微调迁移到时序任务
- 代表：GPT4TS（NeurIPS 2023）、Time-LLM（ICLR 2024）、TEMPO（ICLR 2024）

**主线 C：自监督表征学习（作为预训练基础）**
> 不直接以预测为目标，学习通用时序表征，下游任务再微调
- 代表：TS2Vec、CoST、SimTS、SOFT Contrastive

### 2.2 代表性工作时间线

```
2021  Monash 数据集库 ——————————————— 数据基础设施
2022  TS2Vec, CoST ——————————————— 自监督表征预训练探索
2023  ForecastPFN (NeurIPS)
      GPT4TS / One Fits All (NeurIPS Spotlight)
      TimeGPT-1 (Nixtla, arXiv)
      Lag-Llama (arXiv, ServiceNow/Mila)
2024  PatchTST → 基础架构确立 (ICLR 2023 奠基)
      iTransformer (ICLR 2024 Spotlight, Tsinghua)
      TimesFM (ICML 2024, Google)
      Moirai (ICML 2024 Oral, Salesforce)
      Timer (ICML 2024, Tsinghua THUML)
      MOMENT (ICML 2024, CMU AutonLab)
      TEMPO (ICLR 2024)
      Time-LLM (ICLR 2024)
      TTM / Tiny Time Mixers (NeurIPS 2024, IBM)
      UniTS (NeurIPS 2024, Harvard)
      TFB Benchmark (PVLDB 2024 Best Paper Nom.)
      Moirai-MoE (arXiv 2024, Salesforce)
      GIFT-Eval Benchmark (arXiv 2024, Salesforce)
      Time-MoE 2.4B (ICLR 2025 Spotlight)
2025  Timer-XL (ICLR 2025, Tsinghua)
      Moirai 2.0, Chronos-Bolt, Chronos-2 (Amazon)
      TimesFM 2.0/2.5 (Google)
      Scaling Laws 研究 (NVIDIA)
```

### 2.3 研究热度与趋势

- 2024 年是 TSFMs 爆发年：主流顶会（ICML/NeurIPS/ICLR）均有多篇相关工作
- 2025 年趋势：更大规模模型（2.4B 参数）、多变量通用化、评测标准化
- 社区争议：LLM 用于时序是否有实质增益（NeurIPS 2024 批判性论文出现）
- 工业落地：TimeGPT（API 商业化）、IBM Granite Time Series、AWS Chronos 集成

---

## 3. 核心方法分类

### 3.1 预训练范式

#### 3.1.1 自回归生成预训练（Next Token Prediction）

最主流的范式，直接沿用 GPT 的语言模型预训练思路：

- **输入**：历史时序片段（以 patch 为 token）
- **目标**：预测下一个 patch（自回归方式）
- **优点**：自然支持预测任务，不需要额外下游适配头
- **代表**：TimesFM、Timer、Timer-XL、Time-MoE、Lag-Llama

```
历史序列: [p1][p2][p3] → 预测 [p4]
                [p2][p3][p4] → 预测 [p5]  (自回归)
```

#### 3.1.2 掩码重建预训练（Masked Patch Modeling）

借鉴 BERT/MAE 的掩码重建思路：

- **输入**：随机掩盖部分 patch，用 [MASK] token 替换
- **目标**：重建被掩盖的 patch
- **优点**：可以同时学习局部模式和全局上下文；适合多任务（预测、插补、异常检测）
- **代表**：MOMENT、PatchTST（self-supervised 变体）

#### 3.1.3 量化离散化 + 语言模型预训练

将连续时序值转化为离散 token，复用 NLP 语言模型的词汇预测框架：

- **步骤**：1）对序列按绝对均值缩放 → 2）均匀分箱量化（如 4096 个 bin）→ 3）用 T5/GPT 架构做 cross-entropy 训练
- **优点**：直接复用 LLM 基础设施；支持概率预测（每个 bin 有概率）
- **代表**：Chronos（Amazon）

#### 3.1.4 合成数据预训练（Prior-Fitted Networks）

完全不使用真实时序数据，通过合成数据（先验）训练：

- **原理**：Prior-Data Fitted Networks（PFN）——设计覆盖广泛时序模式的合成先验，训练后近似贝叶斯推断
- **优点**：无需标注数据；单次前向即可完成预测（速度极快）
- **代表**：ForecastPFN（NeurIPS 2023）

#### 3.1.5 对比学习预训练

无监督地学习时序表征，不以预测为目标：

- 正例：同一序列的不同视角（时间掩码、尺度变换等增强）
- 负例：不同序列或不同时间步
- **代表**：TS2Vec（层次对比）、CoST（季节-趋势解耦对比）、SimTS

### 3.2 架构选择

#### 3.2.1 Patch-based Transformer（主流）

将时序划分为固定长度的 patch（子序列段），每个 patch 作为一个 token：

- **PatchTST**（ICLR 2023）：奠定了 patch 化思想的地位，channel-independent 设计
- **TimesFM**：patch_len=32，自回归生成 output_patch_len=128
- **MOMENT**：T5 编码器 + patch masking，patch 长度固定为 8
- **Timer**：GPT-style，S3 格式统一序列

**优点**：显著减少 attention 计算量（序列长度 L 被压缩为 L/p）；局部语义保留；支持更长上下文

#### 3.2.2 Channel-Independent vs. Channel-Dependent

| 策略 | 核心思想 | 优点 | 缺点 | 代表模型 |
|------|---------|------|------|---------|
| **Channel-Independent (CI)** | 每个变量独立建模，共享 Transformer 权重 | 参数效率高；泛化到任意变量数；抗过拟合 | 无法捕捉变量间相关性 | PatchTST、Lag-Llama、TimesFM |
| **Channel-Dependent (CD)** | 联合建模所有变量间的相关性 | 利用多变量信息；理论上表达能力更强 | 变量数固定；更容易过拟合 | iTransformer、Moirai（any-variate attention）|
| **混合/自适应** | 动态决定是否使用变量间信息 | 兼顾两者优点 | 复杂度高 | UniTS、Timer-XL |

iTransformer（ICLR 2024）提出了"倒置 Transformer"：将每个变量的完整时间序列嵌入为一个 token，通过 attention 捕捉变量间关系，实验上超越了大量 CD 基线。

Moirai 提出 **Any-Variate Attention**：通过特殊设计支持任意变量数输入，在不同预测任务间保持通用性。

#### 3.2.3 Decoder-Only vs. Encoder-Only vs. Encoder-Decoder

| 架构 | 代表模型 | 特点 |
|------|---------|------|
| Decoder-only（GPT-style） | TimesFM、Timer、Timer-XL、Time-MoE、Moirai-MoE、Moirai 2.0 | 自回归生成，天然支持预测；flexible context |
| Encoder-only（BERT-style） | MOMENT（T5 encoder）、Moirai 1.0 | 适合多任务；掩码重建预训练 |
| Encoder-Decoder | Chronos（T5）、TTM | 编码历史，解码未来 |

趋势：2025 年新模型多向 decoder-only 收敛，因其自回归特性与预测任务天然对齐，且训练效率更高（一次更新可覆盖多个上下文长度）。

#### 3.2.4 Mamba/SSM（新兴方向）

Mamba（Selective State Space Model）作为 Transformer 的替代架构，近年在时序领域引发关注：

- **线性复杂度**：相对于 Transformer 的 O(L²)，Mamba 为 O(L)
- **S-Mamba**：双向 Mamba 块，同时提取变量间相关性和时间依赖
- **MambaTS**：针对长期预测改进 Mamba，引入变量扫描顺序和 permutation 训练
- 目前 Mamba 在时序上的优势尚未形成压倒性共识，大多数顶会工作仍以 Transformer 为主

#### 3.2.5 Mixture of Experts（MoE）

MoE 在时序基础模型中正快速兴起，解决"一个模型适配所有模式"的矛盾：

- **Time-MoE**（ICLR 2025 Spotlight）：2.4B 参数，稀疏 MoE，以同等激活参数量大幅超越 dense 基线
- **Moirai-MoE**（Salesforce 2024）：Token 级别自动专家分配，基于预训练模型聚类中心引导路由，比 Moirai 提升 17%

### 3.3 Zero-shot / Few-shot 泛化能力

TSFMs 的核心价值之一在于泛化能力，通常通过以下协议评测：

- **Zero-shot**：模型完全未见目标数据集，直接推断
- **Few-shot**：在目标数据集上用少量样本（1%~10%）微调
- **Full-shot**：在目标数据集上完整训练

当前最佳零样本性能：TimesFM/Moirai/Chronos 在多数 benchmark 上接近甚至超越针对特定数据集训练的有监督模型，但在高度异质的领域仍有差距。

---

## 4. 重点文献深度解读

### 4.1 从零预训练的时间序列专用基础模型

#### 4.1.1 TimesFM：Google 的 Decoder-Only 时间序列基础模型

- **来源**：Das et al., Google Research，ICML 2024
- **arXiv**：[2310.10688](https://arxiv.org/abs/2310.10688)

**核心贡献**：
首个在工业规模（1000 亿真实时间点）上预训练、并在 ICML 顶会上发表的时间序列基础模型。实现了 200M 参数规模下接近 SOTA 有监督方法的零样本预测性能。

**模型架构**：
- 架构：Decoder-only Transformer，仿照 GPT 设计
- 关键参数：`input_patch_len=32`，`output_patch_len=128`，`num_layers=20`，`model_dims=1280`
- 每次自回归步生成 128 个未来值，预测 512 步只需 4 次解码
- 点预测（确定性），不直接输出分布（TimesFM 2.0 后支持分位数预测）
- Channel-independent 设计

**训练数据**：
- 100B 真实时间点，来自 Google 内部数据及公开数据集
- 混合合成数据（Gaussian Process 生成）

**关键结果**：
- 零样本性能在多个 benchmark 接近或超越有监督基线
- 参数量（200M）远小于通用 LLM，推理效率高

**局限性**：
- 仅支持点预测（v1.0）
- 原始版本为 channel-independent，无法利用多变量间相关性

---

#### 4.1.2 Moirai：Salesforce 的通用时间序列预测 Transformer

- **来源**：Woo et al., Salesforce AI Research，ICML 2024 Oral
- **arXiv**：[2402.02592](https://arxiv.org/abs/2402.02592)

**核心贡献**：
解决时间序列基础模型的三大挑战：跨频率学习、任意变量数适配、分布多样性建模。提出 LOTSA 数据集（270 亿观测值）和 Moirai 模型系列（Small/Base/Large）。

**模型架构**：
- 架构：Masked Encoder-only Transformer（Moirai 1.0），Decoder-only（Moirai 2.0）
- **多尺度 Patch 投影层**：使用多个不同大小的 patch 投影层，单一模型可捕获不同频率下的时序模式
- **Any-Variate Attention**：特殊 attention 设计，允许单模型处理任意数量的变量
- **混合分布输出头**：融合多种分布（Student-t、负二项、对数正态等），建模复杂预测分布

**训练数据（LOTSA）**：
- Large-scale Open Time Series Archive
- 270 亿观测值，跨 9 个域（能源、金融、交通、气候、销售等）
- 作为开源资源发布

**关键结果**：
- 零样本下与有监督方法竞争或超越
- 作为 ICML 2024 Oral，是该批次认可度最高的时序基础模型之一

**局限性**：
- Moirai 1.0 encoder-only 在自回归生成方面效率不如 decoder-only

---

#### 4.1.3 Chronos：Amazon 的"时间序列语言"

- **来源**：Ansari et al., Amazon，TMLR 2024（arXiv 2403.07815）
- **链接**：[https://arxiv.org/abs/2403.07815](https://arxiv.org/abs/2403.07815)

**核心贡献**：
将时序预测转化为语言模型问题：通过**量化-离散化** tokenization，将连续时间值转换为固定词汇表 token，直接复用 T5 架构和 cross-entropy 损失训练。

**模型架构**：
- 基础架构：T5 Encoder-Decoder，规模从 20M（Mini）到 710M（Large）
- Tokenization：按绝对均值缩放后，均匀分箱量化成 4096 个 bin token
- 特殊 token：PAD（缺失值/填充），EOS（序列结束）
- 输出：每个 bin 的概率分布，从而得到预测分布

**训练数据**：
- 大量公开时间序列数据集
- Gaussian Process 生成的合成数据（提升 OOD 泛化）

**关键结果**：
- 42 个数据集的综合 benchmark：训练集内数据显著超越竞争方法；OOD 数据集上零样本接近有监督基线
- Chronos-Bolt（2024）：5% 更低误差，速度提升 250 倍，内存效率提升 20 倍

**局限性**：
- 原始版本为单变量（univariate only）
- 量化粒度限制了对细粒度连续值的精确建模

---

#### 4.1.4 Lag-Llama：概率时间序列预测的基础模型

- **来源**：Rasul et al., ServiceNow / Mila / Université de Montréal / McGill，2024
- **arXiv**：[2310.08278](https://arxiv.org/abs/2310.08278)

**核心贡献**：
首批开源概率时间序列预测基础模型之一。借鉴 LLaMA 架构，以滞后值（lags）和日历特征作为协变量，输出预测分布，支持零样本和少样本预测。

**模型架构**：
- 架构：Decoder-only Transformer，仿照 LLaMA 设计
- 输入特征：滞后值（lagged values）+ 日历特征（年/月/周/日等）
- 输出：概率分布（Student-t 分布头）
- Channel-independent，仅支持单变量预测

**训练数据**：
- 多个公开时序数据集，跨多个域

**关键结果**：
- 与众多预测模型相比展示出强零样本泛化能力
- 少量微调即可达到 SOTA 水平

**局限性**：
- 仅支持单变量；多变量场景表现受限
- 模型规模相对较小

---

#### 4.1.5 Timer：清华 THUML 的大型时间序列模型

- **来源**：Liu et al., 清华大学 THUML，ICML 2024
- **arXiv**：[2402.02368](https://arxiv.org/abs/2402.02368)

**核心贡献**：
提出 GPT-style 的大型时间序列模型（LTSM），以**单序列格式（Single-series Sequence, S3）** 统一异质多域时序数据，通过下一个 token 预测进行生成式预训练。

**模型架构**：
- 架构：Decoder-only Transformer（GPT-style）
- S3 格式：将不同域、不同频率的时序数据标准化为统一格式
- 统一多任务：预测、插补、异常检测均转化为生成式任务

**训练数据（UTSD）**：
- Unified Time Series Dataset，覆盖 7 个域，最大规模达 12G 时间点（UTSD-12G）

**关键结果**：
- 展现明显的少样本泛化性、可扩展性和任务通用性
- 在预测、插补、异常检测上超越专用 SOTA 模型

**局限性**：
- 仅限单变量，多变量扩展留给 Timer-XL

---

#### 4.1.6 Timer-XL：长上下文时间序列统一预测

- **来源**：Liu et al., 清华大学 THUML，ICLR 2025
- **arXiv**：[2410.04803](https://arxiv.org/abs/2410.04803)

**核心贡献**：
将 Timer 的单变量范式扩展至**多变量统一预测**，提出 TimeAttention 机制，支持任意长度和任意变量数的时序，适配协变量信息。

**模型架构**：
- 架构：Causal Transformer（因果 Transformer）
- **TimeAttention**：统一捕获序列内（intra-series）和序列间（inter-series）的细粒度依赖
- 支持 patch 级 token，位置嵌入嵌入时间因果性
- 支持协变量（exogenous variables）输入

**关键结果**：
- 在单变量、多变量、协变量预测三类场景均达到竞争性性能
- ICLR 2025 接受，代表清华 THUML 在大型时序模型方向的系统化深耕

---

#### 4.1.7 MOMENT：CMU AutonLab 的开源时间序列基础模型家族

- **来源**：Goswami et al., CMU AutonLab，ICML 2024
- **arXiv**：[2402.03885](https://arxiv.org/abs/2402.03885)

**核心贡献**：
提出基于 T5 编码器的开源时间序列基础模型家族，采用**掩码 patch 重建**作为预训练任务，构建 The Time Series Pile 数据集，支持预测、分类、异常检测和插补等多任务。

**模型架构**：
- 基础架构：T5-Large Encoder（约 385M 参数）
- Patch 化：将时序切分为不重叠的固定长度 patch
- 预训练任务：随机掩盖约 30% 的 patch（用 [MASK] token 替换），重建掩盖内容
- 下游适配：轻量级任务特定头（reconstruction head / prediction head）

**训练数据（The Time Series Pile）**：
- 整合多个公开域的时序数据集（医疗、工程、金融等）

**关键结果**：
- 在有限监督下（minimal fine-tuning）展现强劲性能
- 开源生态友好：Hugging Face 上提供预训练模型（AutonLab/MOMENT-1-large）

**局限性**：
- Encoder-only 架构在自回归预测上需要额外适配
- 对超长序列支持有限

---

#### 4.1.8 Time-MoE：十亿级时间序列基础模型

- **来源**：Shi et al., ICLR 2025 Spotlight（Top 5.1%）
- **arXiv**：[2409.16040](https://arxiv.org/abs/2409.16040)

**核心贡献**：
首个超过 **10 亿参数**的时序基础模型（最大版本 2.4B），通过**稀疏混合专家（Sparse MoE）** 在保持推理效率的同时大幅提升模型容量。提出 Time-300B 数据集（3000 亿时间点）。验证了时序领域的 Scaling Law。

**模型架构**：
- 架构：Decoder-only Transformer + Sparse MoE 层
- MoE 设计：每次推理只激活部分专家子网络（spare activation），降低计算成本
- 自回归生成，支持灵活预测 horizon 和输入上下文长度

**训练数据（Time-300B）**：
- 3000 亿时间点，跨 9 个以上域
- 迄今最大的开放时序预训练数据集

**关键结果**：
- 与同等激活参数量的 dense 模型相比，Time-MoE 大幅领先
- 验证了时序领域的 data scaling law 和 model scaling law

---

#### 4.1.9 Tiny Time Mixers（TTM）：IBM 的高效轻量基础模型

- **来源**：Ekambaram et al., IBM Research，NeurIPS 2024
- **arXiv**：[2401.03955](https://arxiv.org/abs/2401.03955)

**核心贡献**：
极致轻量的时序基础模型（起步 1M 参数），基于 TSMixer 架构而非 Transformer。通过**自适应 patch**、**多分辨率采样**和**分辨率前缀微调**，在 CPU 上即可运行，性价比极高。

**模型架构**：
- 架构：TSMixer（MLP-Mixer 变体，非 Transformer）
- 核心创新：adaptive patching（自适应 patch 长度）、resolution prefix tuning
- 参数规模：1M（轻量版）~数十 MB

**关键结果**：
- 零样本/少样本预测比 TimesFM、Moirai、Chronos、Lag-Llama、MOMENT、GPT4TS 和 Time-LLM 提升 4%~40%
- 可在仅 5% 训练数据下微调并达到竞争性性能
- IBM Granite Time Series 系列的基础

---

#### 4.1.10 UniTS：Harvard MIMS 的统一多任务时序模型

- **来源**：Gao et al., Harvard University MIMS Lab，NeurIPS 2024
- **arXiv**：[2403.00131](https://arxiv.org/abs/2403.00131)

**核心贡献**：
统一处理预测、分类、插补和异常检测四类任务的单一模型，通过**任务 token 化（task tokenization）** 将不同任务统一表达，在 38 个数据集上均表现优异。

**模型架构**：
- 改进型 Transformer 块
- 任务 token：特定任务通过可学习的 task token 注入，指导模型行为
- 共享权重，单模型服务所有任务

**关键结果**：
- 在插补任务上超越最强基线 12.4%（MSE）
- 在异常检测上超越最强基线 2.3%（F1）
- 预训练于异质多域多任务数据，具备强迁移能力

---

#### 4.1.11 TimeGPT-1：首个商业化时序基础模型

- **来源**：Garza et al., Nixtla，arXiv 2023（[2310.03589](https://arxiv.org/abs/2310.03589)）

**核心贡献**：
首个以 API 形式对外提供的时序基础模型，支持零样本预测和异常检测。训练于超过 1000 亿时间点。

**模型架构**：
- 专有架构（未完全公开）
- API 服务：用户通过 REST API 直接调用，无需本地部署

**关键结果**：
- 在多个零样本 benchmark 上达到或超越有监督基线
- 已有企业用于零代码快速部署时序预测

**局限性**：
- 闭源，无法复现和研究内部机制
- 商业 API 存在访问成本

---

### 4.2 基于 LLM 适配的时间序列方法

#### 4.2.1 GPT4TS / One Fits All：冻结 GPT-2 的时序通用分析

- **来源**：Zhou et al., Alibaba DAMO Academy，NeurIPS 2023 Spotlight
- **arXiv**：[2302.11939](https://arxiv.org/abs/2302.11939)

**核心贡献**：
冻结预训练 GPT-2 的 Self-Attention 和 FFN 层（仅微调位置嵌入、LayerNorm 和输出线性层），证明语言模型的预训练权重对时序任务有迁移价值。适用于长短期预测、分类、插补、异常检测和少样本学习。

**技术方案**：
- Instance Norm + Patching 对原始时序预处理
- 线性投影层将 patch 映射到 GPT-2 隐层维度
- 仅使用 GPT-2 的前 6 层（12 层中的前半部分）
- 冻结 attention 和 FFN，仅调整 positional embedding 和 LayerNorm

**关键结果**：
- 在 6 类任务上超越多数任务专用 SOTA 模型
- NeurIPS 2023 Spotlight 认可

**局限性**：
- 对 GPT-2 的利用程度存在争议（NeurIPS 2024 批判性工作指出 LLM 组件或可替换）

---

#### 4.2.2 Time-LLM：重编程 LLM 用于时序预测

- **来源**：Jin et al., Monash University / Ant Group，ICLR 2024
- **arXiv**：[2310.01728](https://arxiv.org/abs/2310.01728)

**核心贡献**：
提出**重编程（Reprogramming）**框架，将时序 patch 映射为"文本原型表示（text prototypes）"，使冻结的 LLM（LLaMA/GPT-2）能够感知时序输入，同时引入 **Prompt-as-Prefix（PaP）** 丰富上下文。

**技术方案**：
- Patch Reprogramming：将时序 patch 通过注意力机制对齐到 LLM 词嵌入空间中的文本原型
- Prompt-as-Prefix：自然语言描述（任务类型、领域、趋势描述）作为 prefix 注入 LLM
- LLM 骨干保持冻结（Frozen），降低微调成本

**关键结果**：
- 平均比 GPT4TS 提升 12%，比 TimesNet 提升 20%
- 零样本场景比 GPT4TS 提升 22%

**局限性**：
- 推理开销大（依赖完整 LLM 前向）
- NeurIPS 2024 批判性研究质疑 LLM 组件的实质贡献

---

#### 4.2.3 TEMPO：基于提示的生成式预训练 Transformer

- **来源**：Cao et al., Georgia Tech，ICLR 2024
- **arXiv**：[2310.04948](https://arxiv.org/abs/2310.04948)

**核心贡献**：
结合**时序分解**（趋势 + 季节性 + 残差）和**可学习软提示（soft prompts）** 的 GPT 微调框架，是首批开源时序基础模型之一。

**技术方案**：
- STL 分解：将时序分解为趋势、季节性、残差三个分量
- 对每个分量分别使用 GPT 骨干编码
- 软提示：learnable prompt vectors 编码时序的趋势和季节性先验知识
- 分布适配：提示辅助模型适应不同类型的时序分布

**关键结果**：
- 零样本场景下超越 SOTA 方法
- 支持多模态输入（时序 + 文本）

---

### 4.3 关键预训练前置工作

#### 4.3.1 ForecastPFN：合成数据训练的零样本预测

- **来源**：Dooley et al., Abacus AI，NeurIPS 2023
- **arXiv**：[2311.01933](https://arxiv.org/abs/2311.01933)

**核心贡献**：
首个完全基于合成数据（无真实时序）预训练的零样本预测模型，近似贝叶斯推断。单次前向 0.2 秒，比 Transformer 方法快 100 倍以上。

---

#### 4.3.2 PatchTST：Patch 化时序 Transformer 的奠基作

- **来源**：Nie et al., Princeton / IBM，ICLR 2023
- **arXiv**：[2211.14730](https://arxiv.org/abs/2211.14730)

**核心贡献**：
提出时序 patch 化方法（一条时序值 64 个 word），channel-independent 设计，在 LTSF benchmark 上大幅超越前代 Transformer 方法。为后续几乎所有 TSFMs 奠定了 patch-based 范式。

---

#### 4.3.3 iTransformer：倒置 Transformer

- **来源**：Liu et al., 清华大学 THUML，ICLR 2024 Spotlight
- **arXiv**：[2310.06625](https://arxiv.org/abs/2310.06625)

**核心贡献**：
颠覆传统 Transformer 在时序上的应用方式：将每个变量的完整时间序列嵌入为一个 token（variate token），用 attention 建模变量间关系，FFN 用于学习每个变量的非线性表示。在多变量预测 benchmark 上持续超越其他方法。

---

## 5. 技术方案横向对比

### 5.1 主要时序基础模型对比表

| 模型 | 机构 | 发表 | 架构 | 预训练范式 | 规模 | 变量处理 | 输出类型 | 开源 |
|------|------|------|------|-----------|------|---------|---------|------|
| **TimesFM** | Google | ICML 2024 | Decoder-only | 自回归 NTP | 200M | CI | 点预测(v1)/分位数(v2) | 部分 |
| **Moirai** | Salesforce | ICML 2024 Oral | Encoder (v1)/Decoder (v2) | 掩码重建/自回归 | Small~Large | Any-variate | 混合分布 | 是 |
| **Chronos** | Amazon | TMLR 2024 | T5 Enc-Dec | 量化+NLP LM | 20M~710M | CI（v1） | 分布 | 是 |
| **Lag-Llama** | ServiceNow/Mila | 2024 | Decoder-only | 自回归 NTP | 小 | CI | 概率分布 | 是 |
| **Timer** | Tsinghua THUML | ICML 2024 | Decoder-only | 自回归 NTP | 84M base | CI | 点预测 | 是 |
| **Timer-XL** | Tsinghua THUML | ICLR 2025 | Causal Transformer | 自回归 NTP | ~ | 多变量 | 点预测 | 是 |
| **MOMENT** | CMU AutonLab | ICML 2024 | T5 Encoder | 掩码重建 | 385M | CI | 多任务 | 是 |
| **Time-MoE** | 多机构 | ICLR 2025 Spotlight | Decoder-only + MoE | 自回归 NTP | 0.1B~2.4B | CI | 点预测 | 是 |
| **TTM** | IBM | NeurIPS 2024 | TSMixer | 多分辨率预训练 | 1M+ | 多变量 | 点预测 | 是 |
| **UniTS** | Harvard | NeurIPS 2024 | Transformer | 多任务预训练 | ~ | 多变量 | 多任务 | 是 |
| **TimeGPT** | Nixtla | arXiv 2023 | 专有 | ~ | ~ | 多变量 | 点+区间 | 否（API）|
| **Moirai-MoE** | Salesforce | arXiv 2024 | Decoder+MoE | 自回归 | ~ | Any-variate | 混合分布 | 是 |

### 5.2 LLM 适配方法对比

| 方法 | 发表 | LLM 骨干 | LLM 是否冻结 | 核心机制 | 任务 |
|------|------|---------|------------|---------|------|
| **GPT4TS** | NeurIPS 2023 Spotlight | GPT-2（前 6 层）| 部分冻结（att+FFN 冻结）| Patch + 线性投影 | 6 类任务 |
| **Time-LLM** | ICLR 2024 | LLaMA/GPT-2 | 完全冻结 | Patch Reprogramming + PaP | 预测 |
| **TEMPO** | ICLR 2024 | GPT-2 | 部分微调 | STL 分解 + Soft Prompt | 预测 |

### 5.3 预训练范式优劣对比

| 范式 | 代表 | 优势 | 劣势 |
|------|------|------|------|
| 自回归 NTP | TimesFM, Timer, Chronos | 天然适配预测；灵活 context 长度 | 需要大规模真实数据；单任务偏向 |
| 掩码重建 | MOMENT | 多任务友好；双向上下文 | 预测时需额外适配 |
| 量化 LM | Chronos | 复用 NLP 基础设施；概率预测 | 量化损失精度；单变量限制（v1）|
| 合成数据 PFN | ForecastPFN | 不依赖真实数据；推理极快 | 合成-真实 gap；场景有限 |
| LLM 重编程 | Time-LLM, GPT4TS | 利用语言知识；少样本友好 | 推理开销大；LLM 实质贡献存疑 |
| MoE | Time-MoE, Moirai-MoE | 大容量+高效；专家自适应 | 训练复杂；路由稳定性 |

---

## 6. Benchmark 与评测体系

### 6.1 经典数据集

| 数据集/集合 | 类型 | 规模 | 特点 |
|-----------|------|------|------|
| **ETT（ETTh1/h2/m1/m2）** | 电力变压器温度 | 4 个数据集，7 变量 | LTSF 标准 benchmark，中等规模 |
| **Weather** | 气象 | 21 变量，52696 时间步 | 常用多变量 benchmark |
| **Traffic** | 交通流量 | 862 变量 | 高维多变量 |
| **Electricity（ECL）** | 用电量 | 321 变量 | 工业用电预测 |
| **Exchange Rate** | 汇率 | 8 国，7588 时间步 | 金融时序 |
| **M4** | 混合 | 10 万条序列，6 种频率 | M 系列竞赛数据 |
| **M5** | 零售销量 | 约 3 万条 | Kaggle 竞赛 |

### 6.2 大规模开放数据集库

| 数据集 | 机构 | 规模 | 说明 |
|--------|------|------|------|
| **Monash Forecasting Archive** | Monash University | 58+ 变种数据集 | 第一个正式时序预测 benchmark 库（NeurIPS 2021 Datasets Track）|
| **LOTSA** | Salesforce | 270 亿观测值，9 域 | Moirai 预训练数据，已开源 |
| **Time-300B** | Time-MoE 团队 | 3000 亿时间点，9+ 域 | 目前最大开放时序预训练集 |
| **UTSD** | Tsinghua THUML | 最大 12G 时间点，7 域 | Timer 预训练数据 |

### 6.3 新一代 Foundation Model 专属 Benchmark

#### TFB（PVLDB 2024，Best Paper Nomination）
- **全称**：Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods
- **arXiv**：[2403.20150](https://arxiv.org/abs/2403.20150)
- **覆盖**：10 个域，21 种单变量方法（8068 条序列），14 种多变量方法（25 数据集）
- **创新**：Fixed + Rolling 两种评估策略；8 个误差指标；减少"刻板偏见"（统计方法被低估）
- **定位**：全面公平评测传统+深度+基础模型方法

#### GIFT-Eval（Salesforce 2024）
- **全称**：A Benchmark For General Time Series Forecasting Model Evaluation
- **arXiv**：[2410.10393](https://arxiv.org/abs/2410.10393)
- **覆盖**：23 个数据集，14.4 万条序列，1.77 亿数据点；7 个域；10 种频率
- **预训练数据**：提供非泄露的约 2300 亿数据点预训练集
- **基线**：17 个基线（统计/深度/基础模型）
- **定位**：专为 zero-shot/few-shot Foundation Model 评测设计

#### TSFM-Bench（arXiv 2024）
- **arXiv**：[2410.11802](https://arxiv.org/abs/2410.11802)
- 支持 zero-shot、few-shot、full-shot 三种评测协议

#### FoundTS（ICLR 2025 Workshop）
- 同样支持 zero/few/full-shot 评测，涵盖更广泛的基础模型

### 6.4 评测指标现状

| 指标 | 用途 | 说明 |
|------|------|------|
| MAE / MSE / RMSE | 点预测误差 | 最常用，但量纲敏感 |
| MAPE / SMAPE | 相对误差 | 零值敏感 |
| MASE | 相对精度 | 以朴素预测为基准，跨数据集可比 |
| CRPS | 概率预测 | 评估预测分布质量，GIFT-Eval 主指标 |
| WQL | 分位数损失 | Moirai/Chronos 等概率模型用 |

**评测标准不统一问题**：各工作使用不同数据集划分、归一化方式和评测指标，导致论文间结果难以横向比较。TFB 和 GIFT-Eval 正试图建立标准，但尚未成为领域共识。

---

## 7. 最新进展（2024-2025）

### 7.1 模型迭代

**TimesFM 系列（Google）**：
- TimesFM 1.0（ICML 2024）：200M 参数，点预测，零样本接近有监督
- TimesFM 2.0：扩展上下文长度，支持分位数预测
- TimesFM 2.5：进一步提升可扩展性，支持连续分位数输出，参数效率更高

**Moirai 系列（Salesforce）**：
- Moirai 1.0（ICML 2024）：LOTSA 27B 数据，encoder-only，混合分布
- Moirai-MoE（2024 年底）：稀疏 MoE 版本，token 级专家分配，+17% 性能提升，65 倍参数效率
- Moirai 2.0（arXiv 2511）：改为 decoder-only，36M 条序列更大规模预训练，量化分位数预测，多 token 预测（MTP）

**Chronos 系列（Amazon）**：
- Chronos v1（T5，2024 年初）：单变量，量化离散化
- Chronos-Bolt（2024 年中）：混合 encoder-decoder，推理速度 250 倍提升，错误率降 5%
- Chronos-2（2025）：从单变量扩展到通用多变量预测，支持外部协变量，在 GIFT-Eval 上超越 TimesFM 2.5

### 7.2 规模化突破

**Time-MoE 2.4B（ICLR 2025 Spotlight）**：
- 时序基础模型参数首次突破 10 亿量级
- 验证了时序域的 Scaling Law（训练数据量和模型参数量均表现出明确的 power-law 缩放关系）

**Scaling Laws 研究（NVIDIA 2025）**：
- 系统研究 TSFMs 的神经网络 Scaling Laws
- 发现 OOD（out-of-distribution）和 ID（in-distribution）下的 log-likelihood 损失均遵循相似的 scaling 行为
- 模型架构对 scaling 系数有显著影响

### 7.3 评测标准化

- GIFT-Eval（Salesforce）提供标准化零样本预测 leaderboard，自发布以来已收到超过 25 个基础模型提交
- TFB 获 PVLDB 2024 最佳论文提名，开源完整评测流程
- ICLR/NeurIPS 2024 Workshop："Time Series in the Age of Large Models" 专题探讨标准化问题

### 7.4 批判性研究与反思

**"Are Language Models Actually Useful for Time Series Forecasting?"（NeurIPS 2024）**：
- 对 3 种流行 LLM 时序方法做消融实验
- 发现：移除 LLM 组件或用简单 attention 层替换，性能不降反升
- 结论：预训练语言模型权重对时序预测无实质贡献；计算成本与收益不匹配
- 意义：引发社区对"LLM for Time Series"路线的深度反思

### 7.5 NeurIPS 2025 Workshop 信号

"Recent Advances in Time Series Foundation Models: Have We Reached the 'BERT Moment'?" 的 NeurIPS 2025 Workshop 表明社区正集体讨论 TSFMs 是否已达到类似 NLP 中 BERT 的通用表征里程碑。

---

## 8. 核心技术挑战与未来方向

### 8.1 核心开放挑战

#### 8.1.1 时序异质性（Heterogeneity）

时序数据的异质性远超文本数据：
- **频率异质性**：秒级（金融行情）、分钟级（用电）、小时级（气象）、天/月级（宏观经济）
- **量纲异质性**：不同传感器物理量纲不可比，直接拼接导致分布冲突
- **模式异质性**：不同域的季节性周期、趋势强度、噪声水平差异巨大

**现有缓解方案**：实例归一化（Instance Norm）、可逆归一化（RevIN）、多尺度 patch 投影（Moirai）、频率嵌入

#### 8.1.2 跨域泛化（Cross-domain Generalization）

与文本的域迁移不同，时序的跨域迁移面临更严峻挑战：
- 气候数据的物理约束与金融数据的随机游走性质几乎没有共性
- "通用"模型在特定域往往不如专域模型
- 联邦学习（Federated Learning）被提出作为一种缓解数据孤岛和域偏移的方案（AAAI 2025）

#### 8.1.3 长上下文建模

时序预测中更长的历史上下文理论上应带来更好性能，但：
- Transformer 的 O(L²) attention 在超长序列（>10K 时间步）上计算不可行
- Mamba/SSM 提供线性复杂度替代
- Timer-XL 等工作专注于长上下文建模

#### 8.1.4 评测标准不统一

- 不同论文使用不同数据集划分（train/val/test 比例）
- 归一化策略不统一（global norm vs. instance norm）
- 评测指标不统一（MAE vs. MSE vs. MASE vs. CRPS）
- 有的工作在训练集中"泄漏"了测试域数据

**进展**：GIFT-Eval、TFB 正在推动标准化，但尚未被所有工作采纳。

#### 8.1.5 概率预测 vs. 点预测

许多基础模型（TimesFM v1、Timer）仅输出点预测，而实际场景需要不确定性估计：
- Chronos：通过量化分箱天然输出分布
- Moirai：混合分布输出
- 趋势：新模型（Moirai 2.0、TimesFM 2.5）均在向概率预测迁移

#### 8.1.6 多变量建模的"诅咒"

- Channel-independent 方法忽略变量间关联，性能受限于高相关性场景
- Channel-dependent 方法变量数固定，无法泛化到新域
- Any-variate attention（Moirai）和 TimeAttention（Timer-XL）是当前最有力的解

### 8.2 未来研究方向

#### 8.2.1 Scaling Law 探索

时序领域的 Scaling Law 研究刚刚起步（Time-MoE、NVIDIA 2025）：
- 时序 token 数量 vs. 性能的 power-law 关系
- 模型参数量 vs. 性能 scaling
- 上下文长度 vs. 性能 scaling（尚未系统研究）

#### 8.2.2 多模态时序理解

将时序与文本、图像、元数据结合：
- 使用自然语言描述（任务信息、领域知识）辅助时序预测（Time-LLM、TEMPO 的方向）
- 时序 + 图（知识图谱）+ 关系图（传感器拓扑）的联合建模

#### 8.2.3 面向特定下游任务的对齐

类比 RLHF 在 LLM 中的作用，时序基础模型可能需要：
- 任务特定的偏好对齐（如降低尾部风险而非均方误差）
- 人类反馈引导预测方向

#### 8.2.4 可解释性与可靠性

基础模型的"黑盒"特性在高风险领域（医疗、电力调度）是重大障碍：
- 时序基础模型的 attention 可视化解释
- 不确定性校准（Calibration）研究

#### 8.2.5 异常检测与因果推断的统一

超越纯预测任务，构建真正通用的时序基础模型：
- 统一预测、异常检测、根因分析、因果发现
- UniTS 是初步探索，但与专域方法仍有差距

#### 8.2.6 高效部署与边缘推理

随着 TSFMs 规模增大，轻量化成为重要方向：
- 知识蒸馏：将大型 TSFM 蒸馏到边缘可用的小模型
- 量化推理（INT8/INT4）
- TTM 的方向（CPU 可用的 1M 参数模型）已展示出这条路线的潜力

#### 8.2.7 数据质量与数据飞轮

时序基础模型的质量高度依赖预训练数据：
- 高质量多域时序数据的采集、清洗和标注
- 合成数据生成（Gaussian Process、扩散模型）作为数据增强
- 数据多样性度量框架的建立

---

## 9. 参考文献索引

### 核心基础模型论文

1. **TimesFM**：Das, A. et al. "A decoder-only foundation model for time-series forecasting." ICML 2024. [arXiv:2310.10688](https://arxiv.org/abs/2310.10688) | [Google Research Blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)

2. **Moirai**：Woo, G. et al. "Unified Training of Universal Time Series Forecasting Transformers." ICML 2024 Oral. [arXiv:2402.02592](https://arxiv.org/abs/2402.02592) | [Salesforce Blog](https://www.salesforce.com/blog/moirai/)

3. **Chronos**：Ansari, A. F. et al. "Chronos: Learning the Language of Time Series." TMLR 2024. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815) | [Amazon Science](https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting)

4. **Lag-Llama**：Rasul, K. et al. "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting." 2024. [arXiv:2310.08278](https://arxiv.org/abs/2310.08278)

5. **Timer**：Liu, Y. et al. "Timer: Generative Pre-trained Transformers Are Large Time Series Models." ICML 2024. [arXiv:2402.02368](https://arxiv.org/abs/2402.02368) | [GitHub](https://github.com/thuml/Large-Time-Series-Model)

6. **Timer-XL**：Liu, Y. et al. "Timer-XL: Long-Context Transformers for Unified Time Series Forecasting." ICLR 2025. [arXiv:2410.04803](https://arxiv.org/abs/2410.04803) | [GitHub](https://github.com/thuml/Timer-XL)

7. **MOMENT**：Goswami, M. et al. "MOMENT: A Family of Open Time-series Foundation Models." ICML 2024. [arXiv:2402.03885](https://arxiv.org/abs/2402.03885) | [GitHub](https://github.com/moment-timeseries-foundation-model/moment)

8. **Time-MoE**：Shi, X. et al. "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts." ICLR 2025 Spotlight. [arXiv:2409.16040](https://arxiv.org/abs/2409.16040) | [GitHub](https://github.com/Time-MoE/Time-MoE)

9. **TTM (Tiny Time Mixers)**：Ekambaram, V. et al. "Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series." NeurIPS 2024. [arXiv:2401.03955](https://arxiv.org/abs/2401.03955)

10. **UniTS**：Gao, S. et al. "UniTS: A Unified Multi-Task Time Series Model." NeurIPS 2024. [arXiv:2403.00131](https://arxiv.org/abs/2403.00131) | [GitHub](https://github.com/mims-harvard/UniTS)

11. **TimeGPT-1**：Garza, A. et al. "TimeGPT-1." arXiv 2023. [arXiv:2310.03589](https://arxiv.org/abs/2310.03589)

12. **Moirai-MoE**：Liu, X. et al. "Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts." arXiv 2024. [arXiv:2410.10469](https://arxiv.org/abs/2410.10469) | [Salesforce Blog](https://www.salesforce.com/blog/time-series-morai-moe/)

### LLM 适配方法

13. **GPT4TS / One Fits All**：Zhou, T. et al. "One Fits All: Power General Time Series Analysis by Pretrained LM." NeurIPS 2023 Spotlight. [arXiv:2302.11939](https://arxiv.org/abs/2302.11939) | [GitHub](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)

14. **Time-LLM**：Jin, M. et al. "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." ICLR 2024. [arXiv:2310.01728](https://arxiv.org/abs/2310.01728) | [GitHub](https://github.com/KimMeen/Time-LLM)

15. **TEMPO**：Cao, D. et al. "TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting." ICLR 2024. [arXiv:2310.04948](https://arxiv.org/abs/2310.04948) | [GitHub](https://github.com/DC-research/TEMPO)

### 基础架构论文

16. **PatchTST**：Nie, Y. et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)

17. **iTransformer**：Liu, Y. et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR 2024 Spotlight. [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) | [GitHub](https://github.com/thuml/iTransformer)

18. **ForecastPFN**：Dooley, S. et al. "ForecastPFN: Synthetically-Trained Zero-Shot Forecasting." NeurIPS 2023. [arXiv:2311.01933](https://arxiv.org/abs/2311.01933)

19. **TS2Vec**：Yue, Z. et al. "TS2Vec: Towards Universal Representation of Time Series." AAAI 2022. [arXiv:2106.10466](https://arxiv.org/abs/2106.10466)

### Benchmark 与评测

20. **Monash Archive**：Godahewa, R. et al. "Monash Time Series Forecasting Archive." NeurIPS 2021 Datasets & Benchmarks. [arXiv:2105.06643](https://arxiv.org/abs/2105.06643) | [forecastingdata.org](https://forecastingdata.org/)

21. **TFB**：Hu, Y. et al. "TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods." PVLDB 2024 (Best Paper Nomination). [arXiv:2403.20150](https://arxiv.org/abs/2403.20150) | [GitHub](https://github.com/decisionintelligence/TFB)

22. **GIFT-Eval**：Aksu, T. et al. "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation." arXiv 2024. [arXiv:2410.10393](https://arxiv.org/abs/2410.10393) | [GitHub](https://github.com/SalesforceAIResearch/gift-eval)

23. **TSFM-Bench**：[arXiv:2410.11802](https://arxiv.org/abs/2410.11802)

### 综述论文

24. **综述（Ye et al.）**：Ye, J. et al. "Empowering Time Series Analysis with Foundation Models: A Comprehensive Survey." arXiv 2024. [arXiv:2405.02358](https://arxiv.org/abs/2405.02358)

25. **综述（LLM4TS）**：Jiang, Y. et al. "Large Language Models for Time Series: A Survey." IJCAI 2024. [会议主页](https://www.ijcai.org/proceedings/2024/921)

26. **综述（Foundation Models for TS）**：[arXiv:2504.04011](https://arxiv.org/abs/2504.04011)（2025）

### 批判性研究

27. **LLM 无效论**：Tan, M. et al. "Are Language Models Actually Useful for Time Series Forecasting?" NeurIPS 2024. [论文主页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html)

### 最新迭代工作（2024-2025）

28. **Moirai 2.0**：[arXiv:2511.11698](https://arxiv.org/abs/2511.11698)

29. **Chronos-2**：[Amazon Science Blog](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting) | [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)

30. **Scaling Laws for TSFMs**：[arXiv:2410.12360](https://arxiv.org/abs/2410.12360)（NVIDIA）

31. **S-Mamba**：时间序列的双向 Mamba 模型

32. **MambaTS**：[arXiv:2405.16440](https://arxiv.org/abs/2405.16440)

33. **LLM4TS（ACM TIST）**：[ACM DL](https://dl.acm.org/doi/10.1145/3719207)

34. **FoundTS**：[OpenReview](https://openreview.net/forum?id=B4OaA0aJ4Z)

35. **How Foundational are Foundation Models for TS?**：[arXiv:2510.00742](https://arxiv.org/abs/2510.00742)

---

*本报告由 Academic Research Surveyor 生成，基于截至 2025 年底的公开学术文献和技术博客整理。建议结合原始论文深入阅读各模型的技术细节，尤其关注 ICML/NeurIPS/ICLR 各年度会议记录及 arXiv 上的最新预印本。*
