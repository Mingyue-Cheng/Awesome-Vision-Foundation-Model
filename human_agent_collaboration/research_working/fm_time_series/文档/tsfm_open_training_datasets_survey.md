# Time Series Foundation Model 开源训练数据集调研

**生成日期**：2026-04-07  
**主题**：现有 Time Series Foundation Model（TSFM）可公开获取的训练数据集 / 训练语料集合  
**统计口径**：优先采用官方数据集卡、官方博客、官方模型卡、官方代码仓库中的公开说明；截至 **2026-04-07**。  

---

## 1. 结论速览

如果只看“**当前能公开拿到、且明确服务于 TSFM 训练/预训练**”的数据，最值得关注的是下面 8 类：

1. **LOTSA**：Moirai 1.x 的代表性开源预训练语料库，也是目前最常被引用的通用 forecasting 语料之一。
2. **GIFT-Eval Pretrain**：当前最重要的“**无泄漏**”预训练集合之一，已经被 Chronos-2、Moirai 2.0 等后续模型使用。
3. **UTSD**：Timer / Sundial 系列的重要统一训练数据集，强调可扩展性和分层容量设计。
4. **Chronos Datasets**：Amazon/AutoGluon 公开的数据集合，包含真实数据集合以及 `TSMixup`、`KernelSynth` 等训练语料。
5. **Time-series PILE**：MOMENT 的核心开源训练语料，覆盖 forecasting / classification / anomaly detection 等多任务来源。
6. **Time-300B**：Time-MoE 对应的超大规模开源训练语料。
7. **Monash Forecasting Archive**：不是为 TSFM 专门设计，但它仍然是很多 TSFM 数据构造与评测的重要上游来源。
8. **Time-MMD**：面向多模态时间序列 foundation model 的公开数据集，适合“数值序列 + 文本上下文”的研究。

同时要明确一个现实情况：

- **开源模型 != 开源训练数据**。  
  例如 **TimesFM** 模型开源，但 Google 只公开说明其训练语料主要来自 **Google Trends、Wikipedia Pageviews 与 synthetic data**，并**没有**把完整 100B 训练语料作为单一公开数据集发布。  
- **部分模型是“半开放训练语料”**。  
  例如 **Moirai 2.0** 明确写出其训练语料里同时包含开源部分（GIFT-Eval Pretrain、Chronos mixup、KernelSynth）和 **internal Salesforce operational data**，因此无法按官方配方完全复现。  

---

## 2. 纳入标准

本调研纳入的数据对象满足以下至少一项：

- 有官方公开入口，且可直接下载 / 通过 Hugging Face 或 GitHub 获取；
- 官方明确声明其用于 TSFM 的训练、预训练或训练语料构造；
- 虽非专为 TSFM 设计，但已被多个 TSFM 当作上游训练数据源或训练数据仓库的重要组成。

不纳入：

- 只有论文提到、但没有公开入口的数据；
- 纯 benchmark、且与训练语料没有直接关系的数据；
- 明显是闭源内部数据、无法公开获取的数据。

---

## 3. 核心开源训练数据集总表

| 数据集 / 语料集合 | 主要用途 | 官方公开规模 | 许可证 | 下载方式 | 适合任务类型 | 公开入口 | 代表性关联模型 | 复现价值判断 |
|---|---|---:|---|---|---|---|---|---|
| **LOTSA** | 通用 forecasting 预训练 | 27B observations，9 个 domain | Apache-2.0 | Hugging Face `datasets` / Xet 大文件下载 | zero-shot forecasting、probabilistic forecasting、multivariate / any-variate forecasting | [HF 数据集卡](https://huggingface.co/datasets/Salesforce/lotsa_data)、[Moirai 官方博客](https://www.salesforce.com/blog/moirai) | Moirai 1.x，Timer，Sundial | 很高 |
| **GIFT-Eval Pretrain** | 无泄漏 forecasting 预训练 | 71 个单变量 + 17 个多变量数据集；4.5M time series；230B data points | Apache-2.0 | Hugging Face `datasets` / Xet 大文件下载 | zero-shot forecasting、非泄漏预训练、单变量与多变量 forecasting | [HF 数据集卡](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)、[官方博客](https://www.salesforce.com/blog/gift-eval-time-series-benchmark/) | Chronos-2，Moirai 2.0 | 很高 |
| **UTSD** | 通用 forecasting 预训练 / scaling 实验 | 7 个 domain，最高到 1B time points，分层 1G/2G/4G/12G | Apache-2.0 | Hugging Face 下载；官方也提供 `download_dataset.py` | forecasting 预训练、scaling law、长上下文 / 统一序列训练 | [HF 数据集卡](https://huggingface.co/datasets/thuml/UTSD)、[Timer 官方仓库](https://github.com/thuml/timer) | Timer，Timer-XL，Sundial | 很高 |
| **Chronos Datasets** | Chronos 系列训练与评测语料 | 公开数据集合 + `TSMixup 10M` + `KernelSynth 1M` | 按子数据集而异；主仓说明为 `other` | Hugging Face `datasets`；部分补充集从 `chronos_datasets_extra` 加载 | univariate forecasting、multivariate forecasting、synthetic pretraining、data augmentation | [HF 数据集卡](https://huggingface.co/datasets/autogluon/chronos_datasets)、[extra repo](https://huggingface.co/datasets/autogluon/chronos_datasets_extra) | Chronos，Chronos-2，Sundial | 很高 |
| **Time-series PILE** | 多任务 TSFM 预训练 | 13 个 domain；13M unique series；1.23B timestamps；20.085 GB | MIT | Hugging Face 数据集直接下载 | forecasting、classification、anomaly detection、imputation、自监督表征学习 | [HF 数据集卡](https://huggingface.co/datasets/AutonLab/Timeseries-PILE) | MOMENT | 很高 |
| **Time-300B** | 超大规模 forecasting 预训练 | 超过 300B time points；当前 HF 主仓已到 TB 级 | Apache-2.0 | Hugging Face / Xet 大文件下载 | 大规模 decoder-only 预训练、MoE 预训练、长上下文 forecasting | [HF 数据集卡](https://huggingface.co/datasets/Maple728/Time-300B)、[Time-MoE 官方仓库](https://github.com/Time-MoE/Time-MoE) | Time-MoE | 很高 |
| **Monash Forecasting Archive** | 上游通用 forecasting 数据仓库 | 当前站点列出 30 个数据集、58 个变体；论文版本为 25 个公开数据集 | 研究用途为主；具体按原始子数据集来源而异 | 官网直接下载 `.tsf`；配套 GitHub wrapper / 脚本加载 | forecasting benchmark、自建预训练语料、轻量 corpus 组装 | [Monash Repository](https://forecastingdata.org/)、[Monash 论文页面](https://research.monash.edu/en/publications/monash-time-series-forecasting-archive/) | Chronos，TimesFM，Moirai，TTM 等广泛引用 | 高 |
| **Time-MMD** | 多模态 TSFM 训练 / 评测 | 9 个 primary domains，数值序列 + 文本序列 | ODC-By v1.0 | GitHub 直接下载 CSV；MM-TSFlib 提供读取与预处理示例 | multimodal forecasting、event-aware forecasting、imputation、anomaly detection | [官方 GitHub](https://github.com/AdityaLab/Time-MMD)、[NeurIPS 2024 页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8e7768122f3eeec6d77cd2b424b72413-Abstract-Datasets_and_Benchmarks_Track.html) | 多模态 TSFM / MM-TSFlib | 中高 |

---

## 4. 逐个数据集解读

### 4.1 LOTSA

**定位**：目前最有代表性的开源“通用 forecasting 预训练语料库”之一。  

**官方信息**

- Salesforce 在 Moirai 官方博客中将 LOTSA 描述为“**the largest collection of open time series datasets**”。
- 官方博客给出的规模是：**27B observations，覆盖 9 个 domain**。
- Hugging Face 数据集仓库为 `Salesforce/lotsa_data`。

**为什么重要**

- LOTSA 不是单一原始数据集，而是把多个公开来源统一为适合 TSFM 训练的格式。
- 它解决的是 TSFM 最核心的训练痛点：**跨 domain、跨频率、跨变量维度** 的统一预训练语料组织。

**适合场景**

- 想做 Moirai 路线的通用 forecasting 预训练；
- 想研究 any-variate / multi-frequency 统一建模；
- 想要一个公开程度较高、社区认可度高的 forecasting 语料集合。

**局限**

- 它是“数据集集合”，不是一个单体数据源；
- 实际落地时仍要关注各子数据源的许可证与清洗规则。

---

### 4.2 GIFT-Eval Pretrain

**定位**：当前最关键的“**无测试泄漏**”预训练数据集集合之一。  

**官方信息**

- `Salesforce/GiftEvalPretrain` 官方卡写明：
  - **71 个单变量 + 17 个多变量数据集**
  - **4.5 million time series**
  - **230 billion data points**
  - 覆盖 **7 个 domain、13 种 frequency**
  - 明确强调 **no leakage issue with the train/test split**
- Salesforce GIFT-Eval 官方博客同时写到：预训练部分包含 **230B data points spread over 88 datasets**。

**为什么重要**

- TSFM 领域目前最大的争议之一就是 **data contamination / benchmark leakage**。
- GIFT-Eval Pretrain 的价值不只是“大”，而是它从设计上尽量服务“**可公平评测**”。

**适合场景**

- 想做 zero-shot forecasting TSFM；
- 想尽量避免“模型训练集和 benchmark 测试集重叠”的争议；
- 想做面向 GIFT-Eval leaderboard 的可复现训练。

**局限**

- 仍以 forecasting 为中心；
- 如果目标是多任务时序理解（分类、异常检测、插补），它不如 Time-series PILE 直接。

---

### 4.3 UTSD

**定位**：Timer 系列最核心的公开训练数据集。  

**官方信息**

- `thuml/UTSD` 数据集卡写明：
  - 覆盖 **7 个 domains**
  - 规模 **up to 1 billion time points**
  - 提供分层容量：**UTSD-1G / 2G / 4G / 12G**
- 数据集卡说明其来源于 **publicly accessible online data repositories** 与真实机器运行经验数据的混合整理。

**为什么重要**

- UTSD 的独特价值不是单纯追求“大”，而是强调 **分层容量设计**，适合做 scaling 实验。
- 对 Timer 这类统一 1D 序列建模路线很友好。

**适合场景**

- 想复现 Timer / Timer-XL / Sundial 路线；
- 想研究“数据规模提升是否带来稳定增益”；
- 想做较标准、结构统一的 forecasting 预训练。

**局限**

- 更偏 THUML / Timer 系列训练范式；
- 多任务维度不如 Time-series PILE 丰富。

---

### 4.4 Chronos Datasets

**定位**：目前最工程化、最适合直接接入训练代码的公开 TSFM 训练语料之一。  

**官方信息**

- `autogluon/chronos_datasets` 的官方卡直接写明：  
  **“Time series datasets used for training and evaluation of the Chronos forecasting models.”**
- 其公开说明里包含：
  - 真实世界数据集合；
  - `training_corpus_tsmixup_10m`：**10M TSMixup augmentations of real-world data**
  - `training_corpus_kernel_synth_1m`：**1M synthetic time series generated with KernelSynth**
- 对于不能直接托管在主仓库的数据，还提供 `autogluon/chronos_datasets_extra`。

**为什么重要**

- 它把“原始公开数据 + 训练增强语料 + synthetic 生成语料”放在同一套可加载接口里。
- 从可复现实验角度看，Chronos Datasets 是目前最接近“**官方训练输入**”的开源方案之一。

**适合场景**

- 想复现 Chronos / Chronos-2 路线；
- 想研究离散化建模、synthetic pretraining、数据增强；
- 想直接用 Hugging Face `datasets` 风格管线喂给训练脚本。

**局限**

- 官方卡已明确说明不同子数据集许可证不同；
- 部分特殊数据通过 `chronos_datasets_extra` 侧仓构建，不是一个完全统一的单仓下载体验。

---

### 4.5 Time-series PILE

**定位**：偏“多任务时序基础模型”的开源训练语料。  

**官方信息**

- `AutonLab/Timeseries-PILE` 官方卡写明：
  - **13 unique domains**
  - **13M unique time series**
  - **1.23 billion timestamps**
  - 数据来自 **5+ public repositories**
- 其数据来源明确包括：
  - Informer long-horizon forecasting datasets
  - Monash archive
  - UCR/UEA classification archive
  - TSB-UAD anomaly benchmark

**为什么重要**

- 与 LOTSA、GIFT-Eval Pretrain 主要面向 forecasting 不同，Time-series PILE 从一开始就更偏 **foundation representation + multi-task**。
- 它是 MOMENT 这类“一个 backbone 服务多类时序任务”的关键训练基础。

**适合场景**

- 想训练可迁移到 forecasting / classification / anomaly detection / imputation 的统一模型；
- 想做 masked modeling、自监督表征学习；
- 想做多任务预训练而不是单纯零样本 forecasting。

**局限**

- 如果只追求 forecasting leaderboard，GIFT-Eval Pretrain 更有针对性；
- 数据异质性更高，清洗与任务统一成本也更高。

---

### 4.6 Time-300B

**定位**：目前公开可拿到的超大规模 forecasting 训练语料之一。  

**官方信息**

- Time-MoE 官方仓库将其描述为：  
  **“the largest open-access time series data collection comprising over 300 billion time points across >9 domains.”**
- 对应公开仓库为 `Maple728/Time-300B`，Hugging Face 当前主仓显示体量已到 **TB 级**。

**为什么重要**

- Time-300B 代表的是“按 LLM 风格做大规模 time-series pretraining”的路线。
- 对做数据规模律、MoE 训练、长上下文 forecasting 都有直接价值。

**适合场景**

- 想做大规模 decoder-only 或 MoE 类 TSFM；
- 想研究参数规模与数据规模共同扩张下的收益；
- 具备较强算力与存储条件。

**局限**

- 太大，数据工程门槛高；
- 实操上比 LOTSA / GIFT-Eval Pretrain / UTSD 更重。

---

### 4.7 Monash Forecasting Archive

**定位**：不是专为 TSFM 发明，但几乎是当前开源 forecasting 语料体系的“基础设施”。  

**官方信息**

- Monash 官方站点当前写明：**30 个 datasets、58 个 variations**。
- 论文版本则写的是 **25 publicly available time series datasets**。

**为什么重要**

- 大量 TSFM 不是直接“训练在 Monash 上”，而是：
  - 把 Monash 当成上游公开数据源；
  - 把 Monash 当作 zero-shot / few-shot 评测基准；
  - 用它构造更大的训练集合。

**适合场景**

- 想快速获得覆盖多领域、多频率的公开 forecasting 数据；
- 想做中等规模可复现实验；
- 想自己组装一版“轻量 LOTSA / 轻量 Chronos corpus”。

**局限**

- 它更像“数据仓库”，不是为 foundation pretraining 直接清洗好的统一语料；
- 需要自己处理格式统一、频率对齐、切分、缺失值等问题。

---

### 4.8 Time-MMD

**定位**：当前公开可见的多模态时间序列训练数据代表。  

**官方信息**

- 官方 GitHub 与 NeurIPS 2024 页面均写明：
  - Time-MMD 是 **first multi-domain, multimodal time series dataset**
  - 覆盖 **9 primary data domains**
  - 同时包含 **numerical sequences** 与 **textual sequences**

**为什么重要**

- 传统 TSFM 基本只看数值序列；
- 但现实业务里经常同时依赖：
  - 报告文本
  - 搜索趋势
  - 事件描述
  - 新闻与运维上下文
- 如果要做“时序 + 文本”的 foundation model，Time-MMD 是目前最直接的公开起点之一。

**适合场景**

- 多模态 forecasting；
- 事件驱动预测；
- 检索增强 / 文本条件的时间序列建模。

**局限**

- 它目前还不是 forecasting TSFM 主流训练语料；
- 社区生态和复现成熟度还不如 LOTSA / GIFT-Eval / Chronos 系列。

---

## 5. 开源模型与公开训练数据的对应关系

这一节专门回答一个最容易混淆的问题：**“模型开源了，训练数据是不是也开源了？”**

| 模型 | 官方公开训练数据情况 | 结论 |
|---|---|---|
| **Chronos / Chronos-2** | 官方模型卡与数据集卡已明确关联 `autogluon/chronos_datasets`；`chronos-2` 模型卡还直接列出 `Salesforce/GiftEvalPretrain` | 训练数据开放度高，可较高程度复现 |
| **Moirai 1.x** | 官方博客明确写明预训练依赖 **LOTSA** | 开放度高 |
| **Moirai 2.0** | 官方博客 / 模型卡写明：`GIFT-Eval Pretrain + Train`、`Chronos mixup`、`KernelSynth`、`internal Salesforce operational data` | **部分开源，非完全公开复现** |
| **MOMENT** | 官方明确使用 **Time-series PILE** | 开放度高 |
| **Timer / Timer-XL** | 官方公开 **UTSD**；当前 HF 模型卡标签也关联 `Salesforce/lotsa_data` | 开放度高 |
| **Sundial** | HF 模型卡直接列出 `thuml/UTSD`、`Salesforce/lotsa_data`、`autogluon/chronos_datasets` | 开放度高 |
| **Time-MoE** | 官方公开 **Time-300B** | 开放度高，但工程门槛高 |
| **TimesFM** | Google 官方博客只说明训练在 **100B real-world time-points** 上，且多数来自 **Google Trends** 与 **Wikipedia Pageviews**，另加 synthetic data；**未发布完整训练语料集** | **模型开源，但训练语料未完整公开** |
| **Tiny Time Mixers (TTM)** | IBM 官方说明模型“trained exclusively on public TS datasets”；模型卡列出一批来自 **Monash repository** 的公开数据，但**没有发布成单一统一预训练语料包** | **数据来源公开，但官方训练混合包未完整独立发布** |

### 5.1 按模型分类的表格版

| 模型 / 模型家族 | 官方明确公开的训练数据 | 数据开放程度 | 下载入口 / 组织方式 | 适合复现的方向 | 备注 |
|---|---|---|---|---|---|
| **Chronos** | `autogluon/chronos_datasets`，含真实数据、`TSMixup`、`KernelSynth` | 高 | Hugging Face 数据集主仓 + `chronos_datasets_extra` | 离散化 forecasting、synthetic pretraining、概率预测 | 是目前最接近“官方训练输入”公开化的 TSFM 之一 |
| **Chronos-2** | `autogluon/chronos_datasets` + `Salesforce/GiftEvalPretrain` | 高 | Hugging Face 数据集 + 模型卡直接关联 | 非泄漏 forecasting、跨域 zero-shot forecasting | 官方模型卡明确列出了这两个训练数据来源 |
| **Moirai 1.x** | `Salesforce/lotsa_data` | 高 | Hugging Face LOTSA 数据集 | 通用 universal forecasting、any-variate forecasting | 是 LOTSA 最典型的官方使用者 |
| **Moirai 2.0** | `GIFT-Eval Pretrain + Train`、Chronos mixup、KernelSynth，外加 `internal Salesforce operational data` | 中 | 开源部分可下载；内部运营数据不可获得 | 可做“部分复现”，难做“完全复现” | 这是典型的“模型开源但训练配方不完全开源” |
| **MOMENT** | `AutonLab/Timeseries-PILE` | 高 | Hugging Face 数据集 | 多任务 TSFM、自监督表征学习、masked modeling | 比 forecasting-only 路线更偏通用时序表征 |
| **Timer** | `thuml/UTSD`；`timer-base-84m` 模型卡还标注了 `Salesforce/lotsa_data` | 高 | Hugging Face UTSD / LOTSA | 统一序列生成建模、zero-shot forecasting、规模扩展实验 | 官方说明其是大规模 generative TS model 路线 |
| **Timer-XL / OpenLTM 路线** | 公开资料延续 UTSD 系列语料；部分公开模型卡沿用 LOTSA / UTSD 标签 | 中高 | 以 THUML / HF 公开入口为主 | 更长上下文 forecasting、生成式 TSFM | 这一行是基于公开模型卡与 Timer 系列延续关系做的归纳 |
| **Sundial** | `thuml/UTSD`、`Salesforce/lotsa_data`、`autogluon/chronos_datasets` | 高 | Hugging Face 模型卡直接列数据集标签 | 混合语料预训练、通用 zero-shot forecasting | 是目前少数直接把多套公开语料一起挂到模型卡上的例子 |
| **Time-MoE** | `Maple728/Time-300B` | 高 | Hugging Face TB 级大数据集 | MoE 预训练、超大规模 forecasting foundation model | 复现门槛主要在算力与存储，而不是数据获取 |
| **TimesFM** | 官方只公开描述：100B real-world time-points，主要来自 Google Trends 与 Wikipedia Pageviews，另含 synthetic data | 低 | 无完整公开训练语料包 | 只能复现模型推理，难复现原始预训练 | 是“模型开源、完整训练语料未公开”的代表案例 |
| **TTM / Tiny Time Mixers** | 官方声明仅用 public TS datasets；模型卡列出多项来自 Monash repository 的公开数据 | 中 | 需根据模型卡列出的上游公开数据自行重建 | 轻量 TSFM、few-shot / zero-shot forecasting | 数据来源公开，但没有单独发布官方统一训练混合包 |
| **多模态 TSFM（基于 Time-MMD）** | `Time-MMD` | 高 | GitHub CSV + MM-TSFlib | 时序 + 文本、多模态 forecasting、事件增强建模 | 当前更像研究起点，而不是单一主流模型家族 |

---

## 6. 如果你要自己训练 TSFM，怎么选数据

### 6.1 目标是“做零样本 forecasting foundation model”

优先级建议：

1. **GIFT-Eval Pretrain**
2. **LOTSA**
3. **Chronos Datasets**
4. **UTSD**

原因：

- GIFT-Eval Pretrain 在“公平评测、避免泄漏”上最好；
- LOTSA 在“多域通用 forecasting 预训练”上最经典；
- Chronos Datasets 最贴近可复现实验；
- UTSD 很适合做规模扩展和统一序列训练。

### 6.2 目标是“做多任务 TS foundation model”

优先级建议：

1. **Time-series PILE**
2. **UTSD**
3. **Monash Archive**

原因：

- Time-series PILE 天然覆盖 forecasting / classification / anomaly detection；
- 更适合 MOMENT 这类表征学习或 masked reconstruction 路线。

### 6.3 目标是“做超大规模预训练”

优先级建议：

1. **Time-300B**
2. **LOTSA + GIFT-Eval Pretrain + Chronos synthetic**

原因：

- Time-300B 是当前公开规模最激进的选项之一；
- 但算力不足时，组合式方案更务实。

### 6.4 目标是“做多模态 TS foundation model”

优先级建议：

1. **Time-MMD**
2. **Time-MMD + 数值 forecasting corpus（如 LOTSA / GIFT-Eval Pretrain）**

原因：

- Time-MMD 适合作为文本-时序对齐的起点；
- 如果想兼顾通用 forecasting 能力，最好和纯数值大语料混合训练。

---

## 7. 实务判断：哪些是真正值得优先下载的

如果你现在就要开始搭一个可复现 TSFM 训练栈，我建议按下面顺序考虑：

### 第一梯队：最值得优先落地

- **GIFT-Eval Pretrain**
- **LOTSA**
- **Chronos Datasets**
- **UTSD**

这 4 个数据源基本覆盖了当前主流 forecasting TSFM 的公开训练路线。

### 第二梯队：按研究方向补充

- **Time-series PILE**：适合多任务 / 表征学习
- **Time-300B**：适合超大规模实验
- **Time-MMD**：适合多模态路线

### 第三梯队：基础仓库 / 上游源

- **Monash Forecasting Archive**

它更像“原料库”，适合自己再加工。

---

## 8. 关键观察

### 观察 1：TSFM 训练数据正在从“拼 benchmark”转向“专门的预训练语料库”

早期很多方法只是把 Monash、ETT、Traffic、Weather 等常见数据拼起来。  
近两年则明显转向：

- LOTSA
- GIFT-Eval Pretrain
- UTSD
- Time-300B
- Chronos Datasets

也就是开始有“**面向 foundation model 训练而设计**”的数据组织方式。

### 观察 2：真正重要的不只是规模，而是“能否说明没有泄漏”

现在社区对“zero-shot 是否真的 zero-shot”越来越敏感。  
因此：

- **GIFT-Eval Pretrain** 的价值非常高；
- 以后训练语料的一个核心竞争点，不只是大，而是 **可审计、可溯源、可证明无泄漏**。

### 观察 3：多任务与多模态数据会成为下一阶段分水岭

目前 forecasting 仍是主流，但如果往更通用的 TSFM 走：

- 多任务：看 **Time-series PILE**
- 多模态：看 **Time-MMD**

这两条线很可能会决定下一代 TSFM 是否只是“更强的 forecaster”，还是“更通用的时序基础模型”。

---

## 9. 参考来源

### 官方数据集 / 官方仓库

- LOTSA: <https://huggingface.co/datasets/Salesforce/lotsa_data>
- GIFT-Eval Pretrain: <https://huggingface.co/datasets/Salesforce/GiftEvalPretrain>
- GIFT-Eval: <https://huggingface.co/datasets/Salesforce/GiftEval>
- UTSD: <https://huggingface.co/datasets/thuml/UTSD>
- Chronos Datasets: <https://huggingface.co/datasets/autogluon/chronos_datasets>
- Chronos Datasets Extra: <https://huggingface.co/datasets/autogluon/chronos_datasets_extra>
- Time-series PILE: <https://huggingface.co/datasets/AutonLab/Timeseries-PILE>
- Time-300B: <https://huggingface.co/datasets/Maple728/Time-300B>
- Monash Forecasting Repository: <https://forecastingdata.org/>
- Time-MMD: <https://github.com/AdityaLab/Time-MMD>

### 官方博客 / 官方模型卡 / 官方论文入口

- Moirai 官方博客: <https://www.salesforce.com/blog/moirai>
- Moirai 2.0 官方博客: <https://www.salesforce.com/blog/moirai-2-0/>
- Moirai 2.0 模型卡: <https://huggingface.co/Salesforce/moirai-2.0-R-small>
- GIFT-Eval 官方博客: <https://www.salesforce.com/blog/gift-eval-time-series-benchmark/>
- TimesFM 官方博客: <https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/>
- Time-MoE 官方仓库: <https://github.com/Time-MoE/Time-MoE>
- TTM / TinyTimeMixer 官方说明: <https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1>
- TTM 模型卡: <https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1>
- Monash 论文页面: <https://research.monash.edu/en/publications/monash-time-series-forecasting-archive/>
- Time-MMD NeurIPS 页面: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/8e7768122f3eeec6d77cd2b424b72413-Abstract-Datasets_and_Benchmarks_Track.html>

---

## 10. 一句话总结

如果你问“**现在哪些开源训练数据最能代表 TSFM 的主流训练路线**”，答案基本就是：

**LOTSA、GIFT-Eval Pretrain、UTSD、Chronos Datasets、Time-series PILE、Time-300B**。

其中：

- 做 forecasting，优先看 **GIFT-Eval Pretrain / LOTSA / Chronos / UTSD**
- 做多任务，优先看 **Time-series PILE**
- 做超大规模，优先看 **Time-300B**
- 做多模态，优先看 **Time-MMD**
