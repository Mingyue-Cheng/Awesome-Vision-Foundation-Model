# 时间序列基础模型（TSFMs）评测基准调研报告

**生成日期**: 2026-04-07
**关键词**: Time Series Foundation Models, Benchmark, Evaluation Protocol, Zero-shot Forecasting, Data Leakage, GIFT-Eval, TFB, Monash Archive
**信息来源数量**: 35+ 篇文献 / 报告 / 技术博客

---

## 1. 背景综述

### 1.1 研究背景与动机

时间序列分析是金融、气象、能源、交通、医疗等领域的核心任务。过去十年，深度学习方法（Transformer、MLP-Mixer 等）在时序预测领域取得长足进步，但这些方法普遍依赖特定领域的有标注训练数据，跨域泛化能力有限。

2023-2024 年以来，时间序列基础模型（Time Series Foundation Models, TSFMs）作为新范式迅速崛起：Chronos（Amazon）、TimesFM（Google）、Moirai（Salesforce）、MOMENT（CMU）、Time-MoE、TTM（IBM）、Sundial（THU）等在海量异构时序数据上预训练，展现出零样本或少样本泛化能力。然而，**如何公平、可靠地评测这些基础模型**成为了制约领域发展的瓶颈问题。

### 1.2 核心问题定义

TSFM 评测面临的根本挑战可以用一个元问题概括：

> **"在不同的评测设置下，我们真的在测量模型的泛化能力，还是在测量它对评测数据集的记忆程度？"**

具体体现在：
- 训练数据污染（data contamination）：基础模型预训练语料与评测集的重叠
- 评测协议不一致：zero-shot / few-shot / full-shot 的混用与误报
- 数据集选择偏差（cherry-picking）：选择有利于特定模型的数据集
- 指标设计缺陷：MSE/MAE 在跨域、跨粒度比较中的局限性

### 1.3 研究重要性

准确可信的评测体系是：
1. 客观判断 TSFM 技术进展的基础
2. 推动模型在真实场景落地的前提
3. 避免资源浪费在"刷榜"工作上的保障

### 1.4 本报告结构

本报告按如下结构组织：
- 第 2 节：评测基准现状概览与时间线
- 第 3 节：核心 benchmark 逐一深度解析
- 第 4 节：评测维度横向对比分析
- 第 5 节：现有评测问题与挑战
- 第 6 节：最新进展（2024-2025）
- 第 7 节：未来展望

---

## 2. 研究现状概览

### 2.1 评测基准的演进脉络

时序评测基准的发展可分为三个阶段：

**阶段一（2018-2021）：竞赛驱动与领域归档**
- M4/M5 竞赛：100K 时序的大规模竞赛评测，奠定 sMAPE/MASE/OWA 等指标基础
- UCR/UEA Archive（2018）：分类任务的标准数据集
- Monash Forecasting Archive（2021）：跨域预测的首批系统性归档

**阶段二（2022-2023）：针对 Transformer 的基准讨论**
- LTSF-Linear 评测设置（AAAI 2023）：ETT、Traffic、Electricity 等 9 个数据集成为事实标准
- 暴露出 Transformer 评测存在的"过度复杂化"和数据集单一化问题

**阶段三（2024-2025）：面向基础模型的新一代基准**
- GIFT-Eval（NeurIPS 2024）：首个专为 TSFM 设计的大规模 benchmark
- TFB（PVLDB 2024 Best Paper Nomination）：公平性和综合性导向的系统基准
- TSFM-Bench（KDD 2025）：涵盖 zero/few/full-shot 的统一评测框架
- fev-bench（2025）：含协变量的现实主义评测

### 2.2 主要评测方向分类

| 评测方向 | 代表 Benchmark | 核心关注点 |
|---------|--------------|----------|
| 通用预测 | GIFT-Eval, TFB, Monash | 跨域零样本泛化 |
| 公平基准 | TFB, BasicTS | 消除偏差、复现性 |
| 多任务 | MOMENT, TSFM-Bench | 预测+分类+异常检测+插补 |
| 概率预测 | fev-bench, Chronos Benchmark | CRPS/WQL 分布校准 |
| 分类/回归 | UCR/UEA, TSER | 时序理解能力 |
| 异常检测 | TSB-AD, TAB | 检测性能标准化 |

---

## 3. 重点文献深度解读

### 3.1 专为 TSFM 设计的新一代评测基准

#### 文献 1：GIFT-Eval

- **来源**: Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, Doyen Sahoo (Salesforce AI Research). NeurIPS 2024 Datasets and Benchmarks Track. [arXiv:2410.10393](https://arxiv.org/abs/2410.10393)
- **核心贡献**: 首个专为通用时序预测基础模型设计的大规模评测基准，同时提供非泄漏预训练数据集。
- **技术方案**:
  - 收录 **28 个数据集**（部分来源汇总后为 23 个独特数据集），覆盖 **144,000 条时序**、**1.77 亿**数据点
  - 横跨 **7 个领域**（能源、交通、金融、气象、医疗等）、**10 种频率**（秒/分/小时/天/周/月/季/年等）
  - 覆盖 **97 种任务配置**（含单变量/多变量、短/中/长期预测）
  - 单独提供约 **2300 亿数据点**的非泄漏预训练数据集，供模型开发使用
  - 评测协议：**零样本（zero-shot）**为主
  - 指标：**MASE**（点预测）+ **WQL**（概率预测）
- **关键结果**: 对 17 个 baseline 进行评测，涵盖统计模型、深度学习模型、基础模型。结果显示各类模型在不同频率、领域上表现差异显著；Moirai 2.0 在非数据泄漏模型中 MASE 排名第一（截至 2025 年）。
- **优点**: 数据集规模大、多样性高；零样本评测协议明确；提供 leaderboard 方便社区跟踪进展；非泄漏预训练集的设计值得推广。
- **局限性**: 发布时 (2024.10) 仅覆盖单变量场景（不含协变量）；评测场景仍以预测为主，未覆盖分类/异常检测/插补；28 个数据集相对仍然有限。

#### 文献 2：TSFM-Bench

- **来源**: Xiangfei Qiu et al. KDD 2025 Datasets and Benchmarks Track. [arXiv:2410.11802](https://arxiv.org/abs/2410.11802)
- **核心贡献**: 覆盖 zero-shot / few-shot / full-shot 三种评测协议的统一 TSFM 基准，是首批系统比较 LLM-based 与 TS-pretrained 两类基础模型的工作之一。
- **技术方案**:
  - 评测 **14 个 TSFM**，包括 Chronos、TimesFM、Moirai、MOMENT、GPT4TS 等
  - 覆盖三种适应策略：zero-shot（直接推理）、few-shot（极少量微调）、full-shot（完整微调）
  - 多数据集、多领域评测设置
- **关键结果**: 揭示了 LLM-based TSFMs 与 TS-pretrained TSFMs 在不同评测协议下的性能差距；full-shot 下专域模型仍然具有竞争力。
- **优点**: 三种协议同时评测，设计较全面；KDD 发表保证了社区影响力。
- **局限性**: 数据集数量相对有限；评测任务以预测为主。

#### 文献 3：fev-bench

- **来源**: Oleksandr Shchur, Abdul Fatir Ansari, Caner Turkmen, Lorenzo Stella, Nick Erickson, Pablo Guerron, Michael Bohlke-Schneider, Yuyang Wang (Amazon). [arXiv:2509.26468](https://arxiv.org/abs/2509.26468)，2025 年 9 月首发
- **核心贡献**: 首个包含协变量（covariates）的现实主义时序预测 benchmark，弥补 GIFT-Eval 等基准不支持 exogenous variable 的缺陷。
- **技术方案**:
  - **100 个预测任务**，来自 96 个独立数据集，横跨 7 个领域
  - 30 个任务含已知动态协变量（dynamic covariates），24 个含过去动态协变量，19 个含静态协变量
  - 指标：win rate 和 skill score，使用 bootstrap 置信区间做统计显著性检验
  - 配套轻量级 Python 库 `fev` 支持可复现评测
- **关键结果**: Chronos-2 在 fev-bench 上显著优于 TiRex 和 TimesFM 等模型；支持协变量的模型在含 covariate 任务上有明显优势。
- **优点**: 协变量设计贴近实际业务场景；统计检验框架严谨；100 个任务规模较大。
- **局限性**: 发布较新（2025.09），社区采用率尚待提升；覆盖的模型数量有限。

---

### 3.2 公平性与综合性导向的系统基准

#### 文献 4：TFB（Time Series Forecasting Benchmark）

- **来源**: Xiangfei Qiu, Jilin Hu, Lekui Zhou et al. PVLDB 2024（Best Paper Nomination）. [arXiv:2403.20150](https://arxiv.org/abs/2403.20150)
- **核心贡献**: 系统梳理了现有时序预测评测的三大缺陷，并构建了涵盖 8,068 条单变量序列和 25 个多变量数据集的公平评测框架。
- **技术方案**:
  - **单变量（UTSF）**：8,068 条时序，评测 21 种方法
  - **多变量（MTSF）**：25 个数据集，评测 14 种方法
  - 覆盖 **10 个领域**：交通、电力、能源、环境、自然、经济、股市、银行、健康、Web
  - 方法类型：统计、机器学习、深度学习
  - 提供统一、灵活的评测流水线，消除不公平比较
- **三大问题的系统应对**:
  1. **数据域覆盖不足** → 新增多领域数据集
  2. **对传统方法的刻板偏见（stereotype bias）** → 在统一超参设置下公平比较所有方法
  3. **评测流水线不一致** → 标准化 pipeline
- **关键结果**: 在公平评测下，传统统计方法（如 ETS、ARIMA）在多个场景下仍可与深度学习方法竞争；部分"最新"深度学习方法的领先优势在公平设置下显著缩小。
- **优点**: 发表于顶级数据库会议（PVLDB），Best Paper Nomination 证明其质量；开源代码和在线 leaderboard 支持持续评测；UTSF + MTSF 双赛道设计合理。
- **局限性**: 主要面向传统预测方法，对 TSFM 的零样本评测支持有限；不含分类/异常检测等非预测任务。

#### 文献 5：BasicTS

- **来源**: GestaltCogTeam. ECML-PKDD 2023 Workshop. [GitHub](https://github.com/GestaltCogTeam/BasicTS)
- **核心贡献**: 开源的公平、可扩展时序预测 benchmark 工具包，覆盖多种时序分析任务。
- **技术方案**:
  - 支持任务：空间-时序预测（spatial-temporal）、长期时序预测、分类、插补
  - 覆盖方法：统计模型、机器学习、深度学习
  - 标准化训练/评测流水线，方便新方法快速接入
  - 2024 年扩展：IEEE TKDE 收录了基于 BasicTS 的多变量预测综合评测研究
- **优点**: 开源友好，工程实现质量高；支持多种任务类型；被社区广泛使用。
- **局限性**: 主要关注深度学习方法，对 TSFM 的零样本评测支持不够完善；数据集覆盖面相比 TFB 较窄。

---

### 3.3 早期奠基性基准

#### 文献 6：Monash Time Series Forecasting Archive

- **来源**: Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I. Webb, Rob J. Hyndman, Pablo Montero-Manso. NeurIPS 2021 Datasets and Benchmarks Track. [arXiv:2105.06643](https://arxiv.org/abs/2105.06643)
- **核心贡献**: 首个系统化的跨领域时序预测数据集档案库，为 TSFM 的零样本评测奠定了数据基础。
- **技术方案**:
  - 收录 **30+ 个数据集**，覆盖旅游、银行、能源、经济、交通、自然、Web、销售、健康等领域
  - 时序数量从 1 条到 145,063 条不等；序列长度从 11 个点到 7,397,147 个点不等
  - 提供 8 个误差指标下的基线方法性能
  - 数据已标准化为 GluonTS 格式，支持 Python/R 环境加载
- **关键结果**: 在 Monash benchmark 上，Chronos（Amazon）和 Moirai（Salesforce）等 TSFM 的零样本性能接近甚至超越专门训练的模型，成为验证 TSFM 能力的重要平台。
- **优点**: 数据多样性好；已成为 TSFM 零样本评测的事实标准之一；覆盖多种频率。
- **局限性**: 数据集较旧（2021 年）；部分数据集可能已被纳入新 TSFM 的预训练语料（存在泄漏风险）；不含多变量数据集；不含概率预测任务。

#### 文献 7：LTSF-Linear 评测设置（Are Transformers Effective for Time Series Forecasting?）

- **来源**: Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. AAAI 2023 Oral. [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)
- **核心贡献**: 提出 DLinear/NLinear/Linear 简单基线，质疑 Transformer 在长期预测的必要性；同时确立了一套被社区广泛沿用的评测数据集配置。
- **评测数据集配置（"LTSF 标准评测集"）**:
  - ETTh1, ETTh2（电力变压器，小时级）
  - ETTm1, ETTm2（电力变压器，15分钟级）
  - Traffic（旧金山交通传感器）
  - Electricity（ECL 电力消耗）
  - Exchange（8 国汇率）
  - Weather（德国气象站）
  - ILI（美国流感数据，少量数据集）
  - 预测长度：96/192/336/720
- **关键结果**: 简单线性模型在上述 9 个数据集上全面超越 FEDformer、Autoformer、Informer 等 Transformer 变体，平均提升 20%-50%。
- **问题与反思**: 这套评测配置成为后续几乎所有长期预测论文的标准，但也由此引发了"benchmark 过拟合"问题——大量工作在这几个高度相似的数据集上刷新 SOTA，而在新场景下泛化性未知。

#### 文献 8：M4 竞赛

- **来源**: Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos. International Journal of Forecasting, 2018. [论文链接](https://www.sciencedirect.com/science/article/pii/S0169207018300785)
- **核心贡献**: 迄今最大规模的时序预测竞赛之一，确立了 sMAPE/MASE/OWA 作为竞赛指标的权威性。
- **数据规模**: 100,000 条时序，来自经济、金融、人口统计、工业、宏观经济、微观经济领域，频率涵盖年/季/月/周/日/小时。
- **关键发现**:
  - 17 个最优方法中，12 个是"组合方法"（combination）
  - 6 种纯机器学习方法表现差，均不及组合基线
  - 证明统计方法的组合在短-中期预测中的有效性
- **重要意义**: 为 TSFM 的评测提供了历史基准；Chronos、TimesFM 等模型均在 M4 等竞赛数据集上验证了其能力。

#### 文献 9：M5 竞赛

- **来源**: Spyros Makridakis et al. International Journal of Forecasting, 2022. [论文链接](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- **核心贡献**: 首次以层次时序预测为核心，引入 RMSSE 指标，是深度学习在大规模竞赛中首次主导的标志性事件。
- **数据规模**: 42,840 条层次时序，来自 Walmart 美国零售销售数据，12 个跨截面聚合层级。
- **关键发现**: 所有 Top 50 提交均比最优基准提升 14% 以上；Top 5 提升超 20%；LightGBM 及深度神经网络主导排行榜。
- **重要意义**: 代表性零售层次预测场景，TSFM 评测中可作为 few-shot / full-shot 的下游任务基准。

---

### 3.4 分类与多任务评测

#### 文献 10：UCR/UEA Time Series Archive（分类）

- **来源**: Hoang Anh Dau et al. (UCR 2018); Anthony Bagnall et al. (UEA 2018). [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/); [arXiv:1811.00075](https://arxiv.org/abs/1811.00075)
- **核心贡献**: 时序分类的标准评测档案库，UCR 提供单变量、UEA 提供多变量数据集。
- **数据规模**: UCR 2018 版：128 个单变量数据集；UEA 2018 版：30 个多变量数据集，后续持续扩充。
- **评测标准**: 分类准确率（Accuracy），标准 train/test 切分。
- **局限性**: 专注分类，不含预测；数据集较旧且更新频率有限；TSFM 时代的分类评测协议（零样本/线性探针）尚未充分标准化。

#### 文献 11：TSER（Time Series Extrinsic Regression）

- **来源**: Chang Wei Tan et al. Data Mining and Knowledge Discovery, 2021. [arXiv:2006.12672](https://arxiv.org/abs/2006.12672)
- **核心贡献**: 定义并建立了时序外生回归（TSER）任务的评测框架——给定时序，预测连续目标值（而非分类标签）。
- **数据规模**: 初版 19 个数据集（2022 年），后扩展至 63 个，覆盖医疗、环境、工程等领域。
- **评测指标**: RMSE、MAE，标准 train/test 切分。
- **关键发现**: ROCKET 系列算法适应后在 TSER 任务上整体最优，优于 XGBoost、SVR 等经典 ML 方法。

#### 文献 12：MOMENT

- **来源**: Mononito Goswami et al. (CMU AutonLab). ICML 2024. [arXiv:2402.03885](https://arxiv.org/abs/2402.03885)
- **核心贡献**: 首个同时支持预测、分类、异常检测、插补四类任务的开源时序基础模型，兼具评测框架属性。
- **架构**: T5-based encoder，385M 参数，预训练于 Time-series Pile 数据集，使用 patch-level masking。
- **多任务评测结果**:
  - 预测（zero-shot）：在多个 ETT 数据集上接近 SOTA
  - 分类（linear probing）：UCR 数据集上接近 SOTA
  - 插补（zero-shot）：优于线性插值以外的所有统计方法
  - 异常检测（linear probing）：接近 SOTA
- **局限性**: 预训练数据虽包含 Time-series Pile，但与 Monash/ETT 等评测集的泄漏风险存在；多任务单模型的折中可能导致单任务性能不如专用模型。

---

### 3.5 评测挑战与反思性工作

#### 文献 13：Cherry-Picking in Time Series Forecasting

- **来源**: Luis Roque, Carlos Soares, Vitor Cerqueira, Luis Torgo. arXiv 2024.12. [arXiv:2412.14435](https://arxiv.org/abs/2412.14435)
- **核心贡献**: 首次通过大规模实验量化了时序预测领域"数据集选择偏差"（cherry-picking）的影响。
- **关键发现**:
  - 仅用 4 个数据集（典型发论文用量），**46% 的方法**可以被呈现为"最优"，**77%** 可以进入 top-3
  - 深度学习方法对数据集选择高度敏感，经典统计方法更具鲁棒性
  - 将数据集数量从 3 个增加到 6 个，可将"错误识别某方法为最优"的概率降低约 40%
- **建议**: 至少使用 6 个以上数据集进行评测；报告置信区间；采用 Nemenyi / Wilcoxon 等统计检验。

#### 文献 14：There are no Champions in Supervised LTSF

- **来源**: Lorenzo Brigato et al. TMLR 2025. [arXiv:2502.14045](https://arxiv.org/abs/2502.14045)
- **核心贡献**: 大规模超参数搜索实验证明，LTSF 领域"新模型超过旧模型"的声称在不一致实验设置下不可靠。
- **方法**: 评测 8 个代表性监督模型，14 个数据集，约 5,000 个训练网络。
- **关键发现**:
  - 微小的实验设置变化（学习率、窗口大小、normalization）可以显著改变模型排名
  - 现有评测方法夸大了架构创新的实际价值
  - 复现性差、超参数报告不完整是系统性问题
- **建议**: 标准化超参搜索协议；使用统计显著性检验；聚焦于诊断性评测（在什么条件下哪个模型更好）而非单一排名。

#### 文献 15：Time Series Foundation Models: Benchmarking Challenges and Requirements

- **来源**: Marcel Meyer, Sascha Kaltenpoth, Kevin Zalipski, Oliver Müller. arXiv 2025.10. [arXiv:2510.13654](https://arxiv.org/abs/2510.13654)
- **核心贡献**: 系统梳理 TSFM 评测的特有挑战，提出信息泄漏的两种具体类型。
- **两类信息泄漏**:
  1. **训练-测试样本重叠**：同一数据集被重复用于预训练和评测
  2. **时序相关重叠**：训练序列与测试序列在时间上相邻，导致全局模式被记忆
- **其他挑战**:
  - 评测数据集代表性不足（frequency bias、domain bias）
  - 缺乏时空（spatiotemporal）评测
  - 外部冲击（经济危机、疫情）被模型"记忆"导致评测失真
- **建议**: 使用真正的未来数据（out-of-sample future data）做评测；清晰标注预训练语料与评测集的重叠情况。

#### 文献 16：Are We Winning the Wrong Game? Revisiting Evaluation Practices for LTSF

- **来源**: arXiv 2025.03. [arXiv:2603.08156](https://arxiv.org/abs/2603.08156)
- **核心贡献**: 从根本上质疑以 MSE/MAE 为核心的 LTSF "游戏规则"是否与真实业务目标一致。
- **关键论点**:
  - LTSF 已演变为一个由边际 MSE 下降驱动的"基准游戏"，与真实预测目标结构性脱节
  - 真实预测通常需要：时序结构保真（趋势、季节性）、对制度切换的鲁棒性、支持下游决策
  - 优化聚合点误差并不意味着保留上述结构特性
- **建议**: 采用多维评测视角（统计保真性 + 结构一致性 + 决策相关性）；迈向"诊断性报告"而非单一排名。

#### 文献 17：How Foundational are Foundation Models for Time Series?

- **来源**: arXiv 2025.10. [arXiv:2510.00742](https://arxiv.org/abs/2510.00742)
- **核心贡献**: 通过细致的实证研究质疑 TSFM 在迁移学习中的实际价值。
- **关键发现**:
  - TSFM 的零样本能力与其预训练域高度相关，在训练域之外泛化能力急剧下降
  - 微调后的 TSFM 并不一致地优于参数量更小、从头训练的专用模型
  - 在一个新构建的每日电力数据集上，小型专用网络优于微调 TSFM

---

### 3.6 异常检测评测基准

#### 文献 18：TSB-AD

- **来源**: thedatumorg. [GitHub](https://github.com/thedatumorg/TSB-AD)
- **核心贡献**: 包含 40 个数据集、1070 条高质量时序的异常检测基准，评测 40 种算法，识别 VUS-PR 为最可靠评测指标。
- **关键设计**: 支持统计方法到最新基础模型的全谱系评测；使用 VUS-PR（Volume Under the Surface for Precision-Recall）代替传统 F1，避免阈值选择偏差。

#### 文献 19：TAB（Unified Benchmarking of TS Anomaly Detection）

- **来源**: Xiangfei Qiu et al. arXiv 2025.06. [arXiv:2506.18046](https://arxiv.org/abs/2506.18046)
- **核心贡献**: 统一异常检测 benchmark，1,635 条单变量 + 29 个多变量数据集，评测 40 个多变量方法和 46 个单变量方法，覆盖 zero/few/full-shot。
- **关键问题**: 指出现有 TSAD 评测中存在严重指标缺陷——流行指标会显著高估随机模型性能。

---

## 4. 评测维度横向对比分析

### 4.1 Benchmark 核心特征对比总表

| Benchmark | 论文/来源 | 发表时间 | 数据集数量 | 时序总数 | 任务类型 | 零样本支持 | 协变量 | 多变量 | 概率预测 | 开源 |
|-----------|----------|---------|-----------|---------|---------|----------|--------|--------|---------|------|
| **GIFT-Eval** | NeurIPS 2024 | 2024.10 | 28 数据集 | 144,000+ | 预测 | ✓ | ✗ | ✓ | ✓ (WQL) | ✓ |
| **TFB** | PVLDB 2024 | 2024.03 | 8,068单变量+25多变量 | 8,000+ | 预测 | ✗ | ✗ | ✓ | ✗ | ✓ |
| **TSFM-Bench** | KDD 2025 | 2024.10 | 多个 | 多个 | 预测 | ✓ | ✗ | ✓ | ✗ | ✓ |
| **fev-bench** | arXiv 2025 | 2025.09 | 96 | 100 任务 | 预测(含协变量) | ✓ | ✓ | ✓ | ✓ (CRPS) | ✓ |
| **Monash Archive** | NeurIPS 2021 | 2021.05 | 30+ | 多变 | 预测 | ✓(被动) | ✗ | ✗ | ✗ | ✓ |
| **LTSF 标准集** | AAAI 2023 | 2022.05 | 9 | 中等 | 长期预测 | ✗ | ✗ | ✓ | ✗ | ✓ |
| **M4** | IJF 2018 | 2018 | 1 | 100,000 | 中短期预测 | ✗ | ✗ | ✗ | ✓ (区间) | ✓ |
| **M5** | IJF 2022 | 2020 | 1 | 42,840 | 层次预测 | ✗ | ✓ | ✓ | ✓ | ✓ |
| **UCR/UEA** | 2018 | 2018 | 128+30 | 万级 | 分类 | ✓(被动) | ✗ | ✓(UEA) | ✗ | ✓ |
| **TSER** | DMKD 2021 | 2021 | 63 | 多变 | 回归 | ✗ | ✗ | ✓ | ✗ | ✓ |
| **BasicTS** | ECML-WS 2023 | 2023 | 多个 | 多变 | 预测+分类+插补 | ✗ | ✗ | ✓ | ✗ | ✓ |
| **TSB-AD** | 2024 | 2024 | 40 | 1,070 | 异常检测 | ✓ | ✗ | ✓ | ✗ | ✓ |
| **TAB** | arXiv 2025 | 2025.06 | 多个 | 1,635+ | 异常检测 | ✓ | ✗ | ✓ | ✗ | ✓ |

### 4.2 评测协议对比

| 协议类型 | 定义 | 典型应用 benchmark | 核心问题 |
|---------|------|------------------|---------|
| **Zero-shot** | 模型在评测数据集上**无任何梯度更新**直接推理 | GIFT-Eval, Monash, fev-bench | 数据泄漏风险最高；最能体现基础模型价值 |
| **Few-shot** | 仅用极少量（如 8-64 条）样本微调后评测 | TSFM-Bench, fev-bench | 协议定义不统一（样本量定义各异）|
| **Full-shot** | 在完整目标数据集训练集上微调评测 | TFB, BasicTS, LTSF标准集 | 最公平但无法体现迁移能力 |
| **Linear Probing** | 冻结主干，仅训练线性头 | MOMENT (分类/异常检测) | 测试特征表示质量 |

### 4.3 指标体系

| 指标 | 类型 | 公式特征 | 适用场景 | 注意事项 |
|-----|------|---------|---------|---------|
| **MSE** | 点预测 | 平方误差均值 | 深度学习论文标配 | 对量纲敏感，跨数据集不可比 |
| **MAE** | 点预测 | 绝对误差均值 | 与 MSE 搭配 | 同上 |
| **MASE** | 点预测（归一化） | MAE / naive_MAE | 跨序列/数据集比较 | GIFT-Eval 主指标；需要历史数据做归一化 |
| **sMAPE** | 点预测（相对） | 2\|y-f\|/(y+f) | M4/M5 竞赛 | 当值接近 0 时数值不稳定 |
| **RMSSE** | 点预测（归一化） | 平方版 MASE | M5 竞赛 | 层次预测场景 |
| **WQL** | 概率预测 | 加权分位数损失 | GIFT-Eval 概率指标 | 尺度依赖，跨数据集需归一化 |
| **CRPS** | 概率预测 | 连续排名概率分 | fev-bench, 通用概率评测 | 同时评估校准性和锐度 |
| **OWA** | 综合 | sMAPE+MASE 加权 | M4 竞赛 | 竞赛特有，难迁移 |
| **F1/AUPRC** | 异常检测 | 精确率-召回率 | TSAD 任务 | 点调整（point adjustment）存在争议 |
| **VUS-PR** | 异常检测 | PR曲线下面积（三维） | TSB-AD | 对阈值选择不敏感，更可靠 |

### 4.4 主要 TSFM 评测协议汇总

| 模型 | 机构 | 发布时间 | 评测 Benchmark | 零样本指标 | 特点 |
|-----|------|---------|--------------|----------|------|
| **Chronos** | Amazon | 2024.03 | Monash(27集) | MASE, WQL (vs. Seasonal Naive) | T5 架构；量化 tokenization |
| **Chronos-2** | Amazon | 2024.10 | GIFT-Eval, fev-bench, Chronos Bench II | MASE, CRPS | 扩展多变量+协变量 |
| **TimesFM** | Google | 2024.05 | Monash, LTSF标准集 | MAE, MSE | ICML 2024；decoder-only |
| **Moirai** | Salesforce | 2024.03 | Monash, GIFT-Eval | MASE, CRPS | 训练于 LOTSA(27B pts) |
| **Moirai 2.0** | Salesforce | 2024.11 | GIFT-Eval (#1 MASE) | MASE, WQL | 30x 小于前代，2x 快 |
| **MOMENT** | CMU | 2024.02 | UCR/UEA, ETT, Monash | Accuracy, MSE | 多任务；T5-encoder |
| **Time-MoE** | 多机构 | 2024.09 | LTSF标准集(6个) | MSE, MAE | ICLR 2025 Spotlight；2.4B 参数 |
| **TTM** | IBM | 2024.01 | 11个数据集 | MSE, MAE | NeurIPS 2024；轻量级(1M+) |
| **Sundial** | THU | 2025.02 | GIFT-Eval (#1 MASE), FEV | MASE, CRPS | ICML 2025 Oral；Flow Matching |
| **Lag-Llama** | 多机构 | 2023.10 | Monash | CRPS | 首批 LLM-for-TS 之一；概率预测 |

---

## 5. 现有评测问题与挑战

### 5.1 数据污染与泄漏问题

这是 TSFM 评测最核心、最难解决的问题。

**问题一：预训练语料与评测集重叠**

TSFM 通常在海量公开时序数据上预训练，而评测数据集（如 ETT、Monash 中的部分数据集）本身也是公开的。这导致：
- 模型可能在"零样本"测试时，实际上已经在训练中见过这些数据
- 不同 TSFM 对泄漏数据集的处理方式不同，缺乏统一标准

Moirai 的 LOTSA 数据集已在论文中声明纳入了 Monash 中的 29 个数据集的训练集；Chronos 的训练数据也包含部分公开评测集。

**问题二：时序相关性泄漏（Meyer et al., 2025）**

即使训练集和测试集在数据点层面没有重叠，如果它们来自同一时序的相邻时间段，模型可能记忆了该时序的全局模式（趋势、季节性周期等），从而在"零样本"评测中获得不公平优势。

**缓解策略**：
- GIFT-Eval 提供单独的非泄漏预训练数据集
- 使用真正未来时间段的数据进行评测
- 要求模型公开详细的预训练语料成分

### 5.2 Cherry-Picking 与数据集选择偏差

Roque et al. (2024) 的定量研究表明，在 4 个数据集上进行评测时，46% 的方法可以通过选择合适的数据集看起来最优。这解释了为什么：
- ETT/Traffic/Electricity/Weather 等"标准"数据集被反复使用
- 新方法在特定数据集上的"显著提升"可能只是数据集选择的产物
- 深度学习方法普遍比统计方法更易受数据集选择影响

**典型案例**：研究表明，Exchange Rate 数据集本质上接近随机游走，过去持续用于 Transformer 评测存在争议。

### 5.3 评测协议不一致

TSFM 论文中常见的不一致性：

| 不一致来源 | 具体表现 | 影响 |
|----------|---------|-----|
| Zero-shot 定义 | 是否允许使用上下文窗口；是否允许少量适应 | 不同论文的"零样本"不可比 |
| Few-shot 样本量 | 8 条 vs 64 条 vs "10% 训练集" | 同为 few-shot 但协议差异巨大 |
| 归一化方式 | 每序列 vs 全局 vs 无归一化 | 显著影响 MSE/MAE 数值 |
| 预测窗口 | 固定 horizon vs 多 horizon 取均值 | 单 horizon 优化 vs 多 horizon 泛化 |
| Lookback 窗口 | 固定 vs 自适应 | 影响季节性模式捕捉 |

### 5.4 指标设计缺陷

- **MSE/MAE 的跨序列不可比性**：不同量纲的时序直接平均 MSE 无意义，优化 MSE 的模型倾向于在高方差序列上分配更多容量
- **F1 在异常检测中的点调整（point adjustment）偏差**：常用的 "点调整 F1" 被证明会显著高估性能，应转向 VUS-PR 等更鲁棒的指标
- **sMAPE 的数值不稳定性**：当真实值或预测值接近 0 时，sMAPE 极度不稳定

### 5.5 数据集代表性不足

- **频率偏差**：现有 benchmark 高度偏向日/周/月级数据，分钟/秒级高频数据（如金融 tick 数据）严重不足
- **领域偏差**：能源、交通、气象数据占主导，生物医疗、工业传感器等数据稀少
- **多变量结构多样性不足**：大多数多变量数据集是同质的多传感器数据，复杂关联结构（如带社交网络拓扑的时序）缺失
- **时序长度分布失衡**：超长（百万时间步）和极短时序在 benchmark 中代表性不足

### 5.6 评测不可复现性

- 超参数未完全报告（Brigato et al., 2025 发现这是普遍问题）
- 随机种子不固定
- 代码未开源或依赖特定硬件配置
- 数据预处理细节（归一化、缺失值处理）描述不清

---

## 6. 最新进展（2024-2025）

### 6.1 GIFT-Eval Leaderboard 现状（截至 2025 年初）

GIFT-Eval 已成为 TSFM 社区的事实标准 benchmark，leaderboard 持续更新：

| 排名 | 模型 | MASE 排名分 | 特点 |
|-----|------|-----------|------|
| 1 | **Toto** | 5.495（平均排名） | 全模型最优（含可能数据泄漏模型） |
| Top 5 | **Moirai 2.0** | #1 MASE（非泄漏模型） | 30x 小体积，2x 速度提升 |
| Top 5 | **Sundial (Base)** | #1 MASE + #2 CRPS（unseen数据集） | ICML 2025 Oral，Flow Matching |
| 竞争者 | **Chronos-2** | fev-bench 显著最优 | 支持多变量+协变量 |
| 竞争者 | **TiRex** | fev-bench 第二 | — |
| 竞争者 | **TimesFM 2.5** | leaderboard 竞争者 | Google 内部更新 |

主要竞争模型（截至 2025 年）包括：Chronos-2, TimesFM-2.5, TimesFM-2.0, TiRex, FlowState, Granite-FlowState-R1, Kairos, Toto, Sundial, TabPFN-TS, YingLong。

### 6.2 Chronos 系列的评测演进

- **Chronos v1（2024.03）**：在 27 个 Monash 未训练数据集上评测零样本 WQL 和 MASE，使用 Seasonal Naive 归一化作为基准，成为早期 TSFM 零样本评测的参考方案
- **Chronos-2（2024.10）**：从单变量扩展到多变量，支持协变量；在 fev-bench、GIFT-Eval、Chronos Benchmark II 三个 benchmark 上全面评测，成为最系统的 TSFM 评测报告之一

### 6.3 Time-MoE 的评测设置（ICLR 2025 Spotlight）

- 预训练于 Time-300B（300B 时间步，9 个领域）
- 评测在 6 个标准 benchmark 上（ETTh1/h2/m1/m2, Exchange, Weather）
- 关键声称：zero-shot MSE 平均下降超过 20%；full-shot MSE 平均下降 24%
- 但该评测设置使用的是"LTSF 标准集"——与前述 cherry-picking 问题高度相关

### 6.4 Moirai 2.0 的评测（2024.11，arXiv 2511.11698）

- 在 GIFT-Eval 上位居非数据泄漏模型 MASE 排名第一
- 关键设计：削减训练数据而非增加（"Less is More"），专注高质量数据
- 比 Moirai 1.0-Large 快 2 倍，小 30 倍，性能更优
- 在 Monash(29 datasets) 的 CRPS 上同样竞争力强

### 6.5 Sundial 的评测（ICML 2025 Oral）

- 预训练于 TimeBench（10^12 时间步）
- 采用 Flow Matching（TimeFlow Loss）代替传统离散 tokenization
- 在 GIFT-Eval 上 MASE 第一，FEV leaderboard 排名靠前
- 特点：支持原生概率预测，无需假设分布形式

### 6.6 关键趋势

1. **多 benchmark 评测成为规范**：新模型普遍在 GIFT-Eval + fev-bench + Monash 三个平台同时汇报结果
2. **协变量支持**：fev-bench 的出现推动更多模型增加协变量支持
3. **效率-性能折中**：Moirai 2.0（30x smaller）和 TTM（1M 参数）等轻量模型受到关注
4. **概率预测标准化**：CRPS 和 WQL 逐渐取代单纯的点预测 MSE 作为主指标

---

## 7. 未来展望

### 7.1 理想的 TSFM 评测体系应具备的要素

基于上述分析，理想的评测体系应满足以下设计原则：

**1. 数据无污染原则**
- 严格区分预训练语料与评测数据集，并公开声明重叠情况
- 使用真正的未来数据（评测集在预训练结束时间点之后）
- 建立"防泄漏审计"标准，类似 LLM 领域的 contamination score

**2. 统计严谨性**
- 所有性能比较应附带置信区间（bootstrap 或 Bayesian）
- 提供统计显著性检验（Wilcoxon 符号秩检验、Nemenyi 检验）
- 至少在 20+ 数据集上评测以避免 cherry-picking

**3. 多维指标体系**
- 点预测：MASE（归一化，跨序列可比）
- 概率预测：CRPS（评估分布质量）
- 结构保真：趋势/季节性保留度（参见 Revisiting LTSF 2025 建议）
- 效率：推理速度 / 参数量 / 内存占用（工程实用性）

**4. 多协议公平评测**
- Zero-shot / Few-shot / Full-shot 三种协议统一定义，同时报告
- Few-shot 的样本量标准化（建议：k ∈ {8, 64, 512}）

**5. 领域与频率多样性**
- 覆盖至少 10 个领域和 8 种频率
- 专门补充高频数据（分钟/秒级）和极长时序数据
- 纳入具有复杂多变量结构的时序（空间关联、因果结构）

**6. 任务多样性**
- 预测（短/中/长期，单/多变量，有/无协变量）
- 分类（UCR/UEA 扩展版）
- 异常检测（使用 VUS-PR 等可靠指标）
- 插补（TSI-Bench 风格）
- 时序理解（TSER，外生回归）

**7. 持续更新与社区治理**
- 引入动态数据集更新机制，持续扩充新领域数据
- 建立开放的 leaderboard 同时防止"leaderboard 过拟合"
- 制定数据集贡献标准（类似 Hugging Face datasets 的审核机制）

### 7.2 近期值得关注的研究方向

1. **无污染评测框架**：开发验证预训练数据与评测集不重叠的自动化工具
2. **时序基准的动态更新**：参考 LiveBench（LLM 领域）设计随时间演进的时序评测集
3. **决策导向评测（Decision-Aware Evaluation）**：将预测评测与下游决策（库存管理、能源调度）成本/收益挂钩
4. **时空评测**：针对空间关联时序（如气象网格、交通网络）的专项 benchmark
5. **领域外泛化评测**：专门测试预训练域与评测域完全不同的场景
6. **TSFM 的 scaling laws 研究**：系统评测参数量、预训练数据量与下游任务性能的关系

### 7.3 与相邻领域的交叉方向

- **LLM × 时序**：借鉴 LLM 评测中的数据污染检测方法（n-gram 重叠检测、困惑度异常检测）
- **AutoML × 时序 benchmark**：自动化超参搜索与评测，消除人工超参带来的偏差
- **因果推断 × 时序评测**：超越关联性，评测模型在干预场景下的预测能力

---

## 8. 参考文献索引

### 核心 Benchmark 论文

1. **GIFT-Eval**: Taha Aksu et al. "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation." NeurIPS 2024. [arXiv:2410.10393](https://arxiv.org/abs/2410.10393) | [GitHub](https://github.com/SalesforceAIResearch/gift-eval) | [Leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval)

2. **TFB**: Xiangfei Qiu et al. "TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods." PVLDB 2024 (Best Paper Nomination). [arXiv:2403.20150](https://arxiv.org/abs/2403.20150) | [GitHub](https://github.com/decisionintelligence/TFB)

3. **TSFM-Bench**: Li et al. "TSFM-Bench: A Comprehensive and Unified Benchmark of Foundation Models for Time Series Forecasting." KDD 2025. [arXiv:2410.11802](https://arxiv.org/abs/2410.11802)

4. **fev-bench**: Oleksandr Shchur et al. "fev-bench: A Realistic Benchmark for Time Series Forecasting." arXiv 2025. [arXiv:2509.26468](https://arxiv.org/abs/2509.26468) | [GitHub](https://github.com/autogluon/fev)

5. **Monash Archive**: Rakshitha Godahewa et al. "Monash Time Series Forecasting Archive." NeurIPS 2021. [arXiv:2105.06643](https://arxiv.org/abs/2105.06643)

6. **BasicTS**: GestaltCogTeam. "BasicTS: An Open Source Fair Multivariate Time Series Prediction Benchmark." ECML-PKDD 2023. [GitHub](https://github.com/GestaltCogTeam/BasicTS)

### 评测挑战与反思

7. **Cherry-Picking**: Luis Roque et al. "Cherry-Picking in Time Series Forecasting: How to Select Datasets to Make Your Model Shine." arXiv 2024. [arXiv:2412.14435](https://arxiv.org/abs/2412.14435)

8. **No Champions**: Lorenzo Brigato et al. "There are no Champions in Supervised Long-Term Time Series Forecasting." TMLR 2025. [arXiv:2502.14045](https://arxiv.org/abs/2502.14045)

9. **Benchmarking Challenges**: Marcel Meyer et al. "Time Series Foundation Models: Benchmarking Challenges and Requirements." arXiv 2025. [arXiv:2510.13654](https://arxiv.org/abs/2510.13654)

10. **Wrong Game**: "Are We Winning the Wrong Game? Revisiting Evaluation Practices for Long-Term Time Series Forecasting." arXiv 2025. [arXiv:2603.08156](https://arxiv.org/abs/2603.08156)

11. **How Foundational**: "How Foundational are Foundation Models for Time Series Forecasting?" arXiv 2025. [arXiv:2510.00742](https://arxiv.org/abs/2510.00742)

### 早期基准

12. **LTSF-Linear**: Ailing Zeng et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023. [arXiv:2205.13504](https://arxiv.org/abs/2205.13504)

13. **M4 Competition**: Spyros Makridakis et al. "The M4 Competition: Results, findings, conclusion and way forward." International Journal of Forecasting, 2018. [DOI](https://doi.org/10.1016/j.ijforecast.2018.06.001)

14. **M5 Competition**: Spyros Makridakis et al. "M5 accuracy competition: Results, findings, and conclusions." International Journal of Forecasting, 2022. [DOI](https://doi.org/10.1016/j.ijforecast.2021.11.013)

15. **UCR Archive**: Hoang Anh Dau et al. "The UCR Time Series Archive." 2018. [URL](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

16. **UEA Archive**: Anthony Bagnall et al. "The UEA Multivariate Time Series Classification Archive, 2018." [arXiv:1811.00075](https://arxiv.org/abs/1811.00075)

17. **TSER**: Chang Wei Tan et al. "Time Series Extrinsic Regression." DMKD 2021. [arXiv:2006.12672](https://arxiv.org/abs/2006.12672)

### TSFM 模型论文（含评测方案）

18. **Chronos**: Abdul Fatir Ansari et al. "Chronos: Learning the Language of Time Series." arXiv 2024. [GitHub](https://github.com/amazon-science/chronos-forecasting)

19. **Chronos-2**: Amazon Science. "Technical Report of Chronos-2." arXiv 2024. [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)

20. **TimesFM**: Abhimanyu Das et al. "A Decoder-Only Foundation Model for Time-Series Forecasting." ICML 2024. [GitHub](https://github.com/google-research/timesfm)

21. **Moirai**: Gerald Woo et al. "Unified Training of Universal Time Series Forecasting Transformers." ICML 2024. [arXiv:2402.02592](https://arxiv.org/abs/2402.02592)

22. **Moirai 2.0**: Salesforce AI Research. "Moirai 2.0: When Less Is More for Time Series Forecasting." arXiv 2024. [arXiv:2511.11698](https://arxiv.org/abs/2511.11698)

23. **MOMENT**: Mononito Goswami et al. "MOMENT: A Family of Open Time-series Foundation Models." ICML 2024. [arXiv:2402.03885](https://arxiv.org/abs/2402.03885)

24. **Time-MoE**: "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts." ICLR 2025 Spotlight. [arXiv:2409.16040](https://arxiv.org/abs/2409.16040)

25. **TTM**: "Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting." NeurIPS 2024. [arXiv:2401.03955](https://arxiv.org/abs/2401.03955)

26. **Sundial**: "Sundial: A Family of Highly Capable Time Series Foundation Models." ICML 2025 Oral. [arXiv:2502.00816](https://arxiv.org/abs/2502.00816)

### 异常检测基准

27. **TSB-AD**: thedatumorg. "TSB-AD." [GitHub](https://github.com/thedatumorg/TSB-AD)

28. **TAB**: Xiangfei Qiu et al. "TAB: Unified Benchmarking of Time Series Anomaly Detection Methods." arXiv 2025. [arXiv:2506.18046](https://arxiv.org/abs/2506.18046)

29. **TimeSeriesBench**: "TimeSeriesBench: An Industrial-Grade Benchmark for Time Series Anomaly Detection Models." arXiv 2024. [arXiv:2402.10802](https://arxiv.org/abs/2402.10802)

### 插补基准

30. **TSI-Bench**: "TSI-Bench: Benchmarking Time Series Imputation." arXiv 2024. [arXiv:2406.12747](https://arxiv.org/abs/2406.12747)

---

*本报告由 Academic Research Surveyor 基于系统性 Web 搜索与 arXiv/Semantic Scholar API 检索生成，生成日期 2026-04-07。建议结合原始论文深入阅读核心工作，部分 2025 年后期发布论文（如 TAB 2506.18046）为预印本，以最终发表版本为准。*
