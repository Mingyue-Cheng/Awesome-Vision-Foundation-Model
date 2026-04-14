# TextTS 评测脚本使用说明

本文说明当前仓库里 TextTS 的四个评测入口：

1. `textts/scripts/run_timemmd_eval.sh`
2. `textts/scripts/run_ltsf_eval.sh`
3. `textts/scripts/run_gift_eval.sh`
4. `textts/scripts/run_benchmark_eval.sh`

前三个是具体评测脚本，最后一个是统一入口。

## 1. 总览

当前三套评测对应关系如下：

| 评测类型 | 脚本 | Python 入口 | 适用场景 |
| --- | --- | --- | --- |
| Time-MMD | `textts/scripts/run_timemmd_eval.sh` | `python3 -m textts.eval.forecast_eval` | 评测 Time-MMD 多 domain 数据 |
| LTSF benchmark | `textts/scripts/run_ltsf_eval.sh` | `python3 -m textts.eval.ltsf_eval` | 评测 ETT / Weather / Traffic / Electricity |
| GIFT-Eval | `textts/scripts/run_gift_eval.sh` | `python3 -m textts.eval.gift_eval` | 评测 zero-shot / few-shot 泛化能力 |
| 统一入口 | `textts/scripts/run_benchmark_eval.sh` | 依次调用 `run_gift_eval.sh` / `run_ltsf_eval.sh` | 一次性跑 benchmark 套件 |

推荐使用方式：

- 单独评测某一类数据时，直接调用对应 `run_*.sh`
- 要统一跑 benchmark 时，调用 `run_benchmark_eval.sh`
- Time-MMD 目前不在统一 benchmark 入口里，需要单独跑

## 2. 共同前提

所有脚本都默认在仓库根目录执行，并会自动：

- 计算 `PROJECT_ROOT`
- 设置 `PYTHONPATH=$PROJECT_ROOT`
- 调用 `python3 -m textts.eval.*`

因此通常推荐在仓库根目录下执行：

```bash
cd /Users/chengmingyue/human_agent_collaboration/research_working/fm_time_series
```

### 2.1 模型相关公共变量

下面这些变量在三套评测里都通用，统一入口也会透传它们：

| 环境变量 | 含义 | 典型值 | 默认值 |
| --- | --- | --- | --- |
| `MODEL_NAME` | 基座模型名或本地模型名 | `Qwen/Qwen3-0.6B-Base` | 各脚本内置默认 |
| `CHECKPOINT_DIR` | 微调后 TextTS checkpoint 目录 | `outputs/...` | 空，表示不额外挂 checkpoint |
| `TEXTTS_MODULES_PATH` | 额外 TextTS 模块权重路径 | 自定义目录 | 空 |
| `DEVICE` | 推理设备 | `cpu` / `cuda` | `cpu` |
| `TORCH_DTYPE` | 加载模型时的数据类型 | `float16` / `bfloat16` | 空 |
| `DEVICE_MAP` | HuggingFace `device_map` | `auto` | 空 |
| `LOCAL_FILES_ONLY` | 是否只从本地加载模型 | `1` / `0` | 各脚本不同 |

说明：

- 如果 `CHECKPOINT_DIR` 非空，脚本会追加 `--checkpoint-dir`
- 如果 `TEXTTS_MODULES_PATH` 非空，脚本会追加 `--textts-modules-path`
- `LOCAL_FILES_ONLY=1` 适合离线环境，但要求本地已经有模型缓存
- `LOCAL_FILES_ONLY=0` 允许从 HuggingFace 拉取模型，但需要联网

### 2.2 生成与采样公共变量

三套评测都支持 point forecast，部分也支持 probabilistic forecast。

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `POINT_STRATEGY` | 点预测解码策略 | `greedy` |
| `POINT_TEMPERATURE` | 点预测温度 | `1.0` |
| `POINT_TOP_P` | 点预测 top-p | `1.0` |
| `NUM_PROB_SAMPLES` | 概率预测采样次数 | 脚本相关 |
| `PROB_TEMPERATURE` | 概率预测温度 | `1.0` |
| `PROB_TOP_P` | 概率预测 top-p | `0.9` |
| `MAX_SAMPLES` | 最多评测多少条样本 | 空或脚本内默认 |

说明：

- `NUM_PROB_SAMPLES=0` 通常表示不做概率评测
- `MAX_SAMPLES` 适合快速冒烟测试

### 2.3 输出目录

三套脚本都会把结果写到 `OUTPUT_DIR`：

| 脚本 | 默认输出目录 |
| --- | --- |
| `run_timemmd_eval.sh` | `outputs/timemmd_eval_${RUN_LABEL}` |
| `run_ltsf_eval.sh` | `outputs/ltsf_eval_${RUN_LABEL}` |
| `run_gift_eval.sh` | `outputs/${RUN_LABEL}` |
| `run_benchmark_eval.sh` | `outputs/${RUN_LABEL}_gift` 和 `outputs/${RUN_LABEL}_ltsf` |

## 3. Time-MMD 评测

### 3.1 作用

`run_timemmd_eval.sh` 用于评测 Time-MMD 数据。它最终调用：

```bash
python3 -m textts.eval.forecast_eval --data-source timemmd ...
```

这个脚本主要面向 domain 级别数据，例如 `Energy`。

### 3.2 环境变量

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `TIMEMMD_ROOT` | Time-MMD 数据根目录 | `$PROJECT_ROOT/Time-MMD` |
| `DOMAINS` | 多个 domain，逗号分隔 | 空 |
| `DOMAIN` | 单个 domain；也可作为脚本第一个位置参数传入 | `Energy` |
| `TARGET_COL` | 目标列 | `OT` |
| `LOOKBACK` | 历史窗口长度 | `32` |
| `HORIZON` | 预测长度 | `8` |
| `STRIDE` | 滑窗步长 | `16` |
| `MAX_WINDOWS` | 每个序列最多取多少个窗口 | `4` |
| `SPLIT` | 数据切分 | `test` |
| `SPLIT_VAL_RATIO` | 验证集比例 | `0.1` |
| `SPLIT_TEST_RATIO` | 测试集比例 | `0.1` |
| `MAX_SAMPLES` | 最多评测样本数 | `2` |
| `RUN_LABEL` | 结果标签 | `${DOMAIN:-joint}` |
| `CHECKPOINT_DIR` | 默认 checkpoint 路径 | `outputs/timemmd_pretrain_${RUN_LABEL}` |

补充规则：

- 如果设置了 `DOMAINS`，脚本会传 `--domains`
- 否则传 `--domain`
- 可以直接用位置参数指定单个 domain，例如 `bash textts/scripts/run_timemmd_eval.sh Energy`

### 3.3 示例

单 domain 冒烟：

```bash
MODEL_NAME=Qwen/Qwen3-0.6B-Base \
LOCAL_FILES_ONLY=1 \
MAX_SAMPLES=8 \
DEVICE=cpu \
bash textts/scripts/run_timemmd_eval.sh Energy
```

多 domain：

```bash
DOMAINS=Energy,Weather \
TIMEMMD_ROOT=/path/to/Time-MMD \
CHECKPOINT_DIR=outputs/textts_timemmd_ckpt \
DEVICE=cuda \
TORCH_DTYPE=bfloat16 \
NUM_PROB_SAMPLES=16 \
bash textts/scripts/run_timemmd_eval.sh
```

## 4. LTSF Benchmark 评测

### 4.1 作用

`run_ltsf_eval.sh` 用于标准长序列预测 benchmark，最终调用：

```bash
python3 -m textts.eval.ltsf_eval ...
```

当前脚本面向：

- `ETTh1`
- `ETTh2`
- `ETTm1`
- `ETTm2`
- `Weather`
- `Traffic`
- `Electricity`

### 4.2 环境变量

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `LTSF_ROOT` | LTSF 数据根目录 | `$PROJECT_ROOT/数据集/LTSF` |
| `DATASETS` | 数据集列表，逗号分隔，或 `all` | `all` |
| `HORIZONS` | 预测长度列表，逗号分隔 | `96,192,336,720` |
| `LOOKBACK` | 历史窗口长度 | `96` |
| `SPLIT` | 评测切分 | `test` |
| `STRIDE` | 滑窗步长 | `1` |
| `MAX_WINDOWS` | 每个 dataset/horizon 最多窗口数 | 空 |
| `TARGET_MODE` | 目标列选择模式 | `all` |
| `TARGET_COLS` | 显式指定目标列列表 | 空 |
| `MAX_TARGETS` | 最多评测多少个 target 列 | 空 |
| `VAL_RATIO` | 验证集比例 | `0.1` |
| `TEST_RATIO` | 测试集比例 | `0.2` |
| `NUM_PROB_SAMPLES` | 概率采样数 | `0` |
| `RUN_LABEL` | 标签 | `${DATASETS//,/+}_${SPLIT}` |
| `OUTPUT_DIR` | 输出目录 | `outputs/ltsf_eval_${RUN_LABEL}` |

目标列控制逻辑：

- `TARGET_MODE=all`：评测所有可用 target 列
- `TARGET_MODE` 配合 `TARGET_COLS`：只评测指定列
- `MAX_TARGETS`：只取前若干个 target，适合快速测试

### 4.3 示例

跑全部标准 horizon：

```bash
LTSF_ROOT=/path/to/LTSF \
DATASETS=ETTh1,ETTh2,Weather \
MODEL_NAME=Qwen/Qwen3-0.6B-Base \
CHECKPOINT_DIR=outputs/textts_ltsf_ckpt \
DEVICE=cuda \
TORCH_DTYPE=bfloat16 \
bash textts/scripts/run_ltsf_eval.sh
```

只做小规模冒烟：

```bash
DATASETS=ETTh1 \
HORIZONS=96 \
MAX_WINDOWS=8 \
MAX_TARGETS=1 \
MAX_SAMPLES=8 \
DEVICE=cpu \
LOCAL_FILES_ONLY=1 \
bash textts/scripts/run_ltsf_eval.sh
```

## 5. GIFT-Eval 评测

### 5.1 作用

`run_gift_eval.sh` 用于 GIFT-Eval，最终调用：

```bash
python3 -m textts.eval.gift_eval ...
```

支持两种协议：

- `PROTOCOL=zero-shot`
- `PROTOCOL=few-shot`

其中 `few-shot` 会在训练 split 上做轻量适配，再在 eval split 上测试。

### 5.2 环境变量

#### 数据与协议

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `GIFT_SOURCE` | 数据源，可以是 HuggingFace 数据集名，也可以是本地 `json/jsonl` | `Salesforce/GiftEval` |
| `GIFT_CONFIG` | HF dataset config | 空 |
| `TRAIN_SPLIT` | few-shot 训练 split | `train` |
| `EVAL_SPLIT` | 评测 split | `test` |
| `PROTOCOL` | `zero-shot` 或 `few-shot` | `zero-shot` |
| `DATASET_FILTER` | 只评测指定子数据集 | 空 |
| `MAX_TRAIN_RECORDS` | few-shot 最大训练样本数 | 空 |
| `MAX_EVAL_RECORDS` | 最大评测样本数 | 空 |

#### few-shot 适配

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `FEW_SHOT_RATIO` | 从训练集抽多少比例做适配 | `0.05` |
| `FEW_SHOT_STEPS` | few-shot 更新步数 | `20` |
| `FEW_SHOT_BATCH_SIZE` | batch size | `4` |
| `FEW_SHOT_LEARNING_RATE` | 学习率 | `1e-4` |
| `FEW_SHOT_USE_LORA` | 是否开启 LoRA | `0` |
| `SAVE_FEW_SHOT_CHECKPOINT` | 是否保存适配后 checkpoint | `0` |

#### SFT 上下文构造

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `SFT_CONTEXT_MODE` | 上下文模式 | `mixed` |
| `SFT_CONTEXT_CACHE` | L2 上下文缓存文件 | 空 |

`SFT_CONTEXT_MODE` 对应当前 `sft_dataset.py` 的上下文策略，可取：

- `mixed`
- `l0`
- `l1`
- `l2`
- `all`

### 5.3 示例

zero-shot：

```bash
GIFT_SOURCE=Salesforce/GiftEval \
PROTOCOL=zero-shot \
MODEL_NAME=Qwen/Qwen3-0.6B-Base \
CHECKPOINT_DIR=outputs/textts_gift_ckpt \
DEVICE=cuda \
NUM_PROB_SAMPLES=16 \
bash textts/scripts/run_gift_eval.sh
```

few-shot：

```bash
GIFT_SOURCE=/path/to/gift_eval.jsonl \
PROTOCOL=few-shot \
FEW_SHOT_RATIO=0.1 \
FEW_SHOT_STEPS=50 \
FEW_SHOT_BATCH_SIZE=8 \
FEW_SHOT_LEARNING_RATE=5e-5 \
FEW_SHOT_USE_LORA=1 \
SFT_CONTEXT_MODE=mixed \
SFT_CONTEXT_CACHE=/path/to/context_cache.jsonl \
SAVE_FEW_SHOT_CHECKPOINT=1 \
DEVICE=cuda \
bash textts/scripts/run_gift_eval.sh
```

快速子集测试：

```bash
DATASET_FILTER=m4_hourly \
MAX_TRAIN_RECORDS=64 \
MAX_EVAL_RECORDS=32 \
MAX_SAMPLES=32 \
DEVICE=cpu \
bash textts/scripts/run_gift_eval.sh
```

## 6. 统一入口 run_benchmark_eval.sh

### 6.1 作用

`run_benchmark_eval.sh` 当前只统一收口两类 benchmark：

1. `GIFT-Eval`
2. `LTSF benchmark`

它不会调用 `run_timemmd_eval.sh`。

### 6.2 MODE 规则

| `MODE` | 含义 |
| --- | --- |
| `gift` | 只跑 GIFT-Eval |
| `ltsf` | 只跑 LTSF |
| `all` | 依次跑 GIFT 和 LTSF |
| `auto` | 按环境变量自动推断 |

`MODE=auto` 的推断逻辑：

- `GIFT_SOURCE` 和 `LTSF_ROOT` 都非空 -> `all`
- 只有 `GIFT_SOURCE` 非空 -> `gift`
- 只有 `LTSF_ROOT` 非空 -> `ltsf`
- 都没配 -> `all`

### 6.3 统一入口会透传的公共变量

| 环境变量 | 说明 |
| --- | --- |
| `CHECKPOINT_DIR` | 同时透传给 GIFT / LTSF |
| `TEXTTS_MODULES_PATH` | 同时透传给 GIFT / LTSF |
| `MODEL_NAME` | 同时透传给 GIFT / LTSF |
| `DEVICE` | 同时透传给 GIFT / LTSF |
| `TORCH_DTYPE` | 同时透传给 GIFT / LTSF |
| `DEVICE_MAP` | 同时透传给 GIFT / LTSF |
| `LOCAL_FILES_ONLY` | 同时透传给 GIFT / LTSF |

输出目录使用：

| 环境变量 | 默认值 |
| --- | --- |
| `GIFT_OUTPUT_DIR` | `outputs/${RUN_LABEL}_gift` |
| `LTSF_OUTPUT_DIR` | `outputs/${RUN_LABEL}_ltsf` |

### 6.4 示例

同时跑两套 benchmark：

```bash
MODE=all \
RUN_LABEL=textts_benchmark_v1 \
GIFT_SOURCE=Salesforce/GiftEval \
LTSF_ROOT=/path/to/LTSF \
MODEL_NAME=Qwen/Qwen3-0.6B-Base \
CHECKPOINT_DIR=outputs/textts_ckpt \
DEVICE=cuda \
TORCH_DTYPE=bfloat16 \
bash textts/scripts/run_benchmark_eval.sh
```

只跑 GIFT：

```bash
MODE=gift \
GIFT_SOURCE=Salesforce/GiftEval \
GIFT_OUTPUT_DIR=outputs/gift_only_eval \
bash textts/scripts/run_benchmark_eval.sh
```

只跑 LTSF：

```bash
MODE=ltsf \
LTSF_ROOT=/path/to/LTSF \
DATASETS=ETTh1,Weather \
LTSF_OUTPUT_DIR=outputs/ltsf_only_eval \
bash textts/scripts/run_benchmark_eval.sh
```

## 7. 变量选择建议

如果只是验证链路是否通：

- `DEVICE=cpu`
- `LOCAL_FILES_ONLY=1`
- `MAX_SAMPLES=8`
- LTSF 再加 `MAX_WINDOWS=8`
- GIFT 再加 `MAX_EVAL_RECORDS=32`

如果是正式评测：

- 明确指定 `CHECKPOINT_DIR`
- 明确指定 `OUTPUT_DIR` 或 `RUN_LABEL`
- GPU 环境下设置 `DEVICE=cuda`
- 视模型情况设置 `TORCH_DTYPE=bfloat16` 或 `float16`
- GIFT 概率评测时设置 `NUM_PROB_SAMPLES`

## 8. 常见问题

### 8.1 本地模型找不到

如果看到类似“找不到本地模型缓存”的报错，通常是：

- `LOCAL_FILES_ONLY=1`
- 但本地还没有 `MODEL_NAME` 对应的模型权重

处理方式：

- 改成 `LOCAL_FILES_ONLY=0` 并联网下载
- 或先把模型缓存到本地

### 8.2 数据路径不对

优先检查：

- `TIMEMMD_ROOT`
- `LTSF_ROOT`
- `GIFT_SOURCE`
- `CHECKPOINT_DIR`

统一入口不会替你修正这些路径，只负责把变量转发到子脚本。

### 8.3 想统一跑 Time-MMD + benchmark

目前 `run_benchmark_eval.sh` 只收口 `gift` 和 `ltsf`。如果要同时跑 Time-MMD，需要额外再执行一次：

```bash
bash textts/scripts/run_timemmd_eval.sh
```

## 9. 最小工作流

推荐的最小使用顺序：

1. 先用 `MAX_SAMPLES` / `MAX_WINDOWS` 做冒烟测试
2. 确认模型路径、数据路径、输出路径都正常
3. 再扩大到正式 benchmark
4. 如果要统一跑公开 benchmark，用 `run_benchmark_eval.sh`
5. 如果要补充时序多模态 domain 评测，再单独跑 `run_timemmd_eval.sh`
