# TextTS Full Pipeline README

快速入口：

- 脚本：[textts/scripts/run_timemmd_full_pipeline.sh](../textts/scripts/run_timemmd_full_pipeline.sh)
- 详细说明：[textts_full_pipeline_usage.md](./textts_full_pipeline_usage.md)

这个脚本会顺序执行：

1. `CPT`
2. `post-pretrain eval`
3. `SFT`
4. `post-SFT eval`

## Quick Start

### 单域最小示例

```bash
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

### 单域 smoke test

```bash
MAX_WINDOWS=4 \
PRETRAIN_STEPS=1 \
SFT_STEPS=1 \
POST_PRETRAIN_MAX_SAMPLES=2 \
POST_SFT_MAX_SAMPLES=2 \
DEVICE=cpu \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

### 单域常用实验

```bash
DEVICE=cuda \
LOOKBACK=96 \
HORIZON=24 \
MAX_WINDOWS=128 \
PRETRAIN_STEPS=500 \
SFT_STEPS=200 \
POST_PRETRAIN_NUM_PROB_SAMPLES=16 \
POST_SFT_NUM_PROB_SAMPLES=16 \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

### 多域联合实验

```bash
DOMAINS="Energy Climate Traffic" \
DEVICE=cuda \
MAX_WINDOWS=128 \
PRETRAIN_STEPS=500 \
SFT_STEPS=200 \
bash textts/scripts/run_timemmd_full_pipeline.sh
```

## 常用参数

### 基础参数

- `TIMEMMD_ROOT`
- `TARGET_COL`
- `LOOKBACK`
- `HORIZON`
- `STRIDE`
- `MAX_WINDOWS`
- `BATCH_SIZE`
- `MODEL_NAME`
- `DEVICE`
- `LOCAL_FILES_ONLY`

### CPT 参数

- `PRETRAIN_STEPS`
- `PRETRAIN_LEARNING_RATE`
- `PRETRAIN_PRED_PROBABILITY`

### SFT 参数

- `SFT_STEPS`
- `SFT_LEARNING_RATE`
- `SFT_USE_LORA`

### 独立评估参数

- `FINAL_EVAL_SPLIT`
- `POST_PRETRAIN_MAX_SAMPLES`
- `POST_SFT_MAX_SAMPLES`
- `POST_PRETRAIN_NUM_PROB_SAMPLES`
- `POST_SFT_NUM_PROB_SAMPLES`

如果只要点预测，不做概率评估：

```bash
POST_PRETRAIN_NUM_PROB_SAMPLES=0 \
POST_SFT_NUM_PROB_SAMPLES=0 \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

## 输出目录

默认会生成 4 组结果：

- `PRETRAIN_OUTPUT_DIR`
- `POST_PRETRAIN_EVAL_OUTPUT_DIR`
- `SFT_OUTPUT_DIR`
- `POST_SFT_EVAL_OUTPUT_DIR`

常见文件：

- checkpoint 目录：`metadata.json`、`textts_modules.pt`、`llm/`
- eval 目录：`metrics.json`、`predictions.jsonl`

## 自定义输出目录

```bash
PRETRAIN_OUTPUT_DIR=./outputs/my_pretrain \
POST_PRETRAIN_EVAL_OUTPUT_DIR=./outputs/my_pretrain_eval \
SFT_OUTPUT_DIR=./outputs/my_sft \
POST_SFT_EVAL_OUTPUT_DIR=./outputs/my_sft_eval \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

## 备注

- 第 3 步 SFT 默认从第 1 步 CPT checkpoint 继续。
- 第 2 步和第 4 步是独立 eval，不依赖训练脚本内部的 eval。
- 当前评估是 `forecast eval`，不是单独的 imputation 专项评估。
