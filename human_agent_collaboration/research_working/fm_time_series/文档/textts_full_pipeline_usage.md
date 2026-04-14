# TextTS Full Pipeline 脚本使用说明

本文说明如何使用：

- [textts/scripts/run_timemmd_full_pipeline.sh](../textts/scripts/run_timemmd_full_pipeline.sh)
- README 速查版：[textts_full_pipeline_readme.md](./textts_full_pipeline_readme.md)

这个脚本会把 `Time-MMD` 上的完整训练和评估流程按顺序串起来。

## 1. 脚本做什么

脚本固定执行 4 个阶段：

1. `CPT / pretrain`
2. `post-pretrain eval`
3. `SFT`
4. `post-SFT eval`

其中：

- 第 3 步会自动从第 1 步产出的 checkpoint 继续训练
- 第 2 步评估使用第 1 步的 checkpoint
- 第 4 步评估使用第 3 步的 checkpoint

脚本本身主要是对下面三个已有脚本的串联封装：

- [run_timemmd_pretrain.sh](../textts/scripts/run_timemmd_pretrain.sh)
- [run_timemmd_sft.sh](../textts/scripts/run_timemmd_sft.sh)
- [run_timemmd_eval.sh](../textts/scripts/run_timemmd_eval.sh)

---

## 2. 最简单用法

### 2.1 单域运行

```bash
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

如果不传 domain，默认是：

```bash
Energy
```

### 2.2 多域联合运行

```bash
DOMAINS="Energy Climate Traffic" bash textts/scripts/run_timemmd_full_pipeline.sh
```

在 joint 模式下，不需要再传单个位置参数。

---

## 3. 常用环境变量

### 3.1 通用数据与模型参数

- `TIMEMMD_ROOT`
- `TARGET_COL`
- `LOOKBACK`
- `HORIZON`
- `STRIDE`
- `MAX_WINDOWS`
- `BATCH_SIZE`
- `MODEL_NAME`
- `TORCH_DTYPE`
- `DEVICE_MAP`
- `DEVICE`
- `LOCAL_FILES_ONLY`

示例：

```bash
DEVICE=cuda \
LOOKBACK=96 \
HORIZON=24 \
MAX_WINDOWS=64 \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

### 3.2 数据切分参数

- `USE_FIXED_SPLITS`
- `TRAIN_SPLIT`
- `PRETRAIN_EVAL_SPLIT`
- `SFT_EVAL_SPLIT`
- `FINAL_EVAL_SPLIT`
- `SPLIT_VAL_RATIO`
- `SPLIT_TEST_RATIO`

默认含义：

- 训练阶段通常用 `train`
- 训练内评估通常用 `val`
- 独立最终评估默认用 `test`

---

## 4. 各阶段训练参数

### 4.1 CPT 阶段

- `PRETRAIN_STEPS`
- `PRETRAIN_LEARNING_RATE`
- `PRETRAIN_PRED_PROBABILITY`
- `PRETRAIN_EVAL_RATIO`
- `PRETRAIN_EVAL_MAX_SAMPLES`

示例：

```bash
PRETRAIN_STEPS=200 \
PRETRAIN_LEARNING_RATE=1e-4 \
PRETRAIN_PRED_PROBABILITY=0.7 \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

### 4.2 SFT 阶段

- `SFT_STEPS`
- `SFT_LEARNING_RATE`
- `SFT_USE_LORA`
- `SFT_EVAL_RATIO`
- `SFT_EVAL_MAX_SAMPLES`

示例：

```bash
SFT_STEPS=100 \
SFT_LEARNING_RATE=5e-5 \
SFT_USE_LORA=1 \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

---

## 5. 两次独立评估参数

### 5.1 CPT 后评估

- `POST_PRETRAIN_MAX_SAMPLES`
- `POST_PRETRAIN_POINT_STRATEGY`
- `POST_PRETRAIN_POINT_TEMPERATURE`
- `POST_PRETRAIN_POINT_TOP_P`
- `POST_PRETRAIN_NUM_PROB_SAMPLES`
- `POST_PRETRAIN_PROB_TEMPERATURE`
- `POST_PRETRAIN_PROB_TOP_P`

### 5.2 SFT 后评估

- `POST_SFT_MAX_SAMPLES`
- `POST_SFT_POINT_STRATEGY`
- `POST_SFT_POINT_TEMPERATURE`
- `POST_SFT_POINT_TOP_P`
- `POST_SFT_NUM_PROB_SAMPLES`
- `POST_SFT_PROB_TEMPERATURE`
- `POST_SFT_PROB_TOP_P`

如果你只需要点预测，不需要概率评估，可以把：

```bash
POST_PRETRAIN_NUM_PROB_SAMPLES=0
POST_SFT_NUM_PROB_SAMPLES=0
```

---

## 6. 输出目录

脚本默认会产出 4 组目录：

- `PRETRAIN_OUTPUT_DIR`
- `POST_PRETRAIN_EVAL_OUTPUT_DIR`
- `SFT_OUTPUT_DIR`
- `POST_SFT_EVAL_OUTPUT_DIR`

默认命名大致如下：

```text
outputs/timemmd_pretrain_<label>
outputs/timemmd_eval_<label>
outputs/timemmd_sft_<label>
outputs/timemmd_eval_<label>
```

常用相关变量：

- `BASE_RUN_LABEL`
- `PRETRAIN_RUN_LABEL`
- `POST_PRETRAIN_EVAL_RUN_LABEL`
- `SFT_RUN_LABEL`
- `POST_SFT_EVAL_RUN_LABEL`

如果你想把所有结果写到自定义位置，可以直接覆盖目录变量。

示例：

```bash
PRETRAIN_OUTPUT_DIR=./outputs/my_pretrain \
SFT_OUTPUT_DIR=./outputs/my_sft \
POST_PRETRAIN_EVAL_OUTPUT_DIR=./outputs/my_pretrain_eval \
POST_SFT_EVAL_OUTPUT_DIR=./outputs/my_sft_eval \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

---

## 7. 推荐命令示例

### 7.1 本地快速 smoke run

```bash
MAX_WINDOWS=4 \
PRETRAIN_STEPS=1 \
SFT_STEPS=1 \
POST_PRETRAIN_MAX_SAMPLES=2 \
POST_SFT_MAX_SAMPLES=2 \
DEVICE=cpu \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

### 7.2 单域较完整实验

```bash
DEVICE=cuda \
LOOKBACK=96 \
HORIZON=24 \
MAX_WINDOWS=128 \
PRETRAIN_STEPS=500 \
SFT_STEPS=200 \
FINAL_EVAL_SPLIT=test \
POST_PRETRAIN_NUM_PROB_SAMPLES=16 \
POST_SFT_NUM_PROB_SAMPLES=16 \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

### 7.3 多域联合实验

```bash
DOMAINS="Energy Climate Traffic" \
DEVICE=cuda \
MAX_WINDOWS=128 \
PRETRAIN_STEPS=500 \
SFT_STEPS=200 \
bash textts/scripts/run_timemmd_full_pipeline.sh
```

---

## 8. 结果怎么看

重点看这几类文件：

### 8.1 训练输出

在 checkpoint 目录下通常会有：

- `metadata.json`
- `textts_modules.pt`
- `llm/`

其中 `metadata.json` 里会保存：

- `train_metrics`
- `eval_metrics`
- 运行参数相关 metadata

### 8.2 独立评估输出

在 eval 输出目录下通常会有：

- `metrics.json`
- `predictions.jsonl`

其中：

- `metrics.json` 适合看汇总指标
- `predictions.jsonl` 适合做逐样本误差分析

---

## 9. 注意事项

### 9.1 当前评估类型

当前脚本串起来的评估是 `forecast evaluation`，不是单独的 imputation 专项评估。

### 9.2 训练内评估 vs 独立评估

这个 full pipeline 脚本同时支持两类评估：

- 训练脚本内部附带的 eval
- checkpoint 保存后的独立 eval

如果你只关心独立评估，可以把：

- `PRETRAIN_EVAL_RATIO=0`
- `SFT_EVAL_RATIO=0`

这样第 1 步和第 3 步内部不做 eval，只保留第 2 步和第 4 步独立评估。

### 9.3 joint 与单域模式

- 单域模式：优先使用位置参数或 `DOMAIN`
- 联合模式：设置 `DOMAINS`

不要同时混用相互冲突的 domain 设置。

---

## 10. 一句话总结

如果你想在当前仓库里一键跑完：

`CPT -> CPT评估 -> SFT -> SFT评估`

直接用：

```bash
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```
