# Time-MMD Experiment 1 使用说明

本文说明专用实验脚本：

- `textts/scripts/run_timemmd_experiment_1.sh`

这个脚本不是通用训练入口，而是为下面这组固定实验协议准备的：

1. 在 `Time-MMD train split` 上做跨域自监督预训练
2. 用上一步 checkpoint 在 `Time-MMD test split` 上评估
3. 加载上一步 checkpoint，在 `Time-MMD train split` 上做 SFT
4. 用 SFT 后 checkpoint 在 `Time-MMD test split` 上再次评估

## 1. 脚本做什么

脚本内部会顺序调用：

1. `run_timemmd_full_pipeline.sh`
2. 在运行前先做一次 domain preflight
3. 在运行后自动写出实验清单和汇总文件

和通用脚本相比，它固定了三件事：

- 固定 protocol：`train -> val -> test`
- 固定输出根目录：`outputs/timemmd_experiment_1`
- 固定结果汇总文件结构

## 2. 固定 protocol

这个实验脚本默认强制使用：

- `TRAIN_SPLIT=train`
- `PRETRAIN_EVAL_SPLIT=val`
- `SFT_EVAL_SPLIT=val`
- `FINAL_EVAL_SPLIT=test`
- `SPLIT_VAL_RATIO=0.1`
- `SPLIT_TEST_RATIO=0.1`
- `USE_FIXED_SPLITS=1`

也就是说，它表达的是：

- 预训练使用 `train`
- SFT 使用 `train`
- 中间训练内评估使用 `val`
- 最终对比结果使用 `test`

## 3. Domain Preflight

### 3.1 为什么需要 preflight

Time-MMD 不同 domain 的数据质量并不完全一致。当前 loader 对 target 列有两个硬要求：

- `target_col` 必须是数值列
- 至少要能在 `train` 和 `test` 上各切出一个合法窗口

如果某个 domain 不满足这些条件，直接 joint training / eval 会中途报错。

### 3.2 preflight 做什么

脚本会先扫描 `Time-MMD` 下所有 candidate domains，然后对每个 domain 检查：

1. `train split` 是否能产出至少一个窗口
2. `test split` 是否能产出至少一个窗口
3. `target_col` 是否可正常读取

检查通过的域会进入真正实验；失败的域会被排除，并把原因写入：

- `outputs/timemmd_experiment_1/domain_selection.json`

### 3.3 当前一次实际 smoke run 的结果

在当前仓库数据和默认参数下，脚本实际选出的 included domains 是：

- `Agriculture`
- `Climate`
- `Economy`
- `Energy`
- `Environment`
- `Health_US`
- `Traffic`

被排除的域是：

- `Health_AFR`
- `Security`
- `SocialGood`

排除原因会被结构化记录在 `domain_selection.json` 和 `manifest.json` 里。

## 4. 最简单用法

在仓库根目录执行：

```bash
cd /Users/chengmingyue/human_agent_collaboration/research_working/fm_time_series
```

最小 smoke run：

```bash
TIMEMMD_ROOT=/Users/chengmingyue/human_agent_collaboration/research_working/fm_time_series/数据集/Time-MMD \
MODEL_NAME=Qwen/Qwen3-0.6B-Base \
LOCAL_FILES_ONLY=1 \
DEVICE=cpu \
PRETRAIN_STEPS=1 \
SFT_STEPS=1 \
MAX_WINDOWS=1 \
POST_PRETRAIN_MAX_SAMPLES=1 \
POST_SFT_MAX_SAMPLES=1 \
POST_PRETRAIN_NUM_PROB_SAMPLES=0 \
POST_SFT_NUM_PROB_SAMPLES=0 \
bash textts/scripts/run_timemmd_experiment_1.sh
```

## 5. 常用可调参数

虽然 protocol 和输出结构固定了，但下面这些超参仍然保留为环境变量：

### 5.1 数据与模型

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

### 5.2 预训练阶段

- `PRETRAIN_STEPS`
- `PRETRAIN_LEARNING_RATE`
- `PRETRAIN_PRED_PROBABILITY`
- `PRETRAIN_EVAL_RATIO`
- `PRETRAIN_EVAL_MAX_SAMPLES`

### 5.3 SFT 阶段

- `SFT_STEPS`
- `SFT_LEARNING_RATE`
- `SFT_USE_LORA`
- `SFT_EVAL_RATIO`
- `SFT_EVAL_MAX_SAMPLES`

### 5.4 两次最终评估

- `POST_PRETRAIN_MAX_SAMPLES`
- `POST_PRETRAIN_POINT_STRATEGY`
- `POST_PRETRAIN_POINT_TEMPERATURE`
- `POST_PRETRAIN_POINT_TOP_P`
- `POST_PRETRAIN_NUM_PROB_SAMPLES`
- `POST_PRETRAIN_PROB_TEMPERATURE`
- `POST_PRETRAIN_PROB_TOP_P`
- `POST_SFT_MAX_SAMPLES`
- `POST_SFT_POINT_STRATEGY`
- `POST_SFT_POINT_TEMPERATURE`
- `POST_SFT_POINT_TOP_P`
- `POST_SFT_NUM_PROB_SAMPLES`
- `POST_SFT_PROB_TEMPERATURE`
- `POST_SFT_PROB_TOP_P`

## 6. 输出目录

这个实验脚本固定把所有产物写到：

```text
outputs/timemmd_experiment_1
```

目录结构如下：

```text
outputs/timemmd_experiment_1/
  domain_selection.json
  manifest.json
  summary.json
  summary.tsv
  pretrain_checkpoint/
  post_pretrain_eval/
  sft_checkpoint/
  post_sft_eval/
```

其中：

- `pretrain_checkpoint/`：Step 1 预训练 checkpoint
- `post_pretrain_eval/`：Step 1 后在 test split 的评测结果
- `sft_checkpoint/`：Step 2 SFT checkpoint
- `post_sft_eval/`：Step 2 后在 test split 的评测结果

## 7. 结果文件说明

### 7.1 `domain_selection.json`

记录 preflight 的结果，包括：

- 所有 candidate domains
- 最终纳入实验的 domains
- 被排除的 domains 及原因
- 当前使用的 `target_col / lookback / horizon / stride`

这个文件的作用是：

- 解释为什么某些域没有进入实验
- 固化“本次实验实际用的是哪些域”

### 7.2 `manifest.json`

这是实验协议清单，主要记录：

- 实验名
- 固定 protocol
- 实际纳入的 domains
- 被排除的 domains
- `Time-MMD` 数据根目录
- domain preflight 的详细结果

这个文件更像“实验配置声明”。

### 7.3 `summary.json`

这是最终实验汇总，主要包括：

- protocol
- domain selection
- 四个阶段的路径
- `pretrain` metadata
- `post_pretrain_eval` metrics
- `sft` metadata
- `post_sft_eval` metrics
- `delta_post_sft_minus_post_pretrain`

其中 `delta_post_sft_minus_post_pretrain` 是最重要的对比字段，表示：

- SFT 后指标减去 pretrain 后指标

它只是数值差，不自动判断“变好还是变差”，解释时需要结合指标方向。

### 7.4 `summary.tsv`

这是一个扁平表，方便直接导入表格或后续拼 benchmark 表。

当前每一行对应一个最终评估阶段：

- `post_pretrain_eval`
- `post_sft_eval`

列里包含：

- 指标值
- 对应评测目录
- 对应 checkpoint 目录

## 8. 已知注意事项

### 8.1 `DOMAIN` / `DOMAINS` 不再由用户控制

这个脚本会忽略外部传入的：

- `DOMAIN`
- `DOMAINS`

因为它要自己做 preflight 和固定实验 protocol。

### 8.2 不是所有 Time-MMD 域都会自动进实验

如果某个域：

- target 缺失
- test split 切不出窗口
- 当前 `lookback / horizon` 组合过大

它就会被自动排除。

因此 changing `LOOKBACK` / `HORIZON` 可能改变最终纳入的域集合。

### 8.3 这是实验协议脚本，不是通用脚本

如果你的需求是：

- 单域训练
- 手工指定若干域
- 只跑 pretrain 不跑 SFT
- 做 leave-one-domain-out

应该回到通用脚本：

- `textts/scripts/run_timemmd_pretrain.sh`
- `textts/scripts/run_timemmd_sft.sh`
- `textts/scripts/run_timemmd_eval.sh`
- `textts/scripts/run_timemmd_full_pipeline.sh`

## 9. 推荐使用方式

建议分两步：

1. 先用 smoke 配置确认 domain selection 和输出结构正常
2. 再增大 `PRETRAIN_STEPS / SFT_STEPS / MAX_WINDOWS / MAX_SAMPLES`

一个更完整的本地命令示例：

```bash
TIMEMMD_ROOT=/Users/chengmingyue/human_agent_collaboration/research_working/fm_time_series/数据集/Time-MMD \
MODEL_NAME=Qwen/Qwen3-0.6B-Base \
LOCAL_FILES_ONLY=1 \
DEVICE=cpu \
LOOKBACK=32 \
HORIZON=8 \
MAX_WINDOWS=8 \
PRETRAIN_STEPS=20 \
SFT_STEPS=20 \
POST_PRETRAIN_MAX_SAMPLES=8 \
POST_SFT_MAX_SAMPLES=8 \
POST_PRETRAIN_NUM_PROB_SAMPLES=0 \
POST_SFT_NUM_PROB_SAMPLES=0 \
bash textts/scripts/run_timemmd_experiment_1.sh
```
