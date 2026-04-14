# TextTS 当前实现 Pipeline

本文描述的是当前仓库里 `TextTS` 的实际实现 pipeline，不是设计文档中的理想方案。

## 1. 总览

当前代码中的主流程可以概括为：

`CSV / Time-MMD -> sliding windows record -> text/context + normalized TS patches -> patch encoder / channel mixer / projector -> Qwen3 causal LM -> generate forecast-bin tokens -> dequantize to numeric forecast`

从代码结构看，pipeline 主要分为 6 层：

1. 数据加载与滑窗切分
2. 样本格式化
3. batch 组织
4. 模型前向
5. 训练阶段
6. 推理与评估

---

## 2. 数据构造

原始数据会先被切成统一的 `record` 结构。

### 2.1 CSV benchmark

普通 CSV 数据通过滑动窗口切成 `lookback + horizon` 样本，生成：

- `target_history`
- `target_future`
- `covariates`
- `history_start / history_end / forecast_end`

对应实现：

- [textts/data/benchmark_loader.py](../textts/data/benchmark_loader.py)

核心入口：

- `load_csv_windows()`：见 `textts/data/benchmark_loader.py:90`

### 2.2 Time-MMD

Time-MMD loader 会同时读取：

- numerical 文件中的 target 和数值协变量
- textual 文件中的 `report` / `search` 文本事实

然后构造 `context`，并且只使用 `history_end` 之前的文本，避免未来信息泄漏。

对应实现：

- [textts/data/timemmd_loader.py](../textts/data/timemmd_loader.py)

核心入口：

- `load_timemmd_windows()`：见 `textts/data/timemmd_loader.py:274`

关键点：

- target 和 covariates 来自 numerical CSV
- 文本 context 来自 `fact` 字段
- `pred` / `preds` 被显式忽略，避免 leakage

---

## 3. 样本格式化

`TextTSSequenceFormatter` 负责把 `record` 转成模型真正消费的 sample。

对应实现：

- [textts/data/sequence_formatter.py](../textts/data/sequence_formatter.py)

核心类：

- `TextTSSequenceFormatter`：见 `textts/data/sequence_formatter.py:22`

### 3.1 文本部分

文本 prompt 当前由以下字段拼接而成：

- `[DOMAIN]`
- `[FREQ]`
- `[CONTEXT]`
- `[CHANNEL_META]`

对应实现：

- `build_text_prompt()`：见 `textts/data/sequence_formatter.py:121`

### 3.2 数值序列部分

数值序列会经过以下处理：

1. 计算目标序列统计量 `mean/std`
2. 做 RevIN 风格归一化
3. 每个时间步拼成 9 维特征
4. 按 `patch_len=16` 切成 patch

当前每步的 9 维特征是：

- `value_norm`
- `missing_flag`
- `7-dim time_feature`

对应实现：

- `_build_patch_tensor()`：见 `textts/data/sequence_formatter.py:93`

### 3.3 预测目标 token 化

`target_future` 会被量化成 forecast bin token，并追加 `<EOS_FC>`。

对应实现：

- `format_prediction_sample()`：见 `textts/data/sequence_formatter.py:142`
- `ForecastQuantizer.build_forecast_token_ids()`：见 [textts/tokenization/forecast_quantizer.py](../textts/tokenization/forecast_quantizer.py)

量化逻辑：

- 先按实例统计量归一化
- 再映射到 1024 个离散 bin
- 最终转成 `<TSV_bin_xxx>` token id

---

## 4. Batch 组织

`TextTSCollator` 负责把单条 sample pad 成 batch。

对应实现：

- [textts/data/collator.py](../textts/data/collator.py)

核心类：

- `TextTSCollator`：见 `textts/data/collator.py:28`

输出的 batch 主要包含：

- `text_input_ids`
- `text_attention_mask`
- `channel_patches`
- `channel_mask`
- `patch_mask`
- `prefix_control_token_ids`
- `forecast_token_ids`
- `forecast_attention_mask`
- `forecast_labels`
- `revin_mean`
- `revin_std`

其中：

- `channel_patches[:, 0, ...]` 是 target channel
- `channel_patches[:, 1:, ...]` 是 covariate channels

对应实现：

- `__call__()`：见 `textts/data/collator.py:42`

---

## 5. 模型前向

当前模型主干定义在：

- [textts/model/textts_model.py](../textts/model/textts_model.py)

核心类：

- `TextTSModel`：见 `textts/model/textts_model.py:48`

### 5.1 编码主链路

数值 patch 的编码路径是：

`patch_encoder -> channel_mixer -> projector -> Qwen3 hidden space`

具体过程：

1. `patch_encoder` 对 target / covariate patch 编码
2. `channel_mixer` 做跨通道融合
3. `projector` 投影到 Qwen3 的 hidden size

对应实现：

- `_encode_channels()`：见 `textts/model/textts_model.py:107`

### 5.2 prefix 构造

当前 prefix 的拼接顺序是：

`text tokens -> covariate latents -> <TARGET_START> -> target latents -> <BOS_FC / BOS_IMP>`

对应实现：

- `_build_prefix_only()`：见 `textts/model/textts_model.py:148`

这意味着当前实现里，文本和连续时序 latent 是通过 `inputs_embeds` 混合后一起送进 Qwen3 的。

### 5.3 训练时的完整输入

训练时会在 prefix 后面再拼接：

- `forecast_token_ids`

最终形成：

`prefix embeddings + forecast token embeddings`

对应实现：

- `_build_training_batch()`：见 `textts/model/textts_model.py:213`

### 5.4 loss 计算

当前训练只在 forecast 输出位置计算自回归交叉熵，prefix 部分全部用 `-100` mask 掉。

同时，对 forecast 位置的 logits 会加 vocab mask，只允许：

- forecast bins
- `<EOS_FC>`
- `<FORECAST_PAD>`

对应实现：

- `_apply_forecast_mask_to_shifted_logits()`：见 `textts/model/textts_model.py:257`
- `forward()`：见 `textts/model/textts_model.py:271`

---

## 6. 训练阶段

当前实现里有两条训练支线：

- CPT / pretrain
- SFT

### 6.1 Pretrain / CPT

入口：

- [textts/training/pretrain.py](../textts/training/pretrain.py)

核心逻辑：

- `load_records_from_args()` 读取 CSV 或 Time-MMD 数据
- `build_pretrain_datasets()` 生成 prediction dataset 和 imputation dataset
- `MixedBatchSampler` 在 batch 级别混合两类任务
- `TextTSPretrainer` 执行标准训练循环

关键位置：

- dataloader 构造：`textts/training/pretrain.py:101`
- trainer：`textts/training/pretrain.py:131`
- main 流程：`textts/training/pretrain.py:233`

当前 CPT 混合的两个任务是：

1. prediction
2. imputation

其中 imputation 的做法是：

- 随机 mask 掉部分 target patches
- 用 `<BOS_IMP>` 指示任务类型
- 预测被 mask 掉的值对应的 forecast token

对应实现：

- `format_imputation_sample()`：见 `textts/data/sequence_formatter.py:185`

### 6.2 SFT

入口：

- [textts/training/sft.py](../textts/training/sft.py)

SFT 与 pretrain 的主要区别不在模型结构，而在于：

- 会先对原始 record 做 context-level 构造
- 然后再复用 prediction 形式进行训练

关键位置：

- SFT dataset 构造：`textts/training/sft.py:294`
- trainer：`textts/training/sft.py:116`
- main 流程：`textts/training/sft.py:235`

#### SFT 的 context pipeline

SFT 当前支持三类 context：

- `L0`：空 context
- `L1`：基于 metadata 和简单统计量的模板 context
- `L2`：从 JSONL cache 中读取 rich context，失败时 fallback 到 record context 或 L1

对应实现：

- [textts/data/sft_dataset.py](../textts/data/sft_dataset.py)

关键位置：

- `SFTDatasetConfig`：`textts/data/sft_dataset.py:37`
- `build_template_context()`：`textts/data/sft_dataset.py:183`
- `resolve_l2_context()`：`textts/data/sft_dataset.py:216`
- `build_sft_records()`：`textts/data/sft_dataset.py:282`

当前 `sft.py` 默认支持：

- `mixed`
- `l0`
- `l1`
- `l2`
- `all`

另外，SFT 支持可选 LoRA：

- `maybe_apply_lora()`：见 `textts/training/sft.py:84`

---

## 7. 推理与评估

评估入口：

- [textts/eval/forecast_eval.py](../textts/eval/forecast_eval.py)

核心流程：

1. 对每条 record 调 `formatter.format_prediction_sample()`
2. 用 `collator([sample])` 构成 batch
3. 调 `model.generate_single()` 自回归生成 forecast token
4. 把 token 反量化回数值
5. 计算指标

关键位置：

- `evaluate_forecast_records()`：见 `textts/eval/forecast_eval.py:151`

### 7.1 推理

当前单样本推理逻辑在：

- `generate_single()`：见 `textts/model/textts_model.py:333`

生成方式支持：

- `greedy`
- `sample`

每一步都会对 logits 加 forecast vocab mask，只允许生成合法 forecast token。

### 7.2 反量化

生成出来的 token 会通过 `decode_forecast_token_ids()` 转成 bin id，再反量化回原始数值空间。

对应实现：

- `decode_forecast_token_ids()`：见 `textts/eval/forecast_eval.py:51`

### 7.3 评估指标

当前实现支持：

- `MAE`
- `MSE`
- `RMSE`
- `MAPE`
- `SMAPE`

如果启用概率采样，还支持：

- `CRPS`
- `coverage_80`
- `interval_width_80`

对应实现：

- `summarize_eval_outputs()`：见 `textts/eval/forecast_eval.py:115`

---

## 8. 脚本层默认工作流

当前仓库中的默认脚本工作流是：

1. `run_timemmd_pretrain.sh`
2. `run_timemmd_sft.sh`
3. `run_timemmd_eval.sh`
4. `run_timemmd_full_pipeline.sh`

对应文件：

- [textts/scripts/run_timemmd_pretrain.sh](../textts/scripts/run_timemmd_pretrain.sh)
- [textts/scripts/run_timemmd_sft.sh](../textts/scripts/run_timemmd_sft.sh)
- [textts/scripts/run_timemmd_eval.sh](../textts/scripts/run_timemmd_eval.sh)
- [textts/scripts/run_timemmd_full_pipeline.sh](../textts/scripts/run_timemmd_full_pipeline.sh)

这三步分别对应：

1. CPT 预训练
2. SFT 微调
3. 预测评估

其中新增的 `run_timemmd_full_pipeline.sh` 会把以下 4 步显式串起来：

1. CPT
2. post-pretrain eval
3. SFT
4. post-SFT eval

脚本使用说明见：

- [文档/textts_full_pipeline_usage.md](./textts_full_pipeline_usage.md)

跨域预训练与 batch 组织说明见：

- [文档/textts_cross_domain_batching.md](./textts_cross_domain_batching.md)

---

## 9. 一句话总结

当前仓库中的 TextTS 实现，本质上是一个：

“把文本上下文和连续时序 patch latent 一起拼成 `inputs_embeds`，送入 Qwen3 做自回归 forecast token 生成，再把 forecast token 反量化回数值预测”的多模态时序生成模型。

如果只看已经落地的代码，那么它当前的真实 pipeline 就是：

`数据加载 -> record 构造 -> formatter -> collator -> patch encoder/channel mixer/projector -> Qwen3 -> forecast token generation -> dequantization -> metrics`
