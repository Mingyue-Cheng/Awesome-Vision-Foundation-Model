# TextTS 跨域预训练与 Batch 组织说明

本文整理当前代码中与以下问题相关的实现细节：

- `Time-MMD` 是否支持跨域预训练
- 跨域预训练时 batch 是怎么混的
- `MixedBatchSampler` 的行为是什么
- `text_input_ids / channel_patches / forecast_token_ids` 在 batch 里具体长什么样

本文基于当前仓库中的真实代码实现，不是设计文档中的理想方案。

---

## 1. 当前代码是否支持 Time-MMD 跨域预训练

支持。

当前 `pretrain.py` 已经支持通过 `--domains` 读取多个 `Time-MMD` domain，并进行 joint pretrain。

相关实现：

- [textts/training/pretrain.py](../textts/training/pretrain.py)
- [textts/data/timemmd_loader.py](../textts/data/timemmd_loader.py)
- [textts/scripts/run_timemmd_pretrain_joint.sh](../textts/scripts/run_timemmd_pretrain_joint.sh)

关键逻辑：

1. `pretrain.py` 接收 `--domains`
2. 调 `load_timemmd_multi_domain_windows()`
3. 各个 domain 的 windows 被合并成一个 joint record 列表
4. 在这个 joint dataset 上做 CPT 训练

相关位置：

- `textts/training/pretrain.py:196`
- `textts/training/pretrain.py:262`
- `textts/data/timemmd_loader.py:365`

示例命令：

```bash
DOMAINS="Energy Climate Traffic" bash textts/scripts/run_timemmd_pretrain_joint.sh
```

或者：

```bash
DOMAINS="Energy Climate Traffic" bash textts/scripts/run_timemmd_pretrain.sh
```

---

## 2. 当前跨域预训练是怎么混 domain 的

当前实现不是 domain-balanced sampler，也不是 curriculum。  
它的逻辑更接近：

1. 每个 domain 单独加载 windows
2. 把多个 domain 的 `record` 合并
3. 对合并后的 records 做全局 shuffle
4. 在联合样本池上再做任务级别的 batch 混合

也就是说，domain 层面的处理是：

```text
per-domain load -> merge -> shuffle
```

而不是：

```text
per-domain sampler -> batch 内强制域均衡
```

---

## 3. 跨域预训练完整采样流程

按当前代码，跨域预训练的完整采样过程可以概括为：

```text
多域 Time-MMD windows
    ->
merge 成 merged train_records
    ->
global shuffle
    ->
基于同一份 merged records 派生出：
    - PredictionDataset
    - ImputationDataset
    ->
MixedBatchSampler 在 batch 级别决定当前 batch 做 pred 还是 imp
    ->
从对应任务 dataset 中抽样本
    ->
collator 组装成张量 batch
    ->
送入 TextTSModel
```

---

## 4. `MixedTaskDataset` 和 `MixedBatchSampler` 的真实行为

相关代码：

- [textts/data/sequence_sampler.py](../textts/data/sequence_sampler.py)

### 4.1 `MixedTaskDataset`

`MixedTaskDataset` 本身不负责采样，只负责路由。

它接收的 key 形式是：

- `("pred", idx)`
- `("imp", idx)`

然后分别去：

- `pred_dataset[idx]`
- `imp_dataset[idx]`

取出对应样本。

相关位置：

- `textts/data/sequence_sampler.py:17`

### 4.2 `MixedBatchSampler`

`MixedBatchSampler` 是 batch 级采样器。  
它的核心行为是：

1. 先决定当前整个 batch 属于哪种任务
2. 再从该任务对应的数据集中抽 `batch_size` 个 index
3. 最后产出一组 tuple key

注意：

- 一个 batch 要么全是 `pred`
- 要么全是 `imp`
- 不会在同一个 batch 里混两种任务

相关位置：

- `textts/data/sequence_sampler.py:49`

---

## 5. `MixedBatchSampler` 伪代码

按当前实现，可以写成如下伪代码：

```python
pred_dataset = PredictionDataset(train_records)
imp_dataset = ImputationDataset(train_records)

for batch_id in range(num_batches_per_epoch):
    # 先决定整个 batch 的任务类型
    if random() < pred_probability:
        task = "pred"
        dataset = pred_dataset
    else:
        task = "imp"
        dataset = imp_dataset

    # 再从对应任务 dataset 中抽 index
    if batch_size <= len(dataset):
        batch_indices = sample_without_replacement(len(dataset), batch_size)
    else:
        batch_indices = sample_with_replacement(len(dataset), batch_size)

    # 编码成 tuple key
    batch_keys = [(task, idx) for idx in batch_indices]

    # MixedTaskDataset 再根据 key 去取样本
    batch_samples = [MixedTaskDataset[key] for key in batch_keys]

    yield collate(batch_samples)
```

再补上跨域上游逻辑，可以近似写成：

```python
all_records = []
for domain in domains:
    domain_records = load_timemmd_windows(domain)
    all_records.extend(domain_records)

shuffle(all_records)

pred_dataset = PredictionDataset(all_records)
imp_dataset = ImputationDataset(all_records)

for each_batch:
    task = bernoulli(pred_probability)  # pred or imp
    indices = random_sample_from(task_dataset)
    batch = [task_dataset[idx] for idx in indices]
```

---

## 6. `MixedBatchSampler` 的几个关键点

### 6.1 domain 不参与 batch 决策

当前 batch 采样阶段不会先抽 domain。  
domain 信息只是每条 `record` 自带的字段，已经在 merged records 中。

### 6.2 任务是在 batch 级切换

当前实现中：

- 一个 batch 全是 `prediction`
- 或一个 batch 全是 `imputation`

不是样本级混合。

### 6.3 小数据集时可能重复采样

如果：

```text
batch_size > dataset_size
```

采样器会退化成有放回采样。

相关位置：

- `textts/data/sequence_sampler.py:83`

### 6.4 epoch 长度可人为指定

如果设置了：

- `num_batches_per_epoch`

那么一个 epoch 的长度就是固定的 batch 数，不一定等于“完整扫一遍数据”。

相关位置：

- `textts/data/sequence_sampler.py:75`

---

## 7. 第一张图：跨域预训练采样流程

```text
                Time-MMD Multi-Domain Pretrain Sampling

   +------------------+     +------------------+     +------------------+
   | Domain A windows |     | Domain B windows |     | Domain C windows |
   +------------------+     +------------------+     +------------------+
            \                        |                         /
             \                       |                        /
              \______________________|_______________________/
                                     |
                                     v
                     +----------------------------------+
                     | merged train_records             |
                     | [A1, A2, ..., B1, B2, ..., C1] |
                     +----------------------------------+
                                     |
                                     v
                          +----------------------+
                          | global shuffle       |
                          +----------------------+
                                     |
                                     v
                +---------------------------------------------+
                | same merged records feed two task views     |
                +---------------------------------------------+
                         |                              |
                         v                              v
          +-----------------------------+   +-----------------------------+
          | TextTSPredictionDataset     |   | TextTSImputationDataset     |
          | format_prediction_sample()  |   | format_imputation_sample()  |
          +-----------------------------+   +-----------------------------+
                         |                              |
                         +--------------+---------------+
                                        |
                                        v
                         +-------------------------------+
                         | MixedTaskDataset              |
                         | route key: ("pred", i) /      |
                         |            ("imp", i)         |
                         +-------------------------------+
                                        ^
                                        |
                         +-------------------------------+
                         | MixedBatchSampler             |
                         | batch-level task selection    |
                         +-------------------------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
                    | rng.random() < pred_probability ?     |
                    |                                       |
           +--------+--------+                     +--------+--------+
           | choose "pred"   |                     | choose "imp"    |
           +-----------------+                     +-----------------+
                    |                                       |
                    v                                       v
      +-----------------------------+         +-----------------------------+
      | sample indices from pred    |         | sample indices from imp     |
      | e.g. [7, 21, 4, 19]         |         | e.g. [3, 8, 11, 2]         |
      +-----------------------------+         +-----------------------------+
                    |                                       |
                    v                                       v
      +-----------------------------+         +-----------------------------+
      | batch keys                  |         | batch keys                  |
      | [("pred",7), ("pred",21),   |         | [("imp",3), ("imp",8),     |
      |  ("pred",4), ("pred",19)]   |         |  ("imp",11), ("imp",2)]    |
      +-----------------------------+         +-----------------------------+
                    \                                       /
                     \                                     /
                      \___________________________________/
                                      |
                                      v
                         +-------------------------------+
                         | DataLoader + collator         |
                         | build actual tensor batch     |
                         +-------------------------------+
                                      |
                                      v
                         +-------------------------------+
                         | TextTSModel forward           |
                         | one whole batch = one task    |
                         +-------------------------------+
```

---

## 8. 第二张图：batch 张量结构

下面这张图说明一个 batch 在进入模型前是什么结构。

```text
                    One Mixed Batch Before Model Forward

        batch keys from MixedBatchSampler
        e.g. [("pred", 7), ("pred", 21), ("pred", 4)]
                          or
             [("imp", 3), ("imp", 8), ("imp", 11)]
                               |
                               v
                 +-------------------------------+
                 | MixedTaskDataset              |
                 | fetch raw formatted samples   |
                 +-------------------------------+
                               |
                               v
         each sample after formatter looks roughly like:

   +---------------------------------------------------------------+
   | sample_i                                                      |
   |                                                               |
   | text_input_ids: [t1, t2, t3, ...]                             |
   | target_patches: [N_tgt_patch, patch_len, input_dim]           |
   | covariate_patches: list of [N_cov_patch, patch_len, input_dim]|
   | prefix_control_token_id: <BOS_FC> or <BOS_IMP>                |
   | revin_mean: scalar                                            |
   | revin_std: scalar                                             |
   | forecast_token_ids: [bin_1, bin_2, ..., bin_H, <EOS_FC>]      |
   +---------------------------------------------------------------+

                               |
                               v
                    +---------------------------+
                    | TextTSCollator            |
                    | pad and stack samples     |
                    +---------------------------+
                               |
                               v
   batched tensors passed to model:

   +--------------------------------------------------------------------------------------------------+
   | text_input_ids        : [B, T_text_max]                                                         |
   | text_attention_mask   : [B, T_text_max]                                                         |
   |                                                                                                  |
   | channel_patches       : [B, C_max, N_patch_max, patch_len, input_dim]                           |
   |                        where:                                                                    |
   |                        - channel 0 = target                                                      |
   |                        - channel 1.. = covariates                                                |
   |                                                                                                  |
   | channel_mask          : [B, C_max]                                                               |
   |                        marks which channels exist                                                 |
   |                                                                                                  |
   | patch_mask            : [B, C_max, N_patch_max]                                                  |
   |                        marks which patches in each channel are valid                             |
   |                                                                                                  |
   | prefix_control_token_ids : [B]                                                                   |
   |                            <BOS_FC> for prediction or <BOS_IMP> for imputation                  |
   |                                                                                                  |
   | revin_mean            : [B]                                                                      |
   | revin_std             : [B]                                                                      |
   |                                                                                                  |
   | forecast_token_ids    : [B, H_max]                                                               |
   | forecast_attention_mask: [B, H_max]                                                              |
   | forecast_labels       : [B, H_max]                                                               |
   +--------------------------------------------------------------------------------------------------+
                               |
                               v
                    +---------------------------+
                    | TextTSModel               |
                    +---------------------------+
                               |
             +-----------------+------------------+
             |                                    |
             v                                    v
   text_input_ids                        channel_patches
   -> token embeddings                   -> patch encoder
                                          -> channel mixer
                                          -> projector
             \                                    /
              \                                  /
               \________________________________/
                               |
                               v
         model builds mixed prefix in this order:

   [text tokens]
        +
   [covariate latents]
        +
   [<TARGET_START>]
        +
   [target latents]
        +
   [<BOS_FC> or <BOS_IMP>]
                               |
                               v
         if training, append forecast token embeddings:

   [prefix embeddings] + [forecast_token_embeddings]
                               |
                               v
                        Qwen3 forward
                               |
                               v
                 loss only on forecast positions
```

---

## 9. `channel_patches` 再展开一层

`channel_patches` 在每个样本内可以理解成：

```text
channel_patches[b] =

  channel 0: target
    patch 0: [patch_len, input_dim]
    patch 1: [patch_len, input_dim]
    ...
    patch Nt-1

  channel 1: covariate_1
    patch 0
    patch 1
    ...

  channel 2: covariate_2
    patch 0
    patch 1
    ...
```

其中：

- `channel 0` 固定是 target
- 后续 channel 是 covariates

当前每个时间步的 `input_dim=9`，组成是：

```text
[ normalized_value,
  missing_flag,
  time_feature_1,
  time_feature_2,
  ...
  time_feature_7 ]
```

---

## 10. `forecast_token_ids` 再展开一层

`forecast_token_ids` 在 batch 中可以理解成：

```text
forecast_token_ids[b] =

  [<TSV_bin_381>, <TSV_bin_402>, <TSV_bin_417>, ..., <EOS_FC>, <FORECAST_PAD>, ...]
```

其中：

- 前面是 future value 量化后的 bin token
- `<EOS_FC>` 是结束符
- padding 位置会在 `forecast_labels` 中被 mask 掉

---

## 11. 一句话总结

当前跨域 CPT 的 batch 机制本质上是：

```text
先把多域样本 merge + shuffle
    ->
再用同一份 records 构造 pred / imp 两个任务视图
    ->
每个 batch 先决定任务类型
    ->
再从对应任务 dataset 抽一整批样本
```

而模型前向时的 batch 输入本质上是：

```text
文本 token 张量
+ 多通道时序 patch 张量
+ 预测目标 token 张量
```
