# TextTS

当前仓库中的 `TextTS` 代码包含：

- `CPT / pretrain`
- `SFT`
- `forecast eval`
- `Time-MMD` 相关运行脚本

详细文档见：

- [../文档/README.md](../文档/README.md)
- [../文档/textts_current_pipeline.md](../文档/textts_current_pipeline.md)
- [../文档/textts_full_pipeline_usage.md](../文档/textts_full_pipeline_usage.md)
- [../文档/textts_full_pipeline_readme.md](../文档/textts_full_pipeline_readme.md)
- [../文档/textts_cross_domain_batching.md](../文档/textts_cross_domain_batching.md)

## Full Pipeline

一键串联脚本：

- [scripts/run_timemmd_full_pipeline.sh](./scripts/run_timemmd_full_pipeline.sh)

它会按顺序执行：

1. `CPT`
2. `post-pretrain eval`
3. `SFT`
4. `post-SFT eval`

### Quick Start

单域最小示例：

```bash
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

单域 smoke test：

```bash
MAX_WINDOWS=4 \
PRETRAIN_STEPS=1 \
SFT_STEPS=1 \
POST_PRETRAIN_MAX_SAMPLES=2 \
POST_SFT_MAX_SAMPLES=2 \
DEVICE=cpu \
bash textts/scripts/run_timemmd_full_pipeline.sh Energy
```

单域常用实验：

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

多域联合实验：

```bash
DOMAINS="Energy Climate Traffic" \
DEVICE=cuda \
MAX_WINDOWS=128 \
PRETRAIN_STEPS=500 \
SFT_STEPS=200 \
bash textts/scripts/run_timemmd_full_pipeline.sh
```

## Key Scripts

- `scripts/run_timemmd_pretrain.sh`
- `scripts/run_timemmd_sft.sh`
- `scripts/run_timemmd_eval.sh`
- `scripts/run_timemmd_full_pipeline.sh`

## Outputs

full pipeline 默认会产出 4 组目录：

- `PRETRAIN_OUTPUT_DIR`
- `POST_PRETRAIN_EVAL_OUTPUT_DIR`
- `SFT_OUTPUT_DIR`
- `POST_SFT_EVAL_OUTPUT_DIR`

常见结果文件：

- checkpoint 目录：`metadata.json`、`textts_modules.pt`、`llm/`
- eval 目录：`metrics.json`、`predictions.jsonl`

## Notes

- `SFT` 默认从 `CPT` checkpoint 继续训练。
- 当前独立评估是 `forecast eval`，不是单独的 imputation 专项评估。
- 如果需要完整参数说明，直接看 [../文档/textts_full_pipeline_usage.md](../文档/textts_full_pipeline_usage.md)。
