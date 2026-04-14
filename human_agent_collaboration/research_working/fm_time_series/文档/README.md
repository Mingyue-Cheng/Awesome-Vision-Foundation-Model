# 文档索引

当前目录下与 `TextTS` 相关的主要文档如下。

## TextTS 总览

- [textts_current_pipeline.md](./textts_current_pipeline.md)
  当前仓库里 `TextTS` 的真实实现 pipeline，总览数据、模型、训练、评估与脚本入口。

- [textts_cross_domain_batching.md](./textts_cross_domain_batching.md)
  说明 `Time-MMD` 跨域预训练、`MixedBatchSampler`、batch 混合逻辑，以及 batch 张量结构。

## Full Pipeline 脚本

- [textts_full_pipeline_usage.md](./textts_full_pipeline_usage.md)
  `run_timemmd_full_pipeline.sh` 的完整使用说明，适合查参数和输出目录。

- [textts_full_pipeline_readme.md](./textts_full_pipeline_readme.md)
  README 风格的短版说明，适合直接复制命令。

## 设计与规划

- [../textts方案设计/textts_design.md](../textts方案设计/textts_design.md)
  `TextTS` 设计文档。

- [../textts方案设计/textts_data_plan.md](../textts方案设计/textts_data_plan.md)
  `TextTS` 数据筛选与构建规划。

## 代码入口

- [../textts/README.md](../textts/README.md)
  `textts/` 目录下的代码入口 README。

- [../textts/scripts/run_timemmd_full_pipeline.sh](../textts/scripts/run_timemmd_full_pipeline.sh)
  一键串联 `CPT -> post-pretrain eval -> SFT -> post-SFT eval` 的脚本。
