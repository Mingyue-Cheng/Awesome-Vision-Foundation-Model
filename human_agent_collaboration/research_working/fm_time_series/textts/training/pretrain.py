"""Minimal CPT training scaffold for TextTS."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from textts.data.benchmark_loader import CSVWindowConfig, load_csv_windows
from textts.data.benchmark_loader import train_val_split
from textts.data.datasets import build_pretrain_datasets, build_textts_collator
from textts.data.sequence_formatter import TextTSSequenceFormatter
from textts.data.sequence_sampler import MixedBatchSampler, MixedBatchSamplerConfig, MixedTaskDataset
from textts.data.timemmd_loader import (
    TimeMMDMultiDomainConfig,
    TimeMMDWindowConfig,
    load_timemmd_multi_domain_windows,
    load_timemmd_windows,
)
from textts.model.build import Qwen3BuildConfig, build_textts_from_qwen3
from textts.model.checkpoint import save_textts_checkpoint
from textts.training.distributed import (
    DistributedRuntime,
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    move_batch_to_device,
    require_non_empty_shard,
    resolve_runtime_device,
    seed_everything,
    setup_distributed,
    shard_records_for_rank,
    unwrap_model,
    wrap_model_for_ddp,
)
from textts.tokenization.forecast_quantizer import ForecastQuantizer


@dataclass
class PretrainConfig:
    batch_size: int = 8
    pred_probability: float = 0.7
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 0
    num_batches_per_epoch: Optional[int] = None
    max_grad_norm: float = 1.0
    seed: int = 42


def parse_domains_arg(domains: Optional[str]) -> List[str]:
    if domains is None:
        return []
    parts = [part.strip() for chunk in domains.split(",") for part in chunk.split() if part.strip()]
    seen = set()
    ordered = []
    for part in parts:
        if part not in seen:
            ordered.append(part)
            seen.add(part)
    return ordered


def train_val_split_grouped(
    records: Sequence[dict[str, object]],
    *,
    val_ratio: float,
    seed: int,
    group_key: str = "domain",
) -> tuple[List[dict[str, object]], List[dict[str, object]]]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1).")

    grouped: Dict[str, List[dict[str, object]]] = {}
    for record in records:
        group = str(record.get(group_key, "unknown"))
        grouped.setdefault(group, []).append(record)

    rng = random.Random(seed)
    train_records: List[dict[str, object]] = []
    val_records: List[dict[str, object]] = []
    for _, group_records in sorted(grouped.items()):
        shuffled = list(group_records)
        rng.shuffle(shuffled)
        split = int(len(shuffled) * (1.0 - val_ratio))
        split = min(max(split, 0), len(shuffled))
        train_records.extend(shuffled[:split])
        val_records.extend(shuffled[split:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def build_pretrain_dataloader(
    pred_dataset: Dataset[MutableMapping[str, Any]],
    imp_dataset: Dataset[MutableMapping[str, Any]],
    collator: Any,
    config: PretrainConfig,
) -> DataLoader[MutableMapping[str, Any]]:
    mixed_dataset = MixedTaskDataset(pred_dataset, imp_dataset)
    batch_sampler = MixedBatchSampler(
        pred_dataset_size=len(pred_dataset),
        imp_dataset_size=len(imp_dataset),
        config=MixedBatchSamplerConfig(
            batch_size=config.batch_size,
            pred_probability=config.pred_probability,
            drop_last=False,
            num_batches_per_epoch=config.num_batches_per_epoch,
            seed=config.seed,
        ),
    )
    return DataLoader(
        mixed_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=config.num_workers,
    )


def build_pretrain_optimizer(model: torch.nn.Module, config: PretrainConfig) -> AdamW:
    return AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


class TextTSPretrainer:
    """Small trainer loop for CPT skeleton validation."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: Optional[torch.device] = None,
        max_grad_norm: float = 1.0,
        move_model_to_device: bool = True,
        runtime: Optional[DistributedRuntime] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device("cpu")
        self.max_grad_norm = max_grad_norm
        self.runtime = runtime or DistributedRuntime(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            backend="none",
            device=self.device,
            device_type=self.device.type,
        )
        if move_model_to_device:
            self.model.to(self.device)

    def train_step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        self.model.train()
        batch = move_batch_to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(batch)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {"loss": float(loss.detach().item())}

    def train_epoch(
        self,
        dataloader: Iterable[Mapping[str, Any]],
        *,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        total_loss = 0.0
        step_count = 0
        for step_count, batch in enumerate(dataloader, start=1):
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            if max_steps is not None and step_count >= max_steps:
                break
        if step_count == 0:
            return {"loss": 0.0, "steps": 0.0}
        mean_loss = total_loss / step_count
        mean_loss = all_reduce_mean(mean_loss, self.runtime)
        return {"loss": mean_loss, "steps": float(step_count)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal TextTS CPT training from CSV or Time-MMD.")
    parser.add_argument("--data-source", type=str, default="csv", choices=["csv", "timemmd"])
    parser.add_argument("--csv-path", type=str, default=None, help="Path to a CSV file such as ETTh1.csv")
    parser.add_argument("--timemmd-root", type=str, default=None, help="Path to the Time-MMD root directory.")
    parser.add_argument("--domains", type=str, default=None, help="Comma or space separated Time-MMD domains for joint cross-domain training.")
    parser.add_argument("--target-col", type=str, default="OT", help="Target column name.")
    parser.add_argument("--timestamp-col", type=str, default="date", help="Timestamp column name.")
    parser.add_argument("--domain", type=str, default="generic")
    parser.add_argument("--freq", type=str, default="unknown")
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pred-probability", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.5B-Base")
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--distributed-backend", type=str, default="auto", help="auto, hccl, nccl or gloo when launched with torchrun.")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-fixed-splits", action="store_true", help="Use explicit train/val/test splits instead of legacy eval_ratio slicing.")
    parser.add_argument("--train-split", type=str, default="train", choices=["all", "train", "val", "test"])
    parser.add_argument("--eval-split", type=str, default="val", choices=["all", "train", "val", "test"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--eval-max-samples", type=int, default=16)
    parser.add_argument("--eval-point-strategy", type=str, default="greedy", choices=["greedy", "sample"])
    parser.add_argument("--eval-num-prob-samples", type=int, default=0)
    parser.add_argument("--eval-prob-temperature", type=float, default=1.0)
    parser.add_argument("--eval-prob-top-p", type=float, default=0.9)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def load_records_from_args(
    args: argparse.Namespace,
    *,
    split: Optional[str] = None,
) -> list[dict[str, object]]:
    selected_split = split or "all"
    if args.data_source == "csv":
        if not args.csv_path:
            raise ValueError("--csv-path is required when --data-source=csv.")
        return load_csv_windows(
            args.csv_path,
            CSVWindowConfig(
                target_col=args.target_col,
                timestamp_col=args.timestamp_col,
                domain=args.domain,
                freq=args.freq,
                context=args.context,
                lookback=args.lookback,
                horizon=args.horizon,
                stride=args.stride,
                max_windows=args.max_windows,
                split=selected_split,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            ),
        )

    if not args.timemmd_root:
        raise ValueError("--timemmd-root is required when --data-source=timemmd.")
    domains = parse_domains_arg(getattr(args, "domains", None))
    seed = getattr(args, "seed", 42)
    if domains:
        return load_timemmd_multi_domain_windows(
            TimeMMDMultiDomainConfig(
                root_dir=args.timemmd_root,
                domains=domains,
                lookback=args.lookback,
                horizon=args.horizon,
                stride=args.stride,
                target_col=args.target_col,
                max_windows_per_domain=args.max_windows,
                shuffle_records=True,
                shuffle_seed=seed,
                split=selected_split,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )
        )
    if not args.domain or args.domain == "generic":
        raise ValueError("--domain or --domains must be set to valid Time-MMD domain names when --data-source=timemmd.")
    return load_timemmd_windows(
        TimeMMDWindowConfig(
            root_dir=args.timemmd_root,
            domain=args.domain,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
            target_col=args.target_col,
            max_windows=args.max_windows,
            split=selected_split,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    runtime = setup_distributed(device_name=args.device, backend_name=args.distributed_backend)
    try:
        if runtime.enabled and args.device_map is not None:
            raise ValueError("--device-map is incompatible with torchrun/DDP training. Use one process per device instead.")
        seed_everything(args.seed + runtime.rank)

        if args.use_fixed_splits:
            train_records = load_records_from_args(args, split=args.train_split)
            if not train_records:
                raise ValueError(
                    f"No training windows were produced for data_source={args.data_source} and split={args.train_split}."
                )
            eval_records = load_records_from_args(args, split=args.eval_split) if args.eval_ratio > 0.0 else []
        else:
            records = load_records_from_args(args)
            if not records:
                raise ValueError(f"No training windows were produced for data_source={args.data_source}.")
            train_records = records
            eval_records = []
            if args.eval_ratio > 0.0:
                domains = parse_domains_arg(getattr(args, "domains", None))
                if args.data_source == "timemmd" and domains:
                    train_records, eval_records = train_val_split_grouped(records, val_ratio=args.eval_ratio, seed=args.seed)
                else:
                    train_records, eval_records = train_val_split(records, val_ratio=args.eval_ratio)
                if not train_records:
                    train_records = records
                    eval_records = []

        model, tokenizer, tokenizer_bundle = build_textts_from_qwen3(
            Qwen3BuildConfig(
                base_model_name_or_path=args.model_name,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                local_files_only=args.local_files_only,
                patch_len=16,
                input_dim=9,
                d_patch=256,
            )
        )
        formatter = TextTSSequenceFormatter(tokenizer, tokenizer_bundle, ForecastQuantizer())
        config = PretrainConfig(
            batch_size=args.batch_size,
            pred_probability=args.pred_probability,
            learning_rate=args.learning_rate,
            num_batches_per_epoch=args.steps,
            seed=args.seed + runtime.rank,
        )

        train_records = shard_records_for_rank(train_records, runtime)
        require_non_empty_shard(train_records, runtime, split_name="train")
        if not runtime.is_main_process:
            eval_records = []

        pred_dataset, imp_dataset = build_pretrain_datasets(train_records, formatter)
        collator = build_textts_collator(tokenizer_bundle)
        dataloader = build_pretrain_dataloader(pred_dataset, imp_dataset, collator, config)
        runtime_device = runtime.device if runtime.enabled else resolve_runtime_device(args.device)
        if args.device_map is None:
            model.to(runtime_device)
        model = wrap_model_for_ddp(model, runtime, find_unused_parameters=args.ddp_find_unused_parameters)
        optimizer = build_pretrain_optimizer(model, config)
        trainer = TextTSPretrainer(
            model,
            optimizer,
            device=runtime_device,
            max_grad_norm=config.max_grad_norm,
            move_model_to_device=False,
            runtime=runtime,
        )
        train_metrics = trainer.train_epoch(dataloader, max_steps=args.steps)

        barrier(runtime)
        eval_metrics: Dict[str, float] = {}
        eval_predictions: list[dict[str, object]] = []
        base_model = unwrap_model(model)
        if eval_records and runtime.is_main_process:
            from textts.eval.forecast_eval import ForecastEvalConfig, evaluate_forecast_records, write_eval_outputs

            eval_metrics, eval_predictions = evaluate_forecast_records(
                base_model,
                eval_records,
                formatter,
                collator,
                ForecastQuantizer(),
                tokenizer_bundle.forecast_bin_token_ids,
                config=ForecastEvalConfig(
                    point_strategy=args.eval_point_strategy,
                    num_prob_samples=args.eval_num_prob_samples,
                    prob_temperature=args.eval_prob_temperature,
                    prob_top_p=args.eval_prob_top_p,
                    max_samples=args.eval_max_samples,
                ),
                device=runtime_device,
            )
            if args.output_dir:
                write_eval_outputs(Path(args.output_dir) / "eval", metrics=eval_metrics, predictions=eval_predictions)

        barrier(runtime)
        if args.output_dir and runtime.is_main_process:
            save_textts_checkpoint(
                base_model,
                tokenizer,
                args.output_dir,
                optimizer=optimizer,
                metadata={
                    "task": "pretrain",
                    "data_source": args.data_source,
                    "domain": args.domain if not parse_domains_arg(getattr(args, "domains", None)) else "joint",
                    "domains": parse_domains_arg(getattr(args, "domains", None)),
                    "steps": args.steps,
                    "world_size": runtime.world_size,
                    "backend": runtime.backend,
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                },
            )

        if runtime.is_main_process:
            print({"train": train_metrics, "eval": eval_metrics})
    finally:
        cleanup_distributed(runtime)


if __name__ == "__main__":
    main()
