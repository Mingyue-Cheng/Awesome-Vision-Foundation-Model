"""Minimal SFT training scaffold for TextTS."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from textts.data.benchmark_loader import train_val_split
from textts.data.datasets import build_sft_dataset, build_textts_collator
from textts.data.sequence_formatter import TextTSSequenceFormatter
from textts.data.sft_dataset import SFTDatasetConfig
from textts.model.build import Qwen3BuildConfig, build_textts_from_qwen3
from textts.model.checkpoint import load_textts_modules, save_textts_checkpoint
from textts.tokenization.forecast_quantizer import ForecastQuantizer
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
from textts.training.pretrain import (
    load_records_from_args,
    parse_domains_arg,
    train_val_split_grouped,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]


@dataclass
class SFTConfig:
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 0
    max_grad_norm: float = 1.0
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42


def build_sft_dataloader(
    dataset: Dataset[MutableMapping[str, Any]],
    collator: Any,
    config: SFTConfig,
    *,
    shuffle: bool = True,
) -> DataLoader[MutableMapping[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=config.num_workers,
    )


def build_sft_optimizer(model: torch.nn.Module, config: SFTConfig) -> AdamW:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)


def maybe_apply_lora(model: torch.nn.Module, config: SFTConfig) -> torch.nn.Module:
    """Apply LoRA to the backbone if peft is available and enabled."""

    if not config.use_lora:
        return model
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for LoRA-based SFT.")

    llm = getattr(model, "llm", None)
    if llm is None:
        raise ValueError("Expected TextTSModel with a .llm attribute for LoRA application.")

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model.llm = get_peft_model(llm, peft_config)
    return model


class TextTSSFTTrainer:
    """Small trainer loop for SFT skeleton validation."""

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
    parser = argparse.ArgumentParser(description="Run minimal TextTS SFT training from CSV or Time-MMD.")
    parser.add_argument("--data-source", type=str, default="csv", choices=["csv", "timemmd"])
    parser.add_argument("--csv-path", type=str, default=None, help="Path to a CSV file such as ETTh1.csv")
    parser.add_argument("--timemmd-root", type=str, default=None, help="Path to the Time-MMD root directory.")
    parser.add_argument("--domains", type=str, default=None, help="Comma or space separated Time-MMD domains for joint cross-domain SFT.")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.5B-Base")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional TextTS checkpoint directory to continue from.")
    parser.add_argument("--textts-modules-path", type=str, default=None, help="Optional explicit textts_modules.pt path. Overrides checkpoint-dir default.")
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
    parser.add_argument("--sft-context-mode", type=str, default="mixed", choices=["mixed", "l0", "l1", "l2", "all"])
    parser.add_argument("--sft-context-cache", type=str, default=None, help="Optional JSONL cache for L2 rich text contexts.")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def resolve_sft_init_paths(args: argparse.Namespace) -> tuple[str, Optional[str]]:
    model_name_or_path = args.model_name
    textts_modules_path = args.textts_modules_path
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        model_name_or_path = str(checkpoint_dir / "llm")
        if textts_modules_path is None:
            candidate = checkpoint_dir / "textts_modules.pt"
            if candidate.exists():
                textts_modules_path = str(candidate)
    return model_name_or_path, textts_modules_path


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
                    f"No SFT windows were produced for data_source={args.data_source} and split={args.train_split}."
                )
            eval_records = load_records_from_args(args, split=args.eval_split) if args.eval_ratio > 0.0 else []
        else:
            records = load_records_from_args(args)
            if not records:
                raise ValueError(f"No SFT windows were produced for data_source={args.data_source}.")
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

        model_name_or_path, textts_modules_path = resolve_sft_init_paths(args)
        model, tokenizer, tokenizer_bundle = build_textts_from_qwen3(
            Qwen3BuildConfig(
                base_model_name_or_path=model_name_or_path,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                local_files_only=args.local_files_only,
                patch_len=16,
                input_dim=9,
                d_patch=256,
            )
        )
        if textts_modules_path:
            load_textts_modules(model, textts_modules_path, map_location="cpu")
        formatter = TextTSSequenceFormatter(tokenizer, tokenizer_bundle, ForecastQuantizer())
        config = SFTConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_lora=args.use_lora,
            seed=args.seed + runtime.rank,
        )

        train_records = shard_records_for_rank(train_records, runtime)
        require_non_empty_shard(train_records, runtime, split_name="train")
        if not runtime.is_main_process:
            eval_records = []

        dataset = build_sft_dataset(
            train_records,
            formatter,
            config=SFTDatasetConfig(
                context_mode=args.sft_context_mode,
                l2_context_path=args.sft_context_cache,
            ),
        )
        collator = build_textts_collator(tokenizer_bundle)
        if args.device_map is None:
            model.to(runtime.device if runtime.enabled else resolve_runtime_device(args.device))
        model = maybe_apply_lora(model, config)
        model = wrap_model_for_ddp(model, runtime, find_unused_parameters=args.ddp_find_unused_parameters)
        dataloader = build_sft_dataloader(dataset, collator, config, shuffle=True)
        optimizer = build_sft_optimizer(model, config)
        runtime_device = runtime.device if runtime.enabled else resolve_runtime_device(args.device)
        trainer = TextTSSFTTrainer(
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
                    "task": "sft",
                    "data_source": args.data_source,
                    "domain": args.domain if not parse_domains_arg(getattr(args, "domains", None)) else "joint",
                    "domains": parse_domains_arg(getattr(args, "domains", None)),
                    "init_model_name_or_path": model_name_or_path,
                    "init_checkpoint_dir": args.checkpoint_dir,
                    "init_textts_modules_path": textts_modules_path,
                    "steps": args.steps,
                    "world_size": runtime.world_size,
                    "backend": runtime.backend,
                    "sft_context_mode": args.sft_context_mode,
                    "sft_context_cache": args.sft_context_cache,
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
