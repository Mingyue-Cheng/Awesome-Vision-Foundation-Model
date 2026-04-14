"""GIFT-Eval benchmark runner for TextTS.

This module supports two protocols:
- zero-shot: direct evaluation on the benchmark eval split
- few-shot: lightweight SFT adaptation on a small train subset, then eval

Data loading is intentionally flexible:
- Hugging Face datasets (`datasets.load_dataset`)
- local JSON / JSONL files

The record schema is normalized into the TextTS raw record format expected by
TextTSSequenceFormatter.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch

from textts.data.datasets import build_sft_dataset, build_textts_collator
from textts.data.sequence_formatter import TextTSSequenceFormatter
from textts.data.sft_dataset import SFTDatasetConfig
from textts.eval.forecast_eval import (
    ForecastEvalConfig,
    evaluate_forecast_records,
    load_textts_model_for_eval,
    resolve_runtime_device,
)
from textts.model.checkpoint import save_textts_checkpoint
from textts.tokenization.forecast_quantizer import ForecastQuantizer
from textts.training.sft import (
    SFTConfig,
    TextTSSFTTrainer,
    build_sft_dataloader,
    build_sft_optimizer,
    maybe_apply_lora,
)


DEFAULT_WQL_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "history": ("target_history", "past_target", "history", "past_values", "context"),
    "future": ("target_future", "future_target", "future", "label", "labels", "prediction"),
    "domain": ("dataset", "dataset_name", "domain", "source", "collection", "benchmark_dataset"),
    "freq": ("freq", "frequency"),
    "seasonality": ("seasonality", "seasonal_period", "seasonal_periodicity"),
    "target_name": ("target_name", "item_id", "series_id", "target_col"),
    "context": ("context", "description", "text", "prompt"),
    "history_start": ("history_start", "start", "start_time", "history_start_time"),
    "history_end": ("history_end", "past_end", "history_end_time"),
    "forecast_end": ("forecast_end", "end", "end_time", "future_end"),
}


@dataclass(frozen=True)
class GiftEvalRecord:
    raw_record: dict[str, object]
    dataset_name: str
    seasonality: int


def _is_sequence_like(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _first_present(mapping: Mapping[str, Any], candidates: Sequence[str]) -> tuple[Optional[str], Any]:
    for key in candidates:
        if key in mapping:
            return key, mapping[key]
    return None, None


def _coerce_float_sequence(value: object) -> Optional[list[float]]:
    if _is_sequence_like(value):
        result: list[float] = []
        for item in value:  # type: ignore[assignment]
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                return None
        return result
    return None


def _coerce_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def infer_seasonality(freq: object) -> int:
    if not isinstance(freq, str):
        return 1
    normalized = freq.strip().lower()
    if normalized in {"hourly", "1h", "h"}:
        return 24
    if normalized in {"daily", "1d", "d"}:
        return 7
    if normalized in {"weekly", "1w", "w"}:
        return 52
    if normalized in {"monthly", "1m", "m"}:
        return 12
    if normalized in {"quarterly", "q"}:
        return 4
    if normalized in {"yearly", "annual", "y"}:
        return 1
    if normalized in {"15min", "15m"}:
        return 96
    if normalized in {"30min", "30m"}:
        return 48
    if normalized in {"10min", "10m"}:
        return 144
    return 1


def _estimate_mase_denominator(history: Sequence[float], seasonality: int) -> float:
    if not history:
        return 1.0
    lag = max(1, min(seasonality, len(history) - 1))
    if lag <= 0 or len(history) <= lag:
        lag = 1
    diffs = [abs(history[index] - history[index - lag]) for index in range(lag, len(history))]
    if not diffs:
        return 1.0
    denom = sum(diffs) / len(diffs)
    return denom if denom > 1e-8 else 1.0


def mase_score(
    history: Sequence[float],
    target: Sequence[float],
    prediction: Sequence[float],
    *,
    seasonality: int,
) -> float:
    if not target or not prediction:
        return 0.0
    horizon = min(len(target), len(prediction))
    mae = sum(abs(float(prediction[i]) - float(target[i])) for i in range(horizon)) / horizon
    denom = _estimate_mase_denominator(history, seasonality)
    return mae / denom


def weighted_quantile_loss(
    target: Sequence[float],
    samples: Sequence[Sequence[float]],
    *,
    quantiles: Sequence[float] = DEFAULT_WQL_QUANTILES,
) -> float:
    if not target or not samples:
        return 0.0

    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    sample_tensor = torch.as_tensor(samples, dtype=torch.float32)
    if sample_tensor.ndim != 2 or sample_tensor.shape[1] != target_tensor.shape[0]:
        raise ValueError(
            f"Expected sample tensor with shape [S, H={target_tensor.shape[0]}], got {tuple(sample_tensor.shape)}."
        )

    denom = float(target_tensor.abs().sum().item())
    if denom <= 1e-8:
        denom = float(target_tensor.abs().mean().item()) + 1.0

    losses: list[float] = []
    for quantile in quantiles:
        q_pred = torch.quantile(sample_tensor, quantile, dim=0)
        diff = target_tensor - q_pred
        pinball = torch.maximum(quantile * diff, (quantile - 1.0) * diff)
        losses.append(float(2.0 * pinball.sum().item() / denom))
    return sum(losses) / len(losses)


def _load_rows_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [dict(item) for item in payload["data"] if isinstance(item, dict)]
        return [payload]
    raise ValueError(f"Unsupported JSON structure in {path}.")


def _load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(dict(payload))
    return rows


def _load_rows_auto(source: str, *, split: str, hf_config: Optional[str]) -> list[dict[str, Any]]:
    path = Path(source)
    if path.exists():
        if path.suffix.lower() == ".json":
            return _load_rows_from_json(path)
        if path.suffix.lower() == ".jsonl":
            return _load_rows_from_jsonl(path)
        raise ValueError(f"Unsupported local GIFT source format: {path.suffix}")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "datasets is required to load GIFT-Eval from Hugging Face. "
            "Install it or provide a local JSON/JSONL file."
        ) from exc

    dataset = load_dataset(source, name=hf_config, split=split)
    return [dict(row) for row in dataset]


def _normalize_gift_row(row: Mapping[str, Any]) -> GiftEvalRecord:
    history_key, history_value = _first_present(row, FIELD_CANDIDATES["history"])
    future_key, future_value = _first_present(row, FIELD_CANDIDATES["future"])
    history = _coerce_float_sequence(history_value)
    future = _coerce_float_sequence(future_value)
    if history is None or future is None:
        raise ValueError(
            "Could not normalize GIFT row: missing numeric history/future sequence. "
            f"Keys={sorted(row.keys())} history_key={history_key} future_key={future_key}"
        )

    _, dataset_value = _first_present(row, FIELD_CANDIDATES["domain"])
    _, freq_value = _first_present(row, FIELD_CANDIDATES["freq"])
    _, seasonality_value = _first_present(row, FIELD_CANDIDATES["seasonality"])
    _, target_name_value = _first_present(row, FIELD_CANDIDATES["target_name"])
    _, context_value = _first_present(row, FIELD_CANDIDATES["context"])
    _, history_start = _first_present(row, FIELD_CANDIDATES["history_start"])
    _, history_end = _first_present(row, FIELD_CANDIDATES["history_end"])
    _, forecast_end = _first_present(row, FIELD_CANDIDATES["forecast_end"])

    seasonality = _coerce_int(seasonality_value) or infer_seasonality(freq_value)
    dataset_name = str(dataset_value or "gift_eval")
    target_name = str(target_name_value or "target")
    normalized = {
        "domain": dataset_name,
        "freq": str(freq_value or "unknown"),
        "context": str(context_value or ""),
        "target_name": target_name,
        "history_start": "" if history_start is None else str(history_start),
        "history_end": "" if history_end is None else str(history_end),
        "forecast_end": "" if forecast_end is None else str(forecast_end),
        "target_history": history,
        "target_future": future,
        "target_time_features": [[0.0] * 7 for _ in range(len(history))],
        "covariates": [],
        "gift_dataset_name": dataset_name,
        "gift_seasonality": seasonality,
    }
    return GiftEvalRecord(raw_record=normalized, dataset_name=dataset_name, seasonality=seasonality)


def load_gift_records(
    source: str,
    *,
    split: str,
    hf_config: Optional[str] = None,
    dataset_filter: Optional[Sequence[str]] = None,
    max_records: Optional[int] = None,
) -> list[GiftEvalRecord]:
    rows = _load_rows_auto(source, split=split, hf_config=hf_config)
    normalized: list[GiftEvalRecord] = []
    allowed = set(dataset_filter or [])
    for row in rows:
        record = _normalize_gift_row(row)
        if allowed and record.dataset_name not in allowed:
            continue
        normalized.append(record)
        if max_records is not None and len(normalized) >= max_records:
            break
    return normalized


def _parse_dataset_filter(value: Optional[str]) -> list[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for chunk in value.split(",") for part in chunk.split() if part.strip()]


def _select_few_shot_records(
    records: Sequence[GiftEvalRecord],
    *,
    ratio: float,
    max_records: Optional[int],
) -> list[dict[str, object]]:
    if not (0.0 < ratio <= 1.0):
        raise ValueError("few-shot ratio must be in (0, 1].")
    total = len(records)
    count = max(1, int(round(total * ratio)))
    if max_records is not None:
        count = min(count, max_records)
    return [dict(item.raw_record) for item in records[:count]]


def _weighted_average_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    weights = [float(row.get("num_samples", 0.0)) for row in rows]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return {}
    metric_names = sorted({key for row in rows for key in row if key != "num_samples"})
    result: dict[str, float] = {"num_samples": total_weight}
    for metric_name in metric_names:
        weighted = 0.0
        used = 0.0
        for weight, row in zip(weights, rows):
            if metric_name not in row:
                continue
            weighted += weight * float(row[metric_name])
            used += weight
        if used > 0.0:
            result[metric_name] = weighted / used
    return result


def summarize_gift_outputs(
    records: Sequence[GiftEvalRecord],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]]]:
    overall_rows: list[dict[str, float]] = []
    grouped_rows: dict[str, list[dict[str, float]]] = {}
    detailed_rows: list[dict[str, Any]] = []

    for record, prediction in zip(records, prediction_rows):
        history = record.raw_record["target_history"]
        target = prediction["target"]
        point_prediction = prediction["point_prediction"]
        dataset_name = record.dataset_name
        per_row = {
            "num_samples": 1.0,
            "mase": mase_score(history, target, point_prediction, seasonality=record.seasonality),
        }
        prob_samples = prediction.get("prob_samples")
        if isinstance(prob_samples, list) and prob_samples:
            per_row["wql"] = weighted_quantile_loss(target, prob_samples)

        overall_rows.append(per_row)
        grouped_rows.setdefault(dataset_name, []).append(per_row)
        detailed_rows.append(
            {
                "dataset": dataset_name,
                "seasonality": record.seasonality,
                "target_name": record.raw_record.get("target_name", "target"),
                **prediction,
                **per_row,
            }
        )

    overall = _weighted_average_metrics(overall_rows)
    per_dataset = {name: _weighted_average_metrics(rows) for name, rows in grouped_rows.items()}
    return overall, per_dataset, detailed_rows


def run_few_shot_adaptation(
    model: torch.nn.Module,
    tokenizer: object,
    tokenizer_bundle: Any,
    *,
    records: Sequence[dict[str, object]],
    device: torch.device,
    batch_size: int,
    steps: int,
    learning_rate: float,
    use_lora: bool,
    sft_context_mode: str,
    sft_context_cache: Optional[str],
) -> tuple[torch.nn.Module, dict[str, float]]:
    formatter = TextTSSequenceFormatter(tokenizer, tokenizer_bundle, ForecastQuantizer())
    dataset = build_sft_dataset(
        records,
        formatter,
        config=SFTDatasetConfig(
            context_mode=sft_context_mode,
            l2_context_path=sft_context_cache,
        ),
    )
    collator = build_textts_collator(tokenizer_bundle)
    config = SFTConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_lora=use_lora,
        seed=42,
    )

    model.to(device)
    model = maybe_apply_lora(model, config)
    dataloader = build_sft_dataloader(dataset, collator, config, shuffle=True)
    optimizer = build_sft_optimizer(model, config)
    trainer = TextTSSFTTrainer(
        model,
        optimizer,
        device=device,
        max_grad_norm=config.max_grad_norm,
        move_model_to_device=False,
    )
    metrics = trainer.train_epoch(dataloader, max_steps=steps)
    return model, metrics


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextTS on GIFT-Eval.")
    parser.add_argument("--gift-source", type=str, default="Salesforce/GiftEval")
    parser.add_argument("--gift-config", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--protocol", type=str, default="zero-shot", choices=["zero-shot", "few-shot"])
    parser.add_argument("--dataset-filter", type=str, default=None, help="Optional comma-separated dataset names.")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-eval-records", type=int, default=None)
    parser.add_argument("--few-shot-ratio", type=float, default=0.05)
    parser.add_argument("--few-shot-steps", type=int, default=20)
    parser.add_argument("--few-shot-batch-size", type=int, default=4)
    parser.add_argument("--few-shot-learning-rate", type=float, default=1e-4)
    parser.add_argument("--few-shot-use-lora", action="store_true")
    parser.add_argument("--sft-context-mode", type=str, default="mixed", choices=["mixed", "l0", "l1", "l2", "all"])
    parser.add_argument("--sft-context-cache", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.5B-Base")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--textts-modules-path", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--point-strategy", type=str, default="greedy", choices=["greedy", "sample"])
    parser.add_argument("--point-temperature", type=float, default=1.0)
    parser.add_argument("--point-top-p", type=float, default=1.0)
    parser.add_argument("--num-prob-samples", type=int, default=16)
    parser.add_argument("--prob-temperature", type=float, default=1.0)
    parser.add_argument("--prob-top-p", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-few-shot-checkpoint", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    dataset_filter = _parse_dataset_filter(args.dataset_filter)
    runtime_device = resolve_runtime_device(args.device)

    if args.protocol == "few-shot" and args.device_map is not None:
        raise ValueError("--device-map is not supported for few-shot training. Use --device instead.")

    model_name_or_path = args.model_name
    textts_modules_path = args.textts_modules_path
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        model_name_or_path = str(checkpoint_dir / "llm")
        if textts_modules_path is None:
            candidate = checkpoint_dir / "textts_modules.pt"
            if candidate.exists():
                textts_modules_path = str(candidate)

    model, tokenizer, tokenizer_bundle = load_textts_model_for_eval(
        model_name_or_path=model_name_or_path,
        textts_modules_path=textts_modules_path,
        local_files_only=args.local_files_only,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    formatter = TextTSSequenceFormatter(tokenizer, tokenizer_bundle, ForecastQuantizer())
    collator = build_textts_collator(tokenizer_bundle)
    quantizer = ForecastQuantizer()

    train_records_wrapped: list[GiftEvalRecord] = []
    if args.protocol == "few-shot":
        train_records_wrapped = load_gift_records(
            args.gift_source,
            split=args.train_split,
            hf_config=args.gift_config,
            dataset_filter=dataset_filter,
            max_records=args.max_train_records,
        )
        few_shot_records = _select_few_shot_records(
            train_records_wrapped,
            ratio=args.few_shot_ratio,
            max_records=args.max_train_records,
        )
        model, few_shot_metrics = run_few_shot_adaptation(
            model,
            tokenizer,
            tokenizer_bundle,
            records=few_shot_records,
            device=runtime_device,
            batch_size=args.few_shot_batch_size,
            steps=args.few_shot_steps,
            learning_rate=args.few_shot_learning_rate,
            use_lora=args.few_shot_use_lora,
            sft_context_mode=args.sft_context_mode,
            sft_context_cache=args.sft_context_cache,
        )
    else:
        few_shot_metrics = {}

    eval_records_wrapped = load_gift_records(
        args.gift_source,
        split=args.eval_split,
        hf_config=args.gift_config,
        dataset_filter=dataset_filter,
        max_records=args.max_eval_records,
    )
    eval_records = [dict(item.raw_record) for item in eval_records_wrapped]

    eval_metrics, predictions = evaluate_forecast_records(
        model,
        eval_records,
        formatter,
        collator,
        quantizer,
        tokenizer_bundle.forecast_bin_token_ids,
        config=ForecastEvalConfig(
            point_strategy=args.point_strategy,
            point_temperature=args.point_temperature,
            point_top_p=args.point_top_p,
            num_prob_samples=args.num_prob_samples,
            prob_temperature=args.prob_temperature,
            prob_top_p=args.prob_top_p,
            max_samples=args.max_samples,
        ),
        device=runtime_device,
        move_model_to_device=args.device_map is None,
    )

    selected_wrapped = eval_records_wrapped[: len(predictions)]
    gift_metrics, per_dataset_metrics, detailed_rows = summarize_gift_outputs(selected_wrapped, predictions)
    result = {
        "protocol": args.protocol,
        "train_split": args.train_split if args.protocol == "few-shot" else None,
        "eval_split": args.eval_split,
        "few_shot_metrics": few_shot_metrics,
        "forecast_eval_metrics": eval_metrics,
        "gift_eval_metrics": gift_metrics,
        "per_dataset_metrics": per_dataset_metrics,
        "num_eval_records": len(selected_wrapped),
        "dataset_filter": dataset_filter,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        _write_json(output_dir / "metrics.json", result)
        _write_jsonl(output_dir / "predictions_with_metrics.jsonl", detailed_rows)
        if args.protocol == "few-shot" and args.save_few_shot_checkpoint:
            save_textts_checkpoint(
                model,
                tokenizer,
                output_dir / "few_shot_checkpoint",
                metadata={
                    "task": "gift_eval_few_shot",
                    "few_shot_metrics": few_shot_metrics,
                    "gift_eval_metrics": gift_metrics,
                },
            )


if __name__ == "__main__":
    main()
