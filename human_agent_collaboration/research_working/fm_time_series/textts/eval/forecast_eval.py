"""Forecast evaluation for TextTS."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from textts.data.datasets import build_textts_collator
from textts.data.sequence_formatter import TextTSSequenceFormatter
from textts.model.build import Qwen3BuildConfig, build_textts_from_qwen3
from textts.model.checkpoint import load_textts_modules
from textts.tokenization.forecast_quantizer import ForecastQuantizer, QuantizationStats


@dataclass(frozen=True)
class ForecastEvalConfig:
    point_strategy: str = "greedy"
    point_temperature: float = 1.0
    point_top_p: float = 1.0
    num_prob_samples: int = 0
    prob_temperature: float = 1.0
    prob_top_p: float = 0.9
    max_samples: Optional[int] = None


def resolve_runtime_device(device_name: Optional[str] = None) -> torch.device:
    if device_name and device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> MutableMapping[str, Any]:
    moved: MutableMapping[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def decode_forecast_token_ids(
    token_ids: Sequence[int],
    *,
    quantizer: ForecastQuantizer,
    forecast_bin_token_ids: Sequence[int],
    stats: QuantizationStats,
    horizon: int,
) -> torch.Tensor:
    forecast_token_set = set(int(token_id) for token_id in forecast_bin_token_ids)
    filtered = [int(token_id) for token_id in token_ids if int(token_id) in forecast_token_set]
    if not filtered:
        filtered = [int(forecast_bin_token_ids[0])] * horizon
    if len(filtered) < horizon:
        filtered = filtered + [filtered[-1]] * (horizon - len(filtered))
    filtered = filtered[:horizon]
    bin_ids = quantizer.token_ids_to_bin_ids(filtered, forecast_bin_token_ids)
    return quantizer.dequantize(bin_ids, stats)


def _stack_metric_tensors(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty((0,), dtype=torch.float32)
    return torch.stack([value.to(dtype=torch.float32) for value in values], dim=0)


def _mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return float((y_pred - y_true).abs().mean().item())


def _mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return float(((y_pred - y_true) ** 2).mean().item())


def _rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return float(torch.sqrt(((y_pred - y_true) ** 2).mean()).item())


def _smape(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    denom = y_pred.abs() + y_true.abs()
    ratio = 2.0 * (y_pred - y_true).abs() / denom.clamp_min(1e-6)
    return float(ratio.mean().item())


def _mape(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    ratio = (y_pred - y_true).abs() / y_true.abs().clamp_min(1e-6)
    return float(ratio.mean().item())


def _sample_crps(samples: torch.Tensor, y_true: torch.Tensor) -> float:
    # samples: [S, H], y_true: [H]
    abs_err = (samples - y_true.unsqueeze(0)).abs().mean(dim=0)
    pairwise = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().mean(dim=(0, 1))
    crps = abs_err - 0.5 * pairwise
    return float(crps.mean().item())


def _coverage(samples: torch.Tensor, y_true: torch.Tensor, *, lower_q: float, upper_q: float) -> Tuple[float, float]:
    lower = torch.quantile(samples, lower_q, dim=0)
    upper = torch.quantile(samples, upper_q, dim=0)
    inside = ((y_true >= lower) & (y_true <= upper)).to(dtype=torch.float32)
    width = (upper - lower).mean()
    return float(inside.mean().item()), float(width.item())


def summarize_eval_outputs(
    point_predictions: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    prob_predictions: Optional[Sequence[torch.Tensor]] = None,
) -> Dict[str, float]:
    y_pred = _stack_metric_tensors(point_predictions)
    y_true = _stack_metric_tensors(targets)
    if y_pred.numel() == 0 or y_true.numel() == 0:
        return {}

    metrics: Dict[str, float] = {
        "num_samples": float(y_true.shape[0]),
        "mae": _mae(y_pred, y_true),
        "mse": _mse(y_pred, y_true),
        "rmse": _rmse(y_pred, y_true),
        "mape": _mape(y_pred, y_true),
        "smape": _smape(y_pred, y_true),
    }

    if prob_predictions:
        stacked_prob = [sample.to(dtype=torch.float32) for sample in prob_predictions]
        crps_values = []
        coverage_80 = []
        width_80 = []
        for sample_paths, target in zip(stacked_prob, targets):
            crps_values.append(_sample_crps(sample_paths, target))
            cov, width = _coverage(sample_paths, target, lower_q=0.1, upper_q=0.9)
            coverage_80.append(cov)
            width_80.append(width)
        metrics["crps"] = float(sum(crps_values) / len(crps_values))
        metrics["coverage_80"] = float(sum(coverage_80) / len(coverage_80))
        metrics["interval_width_80"] = float(sum(width_80) / len(width_80))

    return metrics


@torch.no_grad()
def evaluate_forecast_records(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    formatter: TextTSSequenceFormatter,
    collator: Any,
    quantizer: ForecastQuantizer,
    forecast_bin_token_ids: Sequence[int],
    *,
    config: Optional[ForecastEvalConfig] = None,
    device: Optional[torch.device] = None,
    move_model_to_device: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    cfg = config or ForecastEvalConfig()
    runtime_device = device or resolve_runtime_device()
    model.eval()
    if move_model_to_device:
        model.to(runtime_device)

    point_predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    prob_predictions: List[torch.Tensor] = []
    prediction_rows: List[Dict[str, Any]] = []

    selected_records = list(records[: cfg.max_samples]) if cfg.max_samples is not None else list(records)
    for index, record in enumerate(selected_records):
        target_future = record.get("target_future")
        if target_future is None:
            continue

        sample = formatter.format_prediction_sample(record)
        batch = collator([sample])
        batch = move_batch_to_device(batch, runtime_device)
        horizon = len(target_future)

        point_token_ids = model.generate_single(
            batch,
            horizon=horizon,
            strategy=cfg.point_strategy,
            temperature=cfg.point_temperature,
            top_p=cfg.point_top_p,
        )
        stats = QuantizationStats(mean=float(sample["revin_mean"]), std=float(sample["revin_std"]))
        point_pred = decode_forecast_token_ids(
            point_token_ids,
            quantizer=quantizer,
            forecast_bin_token_ids=forecast_bin_token_ids,
            stats=stats,
            horizon=horizon,
        ).cpu()
        target = torch.as_tensor(target_future, dtype=torch.float32)

        sample_paths_tensor: Optional[torch.Tensor] = None
        if cfg.num_prob_samples > 0:
            sample_paths = []
            for _ in range(cfg.num_prob_samples):
                sampled_ids = model.generate_single(
                    batch,
                    horizon=horizon,
                    strategy="sample",
                    temperature=cfg.prob_temperature,
                    top_p=cfg.prob_top_p,
                )
                sampled_pred = decode_forecast_token_ids(
                    sampled_ids,
                    quantizer=quantizer,
                    forecast_bin_token_ids=forecast_bin_token_ids,
                    stats=stats,
                    horizon=horizon,
                )
                sample_paths.append(sampled_pred.cpu())
            sample_paths_tensor = torch.stack(sample_paths, dim=0)
            prob_predictions.append(sample_paths_tensor)

        point_predictions.append(point_pred)
        targets.append(target)
        row: Dict[str, Any] = {
            "index": index,
            "domain": record.get("domain", "unknown"),
            "freq": record.get("freq", "unknown"),
            "target": target.tolist(),
            "point_prediction": point_pred.tolist(),
        }
        if sample_paths_tensor is not None:
            row["prob_samples"] = sample_paths_tensor.tolist()
        prediction_rows.append(row)

    metrics = summarize_eval_outputs(
        point_predictions=point_predictions,
        targets=targets,
        prob_predictions=prob_predictions if prob_predictions else None,
    )
    return metrics, prediction_rows


def write_eval_outputs(
    output_dir: str | Path,
    *,
    metrics: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = output_path / "metrics.json"
    predictions_path = output_path / "predictions.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metrics), handle, ensure_ascii=False, indent=2)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return output_path


def load_textts_model_for_eval(
    *,
    model_name_or_path: str,
    textts_modules_path: Optional[str],
    local_files_only: bool,
    torch_dtype: Optional[str],
    device_map: Optional[str],
) -> Tuple[torch.nn.Module, object, Any]:
    model, tokenizer, tokenizer_bundle = build_textts_from_qwen3(
        Qwen3BuildConfig(
            base_model_name_or_path=model_name_or_path,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            device_map=device_map,
            patch_len=16,
            input_dim=9,
            d_patch=256,
        )
    )
    if textts_modules_path:
        load_textts_modules(model, textts_modules_path, map_location="cpu")
    return model, tokenizer, tokenizer_bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextTS forecast evaluation.")
    parser.add_argument("--data-source", type=str, default="timemmd", choices=["csv", "timemmd"])
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--timemmd-root", type=str, default=None)
    parser.add_argument("--domains", type=str, default=None, help="Comma or space separated Time-MMD domains for joint cross-domain evaluation.")
    parser.add_argument("--target-col", type=str, default="OT")
    parser.add_argument("--timestamp-col", type=str, default="date")
    parser.add_argument("--domain", type=str, default="Energy")
    parser.add_argument("--freq", type=str, default="unknown")
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument("--split", type=str, default="test", choices=["all", "train", "val", "test"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=16)
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
    parser.add_argument("--num-prob-samples", type=int, default=0)
    parser.add_argument("--prob-temperature", type=float, default=1.0)
    parser.add_argument("--prob-top-p", type=float, default=0.9)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main() -> None:
    from textts.data.sequence_formatter import TextTSSequenceFormatter
    from textts.tokenization.forecast_quantizer import ForecastQuantizer
    from textts.training.pretrain import load_records_from_args

    parser = build_arg_parser()
    args = parser.parse_args()

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
    quantizer = ForecastQuantizer()
    formatter = TextTSSequenceFormatter(tokenizer, tokenizer_bundle, quantizer)
    collator = build_textts_collator(tokenizer_bundle)
    records = load_records_from_args(args, split=args.split)

    metrics, predictions = evaluate_forecast_records(
        model,
        records,
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
        device=resolve_runtime_device(args.device),
        move_model_to_device=args.device_map is None,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_dir:
        write_eval_outputs(args.output_dir, metrics=metrics, predictions=predictions)


if __name__ == "__main__":
    main()
