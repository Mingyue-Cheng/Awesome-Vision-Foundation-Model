"""LTSF benchmark evaluation for TextTS.

Supported datasets:
- ETTh1 / ETTh2 / ETTm1 / ETTm2
- Weather
- Traffic
- Electricity

The evaluator follows the common long-term forecasting protocol:
- fixed prediction horizons (default 96/192/336/720)
- train/val/test split by time order (default 7:1:2)
- validation/test windows may use lookback context from earlier splits

Because TextTS currently predicts one target channel at a time, this script
evaluates multivariate LTSF datasets by looping over target columns and
averaging metrics across targets.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from textts.data.datasets import build_textts_collator
from textts.data.sequence_formatter import TextTSSequenceFormatter
from textts.eval.forecast_eval import (
    ForecastEvalConfig,
    evaluate_forecast_records,
    load_textts_model_for_eval,
    resolve_runtime_device,
)
from textts.tokenization.forecast_quantizer import ForecastQuantizer


@dataclass(frozen=True)
class LTSFDatasetSpec:
    name: str
    file_candidates: Sequence[str]
    freq: str
    timestamp_col: str = "date"


LTSF_DATASET_SPECS: dict[str, LTSFDatasetSpec] = {
    "ETTh1": LTSFDatasetSpec("ETTh1", ("ETTh1.csv", "etth1.csv"), "hourly"),
    "ETTh2": LTSFDatasetSpec("ETTh2", ("ETTh2.csv", "etth2.csv"), "hourly"),
    "ETTm1": LTSFDatasetSpec("ETTm1", ("ETTm1.csv", "ettm1.csv"), "15min"),
    "ETTm2": LTSFDatasetSpec("ETTm2", ("ETTm2.csv", "ettm2.csv"), "15min"),
    "Weather": LTSFDatasetSpec("Weather", ("weather.csv", "Weather.csv"), "10min"),
    "Traffic": LTSFDatasetSpec("Traffic", ("traffic.csv", "Traffic.csv"), "hourly"),
    "Electricity": LTSFDatasetSpec("Electricity", ("electricity.csv", "Electricity.csv"), "hourly"),
}


def _parse_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_numeric_column(rows: Sequence[Mapping[str, str]], column: str, *, sample_size: int = 32) -> bool:
    checked = 0
    for row in rows[:sample_size]:
        value = row.get(column)
        if value is None or not str(value).strip():
            return False
        try:
            float(value)
        except ValueError:
            return False
        checked += 1
    return checked > 0


def _infer_numeric_columns(rows: Sequence[Mapping[str, str]], *, timestamp_col: str) -> list[str]:
    if not rows:
        return []
    header = list(rows[0].keys())
    return [column for column in header if column != timestamp_col and _is_numeric_column(rows, column)]


def _resolve_dataset_names(datasets: str) -> list[str]:
    if datasets.strip().lower() == "all":
        return list(LTSF_DATASET_SPECS.keys())
    names = [part.strip() for chunk in datasets.split(",") for part in chunk.split() if part.strip()]
    unknown = [name for name in names if name not in LTSF_DATASET_SPECS]
    if unknown:
        raise ValueError(f"Unknown LTSF datasets: {unknown}. Expected one of {sorted(LTSF_DATASET_SPECS)}.")
    return names


def _parse_horizons(value: str) -> list[int]:
    horizons = [int(part.strip()) for chunk in value.split(",") for part in chunk.split() if part.strip()]
    if not horizons:
        raise ValueError("At least one horizon must be provided.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("All horizons must be positive integers.")
    ordered: list[int] = []
    seen = set()
    for horizon in horizons:
        if horizon not in seen:
            ordered.append(horizon)
            seen.add(horizon)
    return ordered


def _resolve_dataset_csv_path(root_dir: str | Path, spec: LTSFDatasetSpec) -> Path:
    root = Path(root_dir)
    search_paths: list[Path] = []
    for candidate in spec.file_candidates:
        search_paths.extend(
            [
                root / candidate,
                root / spec.name / candidate,
                root / spec.name.lower() / candidate,
            ]
        )
    for path in search_paths:
        if path.exists():
            return path
    for candidate in spec.file_candidates:
        matches = list(root.rglob(candidate))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not locate CSV for dataset={spec.name} under {root}. "
        f"Tried candidates={list(spec.file_candidates)}."
    )


def _split_boundaries(total: int, *, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.")
    train_end = int(total * (1.0 - val_ratio - test_ratio))
    val_end = int(total * (1.0 - test_ratio))
    train_end = min(max(train_end, 0), total)
    val_end = min(max(val_end, train_end), total)
    return train_end, val_end


def _history_end_range(
    total: int,
    *,
    split: str,
    lookback: int,
    horizon: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    train_end, val_end = _split_boundaries(total, val_ratio=val_ratio, test_ratio=test_ratio)
    min_history_end = lookback
    max_history_end = total - horizon

    if split == "all":
        start = min_history_end
        end = max_history_end
    elif split == "train":
        start = max(min_history_end, lookback)
        end = min(train_end, max_history_end)
    elif split == "val":
        start = max(train_end, min_history_end)
        end = min(val_end, max_history_end)
    elif split == "test":
        start = max(val_end, min_history_end)
        end = max_history_end
    else:
        raise ValueError(f"Unsupported split={split!r}.")

    return start, end


def _resolve_target_columns(
    rows: Sequence[Mapping[str, str]],
    *,
    timestamp_col: str,
    target_mode: str,
    target_cols_arg: Optional[str],
) -> list[str]:
    numeric_columns = _infer_numeric_columns(rows, timestamp_col=timestamp_col)
    if not numeric_columns:
        raise ValueError("No numeric target columns found in CSV.")

    if target_cols_arg:
        requested = [part.strip() for chunk in target_cols_arg.split(",") for part in chunk.split() if part.strip()]
        missing = [column for column in requested if column not in numeric_columns]
        if missing:
            raise ValueError(f"Requested target columns not found or non-numeric: {missing}")
        return requested

    if target_mode == "ot":
        if "OT" not in numeric_columns:
            raise ValueError("target_mode='ot' requires an 'OT' column in the dataset.")
        return ["OT"]
    if target_mode != "all":
        raise ValueError(f"Unsupported target_mode={target_mode!r}.")
    return numeric_columns


def load_ltsf_records_from_csv(
    csv_path: str | Path,
    *,
    dataset_name: str,
    freq: str,
    lookback: int,
    horizon: int,
    split: str,
    stride: int,
    target_col: str,
    timestamp_col: str = "date",
    max_windows: Optional[int] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> list[dict[str, object]]:
    """Load LTSF records while preserving lookback context across split boundaries."""

    rows = _parse_csv_rows(csv_path)
    if not rows:
        return []

    header = list(rows[0].keys())
    if timestamp_col not in header:
        raise ValueError(f"timestamp_col={timestamp_col!r} not found in {csv_path}. Header={header}")
    if target_col not in header:
        raise ValueError(f"target_col={target_col!r} not found in {csv_path}. Header={header}")

    numeric_columns = _infer_numeric_columns(rows, timestamp_col=timestamp_col)
    if target_col not in numeric_columns:
        raise ValueError(f"target_col={target_col!r} is not numeric in {csv_path}.")

    target_values = [float(row[target_col]) for row in rows]
    covariate_columns = [column for column in numeric_columns if column != target_col]
    covariate_values = {column: [float(row[column]) for row in rows] for column in covariate_columns}

    start_history_end, end_history_end = _history_end_range(
        len(rows),
        split=split,
        lookback=lookback,
        horizon=horizon,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    if end_history_end < start_history_end:
        return []

    records: list[dict[str, object]] = []
    for history_end in range(start_history_end, end_history_end + 1, stride):
        if max_windows is not None and len(records) >= max_windows:
            break
        history_start = history_end - lookback
        forecast_end = history_end + horizon
        if history_start < 0 or forecast_end > len(rows):
            continue

        covariates = [
            {
                "name": column,
                "values": covariate_values[column][history_start:history_end],
                "time_features": [[0.0] * 7 for _ in range(lookback)],
            }
            for column in covariate_columns
        ]
        records.append(
            {
                "domain": dataset_name,
                "freq": freq,
                "context": "",
                "target_name": target_col,
                "history_start": rows[history_start][timestamp_col],
                "history_end": rows[history_end - 1][timestamp_col],
                "forecast_end": rows[forecast_end - 1][timestamp_col],
                "target_history": target_values[history_start:history_end],
                "target_future": target_values[history_end:forecast_end],
                "target_time_features": [[0.0] * 7 for _ in range(lookback)],
                "covariates": covariates,
            }
        )
    return records


def _weighted_average_metrics(metric_rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    weights = [float(row.get("num_samples", 0.0)) for row in metric_rows]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return {}

    metric_names = sorted({key for row in metric_rows for key in row.keys() if key != "num_samples"})
    averaged: dict[str, float] = {"num_samples": total_weight}
    for metric_name in metric_names:
        weighted_sum = 0.0
        used = 0.0
        for weight, row in zip(weights, metric_rows):
            if metric_name not in row:
                continue
            weighted_sum += float(row[metric_name]) * weight
            used += weight
        if used > 0.0:
            averaged[metric_name] = weighted_sum / used
    return averaged


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
    parser = argparse.ArgumentParser(description="Run TextTS evaluation on classic LTSF benchmarks.")
    parser.add_argument("--ltsf-root", type=str, required=True, help="Root directory containing the LTSF CSV files.")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated dataset names or 'all'.")
    parser.add_argument("--horizons", type=str, default="96,192,336,720", help="Comma-separated prediction horizons.")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--split", type=str, default="test", choices=["all", "train", "val", "test"])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=None, help="Optional cap on windows per target.")
    parser.add_argument("--target-mode", type=str, default="all", choices=["all", "ot"])
    parser.add_argument("--target-cols", type=str, default=None, help="Optional explicit target columns, overriding target-mode.")
    parser.add_argument("--max-targets", type=int, default=None, help="Optional cap on target columns per dataset for debugging.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
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
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on evaluation windows after loading.")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_names = _resolve_dataset_names(args.datasets)
    horizons = _parse_horizons(args.horizons)

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
    runtime_device = resolve_runtime_device(args.device)

    summary_rows: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        spec = LTSF_DATASET_SPECS[dataset_name]
        csv_path = _resolve_dataset_csv_path(args.ltsf_root, spec)
        rows = _parse_csv_rows(csv_path)
        target_columns = _resolve_target_columns(
            rows,
            timestamp_col=spec.timestamp_col,
            target_mode=args.target_mode,
            target_cols_arg=args.target_cols,
        )
        if args.max_targets is not None:
            target_columns = target_columns[: args.max_targets]

        for horizon in horizons:
            per_target_metrics: list[dict[str, float]] = []
            per_target_rows: list[dict[str, Any]] = []
            for target_col in target_columns:
                records = load_ltsf_records_from_csv(
                    csv_path,
                    dataset_name=dataset_name,
                    freq=spec.freq,
                    lookback=args.lookback,
                    horizon=horizon,
                    split=args.split,
                    stride=args.stride,
                    target_col=target_col,
                    timestamp_col=spec.timestamp_col,
                    max_windows=args.max_windows,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                )
                if not records:
                    continue

                metrics, _ = evaluate_forecast_records(
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
                    device=runtime_device,
                    move_model_to_device=args.device_map is None,
                )
                if not metrics:
                    continue
                per_target_metrics.append(metrics)
                per_target_rows.append(
                    {
                        "dataset": dataset_name,
                        "horizon": horizon,
                        "target_col": target_col,
                        "num_windows": metrics.get("num_samples", 0.0),
                        **metrics,
                    }
                )

            aggregate_metrics = _weighted_average_metrics(per_target_metrics)
            summary_row = {
                "dataset": dataset_name,
                "csv_path": str(csv_path),
                "horizon": horizon,
                "lookback": args.lookback,
                "split": args.split,
                "target_count": len(per_target_rows),
                **aggregate_metrics,
            }
            summary_rows.append(summary_row)
            print(json.dumps(summary_row, ensure_ascii=False))

            if args.output_dir:
                horizon_dir = Path(args.output_dir) / dataset_name / f"h{horizon}"
                _write_json(horizon_dir / "metrics.json", summary_row)
                _write_jsonl(horizon_dir / "per_target_metrics.jsonl", per_target_rows)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        _write_jsonl(output_dir / "summary.jsonl", summary_rows)
        summary_payload = {"results": summary_rows}
        _write_json(output_dir / "summary.json", summary_payload)


if __name__ == "__main__":
    main()
