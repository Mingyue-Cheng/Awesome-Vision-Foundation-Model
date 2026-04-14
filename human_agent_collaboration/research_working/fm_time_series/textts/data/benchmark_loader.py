"""CSV benchmark loaders for TextTS.

The goal is not to cover every dataset convention, but to provide a small,
practical bridge from common LTSF-style CSV files to the raw record schema
expected by TextTSSequenceFormatter.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class CSVWindowConfig:
    """Sliding-window extraction config for CSV benchmarks."""

    target_col: str
    timestamp_col: str = "date"
    domain: str = "generic"
    freq: str = "unknown"
    context: str = ""
    lookback: int = 96
    horizon: int = 24
    stride: int = 1
    covariate_cols: Optional[Sequence[str]] = None
    max_windows: Optional[int] = None
    split: str = "all"
    val_ratio: float = 0.1
    test_ratio: float = 0.1


def _read_csv_rows(csv_path: str | Path) -> List[Dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _infer_covariate_cols(
    header: Sequence[str],
    *,
    target_col: str,
    timestamp_col: str,
    covariate_cols: Optional[Sequence[str]],
) -> List[str]:
    if covariate_cols is not None:
        return list(covariate_cols)
    return [col for col in header if col not in {target_col, timestamp_col}]


def _parse_float_column(rows: Sequence[Mapping[str, str]], column: str) -> List[float]:
    return [float(row[column]) for row in rows]


def _build_zero_time_features(length: int) -> List[List[float]]:
    return [[0.0] * 7 for _ in range(length)]


def _slice_rows_for_split(
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    val_ratio: float,
    test_ratio: float,
) -> List[Mapping[str, str]]:
    if split == "all":
        return list(rows)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split={split}. Expected one of: all, train, val, test.")
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    total = len(rows)
    train_end = int(total * (1.0 - val_ratio - test_ratio))
    val_end = int(total * (1.0 - test_ratio))
    train_end = min(max(train_end, 0), total)
    val_end = min(max(val_end, train_end), total)

    if split == "train":
        return list(rows[:train_end])
    if split == "val":
        return list(rows[train_end:val_end])
    return list(rows[val_end:])


def load_csv_windows(
    csv_path: str | Path,
    config: CSVWindowConfig,
) -> List[Dict[str, object]]:
    """Load a CSV into TextTS raw records using sliding windows."""

    rows = _read_csv_rows(csv_path)
    if not rows:
        return []

    header = list(rows[0].keys())
    if config.target_col not in header:
        raise ValueError(f"target_col={config.target_col} not found in CSV header: {header}")
    if config.timestamp_col not in header:
        raise ValueError(f"timestamp_col={config.timestamp_col} not found in CSV header: {header}")

    rows = _slice_rows_for_split(
        rows,
        split=config.split,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    if not rows:
        return []

    covariate_cols = _infer_covariate_cols(
        header,
        target_col=config.target_col,
        timestamp_col=config.timestamp_col,
        covariate_cols=config.covariate_cols,
    )

    target_values = _parse_float_column(rows, config.target_col)
    covariate_values = {col: _parse_float_column(rows, col) for col in covariate_cols}

    total_len = len(rows)
    min_required = config.lookback + config.horizon
    if total_len < min_required:
        return []

    windows: List[Dict[str, object]] = []
    max_start = total_len - min_required
    for start in range(0, max_start + 1, config.stride):
        if config.max_windows is not None and len(windows) >= config.max_windows:
            break

        hist_start = start
        hist_end = start + config.lookback
        fut_end = hist_end + config.horizon

        record: Dict[str, object] = {
            "domain": config.domain,
            "freq": config.freq,
            "context": config.context,
            "target_name": config.target_col,
            "history_start": rows[hist_start][config.timestamp_col],
            "history_end": rows[hist_end - 1][config.timestamp_col],
            "forecast_end": rows[fut_end - 1][config.timestamp_col],
            "target_history": target_values[hist_start:hist_end],
            "target_future": target_values[hist_end:fut_end],
            "target_time_features": _build_zero_time_features(config.lookback),
            "covariates": [],
        }

        covariates = []
        for covariate_col in covariate_cols:
            covariates.append(
                {
                    "name": covariate_col,
                    "values": covariate_values[covariate_col][hist_start:hist_end],
                    "time_features": _build_zero_time_features(config.lookback),
                }
            )
        record["covariates"] = covariates
        windows.append(record)

    return windows


def train_val_split(
    records: Sequence[Dict[str, object]],
    *,
    val_ratio: float = 0.1,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1).")
    split = int(len(records) * (1.0 - val_ratio))
    split = min(max(split, 0), len(records))
    return list(records[:split]), list(records[split:])
