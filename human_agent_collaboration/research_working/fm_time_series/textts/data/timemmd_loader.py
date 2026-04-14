"""Dedicated loader for the Time-MMD dataset."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y%m%d",
)

NUMERICAL_META_COLS = {
    "",
    "date",
    "start_date",
    "end_date",
    "ValidStart",
    "ValidEnd",
    "MapDate",
    "AreaOfInterest",
    "StatisticFormatID",
}


@dataclass(frozen=True)
class TimeMMDWindowConfig:
    root_dir: str | Path
    domain: str
    lookback: int = 96
    horizon: int = 24
    stride: int = 1
    target_col: str = "OT"
    max_windows: Optional[int] = None
    include_report_text: bool = True
    include_search_text: bool = True
    max_text_items_per_source: int = 3
    domain_context_prefix: str = ""
    covariate_cols: Optional[Sequence[str]] = None
    textual_sources: Sequence[str] = field(default_factory=lambda: ("report", "search"))
    split: str = "all"
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass(frozen=True)
class TimeMMDMultiDomainConfig:
    root_dir: str | Path
    domains: Sequence[str]
    lookback: int = 96
    horizon: int = 24
    stride: int = 1
    target_col: str = "OT"
    max_windows_per_domain: Optional[int] = None
    shuffle_records: bool = True
    shuffle_seed: int = 42
    include_report_text: bool = True
    include_search_text: bool = True
    max_text_items_per_source: int = 3
    domain_context_prefix: str = ""
    split: str = "all"
    val_ratio: float = 0.1
    test_ratio: float = 0.1


def _read_csv_rows(csv_path: str | Path) -> List[Dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_datetime(value: str | None) -> Optional[datetime]:
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() == "NA":
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _coerce_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() == "NA":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _is_numeric_column(rows: Sequence[Mapping[str, str]], column: str, *, sample_size: int = 16) -> bool:
    if column in NUMERICAL_META_COLS:
        return False
    checked = 0
    numeric = 0
    for row in rows[:sample_size]:
        checked += 1
        if _coerce_float(row.get(column)) is not None:
            numeric += 1
    return checked > 0 and numeric == checked


def _infer_covariates(
    rows: Sequence[Mapping[str, str]],
    *,
    target_col: str,
    covariate_cols: Optional[Sequence[str]],
) -> List[str]:
    if covariate_cols is not None:
        return list(covariate_cols)
    header = list(rows[0].keys())
    return [col for col in header if col != target_col and _is_numeric_column(rows, col)]


def _extract_row_datetime(row: Mapping[str, str], *, prefer_end: bool) -> Optional[datetime]:
    candidates = ["end_date", "ValidEnd", "date"] if prefer_end else ["start_date", "ValidStart", "date"]
    for column in candidates:
        dt = _parse_datetime(row.get(column))
        if dt is not None:
            return dt
    return None


def _infer_frequency_label(rows: Sequence[Mapping[str, str]]) -> str:
    timestamps: List[datetime] = []
    for row in rows:
        dt = _extract_row_datetime(row, prefer_end=False)
        if dt is not None:
            timestamps.append(dt)
        if len(timestamps) >= 8:
            break

    if len(timestamps) < 2:
        return "unknown"

    deltas = []
    for prev, curr in zip(timestamps[:-1], timestamps[1:]):
        days = (curr - prev).days
        if days > 0:
            deltas.append(days)
    if not deltas:
        return "unknown"

    median_days = sorted(deltas)[len(deltas) // 2]
    if median_days <= 1:
        return "daily"
    if median_days <= 8:
        return "weekly"
    if median_days <= 31:
        return "monthly"
    if median_days <= 100:
        return "quarterly"
    if median_days <= 370:
        return "yearly"
    return "unknown"


def _load_text_entries(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
    rows = _read_csv_rows(csv_path)
    entries: List[Dict[str, object]] = []
    for row in rows:
        fact = (row.get("fact") or "").strip()
        if not fact or fact.upper() == "NA":
            continue
        end_dt = _parse_datetime(row.get("end_date"))
        start_dt = _parse_datetime(row.get("start_date"))
        if end_dt is None:
            continue
        entries.append(
            {
                "start_date": start_dt,
                "end_date": end_dt,
                "fact": fact,
            }
        )
    entries.sort(key=lambda item: item["end_date"])  # type: ignore[index]
    return entries


def _select_recent_facts(
    entries: Sequence[Mapping[str, object]],
    *,
    history_end: datetime,
    max_items: int,
) -> List[str]:
    valid = [entry for entry in entries if entry["end_date"] <= history_end]  # type: ignore[index]
    selected = valid[-max_items:]
    return [str(entry["fact"]) for entry in selected]


def _zero_time_features(length: int) -> List[List[float]]:
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


def _build_context(
    domain: str,
    history_end: datetime,
    *,
    config: TimeMMDWindowConfig,
    report_entries: Sequence[Mapping[str, object]],
    search_entries: Sequence[Mapping[str, object]],
) -> str:
    lines: List[str] = []
    if config.domain_context_prefix:
        lines.append(config.domain_context_prefix.strip())
    else:
        lines.append(f"Time-MMD domain: {domain}. Context only uses text whose end_date <= history end.")
    lines.append(f"History end date: {history_end.date().isoformat()}")

    if config.include_report_text:
        facts = _select_recent_facts(
            report_entries,
            history_end=history_end,
            max_items=config.max_text_items_per_source,
        )
        for fact in facts:
            lines.append(f"[REPORT] {fact}")

    if config.include_search_text:
        facts = _select_recent_facts(
            search_entries,
            history_end=history_end,
            max_items=config.max_text_items_per_source,
        )
        for fact in facts:
            lines.append(f"[SEARCH] {fact}")

    return "\n".join(lines)


def load_timemmd_windows(config: TimeMMDWindowConfig) -> List[Dict[str, object]]:
    """Load one Time-MMD domain into TextTS raw records.

    Numerical data provides target/covariates.
    Textual context only uses the `fact` field from report/search files.
    `pred` / `preds` are intentionally ignored to avoid future leakage.
    """

    root_dir = Path(config.root_dir)
    numerical_path = root_dir / "numerical" / config.domain / f"{config.domain}.csv"
    report_path = root_dir / "textual" / config.domain / f"{config.domain}_report.csv"
    search_path = root_dir / "textual" / config.domain / f"{config.domain}_search.csv"

    rows = _read_csv_rows(numerical_path)
    if not rows:
        return []
    rows = _slice_rows_for_split(
        rows,
        split=config.split,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    if not rows:
        return []

    covariate_cols = _infer_covariates(rows, target_col=config.target_col, covariate_cols=config.covariate_cols)
    report_entries = _load_text_entries(report_path)
    search_entries = _load_text_entries(search_path)
    freq = _infer_frequency_label(rows)

    total_len = len(rows)
    min_required = config.lookback + config.horizon
    if total_len < min_required:
        return []

    target_values = [_coerce_float(row.get(config.target_col)) for row in rows]
    if any(value is None for value in target_values):
        raise ValueError(f"Non-numeric or missing target values found in column {config.target_col}.")
    target_series = [float(value) for value in target_values if value is not None]

    covariate_series: Dict[str, List[float]] = {}
    for covariate_col in covariate_cols:
        values = [_coerce_float(row.get(covariate_col)) for row in rows]
        if any(value is None for value in values):
            continue
        covariate_series[covariate_col] = [float(value) for value in values if value is not None]

    windows: List[Dict[str, object]] = []
    max_start = total_len - min_required
    for start in range(0, max_start + 1, config.stride):
        if config.max_windows is not None and len(windows) >= config.max_windows:
            break

        hist_start = start
        hist_end = start + config.lookback
        fut_end = hist_end + config.horizon

        history_end_dt = _extract_row_datetime(rows[hist_end - 1], prefer_end=True)
        if history_end_dt is None:
            continue
        history_start_dt = _extract_row_datetime(rows[hist_start], prefer_end=False)
        forecast_end_dt = _extract_row_datetime(rows[fut_end - 1], prefer_end=True)

        covariates = []
        for covariate_col, values in covariate_series.items():
            covariates.append(
                {
                    "name": covariate_col,
                    "values": values[hist_start:hist_end],
                    "time_features": _zero_time_features(config.lookback),
                }
            )

        record: Dict[str, object] = {
            "domain": config.domain,
            "freq": freq,
            "context": _build_context(
                config.domain,
                history_end_dt,
                config=config,
                report_entries=report_entries,
                search_entries=search_entries,
            ),
            "target_name": config.target_col,
            "history_start": history_start_dt.isoformat() if history_start_dt is not None else "",
            "history_end": history_end_dt.isoformat(),
            "forecast_end": forecast_end_dt.isoformat() if forecast_end_dt is not None else "",
            "target_history": target_series[hist_start:hist_end],
            "target_future": target_series[hist_end:fut_end],
            "target_time_features": _zero_time_features(config.lookback),
            "covariates": covariates,
        }
        windows.append(record)

    return windows


def load_timemmd_multi_domain_windows(
    config: TimeMMDMultiDomainConfig,
) -> List[Dict[str, object]]:
    """Load and merge multiple Time-MMD domains into one record list."""

    merged: List[Dict[str, object]] = []
    for domain in config.domains:
        domain_records = load_timemmd_windows(
            TimeMMDWindowConfig(
                root_dir=config.root_dir,
                domain=domain,
                lookback=config.lookback,
                horizon=config.horizon,
                stride=config.stride,
                target_col=config.target_col,
                max_windows=config.max_windows_per_domain,
                include_report_text=config.include_report_text,
                include_search_text=config.include_search_text,
                max_text_items_per_source=config.max_text_items_per_source,
                domain_context_prefix=config.domain_context_prefix,
                split=config.split,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
            )
        )
        merged.extend(domain_records)

    if config.shuffle_records:
        rng = random.Random(config.shuffle_seed)
        rng.shuffle(merged)

    return merged
