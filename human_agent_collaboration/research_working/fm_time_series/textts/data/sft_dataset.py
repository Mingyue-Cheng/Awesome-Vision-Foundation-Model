"""SFT dataset construction utilities for TextTS.

This module upgrades SFT from "prediction samples reused for training" to a
context-aware dataset builder with three context levels:

- L0: empty context
- L1: deterministic template context derived from metadata/statistics
- L2: rich context loaded from a cached JSONL file, with fallback to L1
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

from torch.utils.data import Dataset

from textts.data.sequence_formatter import TextTSSequenceFormatter


ARTIFACT_PATTERNS = (
    "i'm sorry",
    "as an ai",
    "i cannot",
    "i can't",
    "i do not have",
    "i don't have",
    "based on the context provided",
    "请提供",
)


@dataclass(frozen=True)
class SFTDatasetConfig:
    """Configuration for SFT context construction."""

    context_mode: str = "mixed"
    l0_ratio: float = 0.2
    l1_ratio: float = 0.3
    l2_ratio: float = 0.5
    l2_context_path: Optional[str] = None
    use_record_context_as_l2_fallback: bool = True
    max_covariates_in_template: int = 8

    def validate(self) -> None:
        valid_modes = {"mixed", "l0", "l1", "l2", "all"}
        if self.context_mode not in valid_modes:
            raise ValueError(f"context_mode must be one of {sorted(valid_modes)}, got {self.context_mode!r}.")
        ratios = (self.l0_ratio, self.l1_ratio, self.l2_ratio)
        if any(ratio < 0.0 for ratio in ratios):
            raise ValueError("SFT context ratios must be non-negative.")
        if self.context_mode == "mixed":
            total = sum(ratios)
            if total <= 0.0:
                raise ValueError("At least one SFT context ratio must be positive when context_mode='mixed'.")

    @classmethod
    def from_env(cls) -> "SFTDatasetConfig":
        cache_path = os.environ.get("TEXTTS_SFT_CONTEXT_CACHE")
        return cls(l2_context_path=cache_path or None)


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_float_list(values: object) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    parsed: list[float] = []
    for value in values:
        numeric = _safe_float(value)
        if numeric is not None:
            parsed.append(numeric)
    return parsed


def _missing_rate(record: Mapping[str, Any]) -> float:
    history = record.get("target_history")
    values = _to_float_list(history)
    if not values:
        return 0.0
    missing_mask = record.get("target_missing_mask")
    if not isinstance(missing_mask, Sequence) or isinstance(missing_mask, (str, bytes)):
        return 0.0
    missing = 0
    total = min(len(values), len(missing_mask))
    if total == 0:
        return 0.0
    for flag in list(missing_mask)[:total]:
        missing += 1 if bool(flag) else 0
    return missing / total


def _summarize_stats(values: Sequence[float]) -> str:
    if not values:
        return "mean=NA, std=NA, min=NA, max=NA"
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / max(count - 1, 1)
    std = variance ** 0.5
    min_value = min(values)
    max_value = max(values)
    return f"mean={mean:.4g}, std={std:.4g}, min={min_value:.4g}, max={max_value:.4g}"


def is_valid_context(text: object) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < 30:
        return False
    lowered = stripped.lower()
    return not any(pattern in lowered for pattern in ARTIFACT_PATTERNS)


def build_context_cache_key(record: Mapping[str, Any]) -> str:
    """Build a stable cache key for one raw SFT record."""

    parts = [
        str(record.get("domain", "unknown")),
        str(record.get("freq", "unknown")),
        str(record.get("target_name", "target")),
        str(record.get("history_start", "")),
        str(record.get("history_end", "")),
        str(record.get("forecast_end", "")),
        str(len(record.get("target_history", []) or [])),
        str(len(record.get("target_future", []) or [])),
    ]
    if any(part for part in parts[3:6]):
        return "|".join(parts)

    history = _to_float_list(record.get("target_history"))
    signature_values = history[:4] + history[-4:]
    signature = ",".join(f"{value:.6g}" for value in signature_values)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return "|".join(parts[:3] + [digest])


def load_l2_context_cache(cache_path: Optional[str | Path]) -> Dict[str, str]:
    """Load cached L2 contexts from JSONL.

    Supported row formats:
    - {"key": "...", "context": "..."}
    - {"context_key": "...", "text": "..."}
    - {"domain": "...", "freq": "...", "target_name": "...", "history_start": "...", "history_end": "...", "context": "..."}
    """

    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}

    cache: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            context = payload.get("context", payload.get("text"))
            if not is_valid_context(context):
                continue
            key = payload.get("key") or payload.get("context_key")
            if not isinstance(key, str) or not key.strip():
                key = build_context_cache_key(payload)
            cache[key] = str(context).strip()
    return cache


def build_template_context(
    record: Mapping[str, Any],
    *,
    max_covariates_in_template: int = 8,
) -> str:
    """Build deterministic L1 context from metadata and simple statistics."""

    history = _to_float_list(record.get("target_history"))
    future = _to_float_list(record.get("target_future"))
    covariates = record.get("covariates", []) or []
    covariate_names = [str(item.get("name", "cov")) for item in covariates if isinstance(item, Mapping)]
    if len(covariate_names) > max_covariates_in_template:
        covariate_names = covariate_names[:max_covariates_in_template] + ["..."]

    covariate_text = ", ".join(covariate_names) if covariate_names else "none"
    missing_rate = _missing_rate(record)
    stats_text = _summarize_stats(history)
    history_start = record.get("history_start")
    history_end = record.get("history_end")

    lines = [
        f"Domain: {record.get('domain', 'unknown')}. Frequency: {record.get('freq', 'unknown')}.",
        f"Target variable: {record.get('target_name', 'target')}. Covariates: {covariate_text}.",
        (
            f"Historical window: {len(history)} steps. Forecast horizon: {len(future)} steps. "
            f"Missing rate: {missing_rate:.1%}. Statistics: {stats_text}."
        ),
    ]
    if history_start or history_end:
        lines.append(f"Observed range: {history_start or 'unknown'} to {history_end or 'unknown'}.")
    return " ".join(line.strip() for line in lines if line.strip())


def resolve_l2_context(
    record: Mapping[str, Any],
    *,
    cache: Mapping[str, str],
    config: SFTDatasetConfig,
) -> str:
    cache_key = build_context_cache_key(record)
    cached = cache.get(cache_key)
    if is_valid_context(cached):
        return str(cached).strip()

    if config.use_record_context_as_l2_fallback:
        record_context = record.get("context")
        if is_valid_context(record_context):
            return str(record_context).strip()

    return build_template_context(record, max_covariates_in_template=config.max_covariates_in_template)


def _context_level_for_record(
    record: Mapping[str, Any],
    *,
    index: int,
    config: SFTDatasetConfig,
) -> str:
    if config.context_mode in {"l0", "l1", "l2"}:
        return config.context_mode.upper()
    if config.context_mode == "all":
        raise ValueError("context_mode='all' must be handled by the caller.")

    key = f"{build_context_cache_key(record)}|{index}"
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    total = config.l0_ratio + config.l1_ratio + config.l2_ratio
    l0_threshold = config.l0_ratio / total
    l1_threshold = (config.l0_ratio + config.l1_ratio) / total
    if bucket < l0_threshold:
        return "L0"
    if bucket < l1_threshold:
        return "L1"
    return "L2"


def _apply_context_level(
    record: Mapping[str, Any],
    *,
    level: str,
    cache: Mapping[str, str],
    config: SFTDatasetConfig,
) -> Dict[str, Any]:
    enriched = dict(record)
    cache_key = build_context_cache_key(record)
    if level == "L0":
        context = ""
    elif level == "L1":
        context = build_template_context(record, max_covariates_in_template=config.max_covariates_in_template)
    elif level == "L2":
        context = resolve_l2_context(record, cache=cache, config=config)
    else:
        raise ValueError(f"Unsupported SFT context level: {level}")

    enriched["context"] = context
    enriched["sft_context_level"] = level
    enriched["sft_context_cache_key"] = cache_key
    return enriched


def build_sft_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Optional[SFTDatasetConfig] = None,
) -> list[Dict[str, Any]]:
    """Convert raw records into SFT-ready records with explicit context levels."""

    cfg = config or SFTDatasetConfig.from_env()
    cfg.validate()
    cache = load_l2_context_cache(cfg.l2_context_path)

    if cfg.context_mode == "all":
        expanded: list[Dict[str, Any]] = []
        for record in records:
            for level in ("L0", "L1", "L2"):
                expanded.append(_apply_context_level(record, level=level, cache=cache, config=cfg))
        return expanded

    prepared: list[Dict[str, Any]] = []
    for index, record in enumerate(records):
        level = _context_level_for_record(record, index=index, config=cfg)
        prepared.append(_apply_context_level(record, level=level, cache=cache, config=cfg))
    return prepared


class TextTSSFTInstructionDataset(Dataset[MutableMapping[str, Any]]):
    """SFT dataset with explicit context construction."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        formatter: TextTSSequenceFormatter,
        *,
        config: Optional[SFTDatasetConfig] = None,
    ) -> None:
        self.records = build_sft_records(records, config=config)
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MutableMapping[str, Any]:
        return self.formatter.format_sft_sample(self.records[index])


__all__ = [
    "ARTIFACT_PATTERNS",
    "SFTDatasetConfig",
    "TextTSSFTInstructionDataset",
    "build_context_cache_key",
    "build_sft_records",
    "build_template_context",
    "is_valid_context",
    "load_l2_context_cache",
]
