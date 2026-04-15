"""Protocol metadata helpers for benchmark evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ALLOWED_REGIMES = {"auto", "id", "ood", "ood_fewshot", "unknown"}


def parse_name_list(value: Optional[str]) -> list[str]:
    if value is None or not value.strip():
        return []
    names = [part.strip() for chunk in value.split(",") for part in chunk.split() if part.strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _coerce_name_list(value: object) -> list[str]:
    if isinstance(value, str):
        return parse_name_list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        names: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                names.append(text)
        seen: set[str] = set()
        ordered: list[str] = []
        for name in names:
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered
    return []


def _extract_train_entities(payload: Mapping[str, Any]) -> list[str]:
    candidates = [
        payload.get("train_datasets"),
        payload.get("datasets"),
        payload.get("domains"),
        payload.get("train_domains"),
    ]
    protocol = payload.get("protocol")
    if isinstance(protocol, Mapping):
        candidates.extend(
            [
                protocol.get("train_datasets"),
                protocol.get("datasets"),
                protocol.get("domains"),
                protocol.get("train_domains"),
            ]
        )
    domain_selection = payload.get("domain_selection")
    if isinstance(domain_selection, Mapping):
        candidates.extend(
            [
                domain_selection.get("included_domains"),
                domain_selection.get("domains"),
            ]
        )
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.extend(
            [
                metadata.get("train_datasets"),
                metadata.get("datasets"),
                metadata.get("domains"),
                metadata.get("train_domains"),
            ]
        )

    for candidate in candidates:
        names = _coerce_name_list(candidate)
        if names:
            return names
    return []


def load_train_entities_from_manifest(manifest_path: Optional[str]) -> tuple[list[str], Optional[dict[str, Any]]]:
    if not manifest_path:
        return [], None
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict-like train manifest in {path}.")
    train_entities = _extract_train_entities(payload)
    return train_entities, payload


def resolve_protocol_metadata(
    *,
    benchmark: str,
    eval_entities: Sequence[str],
    requested_regime: str,
    explicit_train_entities: Sequence[str],
    train_manifest_path: Optional[str],
    enforce_protocol: bool,
) -> dict[str, Any]:
    regime = requested_regime.strip().lower()
    if regime not in ALLOWED_REGIMES:
        raise ValueError(f"Unsupported regime={requested_regime!r}. Expected one of {sorted(ALLOWED_REGIMES)}.")

    manifest_train_entities, manifest_payload = load_train_entities_from_manifest(train_manifest_path)
    train_entities = list(explicit_train_entities) if explicit_train_entities else list(manifest_train_entities)

    eval_set = set(str(item) for item in eval_entities)
    train_set = set(str(item) for item in train_entities)
    overlap = sorted(eval_set & train_set)

    inferred_regime = "unknown"
    if train_entities:
        inferred_regime = "id" if overlap else "ood"

    final_regime = inferred_regime if regime == "auto" else regime
    compatible = True
    violation_reason: Optional[str] = None
    if final_regime == "id" and train_entities and not overlap:
        compatible = False
        violation_reason = "Requested ID protocol, but no overlap was found between train and eval entities."
    elif final_regime == "ood" and overlap:
        compatible = False
        violation_reason = "Requested OOD protocol, but train/eval overlap was detected."
    elif final_regime == "ood_fewshot" and overlap:
        compatible = False
        violation_reason = "Requested OOD few-shot protocol, but train/eval overlap was detected before adaptation."

    if enforce_protocol and not compatible:
        raise ValueError(violation_reason or "Protocol enforcement failed.")

    notes: list[str] = []
    if not train_entities:
        notes.append("No train entities were provided, so regime inference is limited.")
    if final_regime == "ood_fewshot":
        notes.append("OOD few-shot means target benchmark is treated as unseen before adaptation; caller must ensure this upstream.")
    if overlap:
        notes.append(f"Detected overlap entities: {', '.join(overlap)}")

    return {
        "benchmark": benchmark,
        "requested_regime": regime,
        "inferred_regime": inferred_regime,
        "final_regime": final_regime,
        "train_entities": train_entities,
        "eval_entities": sorted(eval_set),
        "overlap_entities": overlap,
        "overlap_count": len(overlap),
        "train_entities_source": "cli" if explicit_train_entities else ("manifest" if manifest_train_entities else "unknown"),
        "train_manifest_path": train_manifest_path,
        "protocol_check_passed": compatible,
        "protocol_violation": violation_reason,
        "notes": notes,
        "train_manifest_preview": {
            "keys": sorted(manifest_payload.keys()) if isinstance(manifest_payload, dict) else [],
        }
        if manifest_payload is not None
        else None,
    }
