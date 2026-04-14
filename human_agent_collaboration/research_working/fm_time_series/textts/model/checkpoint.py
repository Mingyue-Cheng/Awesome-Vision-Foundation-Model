"""Checkpoint helpers for TextTS."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from textts.model.textts_model import TextTSModel


def save_textts_checkpoint(
    model: TextTSModel,
    tokenizer: object,
    output_dir: str | Path,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llm_dir = output_path / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(llm_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(llm_dir)

    modules_payload = {
        "textts_config": asdict(model.config),
        "patch_encoder": model.patch_encoder.state_dict(),
        "channel_mixer": model.channel_mixer.state_dict(),
        "projector": model.projector.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(modules_payload, output_path / "textts_modules.pt")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), output_path / "optimizer.pt")

    with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata or {}), handle, ensure_ascii=False, indent=2)

    return output_path


def load_textts_modules(
    model: TextTSModel,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    model.patch_encoder.load_state_dict(payload["patch_encoder"])
    model.channel_mixer.load_state_dict(payload["channel_mixer"])
    model.projector.load_state_dict(payload["projector"])
    return payload
