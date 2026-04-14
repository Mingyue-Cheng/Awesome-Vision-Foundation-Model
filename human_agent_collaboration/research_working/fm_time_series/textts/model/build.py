"""Construction helpers for loading TextTS on top of Qwen3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from textts.model.textts_model import TextTSModel, TextTSModelConfig
from textts.tokenization.tokenizer import (
    TextTSTokenizerBundle,
    TextTSTokenizerConfig,
    extend_tokenizer_and_embeddings,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional dependency
    snapshot_download = None  # type: ignore[assignment]


@dataclass
class Qwen3BuildConfig:
    base_model_name_or_path: str = "Qwen/Qwen3-1.5B-Base"
    trust_remote_code: bool = True
    torch_dtype: Optional[str] = None
    device_map: Optional[str] = None
    local_files_only: bool = False
    tokenizer_config: TextTSTokenizerConfig = TextTSTokenizerConfig()
    patch_len: int = 16
    input_dim: int = 9
    d_patch: int = 256


def _resolve_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name is None:
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{dtype_name} is not a torch dtype.")
    return dtype


def _resolve_model_source(model_name_or_path: str, *, local_files_only: bool) -> str:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return str(path)
    if not local_files_only:
        return model_name_or_path
    if snapshot_download is None:
        return model_name_or_path
    return snapshot_download(
        repo_id=model_name_or_path,
        local_files_only=True,
    )


def build_textts_from_qwen3(
    config: Optional[Qwen3BuildConfig] = None,
) -> Tuple[TextTSModel, object, TextTSTokenizerBundle]:
    """Load tokenizer + Qwen3 backbone + TextTS wrapper."""

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is required to build TextTS from Qwen3.")

    cfg = config or Qwen3BuildConfig()
    torch_dtype = _resolve_dtype(cfg.torch_dtype)
    resolved_source = _resolve_model_source(cfg.base_model_name_or_path, local_files_only=cfg.local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_source,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
        fix_mistral_regex=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        resolved_source,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=cfg.device_map,
        local_files_only=cfg.local_files_only,
    )

    tokenizer_bundle = extend_tokenizer_and_embeddings(tokenizer, llm, cfg.tokenizer_config)
    textts_config = TextTSModelConfig(
        base_model_name_or_path=cfg.base_model_name_or_path,
        hidden_size=llm.config.hidden_size,
        d_patch=cfg.d_patch,
        patch_len=cfg.patch_len,
        input_dim=cfg.input_dim,
        bos_fc_token_id=tokenizer_bundle.control_token_ids["<BOS_FC>"],
        eos_fc_token_id=tokenizer_bundle.control_token_ids["<EOS_FC>"],
        target_start_token_id=tokenizer_bundle.control_token_ids["<TARGET_START>"],
        forecast_pad_token_id=tokenizer_bundle.control_token_ids["<FORECAST_PAD>"],
        forecast_bin_token_ids=tokenizer_bundle.forecast_bin_token_ids,
        forecast_allowed_token_ids=tokenizer_bundle.forecast_allowed_token_ids,
    )
    model = TextTSModel(llm=llm, config=textts_config)
    return model, tokenizer, tokenizer_bundle
