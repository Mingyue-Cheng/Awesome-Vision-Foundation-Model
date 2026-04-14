"""Tokenizer and vocabulary utilities for TextTS.

This module implements the v0.4 document decisions:
- 1024 forecast bin tokens
- 9 control tokens
- full-vocab tied lm_head + forecast-stage logits mask
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn

try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
except ImportError:  # pragma: no cover - compile-time fallback
    PreTrainedModel = object  # type: ignore[assignment]
    PreTrainedTokenizerBase = object  # type: ignore[assignment]


@dataclass(frozen=True)
class TextTSTokenizerConfig:
    """Configuration for tokenizer extension."""

    num_forecast_bins: int = 1024
    control_tokens: Sequence[str] = field(
        default_factory=lambda: (
            "<BOS_FC>",
            "<BOS_IMP>",
            "<EOS_FC>",
            "<MASK>",
            "<PAD>",
            "<UNK_FREQ>",
            "<UNK_DOMAIN>",
            "<TARGET_START>",
            "<FORECAST_PAD>",
        )
    )

    def forecast_bin_tokens(self) -> List[str]:
        return [f"<TSV_bin_{idx}>" for idx in range(self.num_forecast_bins)]


@dataclass
class TextTSTokenizerBundle:
    """Resolved token ids after tokenizer/model extension."""

    tokenizer: PreTrainedTokenizerBase
    config: TextTSTokenizerConfig
    control_token_ids: Dict[str, int]
    forecast_bin_token_ids: List[int]
    forecast_allowed_token_ids: List[int]
    vocab_size: int


def _as_list(tokens: Iterable[str]) -> List[str]:
    return list(tokens)


def build_forecast_vocab_mask(
    vocab_size: int,
    allowed_token_ids: Sequence[int],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an additive mask for forecast decoding.

    Allowed ids receive 0 and all others receive -inf.
    """

    mask = torch.full((vocab_size,), float("-inf"), device=device, dtype=dtype)
    if len(allowed_token_ids) > 0:
        mask[torch.as_tensor(list(allowed_token_ids), device=device)] = 0.0
    return mask


def _resolve_embedding_statistics(embedding: nn.Embedding) -> tuple[torch.Tensor, torch.Tensor]:
    weight = embedding.weight.detach()
    mean = weight.mean(dim=0)
    std = weight.std(dim=0).clamp_min(1e-4)
    return mean, std


def _initialize_token_rows(
    embedding: nn.Embedding,
    token_ids: Sequence[int],
    *,
    source_ids: Optional[Sequence[int]] = None,
    ordered: bool = False,
) -> None:
    if not token_ids:
        return

    with torch.no_grad():
        if source_ids:
            source = embedding.weight[torch.as_tensor(list(source_ids), device=embedding.weight.device)]
            base = source.mean(dim=0)
            values = base.unsqueeze(0).repeat(len(token_ids), 1)
        else:
            mean, std = _resolve_embedding_statistics(embedding)
            noise = torch.randn(
                len(token_ids),
                embedding.embedding_dim,
                device=embedding.weight.device,
                dtype=embedding.weight.dtype,
            )
            values = mean.unsqueeze(0) + 0.02 * std.unsqueeze(0) * noise

        if ordered and len(token_ids) > 1:
            offsets = torch.linspace(-0.05, 0.05, steps=len(token_ids), device=embedding.weight.device)
            values = values + offsets.unsqueeze(-1) * 0.02

        embedding.weight[torch.as_tensor(list(token_ids), device=embedding.weight.device)] = values


def extend_tokenizer_and_embeddings(
    tokenizer: PreTrainedTokenizerBase,
    model: Optional[PreTrainedModel] = None,
    config: Optional[TextTSTokenizerConfig] = None,
) -> TextTSTokenizerBundle:
    """Extend tokenizer and optionally resize/init model embeddings."""

    cfg = config or TextTSTokenizerConfig()
    control_tokens = _as_list(cfg.control_tokens)
    forecast_tokens = cfg.forecast_bin_tokens()
    existing_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    new_control_tokens = [token for token in control_tokens if token not in existing_vocab]
    new_forecast_tokens = [token for token in forecast_tokens if token not in existing_vocab]

    tokenizer.add_special_tokens({"additional_special_tokens": control_tokens})
    tokenizer.add_tokens(forecast_tokens, special_tokens=False)

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        input_embedding = model.get_input_embeddings()
        if input_embedding is None:
            raise ValueError("Model does not expose input embeddings.")

        control_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in control_tokens}
        forecast_ids = [tokenizer.convert_tokens_to_ids(token) for token in forecast_tokens]

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        unk_id = tokenizer.unk_token_id

        common_source_ids = [idx for idx in (bos_id, eos_id, pad_id, unk_id) if idx is not None]

        for token, token_id in control_ids.items():
            if token not in new_control_tokens:
                continue
            if token == "<EOS_FC>" and eos_id is not None:
                source_ids = [eos_id]
            elif token in {"<PAD>", "<FORECAST_PAD>"} and pad_id is not None:
                source_ids = [pad_id]
            elif token == "<MASK>" and common_source_ids:
                source_ids = common_source_ids
            elif token in {"<UNK_FREQ>", "<UNK_DOMAIN>", "<TARGET_START>"} and common_source_ids:
                source_ids = common_source_ids
            else:
                source_ids = common_source_ids
            _initialize_token_rows(input_embedding, [token_id], source_ids=source_ids, ordered=False)

        new_forecast_ids = [
            tokenizer.convert_tokens_to_ids(token)
            for token in new_forecast_tokens
        ]
        _initialize_token_rows(input_embedding, new_forecast_ids, source_ids=None, ordered=True)

        output_embedding = model.get_output_embeddings()
        if output_embedding is not None and output_embedding.weight.data_ptr() != input_embedding.weight.data_ptr():
            output_embedding.weight.data.copy_(input_embedding.weight.data)

    resolved_control_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in control_tokens}
    resolved_forecast_ids = [tokenizer.convert_tokens_to_ids(token) for token in forecast_tokens]
    eos_fc_id = resolved_control_ids["<EOS_FC>"]
    forecast_pad_id = resolved_control_ids["<FORECAST_PAD>"]
    allowed_ids = resolved_forecast_ids + [eos_fc_id, forecast_pad_id]

    return TextTSTokenizerBundle(
        tokenizer=tokenizer,
        config=cfg,
        control_token_ids=resolved_control_ids,
        forecast_bin_token_ids=resolved_forecast_ids,
        forecast_allowed_token_ids=allowed_ids,
        vocab_size=len(tokenizer),
    )
