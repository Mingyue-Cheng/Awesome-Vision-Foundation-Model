"""Cross-channel mixer for variable-cardinality covariate sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ChannelMixerConfig:
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    ffn_hidden_dim: int = 1024
    dropout: float = 0.0


class CrossChannelBlock(nn.Module):
    def __init__(self, config: ChannelMixerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(config.ffn_hidden_dim, config.d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ChannelMixer(nn.Module):
    """Mix channels at each patch index while preserving patch order.

    Input:
        z_batch: [B, C_max, N_patch, d_model]
        channel_mask: [B, C_max] where 1=valid and 0=pad channel

    Output:
        z_out: [B, C_max, N_patch, d_model]
    """

    def __init__(self, config: Optional[ChannelMixerConfig] = None) -> None:
        super().__init__()
        self.config = config or ChannelMixerConfig()
        self.blocks = nn.ModuleList([CrossChannelBlock(self.config) for _ in range(self.config.num_layers)])

    def forward(self, z_batch: torch.Tensor, channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        squeeze_batch = False
        if z_batch.ndim == 3:
            z_batch = z_batch.unsqueeze(0)
            if channel_mask is not None and channel_mask.ndim == 1:
                channel_mask = channel_mask.unsqueeze(0)
            squeeze_batch = True

        if z_batch.ndim != 4:
            raise ValueError("z_batch must have shape [B, C, N_patch, d_model] or [C, N_patch, d_model].")
        if z_batch.shape[-1] != self.config.d_model:
            raise ValueError(f"Expected d_model={self.config.d_model}, got {z_batch.shape[-1]}.")

        batch_size, c_max, n_patch, d_model = z_batch.shape
        z_t = z_batch.permute(0, 2, 1, 3).reshape(batch_size * n_patch, c_max, d_model)

        key_padding_mask = None
        if channel_mask is not None:
            if channel_mask.shape != (batch_size, c_max):
                raise ValueError(
                    f"channel_mask shape {tuple(channel_mask.shape)} does not match {(batch_size, c_max)}."
                )
            key_padding_mask = (~channel_mask.bool()).unsqueeze(1).expand(-1, n_patch, -1).reshape(batch_size * n_patch, c_max)

        z_out = z_t
        for block in self.blocks:
            z_out = block(z_out, key_padding_mask=key_padding_mask)

        z_out = z_out.reshape(batch_size, n_patch, c_max, d_model).permute(0, 2, 1, 3)
        z_out = z_batch + z_out

        if channel_mask is not None:
            z_out = z_out * channel_mask[:, :, None, None].to(dtype=z_out.dtype)

        if squeeze_batch:
            z_out = z_out.squeeze(0)
        return z_out

