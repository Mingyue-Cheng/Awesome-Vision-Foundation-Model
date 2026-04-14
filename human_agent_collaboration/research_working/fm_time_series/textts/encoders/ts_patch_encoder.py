"""Patch encoder for target and covariate time-series channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class TSPatchEncoderConfig:
    input_dim: int = 9
    patch_len: int = 16
    d_patch: int = 256
    role_vocab_size: int = 2


class ConvNormGELU(nn.Module):
    """Conv1d block followed by channel-wise LayerNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return self.act(x)


class TSPatchEncoder(nn.Module):
    """Shared encoder for target and continuous covariate patches.

    Input shape:
        [..., patch_len, input_dim]
    Output shape:
        [..., d_patch]
    """

    def __init__(self, config: Optional[TSPatchEncoderConfig] = None) -> None:
        super().__init__()
        self.config = config or TSPatchEncoderConfig()
        self.block1 = ConvNormGELU(self.config.input_dim, 64)
        self.block2 = ConvNormGELU(64, 128)
        self.block3 = ConvNormGELU(128, self.config.d_patch)
        self.role_embedding = nn.Embedding(self.config.role_vocab_size, self.config.d_patch)
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        patches: torch.Tensor,
        *,
        role_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if patches.ndim < 3:
            raise ValueError("patches must have at least 3 dims: [..., patch_len, input_dim].")
        if patches.shape[-1] != self.config.input_dim:
            raise ValueError(f"Expected input_dim={self.config.input_dim}, got {patches.shape[-1]}.")

        leading_shape = patches.shape[:-2]
        patch_len = patches.shape[-2]
        if patch_len != self.config.patch_len:
            raise ValueError(f"Expected patch_len={self.config.patch_len}, got {patch_len}.")

        x = patches.reshape(-1, patch_len, self.config.input_dim).permute(0, 2, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        z = x.mean(dim=-1).reshape(*leading_shape, self.config.d_patch)

        if role_ids is not None:
            if role_ids.shape != leading_shape:
                raise ValueError(
                    f"role_ids shape {tuple(role_ids.shape)} must match patch leading shape {tuple(leading_shape)}."
                )
            z = z + self.role_embedding(role_ids)
        return z

