"""Project patch latents into the Qwen hidden space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ProjectorConfig:
    d_patch: int = 256
    d_llm: int = 1536


class Projector(nn.Module):
    """Two-layer LLaVA-style projector."""

    def __init__(self, config: Optional[ProjectorConfig] = None) -> None:
        super().__init__()
        self.config = config or ProjectorConfig()
        self.linear1 = nn.Linear(self.config.d_patch, self.config.d_llm)
        self.norm = nn.LayerNorm(self.config.d_llm)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(self.config.d_llm, self.config.d_llm)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.config.d_patch:
            raise ValueError(f"Expected d_patch={self.config.d_patch}, got {z.shape[-1]}.")
        x = self.linear1(z)
        x = self.norm(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

