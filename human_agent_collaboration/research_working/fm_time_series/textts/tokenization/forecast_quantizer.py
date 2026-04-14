"""Forecast-side quantization and dequantization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class QuantizationStats:
    mean: float
    std: float
    used_fallback: bool = False


@dataclass(frozen=True)
class ForecastQuantizerConfig:
    num_bins: int = 1024
    lower: float = -5.0
    upper: float = 5.0
    eps: float = 1e-5

    @property
    def bin_width(self) -> float:
        return (self.upper - self.lower) / self.num_bins


class ForecastQuantizer:
    """Implements RevIN-style instance normalization + forecast-bin quantization."""

    def __init__(self, config: Optional[ForecastQuantizerConfig] = None) -> None:
        self.config = config or ForecastQuantizerConfig()

    def compute_stats(
        self,
        values: Sequence[float] | torch.Tensor,
        *,
        missing_mask: Optional[Sequence[int] | Sequence[bool] | torch.Tensor] = None,
    ) -> QuantizationStats:
        x = torch.as_tensor(values, dtype=torch.float32)
        if missing_mask is None:
            valid = torch.ones_like(x, dtype=torch.bool)
        else:
            mask = torch.as_tensor(missing_mask, dtype=torch.bool)
            if mask.shape != x.shape:
                raise ValueError(f"missing_mask shape {tuple(mask.shape)} must match values shape {tuple(x.shape)}.")
            valid = ~mask

        valid_values = x[valid]
        if valid_values.numel() < 2:
            return QuantizationStats(mean=0.0, std=1.0, used_fallback=True)

        mean = float(valid_values.mean().item())
        std = float(valid_values.std(unbiased=False).item())
        if std < 1e-5:
            return QuantizationStats(mean=0.0, std=1.0, used_fallback=True)
        return QuantizationStats(mean=mean, std=std, used_fallback=False)

    def normalize(self, values: Sequence[float] | torch.Tensor, stats: QuantizationStats) -> torch.Tensor:
        x = torch.as_tensor(values, dtype=torch.float32)
        return (x - stats.mean) / (stats.std + self.config.eps)

    def denormalize(self, values_norm: Sequence[float] | torch.Tensor, stats: QuantizationStats) -> torch.Tensor:
        x_norm = torch.as_tensor(values_norm, dtype=torch.float32)
        return x_norm * stats.std + stats.mean

    def quantize_normalized(self, values_norm: Sequence[float] | torch.Tensor) -> torch.Tensor:
        x_norm = torch.as_tensor(values_norm, dtype=torch.float32)
        bin_ids = torch.floor((x_norm - self.config.lower) / self.config.bin_width).long()
        return torch.clamp(bin_ids, 0, self.config.num_bins - 1)

    def quantize(
        self,
        values: Sequence[float] | torch.Tensor,
        stats: QuantizationStats,
    ) -> torch.Tensor:
        return self.quantize_normalized(self.normalize(values, stats))

    def dequantize_normalized(self, bin_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        bins = torch.as_tensor(bin_ids, dtype=torch.float32)
        return self.config.lower + (bins + 0.5) * self.config.bin_width

    def dequantize(self, bin_ids: Sequence[int] | torch.Tensor, stats: QuantizationStats) -> torch.Tensor:
        x_norm = self.dequantize_normalized(bin_ids)
        return self.denormalize(x_norm, stats)

    def bin_ids_to_token_ids(
        self,
        bin_ids: Sequence[int] | torch.Tensor,
        forecast_bin_token_ids: Sequence[int],
    ) -> list[int]:
        bins = torch.as_tensor(bin_ids, dtype=torch.long).tolist()
        if len(forecast_bin_token_ids) != self.config.num_bins:
            raise ValueError("forecast_bin_token_ids length must equal num_bins.")
        return [int(forecast_bin_token_ids[idx]) for idx in bins]

    def token_ids_to_bin_ids(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        forecast_bin_token_ids: Sequence[int],
    ) -> torch.Tensor:
        token_to_bin = {int(token_id): idx for idx, token_id in enumerate(forecast_bin_token_ids)}
        ids = torch.as_tensor(token_ids, dtype=torch.long).tolist()
        return torch.as_tensor([token_to_bin[int(token_id)] for token_id in ids], dtype=torch.long)

    def build_forecast_token_ids(
        self,
        values: Sequence[float] | torch.Tensor,
        stats: QuantizationStats,
        forecast_bin_token_ids: Sequence[int],
        *,
        eos_token_id: int,
    ) -> list[int]:
        bin_ids = self.quantize(values, stats)
        token_ids = self.bin_ids_to_token_ids(bin_ids, forecast_bin_token_ids)
        token_ids.append(int(eos_token_id))
        return token_ids

