"""Convert raw multimodal time-series records into TextTS training samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import torch

from textts.tokenization.forecast_quantizer import ForecastQuantizer, QuantizationStats
from textts.tokenization.tokenizer import TextTSTokenizerBundle


@dataclass(frozen=True)
class TextTSSequenceFormatterConfig:
    patch_len: int = 16
    input_dim: int = 9
    time_feature_dim: int = 7
    imputation_mask_ratio: float = 0.3


class TextTSSequenceFormatter:
    """Build token/text sections and patch tensors from raw records.

    Expected record schema:
    - domain: str
    - freq: str
    - context: optional str
    - target_history: sequence[float]
    - target_future: optional sequence[float]
    - target_missing_mask: optional sequence[bool/int]
    - target_time_features: optional [L, 7]
    - covariates: optional list[dict(name, values, missing_mask?, time_features?)]
    - covariate_categories: optional mapping[str, str | sequence[str]]
    """

    def __init__(
        self,
        tokenizer: Any,
        tokenizer_bundle: TextTSTokenizerBundle,
        quantizer: ForecastQuantizer,
        config: Optional[TextTSSequenceFormatterConfig] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.bundle = tokenizer_bundle
        self.quantizer = quantizer
        self.config = config or TextTSSequenceFormatterConfig()

    def _encode_text(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "__call__"):
            encoded = self.tokenizer(text, add_special_tokens=False)
            if isinstance(encoded, Mapping) and "input_ids" in encoded:
                return list(encoded["input_ids"])
        if hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        raise TypeError("Tokenizer must implement __call__ or encode returning input_ids.")

    def _to_bool_mask(
        self,
        values: Sequence[float] | torch.Tensor,
        missing_mask: Optional[Sequence[int] | Sequence[bool] | torch.Tensor],
    ) -> torch.Tensor:
        length = torch.as_tensor(values).shape[0]
        if missing_mask is None:
            return torch.zeros((length,), dtype=torch.bool)
        mask = torch.as_tensor(missing_mask, dtype=torch.bool)
        if mask.shape != (length,):
            raise ValueError(f"missing_mask shape {tuple(mask.shape)} must equal ({length},).")
        return mask

    def _to_time_features(
        self,
        length: int,
        time_features: Optional[Sequence[Sequence[float]] | torch.Tensor],
    ) -> torch.Tensor:
        if time_features is None:
            return torch.zeros((length, self.config.time_feature_dim), dtype=torch.float32)
        feats = torch.as_tensor(time_features, dtype=torch.float32)
        expected_shape = (length, self.config.time_feature_dim)
        if tuple(feats.shape) != expected_shape:
            raise ValueError(f"time_features shape {tuple(feats.shape)} must equal {expected_shape}.")
        return feats

    def _pad_to_patch_multiple(self, x: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        remainder = x.shape[0] % self.config.patch_len
        if remainder == 0:
            return x
        pad_len = self.config.patch_len - remainder
        pad_shape = (pad_len, *x.shape[1:])
        pad = torch.full(pad_shape, pad_value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    def _build_patch_tensor(
        self,
        values: Sequence[float] | torch.Tensor,
        *,
        missing_mask: Optional[Sequence[int] | Sequence[bool] | torch.Tensor] = None,
        time_features: Optional[Sequence[Sequence[float]] | torch.Tensor] = None,
        stats: Optional[QuantizationStats] = None,
    ) -> torch.Tensor:
        x = torch.as_tensor(values, dtype=torch.float32)
        mask = self._to_bool_mask(values, missing_mask)
        feats = self._to_time_features(x.shape[0], time_features)
        stats = stats or self.quantizer.compute_stats(x, missing_mask=mask)
        x_norm = self.quantizer.normalize(x, stats)
        x_norm = x_norm.masked_fill(mask, 0.0)

        per_step = torch.cat(
            [
                x_norm.unsqueeze(-1),
                mask.to(dtype=torch.float32).unsqueeze(-1),
                feats,
            ],
            dim=-1,
        )
        if per_step.shape[-1] != self.config.input_dim:
            raise ValueError(f"Expected per-step dim={self.config.input_dim}, got {per_step.shape[-1]}.")
        per_step = self._pad_to_patch_multiple(per_step, pad_value=0.0)
        return per_step.reshape(-1, self.config.patch_len, self.config.input_dim)

    def build_text_prompt(self, record: Mapping[str, Any]) -> str:
        lines = [
            f"[DOMAIN] {record.get('domain', 'unknown')}",
            f"[FREQ] {record.get('freq', 'unknown')}",
            f"[CONTEXT] {record.get('context', '')}".rstrip(),
        ]

        covariate_categories = record.get("covariate_categories", {}) or {}
        for key, value in covariate_categories.items():
            if isinstance(value, (list, tuple)):
                value = " ".join(map(str, value))
            lines.append(f"[COV_CAT] {key}: {value}")

        covariates = record.get("covariates", []) or []
        if covariates:
            target_name = record.get("target_name", "target")
            cov_names = ", ".join(str(cov.get("name", "cov")) for cov in covariates)
            lines.append(f"[CHANNEL_META] target={target_name}; covariates={cov_names}")

        return "\n".join(lines)

    def format_prediction_sample(self, record: Mapping[str, Any]) -> MutableMapping[str, Any]:
        target_history = torch.as_tensor(record["target_history"], dtype=torch.float32)
        target_missing_mask = self._to_bool_mask(target_history, record.get("target_missing_mask"))
        target_stats = self.quantizer.compute_stats(target_history, missing_mask=target_missing_mask)

        sample: MutableMapping[str, Any] = {
            "text_input_ids": self._encode_text(self.build_text_prompt(record)),
            "target_patches": self._build_patch_tensor(
                target_history,
                missing_mask=target_missing_mask,
                time_features=record.get("target_time_features"),
                stats=target_stats,
            ),
            "covariate_patches": [],
            "prefix_control_token_id": self.bundle.control_token_ids["<BOS_FC>"],
            "revin_mean": target_stats.mean,
            "revin_std": target_stats.std,
        }

        for covariate in record.get("covariates", []) or []:
            cov_values = covariate["values"]
            cov_mask = covariate.get("missing_mask")
            cov_stats = self.quantizer.compute_stats(cov_values, missing_mask=cov_mask)
            sample["covariate_patches"].append(
                self._build_patch_tensor(
                    cov_values,
                    missing_mask=cov_mask,
                    time_features=covariate.get("time_features"),
                    stats=cov_stats,
                )
            )

        future_values = record.get("target_future")
        if future_values is not None:
            sample["forecast_token_ids"] = self.quantizer.build_forecast_token_ids(
                future_values,
                target_stats,
                self.bundle.forecast_bin_token_ids,
                eos_token_id=self.bundle.control_token_ids["<EOS_FC>"],
            )

        return sample

    def format_imputation_sample(
        self,
        record: Mapping[str, Any],
        *,
        seed: Optional[int] = None,
    ) -> MutableMapping[str, Any]:
        target_history = torch.as_tensor(record["target_history"], dtype=torch.float32)
        target_missing_mask = self._to_bool_mask(target_history, record.get("target_missing_mask"))
        target_stats = self.quantizer.compute_stats(target_history, missing_mask=target_missing_mask)

        patch_count = (target_history.shape[0] + self.config.patch_len - 1) // self.config.patch_len
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        mask_count = max(1, int(round(patch_count * self.config.imputation_mask_ratio)))
        patch_indices = torch.randperm(patch_count, generator=generator)[:mask_count]
        corrupted_values = target_history.clone()
        corrupted_mask = target_missing_mask.clone()
        masked_targets: list[torch.Tensor] = []

        for patch_idx in sorted(patch_indices.tolist()):
            start = patch_idx * self.config.patch_len
            end = min((patch_idx + 1) * self.config.patch_len, target_history.shape[0])
            masked_targets.append(target_history[start:end])
            corrupted_values[start:end] = 0.0
            corrupted_mask[start:end] = True

        sample: MutableMapping[str, Any] = {
            "text_input_ids": self._encode_text(self.build_text_prompt(record)),
            "target_patches": self._build_patch_tensor(
                corrupted_values,
                missing_mask=corrupted_mask,
                time_features=record.get("target_time_features"),
                stats=target_stats,
            ),
            "covariate_patches": [],
            "prefix_control_token_id": self.bundle.control_token_ids["<BOS_IMP>"],
            "revin_mean": target_stats.mean,
            "revin_std": target_stats.std,
            "masked_patch_indices": sorted(patch_indices.tolist()),
        }

        for covariate in record.get("covariates", []) or []:
            cov_values = covariate["values"]
            cov_mask = covariate.get("missing_mask")
            cov_stats = self.quantizer.compute_stats(cov_values, missing_mask=cov_mask)
            sample["covariate_patches"].append(
                self._build_patch_tensor(
                    cov_values,
                    missing_mask=cov_mask,
                    time_features=covariate.get("time_features"),
                    stats=cov_stats,
                )
            )

        masked_values = torch.cat(masked_targets, dim=0) if masked_targets else target_history[:1]
        sample["forecast_token_ids"] = self.quantizer.build_forecast_token_ids(
            masked_values,
            target_stats,
            self.bundle.forecast_bin_token_ids,
            eos_token_id=self.bundle.control_token_ids["<EOS_FC>"],
        )
        return sample

    def format_sft_sample(self, record: Mapping[str, Any]) -> MutableMapping[str, Any]:
        return self.format_prediction_sample(record)

