"""Batch collation for TextTS multimodal inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import torch


@dataclass
class TextTSCollatorConfig:
    pad_token_id: int
    forecast_pad_token_id: int
    default_prefix_control_token_id: int
    patch_len: int = 16
    input_dim: int = 9


def _as_long_tensor(values: Sequence[int]) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=torch.long)


def _as_float_tensor(values: Any) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


class TextTSCollator:
    """Pad text, channels and forecast outputs into a batch.

    Expected sample schema:
    - text_input_ids: Sequence[int]
    - target_patches: Tensor/array [N_patch, patch_len, input_dim]
    - covariate_patches: optional list of Tensor/array [N_patch, patch_len, input_dim]
    - forecast_token_ids: optional Sequence[int], expected to include EOS if used for training
    - prefix_control_token_id: optional int, defaults to BOS_FC
    """

    def __init__(self, config: TextTSCollatorConfig) -> None:
        self.config = config

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> MutableMapping[str, torch.Tensor]:
        if not samples:
            raise ValueError("TextTSCollator received an empty batch.")

        text_tensors = [_as_long_tensor(sample["text_input_ids"]) for sample in samples]
        target_patches = [_as_float_tensor(sample["target_patches"]) for sample in samples]
        covariate_lists = [
            [_as_float_tensor(patch_tensor) for patch_tensor in sample.get("covariate_patches", [])]
            for sample in samples
        ]

        batch_size = len(samples)
        text_max_len = max(t.shape[0] for t in text_tensors)
        c_max = max(1 + len(covs) for covs in covariate_lists)
        n_patch_max = max(tp.shape[0] for tp in target_patches)

        text_input_ids = torch.full((batch_size, text_max_len), self.config.pad_token_id, dtype=torch.long)
        text_attention_mask = torch.zeros((batch_size, text_max_len), dtype=torch.long)

        channel_patches = torch.zeros(
            (batch_size, c_max, n_patch_max, self.config.patch_len, self.config.input_dim),
            dtype=torch.float32,
        )
        channel_mask = torch.zeros((batch_size, c_max), dtype=torch.bool)
        patch_mask = torch.zeros((batch_size, c_max, n_patch_max), dtype=torch.bool)

        forecast_token_ids: Optional[torch.Tensor] = None
        forecast_attention_mask: Optional[torch.Tensor] = None
        forecast_labels: Optional[torch.Tensor] = None
        prefix_control_token_ids = torch.full(
            (batch_size,),
            self.config.default_prefix_control_token_id,
            dtype=torch.long,
        )
        revin_mean = torch.zeros((batch_size,), dtype=torch.float32)
        revin_std = torch.ones((batch_size,), dtype=torch.float32)
        if all("forecast_token_ids" in sample for sample in samples):
            forecast_sequences = [_as_long_tensor(sample["forecast_token_ids"]) for sample in samples]
            h_max = max(seq.shape[0] for seq in forecast_sequences)
            forecast_token_ids = torch.full((batch_size, h_max), self.config.forecast_pad_token_id, dtype=torch.long)
            forecast_attention_mask = torch.zeros((batch_size, h_max), dtype=torch.long)
            forecast_labels = torch.full((batch_size, h_max), -100, dtype=torch.long)
        else:
            forecast_sequences = None

        for batch_idx, sample in enumerate(samples):
            text = text_tensors[batch_idx]
            text_input_ids[batch_idx, : text.shape[0]] = text
            text_attention_mask[batch_idx, : text.shape[0]] = 1

            if "prefix_control_token_id" in sample:
                prefix_control_token_ids[batch_idx] = int(sample["prefix_control_token_id"])
            if "revin_mean" in sample:
                revin_mean[batch_idx] = float(sample["revin_mean"])
            if "revin_std" in sample:
                revin_std[batch_idx] = float(sample["revin_std"])

            target = target_patches[batch_idx]
            if target.shape[-2:] != (self.config.patch_len, self.config.input_dim):
                raise ValueError(
                    f"target_patches must end with {(self.config.patch_len, self.config.input_dim)}, got {tuple(target.shape)}."
                )

            channel_patches[batch_idx, 0, : target.shape[0]] = target
            channel_mask[batch_idx, 0] = True
            patch_mask[batch_idx, 0, : target.shape[0]] = True

            for cov_idx, cov in enumerate(covariate_lists[batch_idx], start=1):
                if cov.shape[-2:] != (self.config.patch_len, self.config.input_dim):
                    raise ValueError(
                        f"covariate_patches must end with {(self.config.patch_len, self.config.input_dim)}, got {tuple(cov.shape)}."
                    )
                channel_patches[batch_idx, cov_idx, : cov.shape[0]] = cov
                channel_mask[batch_idx, cov_idx] = True
                patch_mask[batch_idx, cov_idx, : cov.shape[0]] = True

            if forecast_sequences is not None:
                forecast = forecast_sequences[batch_idx]
                assert forecast_token_ids is not None
                assert forecast_attention_mask is not None
                assert forecast_labels is not None
                forecast_token_ids[batch_idx, : forecast.shape[0]] = forecast
                forecast_attention_mask[batch_idx, : forecast.shape[0]] = 1
                forecast_labels[batch_idx, : forecast.shape[0]] = forecast

        batch: MutableMapping[str, torch.Tensor] = {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "channel_patches": channel_patches,
            "channel_mask": channel_mask,
            "patch_mask": patch_mask,
            "prefix_control_token_ids": prefix_control_token_ids,
            "revin_mean": revin_mean,
            "revin_std": revin_std,
        }
        if forecast_token_ids is not None:
            batch["forecast_token_ids"] = forecast_token_ids
            batch["forecast_attention_mask"] = forecast_attention_mask
            batch["forecast_labels"] = forecast_labels
        return batch
