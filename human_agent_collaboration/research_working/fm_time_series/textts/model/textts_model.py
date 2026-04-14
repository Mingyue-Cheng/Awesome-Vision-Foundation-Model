"""Core TextTS model skeleton.

This module wires together:
- shared patch encoder
- cross-channel mixer
- projector into Qwen hidden space
- mixed inputs_embeds construction
- full-vocab tied lm_head training with forecast-stage logits mask
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from textts.encoders.channel_mixer import ChannelMixer, ChannelMixerConfig
from textts.encoders.projector import Projector, ProjectorConfig
from textts.encoders.ts_patch_encoder import TSPatchEncoder, TSPatchEncoderConfig
from textts.tokenization.tokenizer import build_forecast_vocab_mask

try:
    from transformers import AutoModelForCausalLM
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError:  # pragma: no cover - compile-time fallback
    AutoModelForCausalLM = None  # type: ignore[assignment]
    CausalLMOutputWithPast = object  # type: ignore[assignment]


@dataclass
class TextTSModelConfig:
    base_model_name_or_path: str = "Qwen/Qwen3-1.5B-Base"
    hidden_size: int = 1536
    d_patch: int = 256
    patch_len: int = 16
    input_dim: int = 9
    bos_fc_token_id: int = -1
    eos_fc_token_id: int = -1
    target_start_token_id: int = -1
    forecast_pad_token_id: int = -1
    forecast_bin_token_ids: Sequence[int] = field(default_factory=list)
    forecast_allowed_token_ids: Sequence[int] = field(default_factory=list)


class TextTSModel(nn.Module):
    """Minimal end-to-end TextTS forward path."""

    def __init__(
        self,
        llm: nn.Module,
        config: TextTSModelConfig,
        *,
        patch_encoder: Optional[TSPatchEncoder] = None,
        channel_mixer: Optional[ChannelMixer] = None,
        projector: Optional[Projector] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.llm = llm
        self.patch_encoder = patch_encoder or TSPatchEncoder(
            TSPatchEncoderConfig(
                input_dim=config.input_dim,
                patch_len=config.patch_len,
                d_patch=config.d_patch,
                role_vocab_size=2,
            )
        )
        self.channel_mixer = channel_mixer or ChannelMixer(ChannelMixerConfig(d_model=config.d_patch))
        self.projector = projector or Projector(ProjectorConfig(d_patch=config.d_patch, d_llm=config.hidden_size))

        model_hidden = getattr(getattr(self.llm, "config", None), "hidden_size", config.hidden_size)
        if model_hidden != self.config.hidden_size:
            raise ValueError(f"LLM hidden_size={model_hidden} does not match config.hidden_size={config.hidden_size}.")

        if not self.config.forecast_allowed_token_ids:
            raise ValueError("forecast_allowed_token_ids must be provided for forecast-stage logits masking.")

        self.register_buffer(
            "forecast_vocab_mask",
            build_forecast_vocab_mask(
                getattr(self.llm.config, "vocab_size", 0),
                self.config.forecast_allowed_token_ids,
            ),
            persistent=False,
        )

    @classmethod
    def from_pretrained(cls, config: TextTSModelConfig, **kwargs: object) -> "TextTSModel":
        if AutoModelForCausalLM is None:
            raise ImportError("transformers is required to load the backbone model.")
        llm = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, **kwargs)
        return cls(llm=llm, config=config)

    def _embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.llm.get_input_embeddings()
        if embedding is None:
            raise ValueError("LLM does not expose input embeddings.")
        return embedding(token_ids)

    def _control_token_embed(self, token_id: int, device: torch.device) -> torch.Tensor:
        token = torch.tensor([token_id], dtype=torch.long, device=device)
        return self._embed_token_ids(token).squeeze(0)

    def _encode_channels(
        self,
        channel_patches: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, c_max, n_patch, patch_len, input_dim = channel_patches.shape
        if patch_len != self.config.patch_len or input_dim != self.config.input_dim:
            raise ValueError(
                f"Expected channel patches [..., {self.config.patch_len}, {self.config.input_dim}], got {tuple(channel_patches.shape)}."
            )

        role_ids = torch.zeros((batch_size, c_max, n_patch), dtype=torch.long, device=channel_patches.device)
        if c_max > 1:
            role_ids[:, 1:, :] = 1

        z = self.patch_encoder(channel_patches, role_ids=role_ids)
        z_ctx = self.channel_mixer(z, channel_mask=channel_mask)
        return z_ctx

    @staticmethod
    def _flatten_valid_covariates(
        cov_latents: torch.Tensor,
        cov_channel_mask: torch.Tensor,
        cov_patch_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        flattened: List[torch.Tensor] = []
        batch_size = cov_latents.shape[0]
        for batch_idx in range(batch_size):
            sample_chunks: List[torch.Tensor] = []
            for cov_idx in range(cov_latents.shape[1]):
                if not cov_channel_mask[batch_idx, cov_idx]:
                    continue
                valid_patches = cov_patch_mask[batch_idx, cov_idx].bool()
                if valid_patches.any():
                    sample_chunks.append(cov_latents[batch_idx, cov_idx, valid_patches])
            if sample_chunks:
                flattened.append(torch.cat(sample_chunks, dim=0))
            else:
                flattened.append(cov_latents.new_zeros((0, cov_latents.shape[-1])))
        return flattened

    def _build_prefix_only(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        text_input_ids = batch["text_input_ids"]
        text_attention_mask = batch["text_attention_mask"].bool()
        channel_patches = batch["channel_patches"]
        channel_mask = batch["channel_mask"].bool()
        patch_mask = batch["patch_mask"].bool()

        z_ctx = self._encode_channels(channel_patches, channel_mask)
        z_target_ctx = z_ctx[:, 0]
        z_cov_ctx = z_ctx[:, 1:]

        h_target = self.projector(z_target_ctx)
        h_cov = self.projector(z_cov_ctx)

        cov_sequences = self._flatten_valid_covariates(h_cov, channel_mask[:, 1:], patch_mask[:, 1:])

        target_start_embed = self._control_token_embed(self.config.target_start_token_id, channel_patches.device)
        prefix_control_token_ids = batch.get("prefix_control_token_ids")
        if prefix_control_token_ids is None:
            prefix_control_token_ids = torch.full(
                (text_input_ids.shape[0],),
                self.config.bos_fc_token_id,
                dtype=torch.long,
                device=channel_patches.device,
            )
        else:
            prefix_control_token_ids = prefix_control_token_ids.to(device=channel_patches.device, dtype=torch.long)

        sample_embeds: List[torch.Tensor] = []
        sample_lengths: List[int] = []
        for batch_idx in range(text_input_ids.shape[0]):
            text_ids = text_input_ids[batch_idx, text_attention_mask[batch_idx]]
            text_embeds = self._embed_token_ids(text_ids)
            target_valid = h_target[batch_idx, patch_mask[batch_idx, 0]]
            prefix_control_embed = self._embed_token_ids(prefix_control_token_ids[batch_idx : batch_idx + 1]).squeeze(0)
            prefix = torch.cat(
                [
                    text_embeds,
                    cov_sequences[batch_idx],
                    target_start_embed.unsqueeze(0),
                    target_valid,
                    prefix_control_embed.unsqueeze(0),
                ],
                dim=0,
            )
            sample_embeds.append(prefix)
            sample_lengths.append(prefix.shape[0])

        max_len = max(sample_lengths)
        hidden_size = self.config.hidden_size
        padded_embeds = channel_patches.new_zeros((text_input_ids.shape[0], max_len, hidden_size))
        attention_mask = torch.zeros((text_input_ids.shape[0], max_len), dtype=torch.long, device=channel_patches.device)
        position_ids = torch.zeros((text_input_ids.shape[0], max_len), dtype=torch.long, device=channel_patches.device)

        for batch_idx, embeds in enumerate(sample_embeds):
            seq_len = embeds.shape[0]
            padded_embeds[batch_idx, :seq_len] = embeds
            attention_mask[batch_idx, :seq_len] = 1
            position_ids[batch_idx, :seq_len] = torch.arange(seq_len, device=channel_patches.device)

        return padded_embeds, attention_mask, position_ids, sample_lengths

    def _build_training_batch(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if "forecast_token_ids" not in batch:
            raise ValueError("Training batch requires forecast_token_ids.")

        prefix_embeds, prefix_attention_mask, prefix_position_ids, prefix_lengths = self._build_prefix_only(batch)
        forecast_token_ids = batch["forecast_token_ids"]
        forecast_attention_mask = batch["forecast_attention_mask"].bool()
        forecast_labels = batch["forecast_labels"]
        batch_size = forecast_token_ids.shape[0]

        forecast_embeds = self._embed_token_ids(forecast_token_ids)
        forecast_lengths = forecast_attention_mask.sum(dim=1).tolist()
        total_lengths = [p + f for p, f in zip(prefix_lengths, forecast_lengths)]
        max_total_len = max(total_lengths)

        mixed_embeds = prefix_embeds.new_zeros((batch_size, max_total_len, self.config.hidden_size))
        attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long, device=prefix_embeds.device)
        position_ids = torch.zeros((batch_size, max_total_len), dtype=torch.long, device=prefix_embeds.device)
        labels = torch.full((batch_size, max_total_len), -100, dtype=torch.long, device=prefix_embeds.device)

        for batch_idx in range(batch_size):
            prefix_len = prefix_lengths[batch_idx]
            forecast_len = forecast_lengths[batch_idx]
            total_len = prefix_len + forecast_len

            mixed_embeds[batch_idx, :prefix_len] = prefix_embeds[batch_idx, :prefix_len]
            attention_mask[batch_idx, :prefix_len] = 1
            position_ids[batch_idx, :prefix_len] = prefix_position_ids[batch_idx, :prefix_len]

            if forecast_len > 0:
                mixed_embeds[batch_idx, prefix_len:total_len] = forecast_embeds[batch_idx, :forecast_len]
                attention_mask[batch_idx, prefix_len:total_len] = 1
                position_ids[batch_idx, prefix_len:total_len] = torch.arange(
                    prefix_len,
                    total_len,
                    device=prefix_embeds.device,
                )
                labels[batch_idx, prefix_len:total_len] = forecast_labels[batch_idx, :forecast_len]

        return mixed_embeds, attention_mask, position_ids, labels

    def _apply_forecast_mask_to_shifted_logits(
        self,
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
    ) -> torch.Tensor:
        masked_logits = shift_logits.clone()
        active_positions = shift_labels != -100
        if active_positions.any():
            masked_logits[active_positions] = masked_logits[active_positions] + self.forecast_vocab_mask.to(
                device=shift_logits.device,
                dtype=shift_logits.dtype,
            )
        return masked_logits

    def forward(self, batch: Mapping[str, torch.Tensor]) -> CausalLMOutputWithPast:
        mixed_embeds, attention_mask, position_ids, labels = self._build_training_batch(batch)
        outputs = self.llm(
            inputs_embeds=mixed_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_logits = self._apply_forecast_mask_to_shifted_logits(shift_logits, shift_labels)
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled).squeeze(-1)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _pad_or_truncate_to_h(token_ids: List[int], horizon: int, *, fill_token_id: int) -> List[int]:
        if not token_ids:
            return [fill_token_id] * horizon
        if len(token_ids) >= horizon:
            return token_ids[:horizon]
        return token_ids + [token_ids[-1]] * (horizon - len(token_ids))

    @torch.no_grad()
    def generate_single(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        horizon: int,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[int]:
        if batch["text_input_ids"].shape[0] != 1:
            raise ValueError("generate_single currently expects batch size 1.")
        if not self.config.forecast_bin_token_ids:
            raise ValueError("forecast_bin_token_ids must be provided for generation.")

        prefix_embeds, attention_mask, position_ids, prefix_lengths = self._build_prefix_only(batch)
        outputs = self.llm(
            inputs_embeds=prefix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        generated: List[int] = []

        for _ in range(horizon + 1):
            logits = outputs.logits[:, -1, :]
            logits = logits + self.forecast_vocab_mask.to(device=logits.device, dtype=logits.dtype)

            if strategy == "greedy":
                next_token = torch.argmax(logits, dim=-1)
            elif strategy == "sample":
                next_token = self._sample_from_logits(logits, temperature=temperature, top_p=top_p)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

            next_id = int(next_token.item())
            if next_id == self.config.eos_fc_token_id:
                break

            generated.append(next_id)
            outputs = self.llm(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

        return self._pad_or_truncate_to_h(
            generated,
            horizon,
            fill_token_id=int(self.config.forecast_bin_token_ids[0]),
        )
