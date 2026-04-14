"""Minimal smoke test for the TextTS skeleton.

Runs a local no-download pipeline:
record -> formatter -> collator -> model.forward
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import torch
from torch import nn

from textts.data.datasets import build_pretrain_datasets, build_textts_collator
from textts.data.sequence_formatter import TextTSSequenceFormatter, TextTSSequenceFormatterConfig
from textts.model.textts_model import TextTSModel, TextTSModelConfig
from textts.tokenization.forecast_quantizer import ForecastQuantizer
from textts.tokenization.tokenizer import (
    TextTSTokenizerConfig,
    extend_tokenizer_and_embeddings,
)
from textts.training.pretrain import PretrainConfig, TextTSPretrainer, build_pretrain_dataloader, build_pretrain_optimizer


class ToyTokenizer:
    """Tiny tokenizer implementing the subset needed by the skeleton."""

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {
            "<bos>": 0,
            "<eos>": 1,
            "<pad>": 2,
            "<unk>": 3,
        }
        self.additional_special_tokens: List[str] = []
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.unk_token_id = 3

    def add_special_tokens(self, mapping: Dict[str, List[str]]) -> int:
        tokens = mapping.get("additional_special_tokens", [])
        added = 0
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
                added += 1
            self.additional_special_tokens.append(token)
        return added

    def add_tokens(self, tokens: List[str], special_tokens: bool = False) -> int:
        del special_tokens
        added = 0
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_to_id[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        ids: List[int] = []
        for token in text.replace("\n", " \n ").split():
            ids.append(self.token_to_id.get(token, self.unk_token_id))
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def __len__(self) -> int:
        return len(self.token_to_id)


class DummyCausalLM(nn.Module):
    """Small causal LM stub with the APIs TextTS needs."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        old_weight = self.embed_tokens.weight.data
        hidden_size = old_weight.shape[1]
        new_embed = nn.Embedding(new_size, hidden_size)
        with torch.no_grad():
            new_embed.weight[: old_weight.shape[0]].copy_(old_weight)
            if new_size > old_weight.shape[0]:
                nn.init.normal_(new_embed.weight[old_weight.shape[0] :], mean=0.0, std=0.02)
        self.embed_tokens = new_embed
        self.lm_head = nn.Linear(hidden_size, new_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.config.vocab_size = new_size
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
        past_key_values: Any = None,
    ) -> Any:
        del attention_mask, position_ids, past_key_values
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Need input_ids or inputs_embeds.")
            hidden = self.embed_tokens(input_ids)
        else:
            hidden = inputs_embeds
        logits = self.lm_head(hidden)
        output = SimpleNamespace(
            logits=logits,
            past_key_values=((),) if use_cache else None,
            hidden_states=None,
            attentions=None,
        )
        if return_dict:
            return output
        return logits


def build_demo_record() -> Dict[str, Any]:
    history = [100.0 + float(i) for i in range(32)]
    future = [132.0, 133.0, 134.0, 135.0]
    covariate = [20.0 + 0.1 * float(i) for i in range(32)]
    zeros = [[0.0] * 7 for _ in range(32)]
    return {
        "domain": "energy",
        "freq": "hourly",
        "context": "Synthetic load example for smoke testing.",
        "target_name": "load",
        "target_history": history,
        "target_future": future,
        "target_time_features": zeros,
        "covariates": [
            {
                "name": "temperature",
                "values": covariate,
                "time_features": zeros,
            }
        ],
        "covariate_categories": {"is_holiday": ["no"] * 32},
    }


def main() -> None:
    tokenizer = ToyTokenizer()
    llm = DummyCausalLM(vocab_size=len(tokenizer), hidden_size=1536)
    bundle = extend_tokenizer_and_embeddings(tokenizer, llm, TextTSTokenizerConfig())
    quantizer = ForecastQuantizer()
    formatter = TextTSSequenceFormatter(
        tokenizer=tokenizer,
        tokenizer_bundle=bundle,
        quantizer=quantizer,
        config=TextTSSequenceFormatterConfig(),
    )
    collator = build_textts_collator(bundle)

    model = TextTSModel(
        llm=llm,
        config=TextTSModelConfig(
            base_model_name_or_path="dummy",
            hidden_size=1536,
            d_patch=256,
            patch_len=16,
            input_dim=9,
            bos_fc_token_id=bundle.control_token_ids["<BOS_FC>"],
            eos_fc_token_id=bundle.control_token_ids["<EOS_FC>"],
            target_start_token_id=bundle.control_token_ids["<TARGET_START>"],
            forecast_pad_token_id=bundle.control_token_ids["<FORECAST_PAD>"],
            forecast_bin_token_ids=bundle.forecast_bin_token_ids,
            forecast_allowed_token_ids=bundle.forecast_allowed_token_ids,
        ),
    )

    sample = formatter.format_prediction_sample(build_demo_record())
    batch = collator([sample])
    output = model(batch)
    print("loss:", float(output.loss.detach()))
    print("logits_shape:", tuple(output.logits.shape))

    records = [build_demo_record(), build_demo_record()]
    pred_dataset, imp_dataset = build_pretrain_datasets(records, formatter)
    dataloader = build_pretrain_dataloader(
        pred_dataset,
        imp_dataset,
        collator,
        PretrainConfig(batch_size=1, num_batches_per_epoch=1),
    )
    optimizer = build_pretrain_optimizer(model, PretrainConfig())
    trainer = TextTSPretrainer(model, optimizer)
    metrics = trainer.train_epoch(dataloader, max_steps=1)
    print("pretrain_metrics:", metrics)


if __name__ == "__main__":
    main()
