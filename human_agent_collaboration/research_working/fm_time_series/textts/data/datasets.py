"""Dataset wrappers and dataset/collator builders for TextTS."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from textts.data.collator import TextTSCollator, TextTSCollatorConfig
from textts.data.sequence_formatter import TextTSSequenceFormatter
from textts.data.sft_dataset import SFTDatasetConfig, TextTSSFTInstructionDataset
from textts.tokenization.tokenizer import TextTSTokenizerBundle


class TextTSPredictionDataset(Dataset[MutableMapping[str, Any]]):
    """Prediction-mode dataset for CPT/SFT."""

    def __init__(self, records: Sequence[Mapping[str, Any]], formatter: TextTSSequenceFormatter) -> None:
        self.records = list(records)
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MutableMapping[str, Any]:
        return self.formatter.format_prediction_sample(self.records[index])


class TextTSImputationDataset(Dataset[MutableMapping[str, Any]]):
    """Imputation-mode dataset for CPT."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        formatter: TextTSSequenceFormatter,
        *,
        seed_offset: int = 0,
    ) -> None:
        self.records = list(records)
        self.formatter = formatter
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MutableMapping[str, Any]:
        return self.formatter.format_imputation_sample(self.records[index], seed=self.seed_offset + index)


class TextTSSFTDataset(Dataset[MutableMapping[str, Any]]):
    """SFT dataset wrapper."""

    def __init__(self, records: Sequence[Mapping[str, Any]], formatter: TextTSSequenceFormatter) -> None:
        self.records = list(records)
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MutableMapping[str, Any]:
        return self.formatter.format_sft_sample(self.records[index])


def build_pretrain_datasets(
    records: Sequence[Mapping[str, Any]],
    formatter: TextTSSequenceFormatter,
    *,
    imputation_seed_offset: int = 10_000,
) -> Tuple[TextTSPredictionDataset, TextTSImputationDataset]:
    """Build the two datasets used by batch-level CPT mixing."""

    return (
        TextTSPredictionDataset(records, formatter),
        TextTSImputationDataset(records, formatter, seed_offset=imputation_seed_offset),
    )


def build_sft_dataset(
    records: Sequence[Mapping[str, Any]],
    formatter: TextTSSequenceFormatter,
    *,
    config: Optional[SFTDatasetConfig] = None,
) -> Dataset[MutableMapping[str, Any]]:
    return TextTSSFTInstructionDataset(records, formatter, config=config)


def build_textts_collator(
    tokenizer_bundle: TextTSTokenizerBundle,
    *,
    patch_len: int = 16,
    input_dim: int = 9,
) -> TextTSCollator:
    return TextTSCollator(
        TextTSCollatorConfig(
            pad_token_id=tokenizer_bundle.control_token_ids["<PAD>"],
            forecast_pad_token_id=tokenizer_bundle.control_token_ids["<FORECAST_PAD>"],
            default_prefix_control_token_id=tokenizer_bundle.control_token_ids["<BOS_FC>"],
            patch_len=patch_len,
            input_dim=input_dim,
        )
    )
