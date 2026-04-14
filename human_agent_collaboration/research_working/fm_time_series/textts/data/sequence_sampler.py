"""Sampling utilities for TextTS datasets."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

from torch.utils.data import BatchSampler, Dataset


TaskName = Literal["pred", "imp"]
SampleKey = Tuple[TaskName, int]


class MixedTaskDataset(Dataset[MutableMapping[str, Any]]):
    """Dataset wrapper routing tuple keys to prediction or imputation datasets."""

    def __init__(
        self,
        pred_dataset: Dataset[MutableMapping[str, Any]],
        imp_dataset: Dataset[MutableMapping[str, Any]],
    ) -> None:
        self.pred_dataset = pred_dataset
        self.imp_dataset = imp_dataset

    def __len__(self) -> int:
        return len(self.pred_dataset) + len(self.imp_dataset)

    def __getitem__(self, index: SampleKey) -> MutableMapping[str, Any]:
        task_name, item_index = index
        if task_name == "pred":
            return self.pred_dataset[item_index]
        if task_name == "imp":
            return self.imp_dataset[item_index]
        raise KeyError(f"Unknown task_name: {task_name}")


@dataclass
class MixedBatchSamplerConfig:
    batch_size: int
    pred_probability: float = 0.7
    drop_last: bool = False
    num_batches_per_epoch: Optional[int] = None
    seed: int = 42


class MixedBatchSampler(BatchSampler):
    """Batch-level sampler alternating prediction/imputation batches.

    Each yielded batch contains tuple keys of the form:
        [("pred", idx_1), ("pred", idx_2), ...]
    or
        [("imp", idx_1), ("imp", idx_2), ...]
    """

    def __init__(
        self,
        pred_dataset_size: int,
        imp_dataset_size: int,
        config: MixedBatchSamplerConfig,
    ) -> None:
        if pred_dataset_size <= 0 or imp_dataset_size <= 0:
            raise ValueError("Both prediction and imputation dataset sizes must be positive.")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not (0.0 < config.pred_probability < 1.0):
            raise ValueError("pred_probability must be in (0, 1).")

        self.pred_dataset_size = pred_dataset_size
        self.imp_dataset_size = imp_dataset_size
        self.config = config

    def __len__(self) -> int:
        if self.config.num_batches_per_epoch is not None:
            return self.config.num_batches_per_epoch
        total_size = self.pred_dataset_size + self.imp_dataset_size
        if self.config.drop_last:
            return total_size // self.config.batch_size
        return math.ceil(total_size / self.config.batch_size)

    def _draw_indices(
        self,
        population_size: int,
        *,
        batch_size: int,
        rng: random.Random,
    ) -> List[int]:
        if batch_size <= population_size:
            return rng.sample(range(population_size), batch_size)
        return [rng.randrange(population_size) for _ in range(batch_size)]

    def __iter__(self) -> Iterator[List[SampleKey]]:
        rng = random.Random(self.config.seed)
        for _ in range(len(self)):
            task_name: TaskName = "pred" if rng.random() < self.config.pred_probability else "imp"
            dataset_size = self.pred_dataset_size if task_name == "pred" else self.imp_dataset_size
            indices = self._draw_indices(dataset_size, batch_size=self.config.batch_size, rng=rng)
            if self.config.drop_last and len(indices) < self.config.batch_size:
                continue
            yield [(task_name, idx) for idx in indices]

