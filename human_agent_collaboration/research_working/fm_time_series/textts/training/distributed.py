"""Distributed and device helpers for TextTS training."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, TypeVar

import torch
from torch import nn

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except ImportError:  # pragma: no cover - optional dependency
    dist = None  # type: ignore[assignment]
    DDP = None  # type: ignore[assignment]

try:  # pragma: no cover - only available on Ascend environments
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None  # type: ignore[assignment]


RecordT = TypeVar("RecordT")


@dataclass
class DistributedRuntime:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device
    device_type: str

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _npu_module() -> Optional[Any]:
    return getattr(torch, "npu", None)


def is_npu_available() -> bool:
    module = _npu_module()
    return module is not None and bool(module.is_available())


def resolve_runtime_device(device_name: Optional[str] = None, *, local_rank: int = 0) -> torch.device:
    normalized = (device_name or "auto").lower()
    if normalized == "auto":
        if is_npu_available():
            return torch.device(f"npu:{local_rank}")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "npu":
        return torch.device(f"npu:{local_rank}")
    if normalized == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device(device_name)


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> MutableMapping[str, Any]:
    moved: MutableMapping[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if is_npu_available():
        npu_module = _npu_module()
        if npu_module is not None and hasattr(npu_module, "manual_seed_all"):
            npu_module.manual_seed_all(seed)


def _set_current_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
        return
    if device.type == "npu":
        npu_module = _npu_module()
        if npu_module is None or not hasattr(npu_module, "set_device"):
            raise RuntimeError("torch_npu is required for NPU training but is not available.")
        npu_module.set_device(device)


def resolve_distributed_backend(backend_name: str, *, device_type: str) -> str:
    if backend_name != "auto":
        return backend_name
    if device_type == "npu":
        return "hccl"
    if device_type == "cuda":
        return "nccl"
    return "gloo"


def setup_distributed(
    *,
    device_name: str = "auto",
    backend_name: str = "auto",
) -> DistributedRuntime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = resolve_runtime_device(device_name, local_rank=local_rank)
    device_type = device.type

    if world_size <= 1:
        return DistributedRuntime(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            backend="none",
            device=device,
            device_type=device_type,
        )

    if dist is None or DDP is None:
        raise ImportError("torch.distributed is required for multi-card training.")

    backend = resolve_distributed_backend(backend_name, device_type=device_type)
    _set_current_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedRuntime(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        device=device,
        device_type=device_type,
    )


def cleanup_distributed(runtime: DistributedRuntime) -> None:
    if runtime.enabled and dist is not None and dist.is_initialized():
        dist.destroy_process_group()


def barrier(runtime: DistributedRuntime) -> None:
    if runtime.enabled and dist is not None and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def wrap_model_for_ddp(
    model: nn.Module,
    runtime: DistributedRuntime,
    *,
    find_unused_parameters: bool = False,
) -> nn.Module:
    if not runtime.enabled:
        return model
    if DDP is None:
        raise ImportError("torch.distributed is required for DDP wrapping.")
    if runtime.device_type in {"cuda", "npu"}:
        return DDP(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return DDP(model, find_unused_parameters=find_unused_parameters)


def all_reduce_mean(value: float, runtime: DistributedRuntime) -> float:
    if not runtime.enabled or dist is None or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float32, device=runtime.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= runtime.world_size
    return float(tensor.item())


def shard_records_for_rank(records: Sequence[RecordT], runtime: DistributedRuntime) -> list[RecordT]:
    if not runtime.enabled:
        return list(records)
    return list(records[runtime.rank :: runtime.world_size])


def require_non_empty_shard(records: Sequence[Any], runtime: DistributedRuntime, *, split_name: str) -> None:
    if records:
        return
    raise ValueError(
        f"{split_name} shard for rank={runtime.rank} is empty with world_size={runtime.world_size}. "
        "Increase the number of windows or reduce nproc_per_node."
    )
