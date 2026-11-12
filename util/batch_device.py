from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def move_to_device(
    batch: Any, device: torch.device | str, *, non_blocking: bool = False
) -> Any:
    """
    Recursively move tensors in a nested batch to `device`.

    Preserves container structure for dict/list/tuple and leaves non-tensors
    untouched (e.g. strings, ints, paths).

    When ``non_blocking=True`` and tensors live in pinned host memory (DataLoader
    ``pin_memory=True``), H2D copies can overlap with compute.
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)

    if isinstance(batch, Mapping):
        # Preserve key ordering; return a plain dict to avoid surprising custom mapping semantics.
        return {
            k: move_to_device(v, device, non_blocking=non_blocking) for k, v in batch.items()
        }

    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device, non_blocking=non_blocking) for v in batch)

    if isinstance(batch, list):
        return [move_to_device(v, device, non_blocking=non_blocking) for v in batch]

    # Avoid treating strings/bytes as Sequence (they are, but should be left alone).
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
        return type(batch)(
            move_to_device(v, device, non_blocking=non_blocking) for v in batch
        )

    return batch
