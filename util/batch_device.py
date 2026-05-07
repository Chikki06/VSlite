from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def move_to_device(batch: Any, device: torch.device | str) -> Any:
    """
    Recursively move tensors in a nested batch to `device`.

    Preserves container structure for dict/list/tuple and leaves non-tensors
    untouched (e.g. strings, ints, paths).
    """
    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, Mapping):
        # Preserve key ordering; return a plain dict to avoid surprising custom mapping semantics.
        return {k: move_to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)

    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]

    # Avoid treating strings/bytes as Sequence (they are, but should be left alone).
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
        return type(batch)(move_to_device(v, device) for v in batch)

    return batch

