"""Fixed preview-batch helpers.

At training start we capture ``N`` deterministic samples from a serial-mode
**test** (held-out) loader. On every save trigger we run inference-only on the
same batch so sequential ``epoch*_step*_pXX.png`` pair PNGs are visually
comparable across epochs within a run and across seeded train runs.

For streaming eval scripts we accumulate N-sample windows from the loader
instead (so the full eval set gets coverage); see :func:`SampleBuffer`.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import torch


def _stack_field(items: list[Any]) -> Any:
    if not items:
        return None
    head = items[0]
    if torch.is_tensor(head):
        return torch.cat([t.detach().cpu() for t in items], dim=0)
    if isinstance(head, list):
        out: list[Any] = []
        for x in items:
            out.extend(list(x))
        return out
    return items


def gather_fixed_preview_batch(
    loader: Iterable[Any],
    *,
    n: int = 8,
) -> Optional[dict[str, Any]]:
    """Iterate ``loader`` until at least ``n`` samples have been concatenated.

    Returns a single dict-of-tensors (with first dim == ``n``) suitable for
    a forward pass, or ``None`` if the loader yielded nothing.

    Works with two batch shapes:
    - dict/Mapping batches (baseline): each key is concatenated along dim 0.
    - tuple batches ``(x, y)`` (SwinGAN/NAFNet/Restormer): returned as
      ``{"input": x, "target": y}``.
    """
    collected: dict[str, list[Any]] = {}
    seen = 0
    for batch in loader:
        if isinstance(batch, Mapping):
            for k, v in batch.items():
                collected.setdefault(k, []).append(v)
            bs = _infer_batch_size(batch)
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
            collected.setdefault("input", []).append(x)
            collected.setdefault("target", []).append(y)
            bs = int(x.shape[0]) if torch.is_tensor(x) else 1
        else:
            return None
        seen += bs
        if seen >= n:
            break
    if seen == 0:
        return None
    out: dict[str, Any] = {}
    for k, items in collected.items():
        v = _stack_field(items)
        if torch.is_tensor(v):
            v = v[:n]
        elif isinstance(v, list):
            v = v[:n]
        out[k] = v
    return out


def _infer_batch_size(batch: Mapping[str, Any]) -> int:
    for v in batch.values():
        if torch.is_tensor(v):
            return int(v.shape[0])
        if isinstance(v, list):
            return len(v)
    return 1


class SampleBuffer:
    """Append-only buffer that yields windows of size ``n`` for test loops."""

    def __init__(self, n: int = 8) -> None:
        self.n = n
        self._reals: list[torch.Tensor] = []
        self._preds: list[torch.Tensor] = []

    def add(self, real: torch.Tensor, pred: torch.Tensor) -> None:
        if real.ndim == 4:
            for i in range(real.shape[0]):
                self._reals.append(real[i].detach().cpu())
                self._preds.append(pred[i].detach().cpu())
        else:
            self._reals.append(real.detach().cpu())
            self._preds.append(pred.detach().cpu())

    def has_window(self) -> bool:
        return len(self._reals) >= self.n

    def take_window(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        reals = self._reals[: self.n]
        preds = self._preds[: self.n]
        self._reals = self._reals[self.n :]
        self._preds = self._preds[self.n :]
        return reals, preds

    def take_remaining(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        reals = list(self._reals)
        preds = list(self._preds)
        self._reals.clear()
        self._preds.clear()
        return reals, preds
