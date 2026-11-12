"""Unified training preview: fixed test tiles, PNG + TensorBoard.

Captures ``n_tiles`` held-out patches once at init (deterministic loader order /
optional ``PREVIEW_TILE_LIST``). Every save trigger runs inference-only
(``torch.no_grad`` + caller ``eval``) on that same batch so sequential
``epoch*_step*_pXX.png`` files are comparable across epochs and seeded runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

from util.fixed_preview import gather_fixed_preview_batch
from util.sample_grid import save_unified_sample, split_batch_to_list
from util.tb_logger import UnifiedTBLogger


class PreviewManager:
    """Capture fixed test patches once; save pred|real pair PNGs + TB on schedule."""

    def __init__(
        self,
        *,
        preview_loader: Iterable[Any],
        out_dir: str,
        tb: Optional[UnifiedTBLogger] = None,
        n_tiles: int = 8,
        should_save: Optional[Callable[[int], bool]] = None,
        forward_fn: Callable[[Any], tuple[torch.Tensor, torch.Tensor]],
        input_mode: Optional[str] = None,
        preview_tile_list: Optional[str] = None,
    ) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.tb = tb
        self.n_tiles = n_tiles
        self.should_save = should_save
        self.forward_fn = forward_fn
        self.input_mode = input_mode
        self.fixed_batch = gather_fixed_preview_batch(preview_loader, n=n_tiles)
        self._tile_names: list[str] = []
        if preview_tile_list and os.path.isfile(preview_tile_list):
            with open(preview_tile_list, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._tile_names = [str(x) for x in data[:n_tiles]]
        elif self.fixed_batch is not None:
            names = self.fixed_batch.get("name") or self.fixed_batch.get("path")
            if isinstance(names, list):
                self._tile_names = [str(n) for n in names[:n_tiles]]
        tiles_path = Path(out_dir).parent / "preview_tiles.json"
        if self._tile_names:
            tiles_path.write_text(json.dumps(self._tile_names, indent=2))

    def _infer_and_save(self, epoch: int, step: int) -> Optional[list[str]]:
        if self.fixed_batch is None:
            return None
        with torch.no_grad():
            target, pred = self.forward_fn(self.fixed_batch)
        reals = split_batch_to_list(target.detach().cpu())[: self.n_tiles]
        preds = split_batch_to_list(pred.detach().cpu())[: self.n_tiles]
        name = f"epoch{int(epoch):03d}_step{int(step):09d}"
        paths = save_unified_sample(
            self.out_dir,
            name,
            reals,
            preds,
            input_mode=self.input_mode,
        )
        if self.tb is not None and paths:
            self.tb.log_preview_pngs(paths, int(epoch))
        return paths or None

    def maybe_save(self, epoch: int, step: int) -> Optional[list[str]]:
        if self.fixed_batch is None:
            return None
        if self.should_save is not None and not self.should_save(int(step) - 1):
            return None
        return self._infer_and_save(epoch, step)

    def save_epoch_end(self, epoch: int, step: int) -> Optional[list[str]]:
        """Always save at epoch boundary (inference only, no grad)."""
        return self._infer_and_save(epoch, step)
