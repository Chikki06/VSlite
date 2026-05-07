"""Atomic filesystem writes for training artifacts.

Cluster filesystems can occasionally produce partial/truncated files if a job is
preempted mid-write. These helpers write to a temporary file in the same
directory and then atomically replace the destination path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable


def _atomic_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    # os.replace is atomic on POSIX when src/dst are on same filesystem.
    os.replace(str(src), str(dst))


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace(tmp, p)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def atomic_write_text(path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_torch_save(torch_save_fn: Callable[[Any, str], None], obj: Any, path: str | os.PathLike[str]) -> None:
    """Atomic wrapper for torch.save-like functions.

    Pass `torch.save` as torch_save_fn to avoid importing torch in this module.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    os.close(fd)  # torch.save opens the path itself.
    try:
        torch_save_fn(obj, str(tmp))
        _atomic_replace(tmp, p)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def atomic_write_via(path: str | os.PathLike[str], write_fn: Callable[[str], None]) -> None:
    """Atomically write a file by delegating the actual write to `write_fn(tmp_path)`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Keep the destination suffix on the temp path so writers (e.g. torchvision.save_image)
    # can infer encoding from the extension; a bare ``.tmp`` suffix raises "unknown file extension".
    if p.suffix:
        fd, tmp = tempfile.mkstemp(prefix=f".{p.stem}.", suffix=p.suffix, dir=str(p.parent))
    else:
        fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    os.close(fd)
    try:
        write_fn(str(tmp))
        _atomic_replace(tmp, p)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
