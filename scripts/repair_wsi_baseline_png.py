#!/usr/bin/env python3
"""Repair baseline WSI PNGs with blank alternating rows or 2x-tall interleaved output.

Some broken `infer_wsi_png_baseline.py` runs produced images where:
  - every other row is empty (model output written into a 2x-tall canvas), or
  - each tile's bottom half is blank (undersized model tiles in oversized stitch slots).

This script detects those patterns on a vertical sample and writes a corrected PNG.
Re-run `infer_wsi_png_baseline.py` after the padding fix for best quality; use this
utility only to salvage already-saved outputs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _row_means(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=(1, 2))


def detect_repair_mode_from_row_means(rm: np.ndarray) -> str | None:
    """Return 'drop_odd', 'drop_even', or None from per-row means (length >= 4)."""
    if rm.shape[0] < 4:
        return None
    even_m = float(rm[0::2].mean())
    odd_m = float(rm[1::2].mean())
    even_z = float((rm[0::2] < 3.0).mean())
    odd_z = float((rm[1::2] < 3.0).mean())
    if even_m > 15.0 and odd_m < 3.0 and odd_z > 0.85:
        return "drop_odd"
    if odd_m > 15.0 and even_m < 3.0 and even_z > 0.85:
        return "drop_even"
    return None


def detect_repair_mode(arr: np.ndarray, *, sample_rows: int = 4096) -> str | None:
    h = min(int(sample_rows), arr.shape[0])
    return detect_repair_mode_from_row_means(_row_means(arr[:h]))


def sample_repair_mode(path: Path, *, sample_rows: int = 4096, sample_cols: int = 256) -> str | None:
    """Detect stripe pattern from a thin vertical crop (avoids loading gigapixel PNGs)."""
    with Image.open(path) as im:
        w, h = im.size
        x1 = min(int(sample_cols), w)
        chunks: list[np.ndarray] = []
        y = 0
        remaining = min(int(sample_rows), h)
        while remaining > 0 and y < h:
            htake = min(512, remaining, h - y)
            crop = im.crop((0, y, x1, y + htake))
            chunks.append(np.asarray(crop.convert("RGB")))
            y += htake
            remaining -= htake
    if not chunks:
        return None
    arr = np.concatenate(chunks, axis=0)
    return detect_repair_mode_from_row_means(_row_means(arr))


def repair_array(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "drop_odd":
        return np.ascontiguousarray(arr[0::2, :, :])
    if mode == "drop_even":
        return np.ascontiguousarray(arr[1::2, :, :])
    raise ValueError(f"Unknown repair mode: {mode}")


def repair_file(path: Path, out_path: Path | None, *, dry_run: bool, force: bool) -> bool:
    mode = sample_repair_mode(path)
    if mode is None and not force:
        print(f"SKIP (no stripe pattern): {path}")
        return False
    if mode is None:
        print(f"WARN: pattern not detected; pass --force to deinterlace odd rows anyway: {path}")
        mode = "drop_odd"
    with Image.open(path) as im:
        w, h = im.size
        print(f"REPAIR {path.name}: {w}x{h} -> {w}x{h // 2} ({mode})")
        if dry_run:
            return True
        # Deinterlace row-by-row to limit peak memory.
        dst = out_path if out_path is not None else path
        dst.parent.mkdir(parents=True, exist_ok=True)
        out_h = h // 2
        strip = 128
        out_im = Image.new("RGB", (w, out_h))
        for y0 in range(0, out_h, strip):
            y1 = min(out_h, y0 + strip)
            block = im.crop((0, 2 * y0, w, 2 * y1))
            arr = np.asarray(block.convert("RGB"))
            fixed = arr[0::2] if mode == "drop_odd" else arr[1::2]
            out_im.paste(Image.fromarray(fixed), (0, y0))
        out_im.save(dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair striped baseline WSI PNG outputs")
    ap.add_argument("paths", nargs="+", help="PNG file(s) or directory(ies) to repair")
    ap.add_argument("--out-dir", type=str, default=None, help="Write fixed PNGs here (default: in-place)")
    ap.add_argument("--suffix", type=str, default="_fixed", help="Suffix when --out-dir is set (ignored for in-place)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Apply odd-row deinterlace even if not detected")
    args = ap.parse_args()

    pngs: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            pngs.extend(sorted(path.glob("*.png")))
        elif path.is_file():
            pngs.append(path)
        else:
            raise FileNotFoundError(path)

    out_root = Path(args.out_dir) if args.out_dir else None
    n = 0
    for src in pngs:
        if out_root is None:
            out = None
        else:
            out = out_root / f"{src.stem}{args.suffix}.png"
        if repair_file(src, out, dry_run=bool(args.dry_run), force=bool(args.force)):
            n += 1
    print(f"Done. Repaired {n}/{len(pngs)} file(s).")


if __name__ == "__main__":
    main()
