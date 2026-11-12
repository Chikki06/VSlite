#!/usr/bin/env python3
"""Convert IR patch .npy trees from float32 -> float16 into sibling IR_fp16 folders.

Default sources:
  dataset/data_frozen/ir  -> dataset/data_frozen/IR_fp16
  dataset/data_1024/ir    -> dataset/data_1024/IR_fp16

Also builds a training dataroot ``dataset/data_frozen_fp16`` with:
  ir -> ../data_frozen/IR_fp16, he/vis/test.txt/val.txt symlinked from data_frozen.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

VSLITE_ROOT = Path(__file__).resolve().parents[1]


def _convert_one_safe(args: tuple[str, str, bool]) -> tuple[str, str]:
    """Convert one file; write via tempfile then rename (no np.save ext quirks)."""
    src_s, dst_s, skip_existing = args
    src = Path(src_s)
    dst = Path(dst_s)
    try:
        if skip_existing and dst.is_file() and dst.stat().st_size > 0:
            # Verify dtype if possible
            try:
                if np.load(dst, mmap_mode="r").dtype == np.float16:
                    return ("skip", src.name)
            except Exception:
                pass
        arr = np.asarray(np.load(src, mmap_mode="r"), dtype=np.float16)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".partial")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, dst)
        return ("ok", src.name)
    except Exception as e:  # noqa: BLE001
        return ("err", f"{src.name}: {type(e).__name__}: {e}")


def convert_tree(
    src_dir: Path,
    dst_dir: Path,
    *,
    workers: int,
    skip_existing: bool,
    limit: int | None,
) -> None:
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    if not src_dir.is_dir():
        print(f"SKIP missing source: {src_dir}", flush=True)
        return
    files = sorted(src_dir.glob("*.npy"))
    if limit is not None:
        files = files[:limit]
    print(f"Converting {len(files)} files\n  {src_dir}\n  -> {dst_dir}", flush=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(dst_dir / p.name), skip_existing) for p in files]
    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_convert_one_safe, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            status, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"ERR {msg}", flush=True)
            if i % 2000 == 0 or i == len(futs):
                print(f"  progress {i}/{len(futs)} ok={ok} skip={skip} err={err}", flush=True)
    print(f"DONE {src_dir.name}: ok={ok} skip={skip} err={err}", flush=True)
    if err:
        raise SystemExit(1)


def ensure_frozen_fp16_dataroot(root: Path) -> Path:
    """Create dataset/data_frozen_fp16 with ir -> IR_fp16 and he/vis/list symlinks."""
    frozen = root / "dataset" / "data_frozen"
    out = root / "dataset" / "data_frozen_fp16"
    out.mkdir(parents=True, exist_ok=True)
    ir_fp16 = frozen / "IR_fp16"
    links = {
        "ir": ir_fp16,
        "he": frozen / "he",
        "vis": frozen / "vis",
        "test.txt": frozen / "test.txt",
        "val.txt": frozen / "val.txt",
    }
    for name, target in links.items():
        dest = out / name
        if dest.is_symlink() or dest.exists():
            if dest.is_symlink() and dest.resolve() == target.resolve():
                continue
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                raise RuntimeError(f"Refusing to replace non-symlink path: {dest}")
        dest.symlink_to(os.path.relpath(target, start=out))
        print(f"link {dest} -> {target}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=VSLITE_ROOT)
    p.add_argument("--workers", type=int, default=max(4, min(32, (os.cpu_count() or 8))))
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--limit", type=int, default=None, help="Debug: convert only first N per tree")
    p.add_argument(
        "--trees",
        nargs="*",
        default=["frozen", "1024"],
        help="Which trees: frozen, 1024",
    )
    args = p.parse_args()
    root = args.root.resolve()
    mapping = {
        "frozen": (root / "dataset" / "data_frozen" / "ir", root / "dataset" / "data_frozen" / "IR_fp16"),
        "1024": (root / "dataset" / "data_1024" / "ir", root / "dataset" / "data_1024" / "IR_fp16"),
    }
    for key in args.trees:
        if key not in mapping:
            print(f"Unknown tree {key!r}", file=sys.stderr)
            raise SystemExit(2)
        src, dst = mapping[key]
        convert_tree(
            src,
            dst,
            workers=args.workers,
            skip_existing=args.skip_existing,
            limit=args.limit,
        )
    if "frozen" in args.trees:
        dr = ensure_frozen_fp16_dataroot(root)
        print(f"DATAROOT for fp16 training: {dr}", flush=True)


if __name__ == "__main__":
    main()
