"""Deep Zoom (DZI) writers via pyvips.

Preferred path for remote upload: ``write_dzi_from_array`` over a uint8 HWC
canvas straight from the WSI stitch (see inference-server ``create_dzi`` /
``temp.py``). That skips encoding a multi-GB PNG/JPEG only to re-decode it.

``write_dzi_from_file`` remains for re-uploading already-written pred images
(streaming sequential access so both ``dzsave`` and thumbnail can run).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

_libjpeg_preloaded = False


def ensure_conda_libjpeg() -> None:
    """Preload conda ``libjpeg.so.8`` before torchvision can claim that soname.

    ``torchvision.libs/libjpeg*.so.8`` lacks ``jpeg12_*`` symbols. If it is
    loaded first (via ``import torchvision`` during CISLNet init), later
    ``libvips``/``libtiff`` fail with::

        undefined symbol: jpeg12_write_raw_data, version LIBJPEG_8.0

    Call this **before** importing torch/torchvision when the same process will
    also ``dzsave`` via pyvips.
    """
    global _libjpeg_preloaded
    if _libjpeg_preloaded:
        return
    import ctypes
    import os

    prefix = os.environ.get("CONDA_PREFIX") or ""
    path = os.path.join(prefix, "lib", "libjpeg.so.8")
    if prefix and os.path.isfile(path):
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    _libjpeg_preloaded = True


def _pyvips() -> Any:
    ensure_conda_libjpeg()
    try:
        import pyvips
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pyvips required. Install with: conda install -c conda-forge libvips pyvips"
        ) from exc
    return pyvips


def _modality_dirs(scratch_root: Path, output_type: str) -> Tuple[Path, Path]:
    modality_dzi = scratch_root / "dzi" / output_type
    modality_thumb = scratch_root / "thumbnail" / output_type
    modality_dzi.mkdir(parents=True, exist_ok=True)
    modality_thumb.mkdir(parents=True, exist_ok=True)
    return modality_dzi, modality_thumb


def _clear_existing_dzi(dzi_base: Path) -> Tuple[Path, Path]:
    dzi_path = Path(str(dzi_base) + ".dzi")
    tiles_dir = Path(str(dzi_base) + "_files")
    if dzi_path.exists():
        dzi_path.unlink()
    if tiles_dir.is_dir():
        shutil.rmtree(tiles_dir)
    return dzi_path, tiles_dir


def _dzsave(vi: Any, dzi_base: Path) -> None:
    vi.dzsave(
        str(dzi_base),
        overlap=0,
        container="fs",
        suffix=".jpg",
        compression=0,
        tile_size=1024,
    )


def write_dzi_from_array(
    img_hwc_u8: np.ndarray,
    *,
    scratch_root: PathLike,
    sample_id: str,
    output_type: str,
    thumb_size: int = 400,
    thumb_quality: int = 75,
) -> Tuple[Path, Path, Path, float]:
    """Write DZI + thumbnail from a contiguous uint8 HWC RGB array.

    Matches the inference-server ``create_dzi`` contract (``new_from_memory`` →
    ``dzsave`` → ``thumbnail_image``). Returns
    ``(dzi_path, tiles_dir, thumb_path, write_time_s)``.
    """
    pyvips = _pyvips()
    t0 = time.time()
    scratch_root = Path(scratch_root)
    modality_dzi, modality_thumb = _modality_dirs(scratch_root, output_type)

    if img_hwc_u8.ndim != 3 or img_hwc_u8.shape[2] not in (3, 4):
        raise ValueError(
            f"Expected HxWx3/4 uint8 RGB; got shape={getattr(img_hwc_u8, 'shape', None)} "
            f"dtype={getattr(img_hwc_u8, 'dtype', None)}"
        )
    if img_hwc_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 canvas; got {img_hwc_u8.dtype}")

    # Drop alpha if present; force C-contiguous for vips memory view.
    rgb = np.ascontiguousarray(img_hwc_u8[:, :, :3])
    height, width, bands = rgb.shape
    linear = rgb.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands, "uchar")

    dzi_base = modality_dzi / sample_id
    dzi_path, tiles_dir = _clear_existing_dzi(dzi_base)
    _dzsave(vi, dzi_base)

    thumb_path = modality_thumb / f"{sample_id}.jpg"
    vi.thumbnail_image(int(thumb_size)).jpegsave(str(thumb_path), Q=int(thumb_quality))

    write_time = time.time() - t0
    print(f"  Write Time: {write_time:.2f} seconds (array→DZI)")
    return dzi_path, tiles_dir, thumb_path, write_time


def write_dzi_from_file(
    pred_path: PathLike,
    *,
    scratch_root: PathLike,
    sample_id: str,
    output_type: str,
    thumb_size: int = 400,
    thumb_quality: int = 75,
) -> Tuple[Path, Path, Path, float]:
    """Write DZI + thumbnail via pyvips streaming from an on-disk pred image.

    Opens the file twice so each pass can use sequential access (required for
    multi-GB WSIs — a single sequential image cannot be consumed by both
    ``dzsave`` and ``thumbnail``).
    """
    pyvips = _pyvips()
    t0 = time.time()
    pred_path = Path(pred_path)
    scratch_root = Path(scratch_root)
    modality_dzi, modality_thumb = _modality_dirs(scratch_root, output_type)

    def _load_rgb() -> Any:
        vi = pyvips.Image.new_from_file(str(pred_path), access="sequential")
        if vi.bands == 1:
            vi = vi.bandjoin([vi, vi])
        if vi.bands == 2:
            vi = vi.bandjoin(vi.extract_band(0))
        if vi.bands > 3:
            vi = vi.extract_band(0, n=3)
        return vi

    dzi_base = modality_dzi / sample_id
    dzi_path, tiles_dir = _clear_existing_dzi(dzi_base)
    _dzsave(_load_rgb(), dzi_base)

    # shrink-on-load; do not reuse a sequential Image for thumbnail
    thumb_path = modality_thumb / f"{sample_id}.jpg"
    pyvips.Image.thumbnail(str(pred_path), int(thumb_size)).jpegsave(
        str(thumb_path), Q=int(thumb_quality)
    )

    write_time = time.time() - t0
    print(f"  Write Time: {write_time:.2f} seconds (file→DZI)")
    return dzi_path, tiles_dir, thumb_path, write_time
