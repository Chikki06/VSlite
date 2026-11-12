#!/usr/bin/env python3
"""WSI → DZI → CISL Viewer remote upload.

Preferred path (``--infer_cislnet``): run tiled CISLNet inference, stitch to a
uint8 HWC canvas, ``dzsave`` via pyvips ``new_from_memory`` (same as the
inference-server hot path), pack, and upload. **No PNG/JPEG WSI encode.**

Legacy path (``--wsi_pred_dir`` / ``--pred_image``): convert already-written
pred images to DZI then upload (CPU-only re-upload of old ``wsi_pred`` trees).

Auth via ``CISLVIEWER_REMOTE_TOKEN`` / ``CISLVIEWER_SITE`` (see remote_upload.md).

Example (direct, no PNG)::

    source /u/akshahi2/.secrets.sh
    python scripts/remote_wsi_upload.py \\
        --infer_cislnet \\
        --name c13 \\
        --checkpoints_dir results/cislnet \\
        --which_epoch 20 \\
        --parent_folder input/IR \\
        --project campus_c13_wsi \\
        --output_type CISLNet-c13-e20 \\
        --input_nc 13 --num_ir_channels 10 \\
        --create_project --ensure_project
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install requests: conda install -c conda-forge requests") from exc

from util.dzi_write import (  # noqa: E402
    ensure_conda_libjpeg,
    write_dzi_from_array,
    write_dzi_from_file,
)

# Must run before any torch/torchvision import in this process (infer path).
ensure_conda_libjpeg()


DEFAULT_SITE = os.environ.get(
    "CISLVIEWER_SITE", "https://cisl-inference.beckman.illinois.edu"
)
DEFAULT_TOKEN = os.environ.get("CISLVIEWER_REMOTE_TOKEN", "")

# Pred filenames look like 10442-1_13ch.png or 10442-1_4ch.jpg
_PRED_NAME_RE = re.compile(
    r"^(?P<sample>.+?)_(?P<suffix>\d+ch|pred|wsi)\.(?P<ext>png|jpe?g)$",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_output_type(model_type: int, vs_modality: Optional[str]) -> str:
    if vs_modality:
        return vs_modality.strip()
    return "IR(VS)" if int(model_type) == 0 else "IR+VIS(VS)"


def discover_preds(wsi_pred_dir: Path) -> List[Tuple[str, Path]]:
    """Return sorted (sample_id, pred_path) pairs from a wsi_pred directory."""
    found: Dict[str, Path] = {}
    for path in sorted(wsi_pred_dir.iterdir()):
        if not path.is_file():
            continue
        m = _PRED_NAME_RE.match(path.name)
        if m:
            sample_id = m.group("sample")
        else:
            stem = path.stem
            sample_id = stem.split("_")[0] if "_" in stem else stem
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
        if sample_id not in found:
            found[sample_id] = path
    return sorted(found.items(), key=lambda x: x[0])


def pack_dzi_archive(dzi_path: Path, tiles_dir: Path, archive_path: Path) -> Path:
    sample_dzi_name = dzi_path.name
    sample_files_name = tiles_dir.name
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(dzi_path, arcname=sample_dzi_name)
        tf.add(tiles_dir, arcname=sample_files_name)
    return archive_path


def build_upload_metadata(
    *,
    sample_id: str,
    sample_folder: Path,
    project_name: str,
    output_type: str,
    model_type: int,
    use_tensorrt: bool,
    tensorrt_precision: Optional[str],
    overwrite: bool,
    create_project: bool,
    timings: Dict[str, float],
    terminal_output: str,
    started_at: str,
    completed_at: str,
    run_uuid: Optional[str] = None,
    project_display_name: Optional[str] = None,
    project_description: Optional[str] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "sample_id": sample_id,
        "project_name": project_name,
        "output_type": output_type,
        "create_project": create_project,
        "overwrite": overwrite,
        "source_path": str(sample_folder.resolve()),
        "run_type": "remote_upload",
        "input_type": "IR",
        "file_format": ".bsq",
        "model_type": int(model_type),
        "use_tensorrt": bool(use_tensorrt),
        "tensorrt_precision": tensorrt_precision,
        "bsq_load_time": timings.get("bsq_load_time"),
        "inference_time": timings.get("inference_time"),
        "write_time": timings.get("write_time"),
        "total_time": timings.get("total_time"),
        "progress_message": "Completed successfully",
        "terminal_output": terminal_output,
        "started_at": started_at,
        "completed_at": completed_at,
        "run_uuid": run_uuid or str(uuid.uuid4()),
    }
    if project_display_name:
        meta["project_display_name"] = project_display_name
    if project_description:
        meta["project_description"] = project_description
    return meta


def upload_to_viewer(
    *,
    site: str,
    token: str,
    metadata: Dict[str, Any],
    archive_path: Path,
    thumbnail_path: Path,
) -> Dict[str, Any]:
    url = site.rstrip("/") + "/api/remote/upload"
    headers = {"Authorization": f"Bearer cisl-remote {token}"}
    archive_bytes = archive_path.stat().st_size
    if archive_bytes > 2 * 1024**3:
        raise RuntimeError(
            f"DZI archive {archive_path} is {archive_bytes / 1024**3:.2f} GiB "
            "(limit 2 GiB); re-encode with lower JPEG tile quality if needed"
        )
    with archive_path.open("rb") as arch, thumbnail_path.open("rb") as thumb:
        files = {
            "dzi_archive": (archive_path.name, arch, "application/gzip"),
            "thumbnail": (thumbnail_path.name, thumb, "image/jpeg"),
        }
        data = {"metadata": json.dumps(metadata)}
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=600)
    try:
        body = resp.json()
    except Exception:
        body = {"error": resp.text[:500]}
    if resp.status_code >= 400:
        raise RuntimeError(f"Upload failed HTTP {resp.status_code}: {body}")
    return body


def create_project_remote(
    *,
    site: str,
    token: str,
    name: str,
    display_name: Optional[str],
    description: Optional[str],
) -> Dict[str, Any]:
    url = site.rstrip("/") + "/api/remote/projects"
    headers = {
        "Authorization": f"Bearer cisl-remote {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "name": name,
        "display_name": display_name or name,
        "description": description or "",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        body = resp.json()
    except Exception:
        body = {"error": resp.text[:500]}
    if resp.status_code >= 400 and resp.status_code != 409:
        err = json.dumps(body).lower()
        if "exist" not in err and "already" not in err:
            raise RuntimeError(f"Create project failed HTTP {resp.status_code}: {body}")
    return body


def _finalize_and_upload(
    *,
    sample_id: str,
    sample_folder: Path,
    dzi_path: Path,
    tiles_dir: Path,
    thumb_path: Path,
    sample_scratch: Path,
    args: argparse.Namespace,
    output_type: str,
    model_type: int,
    timings: Dict[str, float],
    lines: List[str],
    started_at: str,
) -> int:
    completed_at = utc_now_iso()
    meta = build_upload_metadata(
        sample_id=sample_id,
        sample_folder=sample_folder,
        project_name=args.project,
        output_type=output_type,
        model_type=model_type,
        use_tensorrt=False,
        tensorrt_precision=None,
        overwrite=args.overwrite,
        create_project=args.create_project,
        timings=timings,
        terminal_output="\n".join(lines),
        started_at=started_at,
        completed_at=completed_at,
        project_display_name=args.project_display_name,
        project_description=args.project_description,
    )

    if args.dry_run:
        print(f"dry-run: DZI at {dzi_path}")
        print(f"dry-run: tiles at {tiles_dir}")
        print(f"dry-run: thumb at {thumb_path}")
        print(json.dumps(meta, indent=2))
        return 0

    if not args.token:
        print(
            "ERROR: set CISLVIEWER_REMOTE_TOKEN or pass --token",
            file=sys.stderr,
        )
        return 2

    archive_path = sample_scratch / f"{sample_id}_dzi.tar.gz"
    pack_dzi_archive(dzi_path, tiles_dir, archive_path)
    size_gb = archive_path.stat().st_size / 1024**3
    print(f"archive={archive_path} ({size_gb:.3f} GiB)")
    print(f"uploading → {args.site}")
    result = upload_to_viewer(
        site=args.site,
        token=args.token,
        metadata=meta,
        archive_path=archive_path,
        thumbnail_path=thumb_path,
    )
    print(json.dumps(result, indent=2))
    viewer_url = result.get("viewer_url", "/" + args.project)
    print(f"viewer: {args.site.rstrip('/')}{viewer_url}")

    if not args.keep_scratch:
        shutil.rmtree(sample_scratch, ignore_errors=True)
    return 0


def process_one_pred(
    *,
    sample_id: str,
    pred_path: Path,
    input_root: Path,
    scratch_root: Path,
    args: argparse.Namespace,
) -> int:
    """Legacy: pred image on disk → DZI → upload."""
    folder_t0 = time.time()
    started_at = utc_now_iso()
    sample_folder = input_root / sample_id
    output_type = default_output_type(args.model, args.vs_modality or args.output_type)

    print(f"\n=== {sample_id} ===")
    print(f"pred={pred_path}")
    print(f"sample_folder={sample_folder} exists={sample_folder.is_dir()}")

    lines = [
        f"Remote WSI upload for {sample_id} (pred→DZI)",
        f"pred={pred_path}",
        f"model_type={args.model} output_type={output_type}",
    ]

    sample_scratch = scratch_root / sample_id
    if sample_scratch.exists():
        shutil.rmtree(sample_scratch)
    sample_scratch.mkdir(parents=True, exist_ok=True)

    dzi_path, tiles_dir, thumb_path, write_time = write_dzi_from_file(
        pred_path,
        scratch_root=sample_scratch,
        sample_id=sample_id,
        output_type=output_type,
    )
    total_time = time.time() - folder_t0
    lines.append(f"  Write Time: {write_time:.2f} seconds")
    lines.append(f"  Total Folder Time: {total_time:.2f} seconds")
    print(f"  Total Folder Time: {total_time:.2f} seconds")

    timings = {
        "bsq_load_time": None,
        "inference_time": None,
        "write_time": float(write_time),
        "total_time": float(total_time),
    }
    return _finalize_and_upload(
        sample_id=sample_id,
        sample_folder=sample_folder if sample_folder.is_dir() else pred_path.parent,
        dzi_path=dzi_path,
        tiles_dir=tiles_dir,
        thumb_path=thumb_path,
        sample_scratch=sample_scratch,
        args=args,
        output_type=output_type,
        model_type=int(args.model),
        timings=timings,
        lines=lines,
        started_at=started_at,
    )


def process_one_canvas(
    *,
    sample_id: str,
    canvas_hwc_u8,
    sample_folder: Path,
    scratch_root: Path,
    args: argparse.Namespace,
    output_type: str,
    model_type: int,
    inference_time: float,
    bsq_load_time: Optional[float],
    started_at: str,
    folder_t0: float,
) -> int:
    """Hot path: uint8 HWC stitch → DZI (no PNG) → upload."""
    lines = [
        f"Remote WSI upload for {sample_id} (infer→DZI, no PNG)",
        f"canvas={tuple(canvas_hwc_u8.shape)} dtype={canvas_hwc_u8.dtype}",
        f"model_type={model_type} output_type={output_type}",
        f"  Inference Time: {inference_time:.2f} seconds",
    ]
    if bsq_load_time is not None:
        lines.append(f"  BSQ Load Time: {bsq_load_time:.2f} seconds")

    sample_scratch = scratch_root / sample_id
    if sample_scratch.exists():
        shutil.rmtree(sample_scratch)
    sample_scratch.mkdir(parents=True, exist_ok=True)

    dzi_path, tiles_dir, thumb_path, write_time = write_dzi_from_array(
        canvas_hwc_u8,
        scratch_root=sample_scratch,
        sample_id=sample_id,
        output_type=output_type,
    )

    total_time = time.time() - folder_t0
    lines.append(f"  Write Time: {write_time:.2f} seconds")
    lines.append(f"  Total Folder Time: {total_time:.2f} seconds")
    print(f"  Inference Time: {inference_time:.2f} seconds")
    print(f"  Total Folder Time: {total_time:.2f} seconds")

    timings = {
        "bsq_load_time": float(bsq_load_time) if bsq_load_time is not None else None,
        "inference_time": float(inference_time),
        "write_time": float(write_time),
        "total_time": float(total_time),
    }
    return _finalize_and_upload(
        sample_id=sample_id,
        sample_folder=sample_folder,
        dzi_path=dzi_path,
        tiles_dir=tiles_dir,
        thumb_path=thumb_path,
        sample_scratch=sample_scratch,
        args=args,
        output_type=output_type,
        model_type=model_type,
        timings=timings,
        lines=lines,
        started_at=started_at,
    )


def _load_cislnet_model(args: argparse.Namespace):
    """Build CISLNet TestOptions + model the same way as infer_wsi_png_cislnet."""
    import torch
    from options.test_options import TestOptions
    from models import create_model

    argv = [
        "remote_wsi_upload.py",
        "--name",
        args.name,
        "--checkpoints_dir",
        args.checkpoints_dir,
        "--dataroot",
        str(_REPO_ROOT / "dataset" / "data_frozen"),
        "--dataset_mode",
        args.dataset_mode,
        "--model",
        args.model_family,
        "--which_epoch",
        args.which_epoch,
        "--results_dir",
        args.results_dir,
        "--gpu_ids",
        args.gpu_ids,
        "--num_test",
        str(10**9),
        "--input_nc",
        str(int(args.input_nc)),
        "--num_ir_channels",
        str(int(args.num_ir_channels)),
        "--load_from_opt_file",
        "--use_wandb",
        "false",
    ]
    if str(args.ir_channel_indices or "").strip():
        argv += ["--ir_channel_indices", str(args.ir_channel_indices)]
    sys.argv = argv

    opt = TestOptions().parse()
    opt.isTrain = False
    opt.phase = "test"
    opt.serial_batches = True
    opt.no_flip = True
    opt.no_flip_vert = True
    opt.no_rotate = True
    try:
        from scripts.infer_wsi_png_cislnet import resolve_wsi_batch_size  # type: ignore
    except ImportError:
        from infer_wsi_png_cislnet import resolve_wsi_batch_size  # type: ignore
    opt.batch_size = resolve_wsi_batch_size(args.batch_size, int(args.tile_size))
    print(f"[WSI] Using batch_size={opt.batch_size} (cli={args.batch_size})")

    model = create_model(opt)
    model.eval()
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and str(args.gpu_ids) != "-1") else "cpu"
    )
    return model, opt, device


def run_infer_cislnet(args: argparse.Namespace, scratch_root: Path) -> int:
    """GPU stitch → array DZI → upload for each slide under --parent_folder."""
    import torch
    from cislnet.wsi_local_padding import resolve_local_padding_config

    try:
        from scripts.infer_wsi_png_cislnet import (  # type: ignore
            VSLiteFolderLoader,
            _loader_model_type,
            infer_folder,
        )
    except ImportError:  # launched as python scripts/remote_wsi_upload.py
        from infer_wsi_png_cislnet import (  # type: ignore
            VSLiteFolderLoader,
            _loader_model_type,
            infer_folder,
        )

    if not args.name or not args.checkpoints_dir:
        print("ERROR: --infer_cislnet requires --name and --checkpoints_dir", file=sys.stderr)
        return 2
    if not args.parent_folder and not args.folder:
        print("ERROR: --infer_cislnet requires --parent_folder or --folder", file=sys.stderr)
        return 2

    lp_cfg = resolve_local_padding_config(
        local_padding=bool(getattr(args, "local_padding", False)),
        local_padding_size=int(getattr(args, "local_padding_size", 1)),
        local_padding_outer=str(getattr(args, "local_padding_outer", "replicate")),
    )

    model, opt, device = _load_cislnet_model(args)
    loader_mt = _loader_model_type(
        int(args.input_nc), int(args.num_ir_channels), str(args.ir_channel_indices or "")
    )
    # Viewer model_type: 0=IR-only, 1=IR+VIS. Prefer explicit --model when set via legacy flag.
    model_type = int(args.model) if args.model is not None else int(loader_mt)
    output_type = default_output_type(model_type, args.vs_modality or args.output_type)

    if args.folder:
        folder_path = Path(args.folder).expanduser().resolve()
        parent = folder_path.parent
        folder_names = [folder_path.name]
    else:
        parent = Path(args.parent_folder).expanduser().resolve()
        want = [x.strip() for x in str(args.folders or "").split(",") if x.strip()]
        all_names = [
            d
            for d in sorted(os.listdir(parent))
            if os.path.isdir(os.path.join(parent, d)) and not d.endswith("_VIS")
        ]
        if want:
            missing = [n for n in want if n not in set(all_names)]
            if missing:
                raise ValueError(f"--folders not found under {parent}: {missing}")
            folder_names = want
        else:
            folder_names = all_names

    if args.samples:
        want = {x.strip() for x in args.samples.split(",") if x.strip()}
        folder_names = [n for n in folder_names if n in want]
        missing = want - set(folder_names)
        if missing:
            print(f"ERROR: samples not found under parent: {sorted(missing)}", file=sys.stderr)
            return 2

    max_folders = int(args.max_folders or 0)
    if max_folders > 0:
        folder_names = folder_names[:max_folders]
    if args.limit and args.limit > 0:
        folder_names = folder_names[: args.limit]

    if not folder_names:
        print("ERROR: no slides to process", file=sys.stderr)
        return 2

    print(
        f"infer→DZI→upload: {len(folder_names)} slide(s) "
        f"name={args.name} epoch={args.which_epoch} output_type={output_type}"
    )
    print(f"slides={folder_names}")

    failures = 0
    for folder_name in folder_names:
        started_at = utc_now_iso()
        folder_t0 = time.time()
        print(f"\n=== {folder_name} ===")
        try:
            t_load = time.time()
            loader = VSLiteFolderLoader(str(parent), folder_name, model_type=loader_mt)
            data = loader.load()
            bsq_load_time = time.time() - t_load

            t_inf = time.time()
            with torch.no_grad():
                pred = infer_folder(
                    model,
                    data,
                    input_nc=int(args.input_nc),
                    num_ir_channels=int(args.num_ir_channels),
                    ir_channel_indices=str(args.ir_channel_indices or ""),
                    tile_size=int(args.tile_size),
                    batch_size=int(opt.batch_size),
                    optim=bool(args.optim),
                    device=device,
                    local_padding=lp_cfg,
                    num_workers=getattr(args, "num_workers", None),
                )
            inference_time = time.time() - t_inf
            # Drop lazy sources before DZI/upload.
            del data, loader

            rc = process_one_canvas(
                sample_id=folder_name,
                canvas_hwc_u8=pred,
                sample_folder=parent / folder_name,
                scratch_root=scratch_root,
                args=args,
                output_type=output_type,
                model_type=model_type,
                inference_time=inference_time,
                bsq_load_time=bsq_load_time,
                started_at=started_at,
                folder_t0=folder_t0,
            )
            del pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if rc != 0:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"ERROR processing {folder_name}: {exc}", file=sys.stderr)

    print(f"\ndone: {len(folder_names) - failures}/{len(folder_names)} ok, {failures} failed")
    return 0 if failures == 0 else 1


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WSI infer→DZI→upload (preferred) or pred→DZI→upload (legacy)",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--infer_cislnet",
        action="store_true",
        help="Run CISLNet WSI stitch → pyvips DZI from array → upload (no PNG)",
    )
    src.add_argument(
        "--wsi_pred_dir",
        help="Legacy: directory of WSI pred PNGs/JPEGs (e.g. results/.../wsi_pred)",
    )
    src.add_argument(
        "--pred_image",
        help="Legacy: single pred image path",
    )

    # Shared upload / placement
    p.add_argument(
        "--sample_id",
        default=None,
        help="Override sample_id when using --pred_image",
    )
    p.add_argument(
        "--samples",
        default=None,
        help="Comma-separated sample ids to process",
    )
    p.add_argument(
        "--input_root",
        default="input/IR",
        help="Parent of sample folders (legacy metadata / alias for --parent_folder)",
    )
    p.add_argument("--project", required=True, help="Viewer project slug")
    p.add_argument("--project_display_name", default=None)
    p.add_argument("--project_description", default=None)
    p.add_argument(
        "--output_type",
        default=None,
        help="DZI modality folder (alias for --vs_modality)",
    )
    p.add_argument("--vs_modality", default=None, help="Override output_type / DZI subfolder")
    p.add_argument(
        "--model",
        type=int,
        default=None,
        choices=[0, 1],
        help="Viewer model_type 0=10ch/IR-only, 1=13ch/IR+VIS (default: derived when inferring)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--create_project", action="store_true")
    p.add_argument("--ensure_project", action="store_true")
    p.add_argument("--scratch", default=None)
    p.add_argument("--keep_scratch", action="store_true")
    p.add_argument("--site", default=DEFAULT_SITE)
    p.add_argument("--token", default=DEFAULT_TOKEN)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)

    # CISLNet infer knobs (used with --infer_cislnet)
    p.add_argument("--name", default=None, help="Checkpoint run name (e.g. c13, vp1)")
    p.add_argument("--checkpoints_dir", default=None)
    p.add_argument("--which_epoch", type=str, default="latest")
    p.add_argument(
        "--results_dir",
        type=str,
        default=str(_REPO_ROOT / "results" / "cislnet"),
    )
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--dataset_mode", type=str, default="VS")
    p.add_argument(
        "--model_family",
        type=str,
        default="contrastive_pix2pix",
        help="pix2pix model class name for TestOptions",
    )
    p.add_argument("--input_nc", type=int, default=13)
    p.add_argument("--num_ir_channels", type=int, default=10)
    p.add_argument("--ir_channel_indices", type=str, default="")
    p.add_argument("--folder", type=str, default=None)
    p.add_argument("--parent_folder", type=str, default=None)
    p.add_argument("--folders", type=str, default="")
    p.add_argument("--max_folders", type=int, default=0)
    p.add_argument("--tile_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Parallel CPU tile readers; overrides WSI_NUM_WORKERS (default 4).",
    )
    p.add_argument("--optim", action="store_true", default=False)
    p.add_argument("--local_padding", action="store_true", default=False)
    p.add_argument("--local_padding_size", type=int, default=1)
    p.add_argument(
        "--local_padding_outer",
        type=str,
        default="replicate",
        choices=("replicate", "constant"),
    )

    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if args.output_type and not args.vs_modality:
        args.vs_modality = args.output_type

    # Default viewer model_type for legacy pred path.
    if args.model is None and not args.infer_cislnet:
        args.model = 1

    if args.parent_folder is None and args.input_root:
        args.parent_folder = args.input_root

    scratch_root = (
        Path(args.scratch).expanduser().resolve()
        if args.scratch
        else Path(tempfile.mkdtemp(prefix="cisl_remote_"))
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    print(f"scratch={scratch_root}")

    if args.ensure_project and not args.dry_run:
        if not args.token:
            print("ERROR: token required for --ensure_project", file=sys.stderr)
            return 2
        print(f"ensuring project {args.project} …")
        body = create_project_remote(
            site=args.site,
            token=args.token,
            name=args.project,
            display_name=args.project_display_name,
            description=args.project_description,
        )
        print(json.dumps(body, indent=2))

    if args.infer_cislnet:
        return run_infer_cislnet(args, scratch_root)

    # --- Legacy pred → DZI → upload ---
    input_root = Path(args.input_root).expanduser().resolve()
    if args.pred_image:
        pred_path = Path(args.pred_image).expanduser().resolve()
        if not pred_path.is_file():
            print(f"ERROR: not a file: {pred_path}", file=sys.stderr)
            return 2
        sample_id = args.sample_id
        if not sample_id:
            m = _PRED_NAME_RE.match(pred_path.name)
            sample_id = m.group("sample") if m else pred_path.stem.split("_")[0]
        items = [(sample_id, pred_path)]
    else:
        wsi_pred_dir = Path(args.wsi_pred_dir).expanduser().resolve()
        if not wsi_pred_dir.is_dir():
            print(f"ERROR: not a directory: {wsi_pred_dir}", file=sys.stderr)
            return 2
        items = discover_preds(wsi_pred_dir)

    if args.samples:
        want = {x.strip() for x in args.samples.split(",") if x.strip()}
        items = [(s, p) for s, p in items if s in want]
        missing = want - {s for s, _ in items}
        if missing:
            print(f"ERROR: samples not found in preds: {sorted(missing)}", file=sys.stderr)
            return 2

    if args.limit and args.limit > 0:
        items = items[: args.limit]

    if not items:
        print("ERROR: no pred images to process", file=sys.stderr)
        return 2

    print(f"will process {len(items)} sample(s): {[s for s, _ in items]}")

    failures = 0
    for sample_id, pred_path in items:
        try:
            rc = process_one_pred(
                sample_id=sample_id,
                pred_path=pred_path,
                input_root=input_root,
                scratch_root=scratch_root,
                args=args,
            )
            if rc != 0:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"ERROR processing {sample_id}: {exc}", file=sys.stderr)

    print(f"\ndone: {len(items) - failures}/{len(items)} ok, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
