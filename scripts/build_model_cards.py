#!/usr/bin/env python3
"""Build model_cards/ from results/** checkpoints + metrics.json.

One card per canonical weight set (latest_net_G.pth / ckpt_latest.pth|pt).
Re-runnable: overwrites cards/ and index.json. Does not touch schema.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "model_cards"
CARDS_DIR = OUT_DIR / "cards"
SCHEMA_PATH = OUT_DIR / "schema.json"
SCHEMA_VERSION = "1.0.0"
GENERATOR = "scripts/build_model_cards.py"

FAMILY_TASK = {
    "cislnet": "i2i",
    "swingan_i2i": "i2i",
    "swingan": "i2i",
    "restormer_i2i": "i2i",
    "restormer": "i2i",
    "nafnet": "i2i",
    "vqgan_vs": "autoencoder",
    "vqgan": "autoencoder",
    "tcga_encoder": "encoder",
}

METRIC_CORE = (
    "lpips_convnext_tcga_mean",
    "lpips_convnext_tcga_std",
    "lpips_uni_mean",
    "lpips_uni_std",
    "psnr_mean",
    "psnr_std",
    "ssim_mean",
    "ssim_std",
    "fid",
)

# Keys that are metadata on metrics.json, not metric values.
METRIC_META = {
    "n",
    "pred_dir",
    "target_dir",
    "split",
    "region_eval_size",
    "model_tile_size",
    "path",
    "precision",
    "scheme",
}


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def parse_opt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(errors="replace")
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            return {str(k): v for k, v in data.items()}
        except json.JSONDecodeError:
            pass
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("-") or "Options" in line and line.endswith("Options"):
            continue
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        val = raw.split("\t")[0].strip()
        out[key] = _coerce(val)
    return out


def _coerce(val: str) -> Any:
    if val in {"True", "true"}:
        return True
    if val in {"False", "false"}:
        return False
    if val in {"None", "none", ""}:
        return None
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def parse_iter(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    # Formats: "25\n0", "50,93850", "15,300225"
    first = text.splitlines()[0].strip()
    token = first.split(",")[0].strip()
    try:
        return int(float(token))
    except ValueError:
        return None


def sha256_file(path: Path, *, max_bytes: int = 0) -> str | None:
    """Hash checkpoint. max_bytes=0 means full file (can be slow for large ckpts)."""
    if max_bytes == 0:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    data = path.read_bytes()[:max_bytes]
    return hashlib.sha256(data).hexdigest() if data else None


def discover_weights(results: Path) -> list[tuple[str, str, Path]]:
    """Return (family, run, weight_path) for canonical checkpoints."""
    found: list[tuple[str, str, Path]] = []
    if not results.is_dir():
        return found

    # CISLNet-style: results/<family>/<run>/latest_net_G.pth
    for p in results.glob("*/*/latest_net_G.pth"):
        family, run = p.parts[-3], p.parts[-2]
        if run in {"test", "quant", "experiments", "remote_scratch"}:
            continue
        found.append((family, run, p))

    # i2i / encoder: results/<family>/<run>/ckpt_latest.pth
    for p in results.glob("*/*/ckpt_latest.pth"):
        family, run = p.parts[-3], p.parts[-2]
        if "smoke" in run and False:
            pass
        found.append((family, run, p))

    # VQGAN: results/<family>/<run>/checkpoints/ckpt_latest.pt
    for p in results.glob("*/*/checkpoints/ckpt_latest.pt"):
        family, run = p.parts[-4], p.parts[-3]
        found.append((family, run, p))

    # Dedup by (family, run), prefer latest_net_G then ckpt_latest
    by_key: dict[tuple[str, str], Path] = {}
    priority = {"latest_net_G.pth": 0, "ckpt_latest.pth": 1, "ckpt_latest.pt": 2}
    for family, run, path in found:
        key = (family, run)
        prev = by_key.get(key)
        if prev is None or priority.get(path.name, 9) < priority.get(prev.name, 9):
            by_key[key] = path
    return [(f, r, p) for (f, r), p in sorted(by_key.items())]


def discover_metrics(results: Path) -> list[Path]:
    return sorted(results.rglob("metrics.json"))


def run_from_metrics_path(path: Path) -> tuple[str, str] | None:
    """Map metrics.json path → (family, train_run_guess)."""
    parts = path.parts
    try:
        i = parts.index("results")
    except ValueError:
        return None
    rest = parts[i + 1 :]
    if not rest:
        return None
    family = rest[0]

    # results/<fam>/test/<run>/patch_eval_*/metrics.json
    if len(rest) >= 4 and rest[1] == "test":
        tag = rest[2]
        return family, _train_run_from_eval_tag(tag)

    # results/<fam>/quant/<run>/test/<scheme>/patch_eval_*/metrics.json
    if len(rest) >= 5 and rest[1] == "quant":
        return family, rest[2]

    return None


def _train_run_from_eval_tag(tag: str) -> str:
    """c13_oa_all__hold_kidney_20 → c13_oa_all; c10_30 → c10; a13 stays a13."""
    if "__" in tag:
        return tag.split("__", 1)[0]
    # Epoch ladder leaves: c10_5, c10_10, … c10_30 → parent c10 when parent exists later
    m = re.fullmatch(r"(.+)_(\d+)$", tag)
    if m and m.group(1) in {"c10", "c13"} and int(m.group(2)) <= 500:
        # Keep as distinct eval on parent card when parent weights exist; else own card.
        return m.group(1)
    # bestlpips / eN suffixes often point at same train run
    for suffix in ("_bestlpips",):
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    m2 = re.fullmatch(r"(.+)_e(\d+)$", tag)
    if m2:
        return m2.group(1)
    return tag


def eval_id_from_path(path: Path) -> tuple[str, dict[str, Any]]:
    """Build eval id + annotations from path."""
    parts = list(path.parts)
    meta: dict[str, Any] = {}
    name = path.parent.name  # patch_eval_test / patch_eval_train
    split = "test"
    if name.endswith("_train"):
        split = "train"
    elif name.endswith("_val"):
        split = "val"
    elif name.endswith("_test"):
        split = "test"

    # Find run tag folder
    if "quant" in parts:
        qi = parts.index("quant")
        scheme = parts[qi + 3] if len(parts) > qi + 3 else None
        meta["quantization"] = scheme
        eid = f"quant_{scheme}_{split}" if scheme else f"quant_{split}"
        return eid, meta

    # test/<tag>/
    if "test" in parts:
        ti = parts.index("test")
        tag = parts[ti + 1] if len(parts) > ti + 1 else "unknown"
        if "__hold_" in tag:
            region = tag.split("__hold_", 1)[1]
            meta["region"] = region
            eid = f"hold_{region}_{split}"
            return eid, meta
        m = re.fullmatch(r"(.+)_(\d+)$", tag)
        if m and m.group(1) in {"c10", "c13"}:
            meta["epoch"] = int(m.group(2))
            eid = f"epoch_{m.group(2)}_{split}"
            return eid, meta
        if tag.endswith("_bestlpips"):
            eid = f"bestlpips_{split}"
            return eid, meta
        m2 = re.fullmatch(r"(.+)_e(\d+)$", tag)
        if m2:
            meta["epoch"] = int(m2.group(2))
            eid = f"epoch_{m2.group(2)}_{split}"
            return eid, meta
        eid = split if tag == _train_run_from_eval_tag(tag) else f"{tag}_{split}"
        return eid, meta

    return f"{split}", meta


def pack_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for k, v in raw.items():
        if k in METRIC_META:
            continue
        if k in METRIC_CORE:
            metrics[k] = v
        else:
            extras[k] = v
    if extras:
        metrics["extras"] = extras
    return metrics


def primary_metric(metrics: dict[str, Any]) -> dict[str, Any]:
    for name in ("lpips_convnext_tcga_mean", "lpips_uni_mean", "fid", "psnr_mean"):
        if metrics.get(name) is not None:
            lower = name != "psnr_mean"
            return {"name": name, "value": metrics[name], "lower_is_better": lower}
    return {"name": "lpips_convnext_tcga_mean", "value": None, "lower_is_better": True}


def infer_status(run: str, has_eval: bool) -> str:
    if "smoke" in run.lower():
        return "smoke"
    if "nan_bad" in run.lower() or run.endswith(".bak"):
        return "superseded"
    return "published" if has_eval else "draft"


def _guess_input_channels(run: str) -> int | None:
    """Best-effort channel count when opt.txt is missing input_nc."""
    if run.startswith("ir1") or run == "ir1":
        return 1
    if run.startswith("ir3") or run == "ir3":
        return 3
    if run.startswith("ir5") or run == "ir5":
        return 5
    if run.startswith("vs6") or "_c6" in run or run in {"c6", "vs6"}:
        return 6
    if run.startswith("vs10") or "c10" in run or run in {"c10", "vs10"}:
        return 10
    if "c13" in run or run in {"c13", "vs13"} or "visonly" in run:
        return 13
    if run.startswith(("vp", "a", "b", "d", "dec", "enc")):
        return 13
    return None


def losses_from_opt(opt: dict[str, Any], task: str) -> dict[str, Any]:
    losses: dict[str, Any] = {}
    if task == "encoder":
        if "temperature" in opt:
            losses["supcon_temperature"] = opt.get("temperature")
        return losses

    mapping = {
        "L1_loss": "l1",
        "gan_loss": "gan",
        "contrastive_loss": "nce",
        "convnext_loss": "convnext",
        "ganFeat_loss": "gan_feat",
        "lambda_L1": "lambda_l1",
        "lambda_L2": "lambda_l2",
        "lambda_NCE": "lambda_nce",
        "lambda_gan": "lambda_gan",
        "lambda_vgg": "lambda_vgg",
        "lambda_l1": "lambda_l1",
        "lambda_perceptual": "lambda_perceptual",
        "use_perceptual": "perceptual",
        "gan_training": "gan",
    }
    for src, dst in mapping.items():
        if src in opt and opt[src] is not None:
            losses[dst] = opt[src]
    return losses


def architecture_from_opt(opt: dict[str, Any], family: str, task: str) -> dict[str, Any]:
    arch: dict[str, Any] = {"extras": {}}
    if task == "encoder":
        arch["name"] = str(opt.get("arch") or family)
        arch["backbone"] = str(opt.get("arch")) if opt.get("arch") else None
        return arch

    if family.startswith("cislnet"):
        arch["name"] = str(opt.get("netG") or "attunet")
        arch["backbone"] = str(opt.get("netG") or "attunet")
        # Ablation tags often encode attention / structure
        extras = {}
        for k in ("netD", "netF", "netH", "model"):
            if k in opt:
                extras[k] = opt[k]
        arch["extras"] = extras
        return arch

    if "swin" in family or opt.get("use_cbam") is not None:
        arch["name"] = "ProjectedSwinUnet"
        arch["backbone"] = "swin_tiny"
        arch["attention"] = "cbam" if opt.get("use_cbam") else None
        arch["extras"] = {
            k: opt[k]
            for k in ("embed_dim", "depths", "num_heads", "window_size", "discriminator_variant")
            if k in opt
        }
        return arch

    if "restormer" in family:
        arch["name"] = "Restormer"
        arch["backbone"] = "restormer"
        return arch

    if "nafnet" in family:
        arch["name"] = "NAFNet"
        arch["backbone"] = "nafnet"
        return arch

    if task == "autoencoder" or "vqgan" in family:
        arch["name"] = "VQGAN"
        arch["backbone"] = "vqgan"
        return arch

    arch["name"] = family
    return arch


def io_from_opt(opt: dict[str, Any], task: str) -> dict[str, Any]:
    io: dict[str, Any] = {}
    if task == "encoder":
        io["train_resolution"] = opt.get("img_size")
        io["infer_resolution"] = opt.get("img_size")
        io["input_channels"] = 3
        io["output_channels"] = None
        return io

    input_nc = opt.get("input_nc")
    if input_nc is None and opt.get("num_ir_channels") is not None:
        # SwinGAN input_mode ir10 / irvis13
        mode = str(opt.get("input_mode") or "")
        ir = int(opt["num_ir_channels"])
        if "vis" in mode or "13" in mode:
            input_nc = ir + 3
        else:
            input_nc = ir
    io["input_channels"] = int(input_nc) if input_nc is not None else None
    io["input_mode"] = opt.get("input_mode")
    io["output_channels"] = 3 if task in {"i2i", "autoencoder"} else None
    res = opt.get("image_size") or opt.get("img_size") or opt.get("load_size")
    io["train_resolution"] = int(res) if res is not None else None
    io["infer_resolution"] = io["train_resolution"]
    io["color_space"] = opt.get("he_color_space") or opt.get("he_scale")
    io["pixel_size_um"] = 0.69 if task == "i2i" else None
    return io


def training_from_opt(opt: dict[str, Any], epochs_completed: int | None, task: str) -> dict[str, Any]:
    planned = (
        opt.get("total_epochs")
        or opt.get("niter")
        or opt.get("epochs_n")
        or opt.get("epochs")
        or opt.get("n_epochs")
    )
    # CISLNet often uses niter + niter_decay
    if planned is not None and opt.get("niter_decay") is not None and opt.get("total_epochs") is None:
        try:
            planned = int(planned) + int(opt["niter_decay"])
        except (TypeError, ValueError):
            pass

    bs = opt.get("batch_size")
    accum = opt.get("grad_accum") or opt.get("gradient_accumulation_steps") or 1
    try:
        eff = int(bs) * int(accum) if bs is not None else None
    except (TypeError, ValueError):
        eff = None

    lr = opt.get("lr") or opt.get("lr_g") or opt.get("base_lr")
    hypers_skip = {
        "input_nc",
        "image_size",
        "img_size",
        "batch_size",
        "lr",
        "lr_g",
        "base_lr",
        "dataroot",
        "checkpoints_dir",
        "output_dir",
        "results_root",
        "name",
        "run_name",
        "gpu_ids",
        "device",
        "isTrain",
        "gpu_id",
    }
    hypers = {
        k: v
        for k, v in opt.items()
        if k not in hypers_skip and not str(k).startswith("lambda_") and k not in {
            "L1_loss",
            "gan_loss",
            "contrastive_loss",
            "convnext_loss",
            "ganFeat_loss",
            "use_perceptual",
            "gan_training",
            "lambda_L1",
            "lambda_L2",
            "lambda_NCE",
            "lambda_gan",
            "lambda_vgg",
            "lambda_l1",
            "lambda_perceptual",
        }
    }
    # Cap hypers size for catalog readability
    if len(hypers) > 40:
        keep = sorted(hypers.keys())[:40]
        hypers = {k: hypers[k] for k in keep}

    return {
        "epochs_planned": int(planned) if planned is not None else None,
        "epochs_completed": epochs_completed,
        "batch_size": int(bs) if bs is not None else None,
        "grad_accum": int(accum) if accum is not None else None,
        "effective_batch": eff,
        "lr": float(lr) if lr is not None else None,
        "optimizer": opt.get("optimizer"),
        "dataroot": str(opt["dataroot"]) if opt.get("dataroot") not in (None, "") else None,
        "seed": opt.get("seed") or opt.get("split_seed"),
        "losses": losses_from_opt(opt, task),
        "hypers": hypers,
        "duration": {"wall_hours": None, "source": None},
    }


def build_card(
    family: str,
    run: str,
    weight: Path,
    evals_raw: list[tuple[Path, dict[str, Any]]],
    *,
    hash_weights: bool,
) -> dict[str, Any]:
    run_dir = weight.parent if weight.name != "ckpt_latest.pt" else weight.parent.parent
    opt_path = run_dir / "opt.txt"
    iter_path = run_dir / "iter.txt"
    config_path = run_dir / "config.yaml"
    opt = parse_opt(opt_path)
    epochs_completed = parse_iter(iter_path)
    task = FAMILY_TASK.get(family, "other")
    sources = [_rel(weight)]
    if opt_path.is_file():
        sources.append(_rel(opt_path))
    if iter_path.is_file():
        sources.append(_rel(iter_path))
    if config_path.is_file():
        sources.append(_rel(config_path))

    evals: list[dict[str, Any]] = []
    for mpath, raw in evals_raw:
        eid, meta = eval_id_from_path(mpath)
        metrics = pack_metrics(raw)
        n = raw.get("n")
        entry = {
            "id": eid,
            "split": str(raw.get("split") or meta.get("split") or ("train" if "train" in eid else "test")),
            "test_set": None,
            "n": int(n) if n is not None else None,
            "region": meta.get("region"),
            "quantization": meta.get("quantization"),
            "epoch": meta.get("epoch"),
            "metrics_path": _rel(mpath),
            "metrics": metrics,
            "primary_metric": primary_metric(metrics),
        }
        sources.append(_rel(mpath))
        evals.append(entry)

    # Prefer frozen test LPIPS without region/quant/epoch ladder for primary_lpips
    primary_lpips = None
    for prefer in evals:
        if prefer["split"] == "test" and prefer.get("region") is None and prefer.get("quantization") is None:
            if prefer.get("epoch") is None or prefer["id"] in {"test", f"{run}_test"}:
                v = prefer["metrics"].get("lpips_convnext_tcga_mean")
                if v is not None:
                    primary_lpips = float(v)
                    break
    if primary_lpips is None:
        for prefer in evals:
            v = prefer["metrics"].get("lpips_convnext_tcga_mean")
            if v is not None:
                primary_lpips = float(v)
                break

    status = infer_status(run, bool(evals))
    card_id = f"{family}__{run}"
    weights_obj: dict[str, Any] = {
        "path": _rel(weight),
        "kind": "latest",
        "epoch": epochs_completed,
        "bytes": weight.stat().st_size if weight.is_file() else None,
        "sha256": sha256_file(weight) if hash_weights and weight.is_file() else None,
        "companions": [],
    }
    if weight.name == "latest_net_G.pth" and epochs_completed is not None:
        for role in ("D", "F_q", "F", "H"):
            sib = run_dir / f"latest_net_{role}.pth"
            if sib.is_file():
                weights_obj["companions"].append(_rel(sib))

    arch = architecture_from_opt(opt, family, task)
    io = io_from_opt(opt, task)
    # i2i requires input_channels + train_resolution; fill unknowns with null only if present in opt failure
    if task == "i2i":
        if io.get("input_channels") is None:
            io["input_channels"] = _guess_input_channels(run)
        if io.get("train_resolution") is None:
            io["train_resolution"] = 1024 if family == "cislnet" else 512

    tags = [family, task, run]
    if io.get("input_channels"):
        tags.append(f"in{io['input_channels']}")
    if status == "smoke":
        tags.append("smoke")

    links: dict[str, str] = {}
    test_dir = REPO / "results" / family / "test" / run
    if (test_dir / "wsi_pred").is_dir():
        links["wsi_pred"] = _rel(test_dir / "wsi_pred")
    samples = run_dir / "samples"
    if samples.is_dir():
        links["samples"] = _rel(samples)

    card: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": card_id,
        "family": family,
        "run": run,
        "display_name": f"{family}/{run}",
        "task": task,
        "status": status,
        "summary": f"{task} model {family}/{run}",
        "tags": sorted(set(tags)),
        "architecture": arch,
        "io": io,
        "training": training_from_opt(opt, epochs_completed, task),
        "weights": weights_obj,
        "evals": evals,
        "primary_lpips": primary_lpips,
        "links": links,
        "extensions": {},
        "created_from": {
            "generator": GENERATOR,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sources": sorted(set(sources)),
        },
    }
    if config_path.is_file():
        card["extensions"]["config_yaml"] = _rel(config_path)
    return card


def attach_evals(
    weights: list[tuple[str, str, Path]], metrics_paths: list[Path]
) -> dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]]:
    weight_keys = {(f, r) for f, r, _ in weights}
    attached: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = {k: [] for k in weight_keys}
    orphans: list[str] = []

    for mpath in metrics_paths:
        mapped = run_from_metrics_path(mpath)
        if mapped is None:
            orphans.append(_rel(mpath))
            continue
        family, run = mapped
        # Epoch-ladder metrics (c10_30) map to parent c10 when parent exists
        key = (family, run)
        if key not in attached:
            # Try exact tag as its own run (e.g. metrics under a13 for swingan)
            parts = mpath.parts
            if "test" in parts:
                ti = parts.index("test")
                tag = parts[ti + 1]
                key2 = (family, tag)
                if key2 in attached:
                    key = key2
                else:
                    orphans.append(_rel(mpath))
                    continue
            else:
                orphans.append(_rel(mpath))
                continue
        try:
            raw = json.loads(mpath.read_text())
        except (OSError, json.JSONDecodeError):
            orphans.append(_rel(mpath))
            continue
        attached[key].append((mpath, raw))
    return attached


def validate_card(card: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Lightweight validation without jsonschema dependency."""
    errs: list[str] = []
    for req in schema.get("required", []):
        if req not in card:
            errs.append(f"missing required: {req}")
    if card.get("schema_version") != SCHEMA_VERSION:
        errs.append("bad schema_version")
    if card.get("task") == "i2i":
        io = card.get("io") or {}
        if io.get("input_channels") is None:
            errs.append("i2i missing io.input_channels")
        if io.get("train_resolution") is None:
            errs.append("i2i missing io.train_resolution")
    if card.get("task") == "encoder":
        if not (card.get("architecture") or {}).get("name"):
            errs.append("encoder missing architecture.name")
    if "weights" in card and "path" not in card["weights"]:
        errs.append("weights.path missing")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--hash-weights", action="store_true", help="SHA256 every checkpoint (slow)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results = args.results if args.results.is_absolute() else REPO / args.results
    out = args.out if args.out.is_absolute() else args.out
    cards_dir = out / "cards"
    schema = json.loads(SCHEMA_PATH.read_text()) if SCHEMA_PATH.is_file() else {"required": []}

    weights = discover_weights(results)
    metrics_paths = discover_metrics(results)
    attached = attach_evals(weights, metrics_paths)

    cards: list[dict[str, Any]] = []
    errors: list[str] = []
    for family, run, wpath in weights:
        card = build_card(family, run, wpath, attached.get((family, run), []), hash_weights=args.hash_weights)
        errs = validate_card(card, schema)
        if errs:
            errors.append(f"{card['id']}: {', '.join(errs)}")
        cards.append(card)

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": GENERATOR,
        "n_cards": len(cards),
        "n_with_lpips": sum(1 for c in cards if c.get("primary_lpips") is not None),
        "n_with_evals": sum(1 for c in cards if c.get("evals")),
        "cards": [
            {
                "id": c["id"],
                "family": c["family"],
                "run": c["run"],
                "task": c["task"],
                "status": c["status"],
                "primary_lpips": c.get("primary_lpips"),
                "weights_path": c["weights"]["path"],
                "n_evals": len(c.get("evals") or []),
                "card_path": f"cards/{c['id']}.json",
            }
            for c in sorted(
                cards,
                key=lambda c: (
                    c.get("primary_lpips") is None,
                    c.get("primary_lpips") if c.get("primary_lpips") is not None else 9e9,
                    c["id"],
                ),
            )
        ],
        "validation_errors": errors,
    }

    if args.dry_run:
        print(json.dumps({"n_cards": len(cards), "n_errors": len(errors), "sample": index["cards"][:5]}, indent=2))
        return 1 if errors else 0

    cards_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale cards from prior runs
    for old in cards_dir.glob("*.json"):
        old.unlink()
    for card in cards:
        (cards_dir / f"{card['id']}.json").write_text(json.dumps(card, indent=2, sort_keys=False) + "\n")
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    print(f"wrote {len(cards)} cards → {cards_dir}")
    print(f"index → {out / 'index.json'} (with_lpips={index['n_with_lpips']}, errors={len(errors)})")
    if errors:
        for e in errors[:20]:
            print("  ERR", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
