#!/usr/bin/env python3
"""Figures + results_c13.md for a single CISLNet run evaluated every N epochs.

Default run is c13 (frozen all-organ train, canonical test.txt).
Former val.txt samples (10442, 3840) live in test.txt; there is no val split.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/projects/illinois/eng/bioe/rxb/akshat/VSlite")
CKPT = ROOT / "results" / "cislnet"
TEST = CKPT / "test"
UNIFIED = ROOT / "results" / "unified" / "cislnet"
# Sibling of results_c13.md so markdown preview can resolve ./results_figures_c13/
OUT = ROOT / "results_figures_c13"
MD_PATH = ROOT / "results_c13.md"

DEFAULT_RUN = "c13"
DEFAULT_TRAIN_PATCHES = 60047
COLOR = "#1b4f72"
# Frozen split (dataset/data_frozen): all ir/*.npy minus test.txt.
# Former val samples were merged into test.txt.
N_TRAIN = 60047
N_TEST = 6929
N_EVAL = N_TEST


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def load_eval_rows(run: str) -> list[dict]:
    prefix = f"{run}_"
    rows = []
    for p in sorted(TEST.glob(f"{run}_*/patch_eval_test/metrics.json")):
        name = p.parent.parent.name
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        if not rest.isdigit():
            continue
        m = json.loads(p.read_text())
        lp = m.get("lpips_convnext_tcga_mean")
        lp_std = m.get("lpips_convnext_tcga_std")
        rows.append(
            {
                "epoch": int(rest),
                "n": int(m["n"]),
                "psnr": float(m["psnr_mean"]),
                "psnr_std": float(m["psnr_std"]),
                "ssim": float(m["ssim_mean"]),
                "ssim_std": float(m["ssim_std"]),
                "lpips": None if lp is None else float(lp),
                "lpips_std": None if lp_std is None else float(lp_std),
                "fid": float(m["fid"]),
            }
        )
    rows.sort(key=lambda r: r["epoch"])
    return rows


def smooth(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(y) < win:
        return y
    k = np.ones(win, dtype=np.float64) / float(win)
    pad = win // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, k, mode="valid")[: len(y)]


def load_log_json(run: str, n_patches: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    p = CKPT / run / "log.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text())
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, pts in data.get("scalars", {}).items():
        step = np.array([a[0] for a in pts], dtype=np.float64)
        val = np.array([a[1] for a in pts], dtype=np.float64)
        epoch = step / float(max(1, n_patches))
        out[key] = (epoch, val)
    return out


def _read_tfrecords_skip_large(path: Path, max_len: int = 8192):
    from tensorboard.compat.proto.event_pb2 import Event

    with open(path, "rb") as f:
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                return
            length = struct.unpack("<Q", hdr[:8])[0]
            if length > max_len:
                f.seek(length + 4, 1)
                continue
            payload = f.read(length)
            crc = f.read(4)
            if len(payload) < length or len(crc) < 4:
                return
            ev = Event()
            ev.ParseFromString(payload)
            yield ev


def extract_tb_l1(run: str) -> dict[str, list[tuple[int, float]]]:
    d = UNIFIED / run
    files = sorted(d.glob("events.out.tfevents*"))
    want = {
        "l1/train",
        "l1/val",
        "l1/test",
        "lpips_convnext_tcga/train",
        "lpips_convnext_tcga/val",
        "lpips_convnext_tcga/test",
        "meta/num_channels",
    }
    tags: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for fp in files:
        print(f"  TB scan {fp.name} ({fp.stat().st_size / 1e6:.0f} MB)", flush=True)
        n_small = 0
        for ev in _read_tfrecords_skip_large(fp):
            if not ev.HasField("summary"):
                continue
            for v in ev.summary.value:
                if v.HasField("simple_value") and v.tag in want:
                    tags[v.tag].append((int(ev.step), float(v.simple_value)))
                    n_small += 1
        print(f"    kept {n_small} scalar events, tags={list(tags)}", flush=True)
    return dict(tags)


def _epochs_ticks(rows: list[dict]) -> list[int]:
    eps = sorted({r["epoch"] for r in rows})
    if not eps:
        return list(range(2, 36, 2)) + [35]
    if len(eps) <= 12:
        return eps
    return eps[::2] if eps[-1] in eps[::2] else eps[::2] + [eps[-1]]


def fig_fid(rows: list[dict]) -> Path:
    ticks = _epochs_ticks(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    if rows:
        ax.plot(
            [r["epoch"] for r in rows],
            [r["fid"] for r in rows],
            "o-",
            color=COLOR,
            markersize=6,
            linewidth=1.6,
        )
        best = min(rows, key=lambda r: r["fid"])
        ax.scatter([best["epoch"]], [best["fid"]], s=64, zorder=3, color="#c0392b")
        ax.annotate(
            f"best e{best['epoch']}  {best['fid']:.2f}",
            (best["epoch"], best["fid"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="#c0392b",
        )
    ax.set_title(f"Canonical test FID  (n={N_TEST:,})")
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel("FID (lower better)")
    ax.set_xticks(ticks)
    if ticks:
        ax.set_xlim(ticks[0] - 1, ticks[-1] + 1)
    path = OUT / "fig_fid.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_psnr_ssim(rows: list[dict]) -> Path:
    ticks = _epochs_ticks(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.6), sharex=True)
    if rows:
        ep = np.array([r["epoch"] for r in rows])
        axes[0].errorbar(
            ep,
            [r["psnr"] for r in rows],
            yerr=[r["psnr_std"] / np.sqrt(r["n"]) for r in rows],
            fmt="o-",
            color=COLOR,
            markersize=5,
            linewidth=1.5,
            capsize=2,
        )
        axes[1].errorbar(
            ep,
            [r["ssim"] for r in rows],
            yerr=[r["ssim_std"] / np.sqrt(r["n"]) for r in rows],
            fmt="o-",
            color=COLOR,
            markersize=5,
            linewidth=1.5,
            capsize=2,
        )
        lp_rows = [r for r in rows if r.get("lpips") is not None]
        if lp_rows:
            ep_lp = np.array([r["epoch"] for r in lp_rows])
            yerr = [
                (r["lpips_std"] / np.sqrt(r["n"])) if r.get("lpips_std") is not None else 0.0
                for r in lp_rows
            ]
            axes[2].errorbar(
                ep_lp,
                [r["lpips"] for r in lp_rows],
                yerr=yerr,
                fmt="o-",
                color=COLOR,
                markersize=5,
                linewidth=1.5,
                capsize=2,
            )
            best = min(lp_rows, key=lambda r: r["lpips"])
            axes[2].scatter([best["epoch"]], [best["lpips"]], s=48, zorder=3, color="#c0392b")
    axes[0].set_title("PSNR (dB, higher better)")
    axes[1].set_title("SSIM (higher better)")
    axes[2].set_title("LPIPS ConvNeXt-TCGA (lower better)")
    for ax in axes:
        ax.set_xlabel("Checkpoint epoch")
        ax.set_xticks(ticks)
    fig.suptitle(
        f"Canonical test PSNR / SSIM / LPIPS  (n={N_TEST:,}, error bars = SEM)",
        y=1.04,
        fontsize=12,
    )
    path = OUT / "fig_psnr_ssim.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_train_losses(run: str, n_patches: int) -> Path | None:
    series = load_log_json(run, n_patches)
    if "errors/NCE" not in series:
        return None
    ep, nce = series["errors/NCE"]
    win = max(21, len(nce) // 80)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ep, smooth(nce, win), color=COLOR, linewidth=1.5, label="NCE (smoothed)")
    ax.set_title("NCE loss (smoothed)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NCE")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlim(0, 36)
    fig.suptitle("Training NCE from log.json", y=1.03, fontsize=12)
    path = OUT / "fig_train_losses.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_epoch_l1(tags: dict[str, list[tuple[int, float]]]) -> Path | None:
    if not tags:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), sharey=False)
    splits = ["l1/train", "l1/test"]
    titles = ["Train L1", "Test L1"]
    any_pts = False
    for ax, tag, title in zip(axes, splits, titles):
        pts = sorted(tags.get(tag, []), key=lambda x: x[0])
        if pts:
            any_pts = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 36)
    if not any_pts:
        plt.close(fig)
        return None
    axes[0].set_ylabel("L1 (lower better)")
    fig.suptitle("End-of-epoch reconstruction L1 from TensorBoard", y=1.03, fontsize=12)
    path = OUT / "fig_epoch_l1.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_epoch_lpips(tags: dict[str, list[tuple[int, float]]]) -> Path | None:
    if not tags:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), sharey=False)
    splits = ["lpips_convnext_tcga/train", "lpips_convnext_tcga/test"]
    titles = ["Train LPIPS", "Test LPIPS"]
    any_pts = False
    for ax, tag, title in zip(axes, splits, titles):
        pts = sorted(tags.get(tag, []), key=lambda x: x[0])
        if pts:
            any_pts = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 36)
    if not any_pts:
        plt.close(fig)
        return None
    axes[0].set_ylabel("lpips_convnext_tcga (lower better)")
    fig.suptitle("End-of-epoch ConvNeXt-TCGA LPIPS from TensorBoard", y=1.03, fontsize=12)
    path = OUT / "fig_epoch_lpips.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_data_split() -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    labels = ["train", "test\n(held out)"]
    vals = [N_TRAIN, N_TEST]
    colors = [COLOR, "#148f77"]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.55)
    for i, n in enumerate(vals):
        ax.text(i, n + max(vals) * 0.02, f"{n:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Patches")
    ax.set_title("Frozen all-organ split (`dataset/data_frozen`)")
    ax.set_ylim(0, max(vals) * 1.14)
    path = OUT / "fig_data_split.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def dump_metrics_csv(rows: list[dict]) -> Path:
    p = OUT / "epoch_metrics.csv"
    with open(p, "w", encoding="utf-8") as f:
        f.write("epoch,n,psnr,psnr_std,ssim,ssim_std,lpips_convnext_tcga,lpips_std,fid\n")
        for r in rows:
            lp = "" if r.get("lpips") is None else f"{r['lpips']:.6f}"
            lp_std = "" if r.get("lpips_std") is None else f"{r['lpips_std']:.6f}"
            f.write(
                f"{r['epoch']},{r['n']},"
                f"{r['psnr']:.6f},{r['psnr_std']:.6f},{r['ssim']:.6f},{r['ssim_std']:.6f},"
                f"{lp},{lp_std},{r['fid']:.6f}\n"
            )
    return p


def _fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_No `metrics.json` yet — train/eval still running._\n"
    best_fid = min(r["fid"] for r in rows)
    best_psnr = max(r["psnr"] for r in rows)
    best_ssim = max(r["ssim"] for r in rows)
    lp_vals = [r["lpips"] for r in rows if r.get("lpips") is not None]
    best_lpips = min(lp_vals) if lp_vals else None
    lines = [
        "| Ep | N | PSNR | SSIM | LPIPS | FID |",
        "|---:|--:|-----:|-----:|------:|----:|",
    ]
    for r in rows:
        psnr = _fmt(r["psnr"])
        ssim = _fmt(r["ssim"], 3)
        fid = _fmt(r["fid"], 3)
        if r.get("lpips") is None:
            lpips = "—"
        else:
            lpips = _fmt(r["lpips"], 3)
            if best_lpips is not None and r["lpips"] == best_lpips:
                lpips = f"**{lpips}**"
        if r["psnr"] == best_psnr:
            psnr = f"**{psnr}**"
        if r["ssim"] == best_ssim:
            ssim = f"**{ssim}**"
        if r["fid"] == best_fid:
            fid = f"**{fid}**"
        lines.append(f"| {r['epoch']} | {r['n']} | {psnr} | {ssim} | {lpips} | {fid} |")
    return "\n".join(lines) + "\n"


def write_markdown(
    run: str,
    rows: list[dict],
    n_patches: int,
    has_nce: bool,
    has_l1: bool,
    has_lpips_tb: bool,
) -> Path:
    n_done = len(rows)
    planned = list(range(2, 36, 2)) + [35]
    status = (
        f"All **{n_done} / {len(planned)}** planned even-epoch evals have `metrics.json`."
        if n_done == len(planned)
        else f"**{n_done} / {len(planned)}** planned even-epoch evals have `metrics.json` so far "
        f"(epochs {', '.join(str(r['epoch']) for r in rows) or 'none'})."
    )

    interp_lines: list[str] = []
    if rows:
        best_fid = min(rows, key=lambda r: r["fid"])
        last = rows[-1]
        best_psnr = max(rows, key=lambda r: r["psnr"])
        best_ssim = max(rows, key=lambda r: r["ssim"])
        lp_rows = [r for r in rows if r.get("lpips") is not None]
        best_lp = min(lp_rows, key=lambda r: r["lpips"]) if lp_rows else None
        interp_lines.append(
            f"**Best FID is epoch {best_fid['epoch']} ({best_fid['fid']:.3f}).** "
            f"Final checkpoint (e{last['epoch']}) is {last['fid']:.3f}."
        )
        if best_fid["epoch"] != last["epoch"]:
            interp_lines.append(
                f"FID is not monotone through e{last['epoch']}; do not assume later is better."
            )
        disagree = best_psnr["epoch"] != best_fid["epoch"] or best_ssim["epoch"] != best_fid["epoch"]
        if best_lp is not None and best_lp["epoch"] != best_fid["epoch"]:
            disagree = True
        if disagree:
            lp_note = (
                f", LPIPS e{best_lp['epoch']} = {best_lp['lpips']:.3f}" if best_lp is not None else ""
            )
            interp_lines.append(
                f"Pixel / perceptual metrics peak at different checkpoints than FID "
                f"(PSNR e{best_psnr['epoch']} = {best_psnr['psnr']:.3f}, "
                f"SSIM e{best_ssim['epoch']} = {best_ssim['ssim']:.3f}{lp_note}). "
                f"Do not pick the production checkpoint with PSNR/SSIM/LPIPS alone."
            )
    else:
        interp_lines.append(
            "Interpretation will be filled in once the post-train epoch sweep writes `metrics.json`."
        )

    nce_img = (
        '<img src="./results_figures_c13/fig_train_losses.png" alt="Training NCE loss" />\n'
        if has_nce
        else "_Training NCE figure not available yet (`log.json` missing)._\n"
    )
    l1_img = (
        '<img src="./results_figures_c13/fig_epoch_l1.png" alt="End-of-epoch train and test L1" />\n'
        if has_l1
        else "_Epoch-end L1 figure not available yet (TensorBoard scalars missing)._\n"
    )
    lpips_tb_img = (
        '<img src="./results_figures_c13/fig_epoch_lpips.png" alt="End-of-epoch train and test LPIPS" />\n'
        if has_lpips_tb
        else "_Epoch-end LPIPS figure not available yet (TensorBoard scalars missing)._\n"
    )

    md = f"""# Default CISLNet on the frozen all-organ split ({run})

**As of {date.today().isoformat()}.** Patch-level eval on the canonical frozen test set. Figures: [`results_figures_c13/`](./results_figures_c13/).

## Status

{status}

Train job writes numbered `{{epoch}}_net_G.pth` every 2 epochs (and the final epoch 35). Eval is **post-training**: one sweep job scores each saved checkpoint on `dataset/data_frozen/test.txt`.

## Setup

CISLNet `contrastive_pix2pix`, 13-channel IR+VIS, AttUNet G, ConvNeXt NCE, 1024² patches, **35 epochs**, lr 2e-4, batch 10, `save_epoch_freq=2`. Training pool is the default frozen split (no organ-exp allowlist).

| Split | Source | Patches |
|-------|--------|--------:|
| train | all `ir/*.npy` except `test.txt` | {N_TRAIN:,} |
| **test** | `test.txt` (`4466`, `10445`, `4520`, `10442`, `3840`) | **{N_TEST:,}** |

There is **no val split**. Former `val.txt` samples (`10442`, `3840`) were moved into `test.txt` because classic CISLNet never used val for checkpoint selection. `val.txt` is kept present-but-empty so the dataset loader does not hash-split train into a fake val set.

This **is** the canonical `dataset/data_frozen` split, not the organ-exp holdouts in [`results_organ.md`](./results_organ.md). FID and LPIPS both use ConvNeXt-Tiny on TCGA/H&E features (FID = Fréchet distance; LPIPS = Zhang lin0 equal-weight stages — not AlexNet LPIPS, not Inception-FID).

<img src="./results_figures_c13/fig_data_split.png" alt="Frozen split patch counts: 60047 train, 6929 test." />

## Test results

<img src="./results_figures_c13/fig_fid.png" alt="Canonical test FID vs training epoch" />

FID is the metric that moves. Pixel metrics are flatter across checkpoints. LPIPS is the ConvNeXt-TCGA perceptual distance logged in `metrics.json` as `lpips_convnext_tcga_mean`.

<img src="./results_figures_c13/fig_psnr_ssim.png" alt="Canonical test PSNR, SSIM, and LPIPS vs epoch" />

### Full epoch table

{_md_table(rows)}
**Best FID over tested epochs**

"""
    if rows:
        best_fid = min(rows, key=lambda r: r["fid"])
        ranked = sorted(rows, key=lambda r: r["fid"])
        runner = ranked[1] if len(ranked) > 1 else None
        runner_s = f"e{runner['epoch']} ({runner['fid']:.2f})" if runner else "—"
        md += (
            f"| Holdout | Best so far | FID | Runner-up |\n"
            f"|---------|-------------|----:|-----------|\n"
            f"| frozen test | {run} e{best_fid['epoch']} | **{best_fid['fid']:.2f}** | {runner_s} |\n\n"
        )
        lp_rows = [r for r in rows if r.get("lpips") is not None]
        if lp_rows:
            best_lp = min(lp_rows, key=lambda r: r["lpips"])
            ranked_lp = sorted(lp_rows, key=lambda r: r["lpips"])
            runner_lp = ranked_lp[1] if len(ranked_lp) > 1 else None
            runner_lp_s = (
                f"e{runner_lp['epoch']} ({runner_lp['lpips']:.3f})" if runner_lp else "—"
            )
            md += (
                f"**Best LPIPS (ConvNeXt-TCGA) over tested epochs**\n\n"
                f"| Holdout | Best so far | LPIPS | Runner-up |\n"
                f"|---------|-------------|------:|-----------|\n"
                f"| frozen test | {run} e{best_lp['epoch']} | **{best_lp['lpips']:.3f}** | {runner_lp_s} |\n\n"
            )
    else:
        md += "_Pending evals._\n\n"

    md += f"""## Training dynamics

{nce_img}
NCE is the training signal logged every `print_freq` steps. Epoch axis uses train-set size {n_patches:,} patches.

{l1_img}
TensorBoard `l1/train` and `l1/test` are short reconstruction probes at each epoch end (not the FID job). Pixel L1 does **not** automatically pick the best FID checkpoint.

{lpips_tb_img}
TensorBoard `lpips_convnext_tcga/train` and `lpips_convnext_tcga/test` are the same capped epoch-end probes using the frozen ConvNeXt-TCGA Zhang-lin0 metric. Lower is better. These are diagnostics, not a training loss.

## Interpretation

{chr(10).join(interp_lines)}

**Caveats**

- Eval N is {N_TEST:,} patches (`test.txt` only). FID and LPIPS here are ConvNeXt/TCGA, not comparable to Inception-FID or official AlexNet LPIPS.
- Numbered checkpoints include G, D, and F_q (`{{epoch}}_net_{{G,D,F_q}}.pth`); eval loads G only.
- No WSI-level scores in this writeup.

## Reproduce

```bash
# queue 35-epoch train + post-train even-epoch eval (already submitted once)
./jobs/cislnet/submit_c13_epoch_eval.sh

# regenerate figures + this file after metrics exist
python scripts/plot_c13_epoch_eval.py --run {run}
```
"""
    MD_PATH.write_text(md, encoding="utf-8")
    return MD_PATH


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--train-patches", type=int, default=DEFAULT_TRAIN_PATCHES)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--md-path", type=Path, default=MD_PATH)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    global OUT, MD_PATH
    OUT = Path(args.out_dir)
    MD_PATH = Path(args.md_path)
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    rows = load_eval_rows(args.run)
    print(f"{len(rows)} eval metric rows for {args.run}")
    dump_metrics_csv(rows)
    fig_data_split()
    fig_fid(rows)
    fig_psnr_ssim(rows)
    nce_path = fig_train_losses(args.run, int(args.train_patches))
    cached = CKPT / args.run / "tb_l1.json"
    tags: dict[str, list[tuple[int, float]]] = {}
    if cached.is_file():
        tags = json.loads(cached.read_text())
        print(f"reusing cached TB L1 from {cached}")
    else:
        try:
            tags = extract_tb_l1(args.run)
            if tags:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(tags, indent=2))
        except Exception as e:
            print(f"TB extract failed: {e}")
            tags = {}
    l1_path = fig_epoch_l1(tags)
    lpips_tb_path = fig_epoch_lpips(tags)
    (OUT / "tb_l1.json").write_text(json.dumps(tags, indent=2))
    md = write_markdown(
        args.run,
        rows,
        int(args.train_patches),
        has_nce=nce_path is not None,
        has_l1=l1_path is not None,
        has_lpips_tb=lpips_tb_path is not None,
    )
    print("wrote", OUT)
    print("wrote", md)


if __name__ == "__main__":
    main()
