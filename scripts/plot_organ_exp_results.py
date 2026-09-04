#!/usr/bin/env python3
"""Build organ-exp figures from holdout metrics.json, log.json, and TB scalars.

X-axis is training iters = samples_seen / batch_size (the optimizer step
counter). 30 all-sample passes = 159480 iters (~1.59e6 samples).
"""
from __future__ import annotations

import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path("/projects/illinois/eng/bioe/rxb/akshat/VSlite")
sys.path.insert(0, str(ROOT / "scripts"))
from organ_exp_schedule import (  # noqa: E402
    BASE_EPOCHS,
    BATCH_SIZE,
    TEST_EQ_EPOCHS,
    equiv_epoch_from_local,
    samples_from_local,
    samples_per_epoch,
)

CKPT = ROOT / "results" / "cislnet"
TEST = CKPT / "test"
UNIFIED = ROOT / "results" / "unified" / "cislnet"
OUT = ROOT / "results_figures"
SCHED_JSON = ROOT / "dataset" / "organ_exp" / "iter_schedule.json"

RUNS = {
    "oa_all": "c13_oa_all",
    "oa_top3": "c13_oa_top3",
    "os_prostate": "c13_os_prostate",
    "os_kidney": "c13_os_kidney",
    "os_uterus": "c13_os_uterus",
}
TRAIN_PATCHES = {
    "c13_oa_all": 53164,
    "c13_oa_top3": 26944,
    "c13_os_prostate": 17183,
    "c13_os_kidney": 5653,
    "c13_os_uterus": 3840,
}
N_ALL = TRAIN_PATCHES["c13_oa_all"]
C_ALL = "#1b4f72"
C_TOP3 = "#148f77"
C_OS = "#c0392b"
C_OS_P = "#922b21"
C_OS_K = "#c0392b"
C_OS_U = "#6c3483"
COLORS = {
    "c13_oa_all": C_ALL,
    "c13_oa_top3": C_TOP3,
    "c13_os_prostate": C_OS_P,
    "c13_os_kidney": C_OS_K,
    "c13_os_uterus": C_OS_U,
}
LABEL_ALL = "trained on all samples"
LABEL_TOP3 = "trained on prostate, kidney, uterus"
LABEL_OS = "organ-specific"
LABELS = {
    "c13_oa_all": LABEL_ALL,
    "c13_oa_top3": LABEL_TOP3,
    "c13_os_prostate": "prostate only",
    "c13_os_kidney": "kidney only",
    "c13_os_uterus": "uterus only",
}
HOLDOUT_N = {"prostate": 2379, "kidney": 807, "uterus": 538}
XLABEL = "iters"
EQ_TICKS = list(TEST_EQ_EPOCHS)
# Optimizer steps logged during training (total_steps_so_far / batch_size).
ITERS_PER_PASS = samples_per_epoch(N_ALL, BATCH_SIZE) // BATCH_SIZE  # 5316
BUDGET_ITERS = BASE_EPOCHS * ITERS_PER_PASS  # 159480
ITER_TICKS = [e * ITERS_PER_PASS for e in EQ_TICKS]


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


def _load_schedule() -> dict:
    if SCHED_JSON.is_file():
        return json.loads(SCHED_JSON.read_text())
    return {}


def _row_xy(model: str, local_epoch: int) -> tuple[float, int]:
    n = TRAIN_PATCHES[model]
    return (
        equiv_epoch_from_local(local_epoch, n, N_ALL, BATCH_SIZE),
        samples_from_local(local_epoch, n, BATCH_SIZE),
    )


def _has_generator_ckpt(model: str, local_epoch: int) -> bool:
    """Skip holdout rows whose numbered G checkpoint was never written.

    Missing ckpts still produce metrics.json when the test job falls back to
    randomly initialized weights (PSNR~9, FID>100) — drop those points.
    """
    return (CKPT / model / f"{local_epoch}_net_G.pth").is_file()


def load_holdout_rows() -> list[dict]:
    rows = []
    skipped_no_ckpt = 0
    for p in sorted(TEST.glob("c13_*/patch_eval_test/metrics.json")):
        name = p.parent.parent.name
        if "__hold_" not in name:
            continue
        m = json.loads(p.read_text())
        model, rest = name.split("__hold_", 1)
        if model not in TRAIN_PATCHES:
            continue
        organ, epoch_s = rest.rsplit("_", 1)
        if not epoch_s.isdigit():
            continue
        local_epoch = int(epoch_s)
        if not _has_generator_ckpt(model, local_epoch):
            skipped_no_ckpt += 1
            continue
        eq, samples = _row_xy(model, local_epoch)
        lp = m.get("lpips_convnext_tcga_mean")
        lp_std = m.get("lpips_convnext_tcga_std")
        rows.append(
            {
                "model": model,
                "organ": organ,
                "epoch": local_epoch,
                "eq_epoch": eq,
                "samples": samples,
                "iters": samples / float(BATCH_SIZE),
                "n": int(m["n"]),
                "psnr": float(m["psnr_mean"]),
                "psnr_std": float(m["psnr_std"]),
                "ssim": float(m["ssim_mean"]),
                "ssim_std": float(m["ssim_std"]),
                "fid": float(m["fid"]),
                "lpips": None if lp is None else float(lp),
                "lpips_std": None if lp_std is None else float(lp_std),
                "has_lpips": lp is not None,
            }
        )
    if skipped_no_ckpt:
        print(f"skipped {skipped_no_ckpt} holdout rows with no {{epoch}}_net_G.pth")
    return rows


def on_eq_grid(rows: list[dict], tol: float = 0.75) -> list[dict]:
    """Keep checkpoints that sit on the matched-iter eval grid."""
    out = []
    for r in rows:
        if any(abs(r["eq_epoch"] - t) <= tol for t in EQ_TICKS):
            out.append(r)
    return out


def smooth(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(y) < win:
        return y
    k = np.ones(win, dtype=np.float64) / float(win)
    pad = win // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, k, mode="valid")[: len(y)]


def _light_smooth(y: np.ndarray) -> np.ndarray:
    """Short moving average for NCE — light smoothing, not a flat trend line."""
    n = len(y)
    if n < 12:
        return y
    win = max(15, min(51, n // 150))
    if win % 2 == 0:
        win += 1
    return smooth(y, win)


def _spe_run(run: str) -> int:
    return samples_per_epoch(TRAIN_PATCHES[run], BATCH_SIZE)


def _samples_to_iters(samples: np.ndarray) -> np.ndarray:
    """Logged sample count → optimizer iters (one step per batch)."""
    return np.asarray(samples, dtype=np.float64) / float(BATCH_SIZE)


def _last_wins_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep one y per integer step. Later file order wins (resume hops overwrite)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return x, y
    by_x: dict[int, float] = {}
    for xi, yi in zip(x, y):
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        by_x[int(round(xi))] = float(yi)
    keys = np.array(sorted(by_x), dtype=np.float64)
    vals = np.array([by_x[int(k)] for k in keys], dtype=np.float64)
    return keys, vals


def _dedupe_tb_points(run: str, pts: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """One reconstruction point per local pass; sample-count logs overwrite epoch-index logs."""
    if not pts:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    spe = float(_spe_run(run))
    buckets: dict[int, tuple[float, float]] = {}
    for step, val in pts:
        if not np.isfinite(val):
            continue
        samples = float(step) * spe if step < 1000 else float(step)
        local_ep = max(1, int(round(samples / spe)))
        buckets[local_ep] = (samples, float(val))
    items = sorted(buckets.items(), key=lambda kv: kv[1][0])
    samples = np.array([v[0] for _, v in items], dtype=np.float64)
    vals = np.array([v[1] for _, v in items], dtype=np.float64)
    return _samples_to_iters(samples), vals


def load_log_json(run: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    p = CKPT / run / "log.json"
    data = json.loads(p.read_text())
    out = {}
    for key, pts in data.get("scalars", {}).items():
        step = np.array([a[0] for a in pts], dtype=np.float64)
        val = np.array([a[1] for a in pts], dtype=np.float64)
        # log.json NCE is always a sample count (print_freq=100), never a local index.
        x, y = _last_wins_xy(step, val)
        out[key] = (_samples_to_iters(x), y)
    return out


def _read_tfrecords_skip_large(path: Path, max_len: int = 8192):
    """Yield small TFRecord payloads (skip image-sized records)."""
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


def extract_tb_scalars(run: str) -> dict[str, list[tuple[int, float]]]:
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


def _protocol_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=C_ALL, marker="o", linestyle="-", linewidth=1.6, markersize=6, label=LABEL_ALL),
        Line2D([0], [0], color=C_TOP3, marker="s", linestyle="--", linewidth=1.6, markersize=6, label=LABEL_TOP3),
        Line2D([0], [0], color=C_OS, marker="^", linestyle="-", linewidth=1.6, markersize=6, label=LABEL_OS),
    ]


def _series_for_organ(organ: str) -> list[tuple[str, str, str]]:
    return [
        ("c13_oa_all", "o-", C_ALL),
        ("c13_oa_top3", "s--", C_TOP3),
        (f"c13_os_{organ}", "^-", C_OS),
    ]


def _apply_iter_axis(ax, *, include_zero: bool = False) -> None:
    ax.set_xlabel(XLABEL)
    ticks = [0, *ITER_TICKS] if include_zero else ITER_TICKS
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: "0" if abs(v) < 1 else f"{v / 1000:.0f}k")
    )
    ax.tick_params(axis="x", labelsize=8)
    lo = -0.02 * BUDGET_ITERS if include_zero else ITER_TICKS[0] - 0.08 * BUDGET_ITERS
    ax.set_xlim(lo, BUDGET_ITERS + 0.04 * BUDGET_ITERS)


def _sci_y(ax) -> None:
    """LPIPS ticks as mantissa; scale lives in the y-label as ×10^{-3}."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v * 1e3:.2f}"))


def fig_holdout_fid(rows: list[dict]) -> Path:
    organs = ["prostate", "kidney", "uterus"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=False)
    for ax, organ in zip(axes, organs):
        for model, fmt, col in _series_for_organ(organ):
            pts = sorted(
                [r for r in rows if r["model"] == model and r["organ"] == organ],
                key=lambda r: r["iters"],
            )
            if not pts:
                continue
            ax.plot(
                [r["iters"] for r in pts],
                [r["fid"] for r in pts],
                fmt,
                color=col,
                markersize=6,
                linewidth=1.6,
            )
        n = HOLDOUT_N[organ]
        ax.set_title(f"{organ}  (n={n})")
        _apply_iter_axis(ax)
    axes[0].set_ylabel("FID (lower better)")
    fig.legend(
        handles=_protocol_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
    )
    fig.suptitle("Holdout FID vs iters", y=1.18, fontsize=12)
    path = OUT / "fig_holdout_fid.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_holdout_lpips(rows: list[dict]) -> Path:
    pts_lp = [r for r in rows if r["has_lpips"]]
    organs = ["prostate", "kidney", "uterus"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=False)
    for ax, organ in zip(axes, organs):
        for model, fmt, col in _series_for_organ(organ):
            pts = sorted(
                [r for r in pts_lp if r["model"] == model and r["organ"] == organ],
                key=lambda r: r["iters"],
            )
            if not pts:
                continue
            ax.plot(
                [r["iters"] for r in pts],
                [r["lpips"] for r in pts],
                fmt,
                color=col,
                markersize=6,
                linewidth=1.6,
            )
        ax.set_title(f"{organ}  (n={HOLDOUT_N[organ]})")
        _apply_iter_axis(ax)
        _sci_y(ax)
        if not pts_lp:
            ax.text(0.5, 0.5, "queued\n(LPIPS eval)", ha="center", va="center", transform=ax.transAxes, color="#888")
    axes[0].set_ylabel(r"LPIPS ($\times 10^{-3}$, lower better)")
    fig.legend(
        handles=_protocol_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
    )
    fig.suptitle("Holdout LPIPS vs iters", y=1.18, fontsize=12)
    path = OUT / "fig_holdout_lpips.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_holdout_psnr_ssim(rows: list[dict]) -> Path:
    organs = ["prostate", "kidney", "uterus"]
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.8), sharex=True)
    for j, organ in enumerate(organs):
        for model, mk, col in _series_for_organ(organ):
            pts = sorted(
                [r for r in rows if r["model"] == model and r["organ"] == organ],
                key=lambda r: r["iters"],
            )
            if not pts:
                continue
            ep = np.array([r["iters"] for r in pts])
            axes[0, j].errorbar(
                ep,
                [r["psnr"] for r in pts],
                yerr=[r["psnr_std"] / np.sqrt(r["n"]) for r in pts],
                fmt=mk,
                color=col,
                markersize=5,
                linewidth=1.5,
                capsize=2,
            )
            axes[1, j].errorbar(
                ep,
                [r["ssim"] for r in pts],
                yerr=[r["ssim_std"] / np.sqrt(r["n"]) for r in pts],
                fmt=mk,
                color=col,
                markersize=5,
                linewidth=1.5,
                capsize=2,
            )
        axes[0, j].set_title(f"{organ}  (n={HOLDOUT_N[organ]})")
        _apply_iter_axis(axes[1, j])
    axes[0, 0].set_ylabel("PSNR (dB, higher better)")
    axes[1, 0].set_ylabel("SSIM (higher better)")
    fig.legend(
        handles=_protocol_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.suptitle("Holdout PSNR / SSIM vs iters (error bars = SEM)", y=1.10, fontsize=12)
    path = OUT / "fig_holdout_psnr_ssim.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_train_losses() -> Path | None:
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    any_series = False
    x_hi = float(BUDGET_ITERS) * 1.02
    for folder in RUNS.values():
        p = CKPT / folder / "log.json"
        if not p.is_file():
            continue
        series = load_log_json(folder)
        if "errors/NCE" not in series:
            continue
        it, nce = series["errors/NCE"]
        mask = (it >= 0.0) & (it <= x_hi)
        it, nce = it[mask], nce[mask]
        if it.size < 8:
            continue
        ax.plot(it, _light_smooth(nce), color=COLORS[folder], label=LABELS[folder], linewidth=1.2, zorder=2)
        any_series = True
    if not any_series:
        plt.close(fig)
        return None
    ax.set_title("NCE")
    _apply_iter_axis(ax, include_zero=True)
    ax.set_ylabel("NCE")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle("Training NCE vs iters", y=1.03, fontsize=12)
    path = OUT / "fig_train_losses.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_epoch_l1(l1_by_run: dict[str, dict[str, list[tuple[int, float]]]]) -> Path | None:
    if not any(l1_by_run.values()):
        return None
    has_lpips = any(
        tags.get("lpips_convnext_tcga/train") or tags.get("lpips_convnext_tcga/test")
        for tags in l1_by_run.values()
    )
    if has_lpips:
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), sharex=True)
        panels = [
            (axes[0, 0], "l1/train", "Train L1", "L1 (lower better)"),
            (axes[0, 1], "l1/test", "Holdout L1", "L1 (lower better)"),
            (axes[1, 0], "lpips_convnext_tcga/train", "Train LPIPS", r"LPIPS ($\times 10^{-3}$, lower better)"),
            (axes[1, 1], "lpips_convnext_tcga/test", "Holdout LPIPS", r"LPIPS ($\times 10^{-3}$, lower better)"),
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), sharey=False)
        panels = [
            (axes[0], "l1/train", "Train L1", "L1 (lower better)"),
            (axes[1], "l1/test", "Holdout L1", "L1 (lower better)"),
        ]
    for ax, tag, title, ylab in panels:
        is_lpips = tag.startswith("lpips_")
        for folder, tags in l1_by_run.items():
            it, vals = _dedupe_tb_points(folder, tags.get(tag, []))
            if it.size < 3:
                continue
            ax.plot(
                it,
                smooth(vals, 3) if it.size >= 8 else vals,
                color=COLORS[folder],
                label=LABELS[folder],
                linewidth=1.3,
            )
        ax.set_title(title)
        _apply_iter_axis(ax, include_zero=True)
        ax.set_ylabel(ylab)
        if is_lpips:
            _sci_y(ax)
    panels[0][0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle("Reconstruction vs iters", y=1.03, fontsize=12)
    path = OUT / "fig_epoch_l1.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _bar_run_labels() -> list[str]:
    """Category labels for matched-iter budget bars (RUNS order)."""
    return [
        "all\nsamples",
        "prostate,\nkidney,\nuterus",
        "prostate\nonly",
        "kidney\nonly",
        "uterus\nonly",
    ]


def fig_data_split() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})

    train_labels = [
        "prostate\nonly",
        "kidney\nonly",
        "uterus\nonly",
        "prostate,\nkidney,\nuterus",
        "all\nsamples",
    ]
    train = [17183, 5653, 3840, 26944, 53164]
    train_colors = [C_OS, C_OS, C_OS, C_TOP3, C_ALL]
    x0 = np.arange(len(train_labels))
    axes[0].bar(x0, train, color=train_colors, width=0.7)
    axes[0].set_xticks(x0)
    axes[0].set_xticklabels(train_labels, fontsize=8)
    axes[0].set_ylabel("Patches")
    axes[0].set_title("Training pool")
    axes[0].set_xlabel("")

    hold_labels = ["prostate", "kidney", "uterus"]
    hold = [HOLDOUT_N[o] for o in hold_labels]
    x1 = np.arange(len(hold_labels))
    axes[1].bar(x1, hold, color=C_OS, width=0.55)
    for i, n in enumerate(hold):
        axes[1].text(i, n + 40, str(n), ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x1)
    axes[1].set_xticklabels(hold_labels)
    axes[1].set_title("Holdout")
    axes[1].set_ylim(0, max(hold) * 1.18)
    axes[1].set_xlabel("")

    fig.tight_layout()
    path = OUT / "fig_data_split.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_iter_budget(sched: dict) -> Path:
    """Train length vs the shared sample budget."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    order = list(RUNS.values())
    labels = _bar_run_labels()
    colors = [COLORS[r] for r in order]
    runs = (sched.get("runs") or {}) if sched else {}

    local_eps = []
    samples = []
    for r in order:
        info = runs.get(r, {})
        local_eps.append(int(info.get("local_epochs") or round(BASE_EPOCHS * N_ALL / TRAIN_PATCHES[r])))
        samples.append(int(info.get("scheduled_samples") or samples_from_local(local_eps[-1], TRAIN_PATCHES[r])))

    x = np.arange(len(order))
    axes[0].bar(x, local_eps, color=colors, width=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Local passes")
    axes[0].set_title("Passes over each pool to match the iter budget")
    axes[0].set_xlabel("")
    for i, v in enumerate(local_eps):
        axes[0].text(i, v + max(local_eps) * 0.02, str(v), ha="center", va="bottom", fontsize=8)

    y_iters = np.array(samples) / float(BATCH_SIZE)
    axes[1].bar(x, y_iters, color=colors, width=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("iters")
    axes[1].set_title("Shared iteration budget")
    axes[1].set_xlabel("")
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: "0" if abs(v) < 1 else f"{v / 1000:.0f}k")
    )
    tgt_iters = (sched.get("target_samples") if sched else None) or BUDGET_ITERS * BATCH_SIZE
    y_tgt = float(tgt_iters) / float(BATCH_SIZE)
    axes[1].axhline(y_tgt, color="#444", linestyle=":", linewidth=1.0)
    axes[1].annotate(
        f"{y_tgt / 1000:.0f}k",
        xy=(len(order) - 0.55, y_tgt),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        color="#444",
    )

    fig.suptitle("Matched-iteration protocol", y=1.02, fontsize=11)
    fig.tight_layout()
    path = OUT / "fig_iter_budget.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def dump_metrics_csv(rows: list[dict]) -> Path:
    p = OUT / "holdout_metrics.csv"
    with open(p, "w", encoding="utf-8") as f:
        f.write(
            "model,organ,local_epoch,eq_epoch,samples,n,psnr,psnr_std,ssim,ssim_std,fid,"
            "lpips_convnext_tcga_mean,lpips_convnext_tcga_std\n"
        )
        for r in sorted(rows, key=lambda x: (x["model"], x["organ"], x["epoch"])):
            lp = "" if r["lpips"] is None else f"{r['lpips']:.6f}"
            lp_s = "" if r["lpips_std"] is None else f"{r['lpips_std']:.6f}"
            f.write(
                f"{r['model']},{r['organ']},{r['epoch']},{r['eq_epoch']:.4f},{r['samples']},{r['n']},"
                f"{r['psnr']:.6f},{r['psnr_std']:.6f},{r['ssim']:.6f},{r['ssim_std']:.6f},{r['fid']:.6f},"
                f"{lp},{lp_s}\n"
            )
    return p


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    sched = _load_schedule()
    rows = load_holdout_rows()
    grid = on_eq_grid(rows)
    print(f"{len(rows)} holdout metric rows ({len(grid)} on matched-iter grid; {sum(1 for r in rows if r['has_lpips'])} with LPIPS)")
    dump_metrics_csv(rows)
    fig_data_split()
    fig_iter_budget(sched)
    plot_rows = grid if grid else rows
    fig_holdout_fid(plot_rows)
    fig_holdout_psnr_ssim(plot_rows)
    fig_holdout_lpips(plot_rows)
    fig_train_losses()

    cached = ROOT / "results" / "cislnet" / "organ_exp" / "figures" / "tb_l1.json"
    l1_by_run: dict = {}
    if not cached.is_file() and (OUT / "tb_l1.json").is_file():
        cached = OUT / "tb_l1.json"
    if cached.is_file():
        l1_by_run = json.loads(cached.read_text())
        print(f"reusing cached TB L1 from {cached}")
    else:
        for folder in RUNS.values():
            print(f"Extracting TB scalars for {folder}", flush=True)
            try:
                l1_by_run[folder] = extract_tb_scalars(folder)
            except Exception as e:
                print(f"  TB extract failed for {folder}: {e}")
                l1_by_run[folder] = {}
    fig_epoch_l1(l1_by_run)
    (OUT / "tb_l1.json").write_text(json.dumps(l1_by_run, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
