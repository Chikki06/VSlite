#!/usr/bin/env python3
"""Figure for the c13 noattn + objective (loss) ablation family: LPIPS bars.

Quality: results/cislnet/test/<run>/patch_eval_test/metrics.json (frozen test,
n=6929). Writes results_figures/fig_obj_lpips.png + objective_metrics.csv.
Latency (c13_noattn only — the sole architecture change) lands via
jobs/cislnet/experiments/latency_obj.sh and goes into results.md as a table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/projects/illinois/eng/bioe/rxb/akshat/VSlite")
TEST = ROOT / "results" / "cislnet" / "test"
TRACK = ROOT / "results" / "cislnet" / "experiments" / "tracking.csv"
LAT_SCRATCH = ROOT / "results" / "cislnet" / "experiments" / "latency_obj"
OUT = ROOT / "results_figures"

# Column order everywhere: reference first, then the ablation family.
RUNS = ["c13", "c13_noattn", "c13_gan", "c13_ganl2_01", "c13_ganl2_10", "c13_nce", "c13_l2"]
LABELS = {
    "c13": "c13 (ref)",
    "c13_noattn": "no attn",
    "c13_gan": "GAN only",
    "c13_ganl2_01": "GAN+L2 @0.1",
    "c13_ganl2_10": "GAN+L2 @10",
    "c13_nce": "NCE only @10",
    "c13_l2": "L2 only @1",
}
# Reference dark; attn ablation red; objective family warm/cool by loss.
COLORS = {
    "c13": "#264653",
    "c13_noattn": "#9d0208",
    "c13_gan": "#e76f51",
    "c13_ganl2_01": "#f4a261",
    "c13_ganl2_10": "#e9c46a",
    "c13_nce": "#6d597a",
    "c13_l2": "#2a9d8f",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_quality() -> dict[str, dict]:
    rows = {}
    for run in RUNS:
        m = json.loads((TEST / run / "patch_eval_test" / "metrics.json").read_text())
        rows[run] = {
            "run": run,
            "n": int(m["n"]),
            "lpips": float(m["lpips_convnext_tcga_mean"]),
            "psnr": float(m["psnr_mean"]),
            "ssim": float(m["ssim_mean"]),
            "fid": float(m["fid"]),
        }
    return rows


def load_latency(quality: dict[str, dict]) -> dict[str, float]:
    """Ablation runs: latest tracking.csv row with group=noattn_obj.
    c13: same-node scratch leaf (fallback: vs13 tracking row)."""
    lat: dict[str, float] = {}
    if TRACK.is_file():
        with TRACK.open() as f:
            for r in csv.DictReader(f):
                if r["group"] == "noattn_obj" and r["run"] in RUNS and r["latency_ms_mean"]:
                    lat[r["run"]] = float(r["latency_ms_mean"])
    scratch = LAT_SCRATCH / "test" / "c13" / "patch_eval_test" / "metrics.json"
    if "c13" not in lat and scratch.is_file():
        m = json.loads(scratch.read_text())
        if m.get("latency_ms_mean"):
            lat["c13"] = float(m["latency_ms_mean"])
    if "c13" not in lat:  # legacy fallback: vs13 tracking row
        with TRACK.open() as f:
            for r in csv.DictReader(f):
                if r["run"] == "vs13" and r["latency_ms_mean"]:
                    lat["c13"] = float(r["latency_ms_mean"])
    base = lat["c13"]
    for run in RUNS:
        quality[run]["latency_ms"] = lat.get(run)
        quality[run]["speedup"] = base / lat[run] if run in lat else None
    return lat


def fig_lpips(quality: dict[str, dict]) -> None:
    ys = [quality[r]["lpips"] for r in RUNS]
    base = quality["c13"]["lpips"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(
        np.arange(len(RUNS)),
        ys,
        color=[COLORS[r] for r in RUNS],
        width=0.66,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(np.arange(len(RUNS)))
    ax.set_xticklabels(
        [LABELS[r] for r in RUNS],
        fontsize=9,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("LPIPS ↓ (ConvNeXt-TCGA)")
    ax.set_title("Exp 4 — no-attn / objective ablations (frozen test, n=6929)")
    ax.set_ylim(min(ys) * 0.985, max(ys) * 1.10)
    ax.axhline(base, color=COLORS["c13"], ls=":", lw=0.9, alpha=0.55)
    for b, v in zip(bars, ys):
        d = (v - base) / base * 100.0
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.5f}\n({d:+.1f}%)",
            ha="center",
            va="bottom",
            fontsize=7.5,
            linespacing=1.2,
            fontweight="medium",
        )
    fig.savefig(OUT / "fig_obj_lpips.png")
    plt.close(fig)


def write_csv(quality: dict[str, dict]) -> None:
    with (OUT / "objective_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run", "label", "n", "lpips", "psnr", "ssim", "fid",
                "latency_ms", "speedup_vs_c13",
            ],
        )
        w.writeheader()
        for run in RUNS:
            q = quality[run]
            w.writerow(
                {
                    "run": run,
                    "label": q.get("label", LABELS[run]),
                    "n": q["n"],
                    "lpips": round(q["lpips"], 6),
                    "psnr": round(q["psnr"], 3),
                    "ssim": round(q["ssim"], 4),
                    "fid": round(q["fid"], 3),
                    "latency_ms": round(q["latency_ms"], 2) if q.get("latency_ms") else "",
                    "speedup_vs_c13": round(q["speedup"], 3) if q.get("speedup") else "",
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    quality = load_quality()
    load_latency(quality)
    fig_lpips(quality)
    write_csv(quality)
    print(f"Wrote fig_obj_lpips.png + objective_metrics.csv under {OUT}")
    for run in RUNS:
        q = quality[run]
        lat = f"{q['latency_ms']:6.2f}ms {q['speedup']:4.2f}x" if q.get("latency_ms") else "   —      "
        print(
            f"  {run:14s} LPIPS={q['lpips']:.6f} FID={q['fid']:7.2f} "
            f"PSNR={q['psnr']:6.2f} SSIM={q['ssim']:.4f} {lat}"
        )


if __name__ == "__main__":
    main()
