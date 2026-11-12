#!/usr/bin/env python3
"""Figures for Vis+ channel sweep + architecture latency/LPIPS ablations.

Writes under results_figures_inference/ (sibling of results_inference.md).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path("/projects/illinois/eng/bioe/rxb/akshat/VSlite")
TEST = ROOT / "results" / "cislnet" / "test"
TRACK = ROOT / "results" / "cislnet" / "experiments" / "tracking.csv"
OUT = ROOT / "results_figures_inference"

# Distinct categorical palette (A=attention, B=structure, D=stack, IO=fp16).
# C1 omitted from figures (LPIPS/FID outliers crush the scale).
# Legacy run tags b4_eca / b5_simam are group A (A5/A6).
COLORS = {
    "vs13": "#264653",
    "a1_coordatt_c13": "#2a9d8f",
    "a2_triplet_c13": "#1d7874",
    "a3_scse_c13": "#52b788",
    "a4_gcblock_c13": "#40916c",
    "b4_eca_c13": "#b56576",  # A5 ECA
    "b5_simam_c13": "#6d6875",  # A6 SimAM
    "b1_half_c13": "#e9c46a",
    "b2_scconv_c13": "#f4a261",
    "b3_convnext_c13": "#e76f51",
    "b6_depthwise_c13": "#bc6c25",
    "b7_pixelshuffle_c13": "#9c6644",
    "fp16_c13": "#6d597a",
    # D stacks — darker / cooler so they read as combos, not solo A/B.
    "d1_resnet50_triplet_c13_half": "#1b4332",
    "d1_resnet50_triplet_c13_ps": "#2d6a4f",
    "d1_resnet50_eca_c13_half": "#7f1d1d",
    "d1_resnet50_eca_c13_ps": "#9f1239",
    # D followups (manual 2026-10-10 batch: GCBlock stacks + depthwise stacks).
    "d1_resnet50_gcblock_c13_half": "#0b6e4f",
    "d1_resnet50_gcblock_c13_ps": "#5e548e",
    "d1_resnet50_eca_c13_dw": "#9d0208",
    "d1_resnet50_gcblock_c13_dw": "#0077b6",
    "d1_resnet50_triplet_c13_dw": "#3a5a40",
}
# D combos use diamond markers on the Pareto scatter.
D_MARKER = "D"
VISPLUS_COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]


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


def _lpips_from_metrics(name: str) -> dict:
    p = TEST / name / "patch_eval_test" / "metrics.json"
    m = json.loads(p.read_text())
    return {
        "run": name,
        "n": int(m["n"]),
        "lpips": float(m["lpips_convnext_tcga_mean"]),
        "psnr": float(m["psnr_mean"]),
        "ssim": float(m["ssim_mean"]),
        "fid": float(m["fid"]),
    }


def load_visplus_rows() -> list[dict]:
    """c13 baseline = dedicated c13 leaf when present (same frozen test as vp*)."""
    rows = []
    c13_leaf = TEST / "c13" / "patch_eval_test" / "metrics.json"
    if c13_leaf.is_file():
        r = _lpips_from_metrics("c13")
        r["label"] = "c13 (13ch)"
        r["ir_n"] = 10
    else:
        r = _lpips_from_metrics("vs13")
        r["label"] = "c13 / vs13 (13ch)"
        r["ir_n"] = 10
    rows.append(r)
    for name, ir_n, label in [
        ("vp1", 1, "Vis+1 (4ch)"),
        ("vp3", 3, "Vis+3 (6ch)"),
        ("vp5", 5, "Vis+5 (8ch)"),
    ]:
        r = _lpips_from_metrics(name)
        r["label"] = label
        r["ir_n"] = ir_n
        rows.append(r)
    return rows


def load_latency_rows() -> list[dict]:
    """Architecture ablations from tracking.csv (has latency). Exclude Vis+/C1."""
    keep = {
        "vs13",
        "a1_coordatt_c13",
        "a2_triplet_c13",
        "a3_scse_c13",
        "a4_gcblock_c13",
        "b1_half_c13",
        "b2_scconv_c13",
        "b3_convnext_c13",
        "b4_eca_c13",
        "b5_simam_c13",
        "b6_depthwise_c13",
        "b7_pixelshuffle_c13",
        "fp16_c13",
        "d1_resnet50_triplet_c13_half",
        "d1_resnet50_triplet_c13_ps",
        "d1_resnet50_eca_c13_half",
        "d1_resnet50_eca_c13_ps",
        "d1_resnet50_gcblock_c13_half",
        "d1_resnet50_gcblock_c13_ps",
        "d1_resnet50_eca_c13_dw",
        "d1_resnet50_gcblock_c13_dw",
        "d1_resnet50_triplet_c13_dw",
    }
    labels = {
        "vs13": "c13 / CBAM",
        "a1_coordatt_c13": "A1 CoordAtt",
        "a2_triplet_c13": "A2 Triplet",
        "a3_scse_c13": "A3 SCSE",
        "a4_gcblock_c13": "A4 GCBlock",
        "b1_half_c13": "B1 half",
        "b2_scconv_c13": "B2 SCConv",
        "b3_convnext_c13": "B3 ConvNeXt",
        "b4_eca_c13": "A5 ECA",
        "b5_simam_c13": "A6 SimAM",
        "b6_depthwise_c13": "B6 depthwise",
        "b7_pixelshuffle_c13": "B7 PixelShuffle",
        "fp16_c13": "fp16 IR",
        "d1_resnet50_triplet_c13_half": "D Triplet+half",
        "d1_resnet50_triplet_c13_ps": "D Triplet+PS",
        "d1_resnet50_eca_c13_half": "D ECA+half",
        "d1_resnet50_eca_c13_ps": "D ECA+PS",
        "d1_resnet50_gcblock_c13_half": "D GCBlock+half",
        "d1_resnet50_gcblock_c13_ps": "D GCBlock+PS",
        "d1_resnet50_eca_c13_dw": "D ECA+DW",
        "d1_resnet50_gcblock_c13_dw": "D GCBlock+DW",
        "d1_resnet50_triplet_c13_dw": "D Triplet+DW",
    }
    # Last row wins when hop1+hop2 both appended eval.
    by_run: dict[str, dict] = {}
    with TRACK.open() as f:
        for r in csv.DictReader(f):
            if r["run"] not in keep:
                continue
            by_run[r["run"]] = {
                "run": r["run"],
                "label": labels[r["run"]],
                "group": r["group"],
                "attention": r["attention"],
                "lpips": float(r["lpips_convnext_tcga"]),
                "latency_ms": float(r["latency_ms_mean"]),
                "latency_std": float(r["latency_ms_std"]),
                "fid": float(r["fid"]),
                "notes": r["notes"],
            }
    order = [
        "vs13",
        "a1_coordatt_c13",
        "a2_triplet_c13",
        "a3_scse_c13",
        "a4_gcblock_c13",
        "b4_eca_c13",
        "b5_simam_c13",
        "b1_half_c13",
        "b2_scconv_c13",
        "b3_convnext_c13",
        "b6_depthwise_c13",
        "b7_pixelshuffle_c13",
        "fp16_c13",
        "d1_resnet50_triplet_c13_half",
        "d1_resnet50_triplet_c13_ps",
        "d1_resnet50_eca_c13_half",
        "d1_resnet50_eca_c13_ps",
        "d1_resnet50_gcblock_c13_half",
        "d1_resnet50_gcblock_c13_ps",
        "d1_resnet50_eca_c13_dw",
        "d1_resnet50_gcblock_c13_dw",
        "d1_resnet50_triplet_c13_dw",
    ]
    out = [by_run[k] for k in order if k in by_run]
    base = next(r["latency_ms"] for r in out if r["run"] == "vs13")
    for r in out:
        r["speedup"] = base / r["latency_ms"]
    return out


def fig_visplus_lpips(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    vals = [r["lpips"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        vals,
        color=VISPLUS_COLORS[: len(labels)],
        width=0.62,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("LPIPS ↓ (ConvNeXt-TCGA)")
    ax.set_title("Exp 1 — Vis+ IR count vs full c13 (frozen test, n=6929)")
    ax.set_ylim(min(vals) * 0.90, max(vals) * 1.10)
    ax.axhline(vals[0], color="#264653", ls=":", lw=0.9, alpha=0.55)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.5f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="medium",
        )
    fig.savefig(OUT / "fig_visplus_lpips.png")
    plt.close(fig)


# Per-run (dx, dy) in points. Negative dx → label sits to the LEFT (ha=right).
_LABEL_XYTEXT = {
    "vs13": (8, 7),
    "a1_coordatt_c13": (8, 9),
    "a2_triplet_c13": (8, -11),
    "a3_scse_c13": (0, 12),
    "a4_gcblock_c13": (-16, -6),  # good amount left
    "b4_eca_c13": (6, -13),  # A5 ECA — below marker (was above, collided)
    "b5_simam_c13": (8, 7),
    "b1_half_c13": (8, -9),
    "b2_scconv_c13": (8, 6),
    "b3_convnext_c13": (8, 9),
    "b6_depthwise_c13": (-14, 2),  # a little right vs prior (-22)
    "b7_pixelshuffle_c13": (8, 11),
    "fp16_c13": (8, -13),
    "d1_resnet50_eca_c13_half": (8, -10),
    "d1_resnet50_triplet_c13_half": (-8, -13),  # below diamond (was up-left, collided)
    "d1_resnet50_eca_c13_ps": (8, -14),  # a little up vs prior (-22)
    "d1_resnet50_triplet_c13_ps": (6, -13),
    "d1_resnet50_gcblock_c13_half": (-30, 0),  # far left of diamond
    "d1_resnet50_gcblock_c13_ps": (8, -12),
    "d1_resnet50_eca_c13_dw": (8, -12),
    "d1_resnet50_gcblock_c13_dw": (-14, 4),
    "d1_resnet50_triplet_c13_dw": (8, -10),
}

# Subsets for split figures (always include vs13 baseline). No fp16 / D.
A_RUNS = frozenset(
    {
        "vs13",
        "a1_coordatt_c13",
        "a2_triplet_c13",
        "a3_scse_c13",
        "a4_gcblock_c13",
        "b4_eca_c13",  # A5
        "b5_simam_c13",  # A6
    }
)
B_RUNS = frozenset(
    {
        "vs13",
        "b1_half_c13",
        "b2_scconv_c13",
        "b3_convnext_c13",
        "b6_depthwise_c13",
        "b7_pixelshuffle_c13",
    }
)
AB_RUNS = A_RUNS | B_RUNS
D_RUNS = frozenset(
    {
        "vs13",
        "d1_resnet50_triplet_c13_half",
        "d1_resnet50_triplet_c13_ps",
        "d1_resnet50_eca_c13_half",
        "d1_resnet50_eca_c13_ps",
        "d1_resnet50_gcblock_c13_half",
        "d1_resnet50_gcblock_c13_ps",
        "d1_resnet50_eca_c13_dw",
        "d1_resnet50_gcblock_c13_dw",
        "d1_resnet50_triplet_c13_dw",
    }
)


def _filter_runs(rows: list[dict], keep: frozenset[str]) -> list[dict]:
    return [r for r in rows if r["run"] in keep]


def fig_latency_lpips(
    rows: list[dict],
    *,
    outfile: str = "fig_latency_lpips.png",
    title: str = "Exp 2 — A/B ablations + D stacks (Pareto: lower-left better)",
) -> None:
    """Pareto scatter. No error bars — B2's std dwarfs the others and looks like a bug."""
    base = next(r for r in rows if r["run"] == "vs13")
    # Extra width on the right for an outside legend so it never covers points.
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    fig.subplots_adjust(left=0.11, right=0.78, top=0.92, bottom=0.12)
    legend_handles: list[Line2D] = []
    for r in rows:
        is_d = str(r.get("group", "")).upper() == "D" or r["run"].startswith("d1_")
        marker = D_MARKER if is_d else "o"
        color = COLORS[r["run"]]
        ax.scatter(
            r["latency_ms"],
            r["lpips"],
            s=55 if is_d else 45,
            color=color,
            marker=marker,
            zorder=4 if is_d else 3,
            edgecolors="white",
            linewidths=0.6,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="#666",
                markeredgewidth=0.25,
                markersize=4.0,
                linestyle="None",
                label=r["label"],
            )
        )
        dx, dy = _LABEL_XYTEXT.get(r["run"], (8, 5))
        ax.annotate(
            r["label"],
            (r["latency_ms"], r["lpips"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.5,
            color="#333",
            ha="right" if dx < 0 else "left",
            va="center",
            clip_on=True,
            annotation_clip=True,
        )
    ax.axvline(base["latency_ms"], color="#888", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(base["lpips"], color="#888", ls=":", lw=0.8, alpha=0.7)
    xs = [r["latency_ms"] for r in rows]
    ys = [r["lpips"] for r in rows]
    # Room for left-side labels inside the axes (never past the y-axis).
    # Extra left pad so left-of-marker labels (D Triplet+half, B6) stay in-frame.
    ax.set_xlim(min(xs) - 16.0, max(xs) + 12.0)
    ypad = (max(ys) - min(ys)) * 0.16
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    ax.set_xlabel("Latency (ms / 1024² patch, batch=1)")
    ax.set_ylabel("LPIPS ↓ (ConvNeXt-TCGA)")
    ax.set_title(title)
    # Outside axes on the right — small markers + airy spacing, never covers points.
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        edgecolor="#ddd",
        framealpha=0.95,
        fontsize=6.5,
        ncol=1,
        borderpad=0.5,
        labelspacing=0.7,
        handletextpad=0.5,
        handlelength=1.0,
    )
    fig.savefig(OUT / outfile)
    plt.close(fig)


def fig_latency_bars(
    rows: list[dict],
    *,
    outfile: str = "fig_latency_bars.png",
    title: str = "Exp 2 — Inference latency incl. D stacks (10 warmup + 100 timed)",
) -> None:
    """Single-axis latency bars with ms + speedup labels on each bar (baseline = 1.00×)."""
    # Sort bars by latency so D speed wins are visually obvious.
    rows = sorted(rows, key=lambda r: r["latency_ms"])
    labels = [r["label"] for r in rows]
    lats = [r["latency_ms"] for r in rows]
    speeds = [r["speedup"] for r in rows]
    colors = [COLORS[r["run"]] for r in rows]

    # Narrower figure when fewer bars (A/B subsets).
    fig_w = 6.4 if len(rows) <= 7 else (8.4 if len(rows) <= 12 else 11.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, lats, color=colors, width=0.72, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7.5)
    ax.set_title(title)
    ax.set_ylim(0, max(lats) * 1.28)

    for b, lat, spd in zip(bars, lats, speeds):
        ax.text(
            b.get_x() + b.get_width() / 2,
            lat,
            f"{lat:.1f}\n{spd:.2f}×",
            ha="center",
            va="bottom",
            fontsize=6.5,
            linespacing=1.1,
            fontweight="medium",
        )

    base_lat = next(r["latency_ms"] for r in rows if r["run"] == "vs13")
    ax.axhline(base_lat, color="#264653", ls=":", lw=0.9, alpha=0.5)
    fig.savefig(OUT / outfile)
    plt.close(fig)


def write_csv(vis: list[dict], lat: list[dict]) -> None:
    with (OUT / "visplus_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "label", "ir_n", "n", "lpips", "psnr", "ssim", "fid"])
        w.writeheader()
        for r in vis:
            w.writerow({k: r[k] for k in w.fieldnames})
    with (OUT / "latency_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "label",
                "group",
                "attention",
                "lpips",
                "latency_ms",
                "latency_std",
                "fid",
                "speedup_vs_c13",
            ],
        )
        w.writeheader()
        for r in lat:
            w.writerow(
                {
                    "run": r["run"],
                    "label": r["label"],
                    "group": r["group"],
                    "attention": r["attention"],
                    "lpips": r["lpips"],
                    "latency_ms": r["latency_ms"],
                    "latency_std": r["latency_std"],
                    "fid": r["fid"],
                    "speedup_vs_c13": round(r["speedup"], 3),
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    vis = load_visplus_rows()
    lat = load_latency_rows()
    lat_a = _filter_runs(lat, A_RUNS)
    lat_b = _filter_runs(lat, B_RUNS)
    lat_ab = _filter_runs(lat, AB_RUNS)
    lat_d = _filter_runs(lat, D_RUNS)
    fig_visplus_lpips(vis)
    fig_latency_lpips(lat)
    fig_latency_bars(lat)
    fig_latency_lpips(
        lat_a,
        outfile="fig_latency_lpips_A.png",
        title="Exp 2 — Group A attention (Pareto: lower-left better)",
    )
    fig_latency_bars(
        lat_a,
        outfile="fig_latency_bars_A.png",
        title="Exp 2 — Group A attention latency (10 warmup + 100 timed)",
    )
    fig_latency_lpips(
        lat_b,
        outfile="fig_latency_lpips_B.png",
        title="Exp 2 — Group B structure (Pareto: lower-left better)",
    )
    fig_latency_bars(
        lat_b,
        outfile="fig_latency_bars_B.png",
        title="Exp 2 — Group B structure latency (10 warmup + 100 timed)",
    )
    fig_latency_lpips(
        lat_ab,
        outfile="fig_latency_lpips_AB.png",
        title="Exp 2 — Groups A + B (Pareto: lower-left better)",
    )
    fig_latency_bars(
        lat_ab,
        outfile="fig_latency_bars_AB.png",
        title="Exp 2 — Groups A + B latency (10 warmup + 100 timed)",
    )
    fig_latency_lpips(
        lat_d,
        outfile="fig_latency_lpips_D.png",
        title="Exp 2 — Group D stacks (Pareto: lower-left better)",
    )
    fig_latency_bars(
        lat_d,
        outfile="fig_latency_bars_D.png",
        title="Exp 2 — Group D stack latency (10 warmup + 100 timed)",
    )
    write_csv(vis, lat)
    print(f"Wrote figures + CSV under {OUT}")
    for r in lat:
        print(f"  {r['label']:14s}  {r['latency_ms']:6.2f} ms  {r['speedup']:.2f}×  LPIPS={r['lpips']:.6f}")


if __name__ == "__main__":
    main()
