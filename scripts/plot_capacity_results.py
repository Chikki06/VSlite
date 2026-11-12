#!/usr/bin/env python3
"""Figures for Experiment 3: VIS-only input + decoder-capacity ladder.

Writes under results_figures_inference/ (sibling of results_inference.md):
  fig_ir_need.png       - does the model need IR bands if VIS is always on?
  fig_decoder_ladder.png - latency + quality vs decoder width (10/25/50/75/100%)
  fig_capacity_pareto.png - capacity/IR family vs prior A/D standouts (latency vs LPIPS)
  capacity_metrics.csv  - one row per run (intuitive names + raw run tags)
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

# Intuitive name -> raw run tag. Decoder 50% = Exp 2's B1 half-decoder.
IR_FAMILY = [
    ("VIS-only (0 IR)", "visonly_c13", 0),
    ("Vis+1 (1 IR)", "vp1", 1),
    ("Vis+3 (3 IR)", "vp3", 3),
    ("Vis+5 (5 IR)", "vp5", 5),
    ("Full CISLNet (10 IR)", "c13", 10),
]
LADDER = [
    ("Decoder 10%", "dec10_c13", 0.10),
    ("Decoder 25%", "dec25_c13", 0.25),
    ("Decoder 50% (B1)", "b1_half_c13", 0.50),
    ("Decoder 75%", "dec75_c13", 0.75),
    ("Full (100%)", "c13", 1.00),
]
ENCDEC = ("Enc+Dec 50%", "encdec50_c13", 0.50)
PRIOR_STANDOUTS = [
    ("A2 Triplet", "a2_triplet_c13"),
    ("A5 ECA", "b4_eca_c13"),
    ("A4 GCBlock", "a4_gcblock_c13"),
    ("GCBlock+DW", "d1_resnet50_gcblock_c13_dw"),
]

COLORS = {
    "visonly_c13": "#e76f51",
    "vp1": "#2a9d8f",
    "vp3": "#6ec6b3",
    "vp5": "#118ab2",
    "c13": "#264653",
    "dec10_c13": "#8ecae6",
    "dec25_c13": "#219ebc",
    "b1_half_c13": "#e9c46a",
    "dec75_c13": "#2a9d8f",
    "encdec50_c13": "#9f1239",
    "a2_triplet_c13": "#1d7874",
    "b4_eca_c13": "#b56576",
    "a4_gcblock_c13": "#40916c",
    "d1_resnet50_gcblock_c13_dw": "#0077b6",
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
        }
    )


def _leaf_metrics(run: str) -> dict:
    p = TEST / run / "patch_eval_test" / "metrics.json"
    if not p.is_file():
        raise FileNotFoundError(f"no patch_eval metrics for {run}: {p}")
    m = json.loads(p.read_text())
    return {
        "psnr": float(m["psnr_mean"]),
        "ssim": float(m["ssim_mean"]),
        "fid": float(m["fid"]),
        "lpips": float(m["lpips_convnext_tcga_mean"]),
    }


def _latency_map() -> dict[str, float]:
    """Latest tracking.csv latency per run (last row wins)."""
    lat: dict[str, float] = {}
    with TRACK.open() as f:
        for r in csv.DictReader(f):
            lat[r["run"]] = float(r["latency_ms_mean"])
    return lat


def build_rows() -> list[dict]:
    lat = _latency_map()
    full_lat = lat.get("vs13", lat.get("c13"))
    rows = []
    for label, run, *extra in [*IR_FAMILY, *LADDER, ENCDEC, *PRIOR_STANDOUTS]:
        width = extra[0] if extra else None
        r = {**_leaf_metrics(run), "label": label, "run": run, "width": width}
        r["latency_ms"] = lat.get(run, full_lat if run == "c13" else None)
        rows.append(r)
    for r in rows:
        r["speedup"] = full_lat / r["latency_ms"] if r["latency_ms"] else None
    return rows


def _want(rows: list[dict], names: list[str]) -> list[dict]:
    # Key by intuitive label: the raw tag "c13" appears in BOTH the IR family
    # and the decoder ladder with different widths/labels.
    by = {r["label"]: r for r in rows}
    return [by[k] for k in names if k in by]


def fig_ir_need(rows: list[dict]) -> None:
    fam = _want(rows, [r[0] for r in IR_FAMILY])
    # vp1/vp3/vp5 have no latency measurement - LPIPS/FID only, fine here.
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = [r["width"] for r in fam]
    lp = [r["lpips"] for r in fam]
    fi = [r["fid"] for r in fam]
    ax.plot(x, lp, "o-", color="#264653", lw=1.6, ms=6, label="LPIPS (ConvNeXt)", zorder=4)
    for xi, v, r in zip(x, lp, fam):
        ax.annotate(
            f"{v:.5f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=7,
            color="#264653",
        )
    ax.set_ylabel("LPIPS ↓ (ConvNeXt-TCGA)")
    ax.set_xlabel("IR bands kept in the input (VIS RGB always on)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in fam], rotation=12, ha="right")
    ax2 = ax.twinx()
    ax2.plot(x, fi, "s--", color="#e76f51", lw=1.4, ms=5.5, label="FID (ConvNeXt)")
    for xi, v in zip(x, fi):
        ax2.annotate(
            f"{v:.2f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7,
            color="#e76f51",
        )
    ax2.set_ylabel("FID ↓ (ConvNeXt)", color="#e76f51")
    ax2.spines["right"].set_visible(True)
    ax.set_title("Exp 3 — How many IR bands does VIS still need?")
    # Annotate the Vis+1 dip.
    i1 = x.index(1)
    ax2.annotate(
        "Vis+1 is the worst spot:\nadding one band alone buys nothing",
        (x[i1], fi[i1]),
        textcoords="offset points",
        xytext=(30, 26),
        fontsize=7.5,
        color="#555",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ir_need.png")
    plt.close(fig)


def fig_decoder_ladder(rows: list[dict]) -> None:
    lad = _want(rows, [r[0] for r in LADDER])
    ed = _want(rows, [ENCDEC[0]])[0]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    # ---- left: latency bars (decoder ladder + the Enc+Dec stack) ----
    labels = [r["label"] for r in lad] + [ed["label"]]
    lats = [r["latency_ms"] for r in lad] + [ed["latency_ms"]]
    speeds = [r["speedup"] for r in lad] + [ed["speedup"]]
    cols = [COLORS[r["run"]] for r in lad] + [COLORS[ed["run"]]]
    x = np.arange(len(labels))
    bars = axl.bar(x, lats, width=0.68, edgecolor="white", linewidth=0.6)
    for b, c in zip(bars, cols):
        b.set_color(c)
    for b, lat, spd in zip(bars, lats, speeds):
        axl.text(
            b.get_x() + b.get_width() / 2,
            lat,
            f"{lat:.1f}\n{spd:.2f}×",
            ha="center",
            va="bottom",
            fontsize=7,
            linespacing=1.1,
            fontweight="medium",
        )
    axl.set_xticks(x)
    axl.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    axl.set_ylabel("Latency (ms / 1024² patch)")
    axl.set_title("Decoder width → inference latency\n(last bar: encoder width also 50%)")
    # ---- right: quality curves vs decoder width ----
    wx = [r["width"] for r in lad]
    lp = [r["lpips"] for r in lad]
    fi = [r["fid"] for r in lad]
    axr.plot(wx, lp, "o-", color="#264653", lw=1.6, ms=6, zorder=4)
    # Per-point label placement, tuned by hand to avoid marker/line collisions.
    _off = {
        "dec10_c13": (10, 8),
        "dec25_c13": (-4, -20),
        "b1_half_c13": (2, 10),
        "dec75_c13": (10, -4),
        "c13": (-4, 14),
    }
    for r, xi, v in zip(lad, wx, lp):
        dx, dy = _off.get(r["run"], (6, -16))
        axr.annotate(
            f"{r['label'].replace(' (B1)', '')}\n{v:.5f}",
            (xi, v),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center" if r["run"] == "c13" else ("right" if dx < 0 else "left"),
            fontsize=6.8,
            color="#264653",
        )
    axr3 = axr.twinx()
    axr3.plot(wx, fi, "s--", color="#e76f51", lw=1.4, ms=5.5, zorder=3)
    for xi, v in zip(wx, fi):
        axr3.annotate(
            f"{v:.2f}",
            (xi, v),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color="#e76f51",
        )
    axr3.set_ylabel("FID ↓", color="#e76f51")
    # Enc+Dec 50% as a diamond, dodged a hair to the right of Decoder 50%.
    axr.scatter(
        [ed["width"] + 0.015],
        [ed["lpips"]],
        s=70,
        marker="D",
        color=COLORS[ed["run"]],
        zorder=5,
        edgecolors="white",
        linewidths=0.6,
    )
    axr.annotate(
        f"{ed['label']}\n{ed['lpips']:.5f}",
        (ed["width"] + 0.015, ed["lpips"]),
        textcoords="offset points",
        xytext=(-30, 18),
        ha="center",
        fontsize=6.8,
        color=COLORS[ed["run"]],
        fontweight="medium",
    )
    axr.set_xlabel("Decoder width multiplier (0.10 → 1.00)")
    axr.set_ylabel("LPIPS ↓ (ConvNeXt-TCGA)")
    axr.set_title("Quality vs decoder width\n(FID right axis)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_decoder_ladder.png")
    plt.close(fig)


def fig_capacity_pareto(rows: list[dict]) -> None:
    fam = _want(
        rows,
        [
            "VIS-only (0 IR)",
            "Decoder 10%",
            "Decoder 25%",
            "Decoder 50% (B1)",
            "Decoder 75%",
            "Enc+Dec 50%",
            "Full (100%)",
        ],
    )
    prior = _want(rows, [r[0] for r in PRIOR_STANDOUTS])
    speed_pos = {p[1]: i for i, p in enumerate(PRIOR_STANDOUTS)}
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.subplots_adjust(left=0.11, right=0.80, top=0.92, bottom=0.12)
    # Neutral pentagon for Encoder+Dec (new family), circles for the rest.
    handles: list[Line2D] = []
    for r in fam:
        label = "Full CISLNet (c13)" if r["run"] == "c13" else r["label"]
        marker = "p" if r["run"] == "encdec50_c13" else "o"
        ax.scatter(
            r["latency_ms"],
            r["lpips"],
            s=60 if r["run"] == "c13" else 45,
            marker=marker,
            color=COLORS[r["run"]],
            zorder=4,
            edgecolors="white",
            linewidths=0.6,
        )
        handles.append(
            Line2D(
                [0], [0], marker=marker, color="none", markerfacecolor=COLORS[r["run"]],
                markeredgecolor="#666", markeredgewidth=0.25, markersize=5.5,
                linestyle="None", label=label,
            )
        )
    for r in prior:
        ax.scatter(
            r["latency_ms"],
            r["lpips"],
            s=48,
            marker="o",
            color=COLORS[r["run"]],
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.9,
        )
        handles.append(
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=COLORS[r["run"]],
                markeredgecolor="#666", markeredgewidth=0.25, markersize=5.5,
                linestyle="None", label=f"{r['label']} (Exp 2)", alpha=0.9,
            )
        )
    base = next(r for r in fam if r["run"] == "c13")
    ax.axvline(base["latency_ms"], color="#888", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(base["lpips"], color="#888", ls=":", lw=0.8, alpha=0.7)
    # Manual point labels: (dx, dy) offset points; negative dx → to the left.
    xy = {
        "visonly_c13": (10, 10),
        "dec10_c13": (9, -13),
        "dec25_c13": (9, 6),
        "b1_half_c13": (-6, 12),
        "dec75_c13": (9, 7),
        "encdec50_c13": (9, -4),
        "c13": (-6, -14),
        "a2_triplet_c13": (8, 8),
        "b4_eca_c13": (9, 3),
        "a4_gcblock_c13": (0, -14),
        "d1_resnet50_gcblock_c13_dw": (-10, 12),
    }
    for r in fam + prior:
        dx, dy = xy.get(r["run"], (8, 5))
        label = "Full CISLNet (c13)" if r["run"] == "c13" else r["label"]
        ax.annotate(
            label,
            (r["latency_ms"], r["lpips"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.5,
            color="#333",
            ha="right" if dx < 0 else "left",
            va="center",
            clip_on=True,
        )
    xs = [r["latency_ms"] for r in fam + prior]
    ys = [r["lpips"] for r in fam + prior]
    ax.set_xlim(min(xs) - 10, max(xs) + 14)
    ypad = (max(ys) - min(ys)) * 0.22
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    ax.set_xlabel("Latency (ms / 1024² patch, batch=1)")
    ax.set_ylabel("LPIPS ↓ (ConvNeXt-TCGA)")
    ax.set_title("Exp 3 — Capacity/IR family vs prior standouts (lower-left better; gray cross = full CISLNet)")
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        edgecolor="#ddd",
        fontsize=6.8,
        ncol=1,
        labelspacing=0.7,
    )
    fig.savefig(OUT / "fig_capacity_pareto.png")
    plt.close(fig)


def write_csv(rows: list[dict]) -> None:
    fam = _want(rows, [r[0] for r in IR_FAMILY])
    lad = _want(rows, [r[0] for r in LADDER])
    ed = _want(rows, [ENCDEC[0]])[0]
    prior = _want(rows, [r[0] for r in PRIOR_STANDOUTS])
    fields = [
        "label",
        "run",
        "ir_bands",
        "decoder_width",
        "psnr",
        "ssim",
        "fid",
        "lpips_convnext",
        "latency_ms",
        "speedup_vs_full",
        "group",
    ]
    irs = {run: n for _, run, n in IR_FAMILY}
    widths = {run: w for _, run, w in [*LADDER, ENCDEC]}
    group: dict[str, str] = {}
    for _, run, _n in IR_FAMILY:
        group[run] = "ir-family"
    for _, run, _w in LADDER:
        group[run] = "decoder-ladder"
    group[ENCDEC[1]] = "encdec-stack"
    for _, run in PRIOR_STANDOUTS:
        group[run] = "prior-standout"

    def _row(r: dict) -> dict:
        return {
            "label": r["label"],
            "run": r["run"],
            "ir_bands": irs.get(r["run"], ""),
            "decoder_width": widths.get(r["run"], ""),
            "psnr": round(r["psnr"], 4),
            "ssim": round(r["ssim"], 4),
            "fid": round(r["fid"], 4),
            "lpips_convnext": round(r["lpips"], 6),
            "latency_ms": round(r["latency_ms"], 2) if r["latency_ms"] else "",
            "speedup_vs_full": round(r["speedup"], 3) if r["speedup"] else "",
            "group": group[r["run"]],
        }

    with (OUT / "capacity_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in [*fam, *lad, ed, *prior]:
            w.writerow(_row(r))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    rows = build_rows()
    fig_ir_need(rows)
    fig_decoder_ladder(rows)
    fig_capacity_pareto(rows)
    write_csv(rows)
    print(f"Wrote capacity figures + capacity_metrics.csv under {OUT}")
    for r in rows:
        lat = f"{r['latency_ms']:6.2f} ms" if r["latency_ms"] else "     -   "
        print(
            f"  {r['label']:22s} {r['run']:28s} LPIPS={r['lpips']:.6f} FID={r['fid']:6.2f}"
            f" PSNR={r['psnr']:6.2f} {lat}"
        )


if __name__ == "__main__":
    main()
