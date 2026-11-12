#!/usr/bin/env python3
"""Pick the saved checkpoint epoch with lowest train-time test LPIPS.

Reads Slurm train logs for lines like::

    End of epoch 15 / 20  samples ... test_lpips_convnext_tcga: 0.001093

Only epochs that have ``{epoch}_net_G.pth`` under the run dir are candidates
(``save_epoch_freq`` typically yields 5/10/15/20).

Prints the epoch integer to stdout (or ``latest`` if none found).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"End of epoch\s+(\d+)\s*/\s*\d+\s+samples.*?"
    r"test_lpips_convnext_tcga:\s*([0-9.eE+-]+)"
)

# Default Slurm log dirs for known experiment job names.
LOG_DIR_BY_RUN = {
    "c10": "cn_tr_c10",
    "vp1": "cn_tr_vp1",
    "vp3": "cn_tr_vp3",
    "vp5": "cn_tr_vp5",
    "a1_coordatt_c13": "cn_a1",
    "a2_triplet_c13": "cn_a2",
    "a3_scse_c13": "cn_a3",
    "a4_gcblock_c13": "cn_a4",
    "b1_half_c13": "cn_b1",
    "b2_scconv_c13": "cn_b2",
    "b3_convnext_c13": "cn_b3",
    "c13": "cn_tr_c13",
    "visonly_c13": "cn_tr_c13",
    "dec25_c13": "cn_tr_c13",
    "dec10_c13": "cn_tr_c13",
    "dec75_c13": "cn_tr_c13",
    "encdec50_c13": "cn_tr_c13",
}


def saved_epochs(run_dir: Path) -> set[int]:
    out: set[int] = set()
    for p in run_dir.glob("*_net_G.pth"):
        m = re.match(r"(\d+)_net_G\.pth$", p.name)
        if m:
            out.add(int(m.group(1)))
    return out


def parse_lpips_from_logs(log_dirs: list[Path]) -> dict[int, float]:
    rows: dict[int, float] = {}
    for d in log_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.out")):
            text = f.read_text(errors="ignore")
            for m in LINE_RE.finditer(text):
                rows[int(m.group(1))] = float(m.group(2))
    return rows


def pick_best(
    *,
    run: str,
    run_dir: Path,
    log_dirs: list[Path],
    metric: str = "test",
) -> tuple[str, dict[int, float], set[int]]:
    del metric  # reserved; always use test_lpips from logs today
    rows = parse_lpips_from_logs(log_dirs)
    have = saved_epochs(run_dir)
    cand = {e: v for e, v in rows.items() if e in have}
    if not cand:
        return "latest", rows, have
    best_e, _ = min(cand.items(), key=lambda kv: kv[1])
    return str(best_e), rows, have


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Checkpoint run name (e.g. vp1, a1_coordatt_c13)")
    p.add_argument(
        "--run_dir",
        default=None,
        help="Path to results/cislnet/<run> (default: <vslite>/results/cislnet/<run>)",
    )
    p.add_argument(
        "--logs_root",
        default="/u/akshahi2/logs",
        help="Parent of per-job Slurm log directories",
    )
    p.add_argument(
        "--log_dir",
        action="append",
        default=None,
        help="Explicit log directory (repeatable). Default: map --run → logs/<jobname>",
    )
    p.add_argument(
        "--vslite_root",
        default="/projects/illinois/eng/bioe/rxb/akshat/VSlite",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print selection details to stderr",
    )
    args = p.parse_args(argv)

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(args.vslite_root) / "results" / "cislnet" / args.run
    )
    if args.log_dir:
        log_dirs = [Path(x) for x in args.log_dir]
    else:
        mapped = LOG_DIR_BY_RUN.get(args.run)
        if not mapped:
            print(
                f"ERROR: unknown run {args.run!r}; pass --log_dir",
                file=sys.stderr,
            )
            return 2
        log_dirs = [Path(args.logs_root) / mapped]

    epoch, rows, have = pick_best(run=args.run, run_dir=run_dir, log_dirs=log_dirs)
    if args.verbose:
        cand = {e: rows[e] for e in sorted(have) if e in rows}
        print(
            f"run={args.run} run_dir={run_dir} saved={sorted(have)} "
            f"cand={cand} → epoch={epoch}",
            file=sys.stderr,
        )
    print(epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
