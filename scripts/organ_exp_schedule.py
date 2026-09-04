#!/usr/bin/env python3
"""Matched-iteration schedule for the organ-agnostic vs organ-specific experiment.

All-samples training for BASE_EPOCHS (default 30) is the compute budget. Every
other run trains for the same number of samples seen (drop_last-aligned), which
means more local epochs on smaller pools.

Checkpoint / holdout evals are placed at the same sample counts as all-samples
epochs 5, 10, 15, 20, 25, 30.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/projects/illinois/eng/bioe/rxb/akshat/VSlite")
DEFAULT_MANIFEST = DEFAULT_ROOT / "dataset" / "organ_exp" / "manifest.json"
BASE_EPOCHS = 30
BATCH_SIZE = 10
TEST_EQ_EPOCHS = (5, 10, 15, 20, 25, 30)

RUNS = (
    "c13_oa_all",
    "c13_oa_top3",
    "c13_os_prostate",
    "c13_os_kidney",
    "c13_os_uterus",
)


def load_patch_counts(manifest: Path) -> dict[str, int]:
    m = json.loads(manifest.read_text())
    os_ = m["train_os"]
    return {
        "c13_oa_all": int(m["train_oa_all"]["train_patches"]),
        "c13_oa_top3": int(m["train_oa_top3"]["train_patches"]),
        "c13_os_prostate": int(os_["prostate"]["train_patches"]),
        "c13_os_kidney": int(os_["kidney"]["train_patches"]),
        "c13_os_uterus": int(os_["uterus"]["train_patches"]),
    }


def samples_per_epoch(n_patches: int, batch_size: int = BATCH_SIZE) -> int:
    """Samples consumed per epoch with drop_last=True."""
    bs = max(1, int(batch_size))
    return (int(n_patches) // bs) * bs


def target_samples(
    n_all: int, base_epochs: int = BASE_EPOCHS, batch_size: int = BATCH_SIZE
) -> int:
    return int(base_epochs) * samples_per_epoch(n_all, batch_size)


def local_epochs_for_budget(
    n_run: int,
    n_all: int,
    base_epochs: int = BASE_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> int:
    spe = samples_per_epoch(n_run, batch_size)
    if spe <= 0:
        return int(base_epochs)
    tgt = target_samples(n_all, base_epochs, batch_size)
    return max(1, int(round(tgt / spe)))


def local_epoch_for_eq(
    eq_epoch: int,
    n_run: int,
    n_all: int,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Local epoch whose sample count best matches all-samples `eq_epoch`."""
    spe_run = samples_per_epoch(n_run, batch_size)
    spe_all = samples_per_epoch(n_all, batch_size)
    if spe_run <= 0:
        return int(eq_epoch)
    return max(1, int(round(int(eq_epoch) * spe_all / spe_run)))


def equiv_epoch_from_local(
    local_epoch: int,
    n_run: int,
    n_all: int,
    batch_size: int = BATCH_SIZE,
) -> float:
    spe_run = samples_per_epoch(n_run, batch_size)
    spe_all = samples_per_epoch(n_all, batch_size)
    if spe_all <= 0:
        return float(local_epoch)
    return float(local_epoch) * float(spe_run) / float(spe_all)


def samples_from_local(
    local_epoch: int, n_run: int, batch_size: int = BATCH_SIZE
) -> int:
    return int(local_epoch) * samples_per_epoch(n_run, batch_size)


def build_schedule(
    *,
    manifest: Path,
    base_epochs: int = BASE_EPOCHS,
    batch_size: int = BATCH_SIZE,
    test_eq_epochs: tuple[int, ...] = TEST_EQ_EPOCHS,
) -> dict[str, Any]:
    counts = load_patch_counts(manifest)
    n_all = counts["c13_oa_all"]
    tgt = target_samples(n_all, base_epochs, batch_size)
    runs: dict[str, Any] = {}
    for name in RUNS:
        n = counts[name]
        local_ep = local_epochs_for_budget(n, n_all, base_epochs, batch_size)
        ckpt_equiv: dict[str, dict[str, int]] = {}
        ckpt_unique: list[int] = []
        seen: set[int] = set()
        for i, eq in enumerate(test_eq_epochs):
            loc = local_ep if i == len(test_eq_epochs) - 1 else local_epoch_for_eq(
                eq, n, n_all, batch_size
            )
            ckpt_equiv[str(eq)] = {
                "local_epoch": loc,
                "samples": samples_from_local(loc, n, batch_size),
            }
            if loc not in seen:
                seen.add(loc)
                ckpt_unique.append(loc)
        runs[name] = {
            "train_patches": n,
            "samples_per_epoch": samples_per_epoch(n, batch_size),
            "local_epochs": local_ep,
            "save_epochs": ckpt_unique,
            "target_samples": tgt,
            "scheduled_samples": samples_from_local(local_ep, n, batch_size),
            "ckpt_equiv": ckpt_equiv,
        }
    return {
        "base_epochs": int(base_epochs),
        "batch_size": int(batch_size),
        "n_all": n_all,
        "target_samples": tgt,
        "test_eq_epochs": list(test_eq_epochs),
        "runs": runs,
    }


def _parse_eq_epochs(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(raw or "").split(",") if p.strip()]
    if not parts:
        return TEST_EQ_EPOCHS
    return tuple(int(p) for p in parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--base-epochs", type=int, default=BASE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument(
        "--test-eq-epochs",
        type=str,
        default=",".join(str(e) for e in TEST_EQ_EPOCHS),
        help="All-samples epoch grid to match (comma-separated).",
    )
    ap.add_argument(
        "--export-bash",
        action="store_true",
        help="Print bash assignments for submit_organ_exp.sh",
    )
    ap.add_argument("--json", action="store_true", help="Print full JSON schedule.")
    args = ap.parse_args()
    eq = _parse_eq_epochs(args.test_eq_epochs)
    sched = build_schedule(
        manifest=args.manifest,
        base_epochs=int(args.base_epochs),
        batch_size=int(args.batch_size),
        test_eq_epochs=eq,
    )
    if args.json:
        print(json.dumps(sched, indent=2))
        return
    if args.export_bash:
        print(f"ORG_N_ALL={sched['n_all']}")
        print(f"ORG_TARGET_SAMPLES={sched['target_samples']}")
        print(f"ORG_BASE_EPOCHS={sched['base_epochs']}")
        print(f"ORG_BATCH_SIZE={sched['batch_size']}")
        print("declare -A ORG_EPOCHS ORG_SAVE_EPOCHS ORG_PATCHES")
        for name, r in sched["runs"].items():
            # Colon-separated: sbatch --export splits on commas.
            save = ":".join(str(e) for e in r["save_epochs"])
            print(f"ORG_EPOCHS[{name}]={r['local_epochs']}")
            print(f"ORG_SAVE_EPOCHS[{name}]={save}")
            print(f"ORG_PATCHES[{name}]={r['train_patches']}")
        return
    print(
        f"Matched-iter budget: {sched['base_epochs']} all-sample epochs × "
        f"{sched['n_all']} patches → {sched['target_samples']} samples "
        f"(batch {sched['batch_size']}, drop_last)"
    )
    print(f"{'run':<18} {'N':>7} {'epochs':>7}  save_epochs")
    for name, r in sched["runs"].items():
        save = ",".join(str(e) for e in r["save_epochs"])
        print(f"{name:<18} {r['train_patches']:>7} {r['local_epochs']:>7}  {save}")


if __name__ == "__main__":
    main()
