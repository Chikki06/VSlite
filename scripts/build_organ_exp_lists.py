#!/usr/bin/env python3
"""Build organ-agnostic / organ-specific CISLNet sample lists.

Writes under dataset/organ_exp/ (default). Does not modify data_frozen/test.txt
or val.txt. Canonical test sample IDs (including former val IDs now in test.txt)
are excluded from the whole organ-exp.
Holdout IDs are a separate hardcoded list (HOLDOUTS) chosen offline so each
top-3 organ is ~11.6% patches; seed 42 was only that search, not a runtime split.

Val lists are carved from each train pool (never from holdouts). Holdouts are
test-only: VS_dataset excludes test IDs from train/val, so reusing holdout as
--val_list_path yields an empty val set and crashes DataLoader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


TOP3 = ("Prostate", "Kidney", "Uterus")

# Hardcoded equalized holdout (frac≈0.1160). IDs come from an offline seed-42
# search over remaining (non-canonical) samples — not data_frozen/test.txt.
HOLDOUTS: Dict[str, Tuple[str, ...]] = {
    "Prostate": (
        "Prost2a",
        "3932-3",
        "3945-3",
        "Prost4a",
        "4249-2",
        "Prost3a",
        "3899-8",
    ),
    "Kidney": ("3974-3", "5387-3"),
    "Uterus": ("10465", "uterus"),
}

# Fraction of each train-pool's patches reserved for validation (sample-level).
VAL_FRAC = 0.05
VAL_SEED = 42

ALIASES = {
    "uterus": "Uterus",
    "Pros1b": "Prostate",
    "Prost2a": "Prostate",
    "Prost3a": "Prostate",
    "Prost4a": "Prostate",
    "SV1b": "Seminal vesicle",
}


def _read_rules(path: Path) -> List[str]:
    rules: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            rules.append(s)
    return rules


def _matches(sid: str, rules: Sequence[str]) -> bool:
    for r in rules:
        if "-" in r:
            if sid == r:
                return True
        elif sid.startswith(r):
            return True
    return False


def _write_list(path: Path, sample_ids: Iterable[str], header: str) -> None:
    ids = sorted(set(sample_ids))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        for sid in ids:
            f.write(f"{sid}\n")


def _carve_val(
    train_ids: Sequence[str],
    patches: Counter,
    *,
    frac: float = VAL_FRAC,
    seed: int = VAL_SEED,
) -> Tuple[List[str], List[str]]:
    """Split train pool into (train_ids, val_ids) by sample, targeting ~frac patches.

    Picks the highest-patch samples whose cumulative size is closest to frac*total
    without exceeding ~1.5*target when possible; always keeps ≥1 val sample when
    the pool has ≥2 samples.
    """
    ids = list(train_ids)
    if len(ids) == 0:
        return [], []
    if len(ids) == 1:
        return ids, []

    total = sum(int(patches[s]) for s in ids)
    target = max(1, int(round(total * float(frac))))

    # Deterministic order: hash rank, then prefer mid-sized samples (stable).
    def _rank(sid: str) -> Tuple[float, int, str]:
        h = hashlib.sha256(f"{seed}:{sid}".encode("utf-8")).hexdigest()
        return (int(h[:8], 16) / float(0xFFFFFFFF), -int(patches[sid]), sid)

    ranked = sorted(ids, key=_rank)
    val: List[str] = []
    val_n = 0
    for sid in ranked:
        n = int(patches[sid])
        if not val:
            val.append(sid)
            val_n += n
            continue
        if val_n >= target:
            break
        # Avoid one huge sample swallowing the whole pool when others exist.
        if val_n + n > max(target * 1.5, target + 1) and len(val) >= 1:
            continue
        val.append(sid)
        val_n += n

    val_set = set(val)
    train_kept = [s for s in ids if s not in val_set]
    if not train_kept:
        # Degenerate: put the largest sample back into train.
        biggest = max(val, key=lambda s: int(patches[s]))
        val = [s for s in val if s != biggest]
        train_kept = [biggest]
    return train_kept, val


def _build_organ_map(db_path: Path) -> Dict[str, str]:
    by_exact: Dict[str, str] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT sample_id, organ FROM tissue__images").fetchall()
    finally:
        conn.close()
    for sid, organ in rows:
        if not organ or not str(organ).strip():
            continue
        organ_s = str(organ).strip()
        base = str(sid)
        for suf in ("_HE", "_IR", "_VIS"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        base2 = re.sub(r"_\d+_\d+$", "", base)
        base3 = re.sub(
            r"_(Prostate|Kidney|Uterus|Ovary|Breast|Lung|Liver|Thyroid|Pancreas|"
            r"Colon|Stomach|Tongue|Tonsil|Penis|Bladder|Overy).*$",
            "",
            base,
            flags=re.I,
        )
        for key in {base, base2, base3, base.split("_")[0]}:
            by_exact.setdefault(key, organ_s)
    by_exact.update(ALIASES)
    return by_exact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--vslite-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
    )
    ap.add_argument(
        "--dataroot",
        type=str,
        default="",
        help="Frozen patch root (default: <root>/dataset/data_frozen).",
    )
    ap.add_argument(
        "--db",
        type=str,
        default="",
        help="tissue_images.db path (default: <root>/tissue_images.db).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output dir for lists (default: <root>/dataset/organ_exp).",
    )
    args = ap.parse_args()

    root = Path(args.vslite_root).resolve()
    dataroot = Path(args.dataroot).resolve() if args.dataroot else root / "dataset" / "data_frozen"
    db_path = Path(args.db).resolve() if args.db else root / "tissue_images.db"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "dataset" / "organ_exp"

    ir_dir = dataroot / "ir"
    if not ir_dir.is_dir():
        raise FileNotFoundError(f"IR dir not found: {ir_dir}")
    if not db_path.is_file():
        raise FileNotFoundError(f"DB not found: {db_path}")

    patches: Counter = Counter()
    for name in os.listdir(ir_dir):
        if name.endswith(".npy"):
            patches[name.split("_", 1)[0]] += 1

    organ_of = _build_organ_map(db_path)
    matched = {sid: organ_of[sid] for sid in patches if sid in organ_of}

    test_rules = _read_rules(dataroot / "test.txt")
    val_rules = _read_rules(dataroot / "val.txt")
    canon_excl = {
        sid for sid in patches if _matches(sid, test_rules) or _matches(sid, val_rules)
    }

    holdout_ids: Set[str] = set()
    holdout_by_organ: Dict[str, List[str]] = {}
    for organ in TOP3:
        ids = list(HOLDOUTS[organ])
        missing = [s for s in ids if s not in patches]
        if missing:
            raise RuntimeError(f"Holdout sample(s) missing from frozen IR: {missing}")
        for sid in ids:
            if matched.get(sid) != organ:
                raise RuntimeError(
                    f"Holdout {sid!r} expected organ={organ!r}, got {matched.get(sid)!r}"
                )
            if sid in canon_excl:
                raise RuntimeError(f"Holdout {sid!r} is in canonical test/val — pick another")
        holdout_by_organ[organ] = ids
        holdout_ids.update(ids)

    # Eligible pools (organ samples not in canonical test/val).
    pools: Dict[str, List[Tuple[str, int]]] = {}
    for organ in TOP3:
        elig = [
            (sid, int(patches[sid]))
            for sid in patches
            if matched.get(sid) == organ and sid not in canon_excl
        ]
        elig.sort(key=lambda x: (-x[1], x[0]))
        pools[organ] = elig

    train_os_pool: Dict[str, List[str]] = {}
    holdout_stats = {}
    for organ in TOP3:
        hold_set = set(holdout_by_organ[organ])
        pool_ids = [sid for sid, _ in pools[organ] if sid not in hold_set]
        train_os_pool[organ] = pool_ids
        hold_n = sum(patches[s] for s in hold_set)
        pool_n = sum(patches[s] for s in pool_ids)
        total = hold_n + pool_n
        holdout_stats[organ] = {
            "holdout_samples": holdout_by_organ[organ],
            "holdout_patches": hold_n,
            "pool_samples": len(pool_ids),
            "pool_patches": pool_n,
            "eligible_patches": total,
            "holdout_frac": hold_n / total if total else 0.0,
            "test_train_ratio": hold_n / pool_n if pool_n else None,
        }

    # Agnostic pools: all / top3, excluding canon + all holdouts.
    all_pool_oa = [
        sid for sid in patches if sid not in canon_excl and sid not in holdout_ids
    ]
    top3_pool = [sid for sid in all_pool_oa if matched.get(sid) in TOP3]

    # Carve val from each pool (disjoint from train; never from holdout/test).
    train_os: Dict[str, List[str]] = {}
    val_os: Dict[str, List[str]] = {}
    for organ in TOP3:
        train_os[organ], val_os[organ] = _carve_val(train_os_pool[organ], patches)

    train_oa_all, val_oa_all = _carve_val(all_pool_oa, patches)
    train_oa_top3, val_oa_top3 = _carve_val(top3_pool, patches)

    def _split_stats(train_ids: Sequence[str], val_ids: Sequence[str]) -> dict:
        return {
            "train_samples": len(train_ids),
            "train_patches": int(sum(patches[s] for s in train_ids)),
            "val_samples": list(val_ids),
            "val_sample_count": len(val_ids),
            "val_patches": int(sum(patches[s] for s in val_ids)),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    for organ, key in (("Prostate", "prostate"), ("Kidney", "kidney"), ("Uterus", "uterus")):
        _write_list(
            out_dir / f"holdout_{key}.txt",
            holdout_by_organ[organ],
            f"organ-exp holdout for {organ}",
        )
        _write_list(
            out_dir / f"train_os_{key}.txt",
            train_os[organ],
            f"organ-specific train pool for {organ}",
        )
        _write_list(
            out_dir / f"val_os_{key}.txt",
            val_os[organ],
            f"organ-specific val pool for {organ} (carved from train; not holdout)",
        )

    _write_list(out_dir / "holdout_all.txt", holdout_ids, "union of organ-exp holdouts")
    _write_list(out_dir / "train_oa_all.txt", train_oa_all, "organ-agnostic all organs train")
    _write_list(out_dir / "val_oa_all.txt", val_oa_all, "organ-agnostic all organs val")
    _write_list(out_dir / "train_oa_top3.txt", train_oa_top3, "organ-agnostic top3 organs train")
    _write_list(out_dir / "val_oa_top3.txt", val_oa_top3, "organ-agnostic top3 organs val")

    by_organ_patches = Counter()
    by_organ_samples = Counter()
    for sid, n in patches.items():
        o = matched.get(sid, "UNKNOWN")
        by_organ_patches[o] += n
        by_organ_samples[o] += 1

    manifest = {
        "dataroot": str(dataroot),
        "db": str(db_path),
        "seed": 42,
        "val_frac": VAL_FRAC,
        "val_seed": VAL_SEED,
        "top3_organs": list(TOP3),
        "frozen_samples": len(patches),
        "frozen_patches": int(sum(patches.values())),
        "canonical_excluded_samples": sorted(canon_excl),
        "organ_patch_counts": dict(by_organ_patches.most_common()),
        "organ_sample_counts": dict(by_organ_samples.most_common()),
        "holdouts": holdout_stats,
        "train_oa_all": _split_stats(train_oa_all, val_oa_all),
        "train_oa_top3": _split_stats(train_oa_top3, val_oa_top3),
        "train_os": {
            k.lower(): _split_stats(train_os[k], val_os[k]) for k in TOP3
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote organ-exp lists to {out_dir}")
    for organ in TOP3:
        st = holdout_stats[organ]
        os_st = manifest["train_os"][organ.lower()]
        print(
            f"  {organ}: hold={st['holdout_patches']} train={os_st['train_patches']} "
            f"val={os_st['val_patches']} frac={st['holdout_frac']:.4f} "
            f"ratio={st['test_train_ratio']:.4f}"
        )
    print(
        f"  oa_all: train={manifest['train_oa_all']['train_patches']} "
        f"val={manifest['train_oa_all']['val_patches']}"
    )
    print(
        f"  oa_top3: train={manifest['train_oa_top3']['train_patches']} "
        f"val={manifest['train_oa_top3']['val_patches']}"
    )


if __name__ == "__main__":
    main()
