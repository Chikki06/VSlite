import argparse
import hashlib
import os
from collections import Counter


def _sorted_npy_files(d: str) -> list[str]:
    files = [f for f in os.listdir(d) if f.endswith(".npy") and not f.startswith(".")]
    files.sort()
    return files


def _hash01(s: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def _parse_sample_id(patch_name: str) -> str:
    base = os.path.splitext(os.path.basename(patch_name))[0]
    return base.split("_", 1)[0]


def _read_rules(path: str) -> tuple[set[str], list[str]]:
    exact: set[str] = set()
    prefixes: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f.read().splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if "-" in s:
                exact.add(s)
            else:
                prefixes.add(s)
    return exact, sorted(prefixes)


def _in_rules(sample_id: str, exact: set[str], prefixes: list[str]) -> bool:
    if sample_id in exact:
        return True
    for p in prefixes:
        if sample_id.startswith(p):
            return True
    return False


def _count_samples(names: list[str]) -> Counter:
    c = Counter()
    for n in names:
        c[_parse_sample_id(n)] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--test_list_path", type=str, default="")
    ap.add_argument("--val_list_path", type=str, default="")
    ap.add_argument("--split_seed", type=int, default=1337)
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.05, help="Unused when test.txt exists (test is fixed).")
    ap.add_argument("--top_k_samples", type=int, default=5, help="Show top-K samples by patch count per split.")
    args = ap.parse_args()

    ir_dir = os.path.join(args.dataroot, "ir")
    if not os.path.isdir(ir_dir):
        raise FileNotFoundError(f"Missing IR directory: {ir_dir}")

    # Locate test.txt (or explicit path).
    test_list_path = str(args.test_list_path or "").strip()
    if test_list_path == "":
        candidate = os.path.join(args.dataroot, "test.txt")
        test_list_path = candidate if os.path.isfile(candidate) else ""
    if test_list_path == "" or not os.path.isfile(test_list_path):
        raise FileNotFoundError("No test list found. Provide --test_list_path or create dataroot/test.txt.")

    test_exact, test_prefixes = _read_rules(test_list_path)

    # Locate val.txt (or explicit path). Optional.
    val_list_path = str(args.val_list_path or "").strip()
    if val_list_path == "":
        candidate = os.path.join(args.dataroot, "val.txt")
        val_list_path = candidate if os.path.isfile(candidate) else ""
    val_rules = None
    if val_list_path != "" and os.path.isfile(val_list_path):
        val_rules = _read_rules(val_list_path)

    names = _sorted_npy_files(ir_dir)

    # test split (always from test rules)
    test_names = [n for n in names if _in_rules(_parse_sample_id(n), test_exact, test_prefixes)]
    rest = [n for n in names if not _in_rules(_parse_sample_id(n), test_exact, test_prefixes)]

    # val split (if val.txt exists): subset of the non-test portion
    if val_rules is not None:
        v_exact, v_prefixes = val_rules
        val_names = [n for n in rest if _in_rules(_parse_sample_id(n), v_exact, v_prefixes)]
        train_names = [n for n in rest if not _in_rules(_parse_sample_id(n), v_exact, v_prefixes)]
    else:
        # Split remaining into train/val like VSLiteI2IDataset:
        # train_cut = train / (train + val)
        tv = float(args.train_frac + args.val_frac)
        if tv <= 0:
            raise ValueError("train_frac + val_frac must be > 0")
        train_cut = float(args.train_frac) / tv

        train_names = []
        val_names = []
        for n in rest:
            r = _hash01(n, int(args.split_seed))
            (train_names if r < train_cut else val_names).append(n)

    c_all = _count_samples(names)
    c_test = _count_samples(test_names)
    c_train = _count_samples(train_names)
    c_val = _count_samples(val_names)

    def _topk(c: Counter, k: int):
        return c.most_common(max(0, int(k)))

    print(f"dataroot: {args.dataroot}")
    print(f"IR patches total: {len(names)} | unique samples: {len(c_all)}")
    print(f"test.txt rules: exact={len(test_exact)} prefixes={len(test_prefixes)}")
    if val_rules is not None:
        v_exact, v_prefixes = val_rules
        print(f"val.txt rules: exact={len(v_exact)} prefixes={len(v_prefixes)}")
    else:
        tv = float(args.train_frac + args.val_frac)
        train_cut = float(args.train_frac) / tv if tv > 0 else 0.0
        print(f"val.txt rules: (none) -> hash-split with split_seed={int(args.split_seed)} train_cut={train_cut:.6f}")
    print(f"unique samples excluded into test: {len(c_test)}")
    print(
        "split sizes (patches | unique samples): "
        f"test={len(test_names)}|{len(c_test)} "
        f"train={len(train_names)}|{len(c_train)} "
        f"val={len(val_names)}|{len(c_val)}"
    )
    if val_rules is None:
        print(f"train/val split_seed={int(args.split_seed)} train_cut={train_cut:.6f}")

    k = int(args.top_k_samples)
    if k > 0:
        print(f"top {k} samples by patch count (test): { _topk(c_test, k) }")
        print(f"top {k} samples by patch count (train): { _topk(c_train, k) }")
        print(f"top {k} samples by patch count (val): { _topk(c_val, k) }")


if __name__ == "__main__":
    main()

