import argparse
import os
from collections import Counter


def _sorted_npy_files(d: str) -> list[str]:
    files = [f for f in os.listdir(d) if f.endswith(".npy") and not f.startswith(".")]
    files.sort()
    return files


def _parse_sample_id(patch_name: str) -> str:
    base = os.path.splitext(os.path.basename(patch_name))[0]
    return base.split("_", 1)[0]


def _read_test_samples(path: str) -> set[str]:
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


def _in_test(sample_id: str, exact: set[str], prefixes: list[str]) -> bool:
    if sample_id in exact:
        return True
    for p in prefixes:
        if sample_id.startswith(p):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--test_list_path", type=str, default="")
    ap.add_argument("--probe_sample", type=str, default="10440-1")
    args = ap.parse_args()

    ir_dir = os.path.join(args.dataroot, "ir")
    if not os.path.isdir(ir_dir):
        raise FileNotFoundError(f"Missing IR directory: {ir_dir}")

    test_list_path = str(args.test_list_path or "").strip()
    if test_list_path == "":
        candidate = os.path.join(args.dataroot, "test.txt")
        test_list_path = candidate if os.path.isfile(candidate) else ""
    if test_list_path == "" or not os.path.isfile(test_list_path):
        raise FileNotFoundError(
            "No test list found. Provide --test_list_path or create dataroot/test.txt."
        )

    test_exact, test_prefixes = _read_test_samples(test_list_path)
    names = _sorted_npy_files(ir_dir)

    test_names = [n for n in names if _in_test(_parse_sample_id(n), test_exact, test_prefixes)]
    train_names = [n for n in names if not _in_test(_parse_sample_id(n), test_exact, test_prefixes)]

    def _count_samples(ns: list[str]) -> Counter:
        c = Counter()
        for n in ns:
            c[_parse_sample_id(n)] += 1
        return c

    c_test = _count_samples(test_names)
    c_train = _count_samples(train_names)

    probe = str(args.probe_sample)
    in_test = c_test.get(probe, 0)
    in_train = c_train.get(probe, 0)

    print(f"IR patches total: {len(names)}")
    print(f"Test rules: exact={len(test_exact)} prefixes={len(test_prefixes)}")
    print(f"Test patches: {len(test_names)} | Train(+val) patches: {len(train_names)}")
    print(f"Probe sample {probe!r}: in_test={in_test} in_train={in_train}")
    if in_test > 0 and in_train == 0:
        print("OK: no leakage (probe only appears in test).")
        raise SystemExit(0)
    if in_test == 0 and in_train == 0:
        print("WARN: probe sample not found in dataset.")
        raise SystemExit(2)
    print("FAIL: leakage detected (probe appears in both splits).")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

