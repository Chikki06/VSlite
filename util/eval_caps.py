from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


# Centralized, conservative defaults.
# - "ITEMS" caps are used by i2i patch trainers (Restormer/SwinGAN/NAFNet) where eval loops
#   iterate over datasets of patches.
# - "BATCHES" caps are used by baseline + VQGAN where eval loops are batch-based.
DEFAULT_EVAL_MAX_ITEMS_TRAIN = 512
DEFAULT_EVAL_MAX_ITEMS_VAL = 512
DEFAULT_EVAL_MAX_ITEMS_TEST = 512

DEFAULT_EVAL_MAX_BATCHES_TRAIN = 20
DEFAULT_EVAL_MAX_BATCHES_VAL = 50
DEFAULT_EVAL_MAX_BATCHES_TEST = 50


def eval_max_items_train(default: int = DEFAULT_EVAL_MAX_ITEMS_TRAIN) -> int:
    return _env_int("UNIFIED_EVAL_MAX_ITEMS_TRAIN", default)


def eval_max_items_val(default: int = DEFAULT_EVAL_MAX_ITEMS_VAL) -> int:
    return _env_int("UNIFIED_EVAL_MAX_ITEMS_VAL", default)


def eval_max_items_test(default: int = DEFAULT_EVAL_MAX_ITEMS_TEST) -> int:
    return _env_int("UNIFIED_EVAL_MAX_ITEMS_TEST", default)


def eval_max_batches_train(default: int = DEFAULT_EVAL_MAX_BATCHES_TRAIN) -> int:
    return _env_int("UNIFIED_EVAL_MAX_BATCHES_TRAIN", default)


def eval_max_batches_val(default: int = DEFAULT_EVAL_MAX_BATCHES_VAL) -> int:
    return _env_int("UNIFIED_EVAL_MAX_BATCHES_VAL", default)


def eval_max_batches_test(default: int = DEFAULT_EVAL_MAX_BATCHES_TEST) -> int:
    return _env_int("UNIFIED_EVAL_MAX_BATCHES_TEST", default)

