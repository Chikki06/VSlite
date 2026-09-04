"""QAT training configuration (extends quant ablation config)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from cislnet.quantization.config import QuantAblationConfig


@dataclass
class QATTrainConfig:
    """Fine-tune fp32 CISLNet generator with fake-quant at 1024px."""

    quant: QuantAblationConfig
    source_ckpt: str
    opt_dir: str
    run: str
    output_dir: str
    dataroot: str
    patch_size: int = 1024
    batch_size: int = 4
    grad_accum_steps: int = 1
    gpu_ids: str = "0"
    save_epoch_freq: int = 1
    print_freq: int = 100
    early_stop_patience: int = 0

    @classmethod
    def from_env(cls) -> QATTrainConfig:
        q = QuantAblationConfig.from_env()
        q.qat_enabled = True
        root = os.environ.get("VSLITE_ROOT", "/projects/illinois/eng/bioe/rxb/akshat/VSlite")
        run = os.environ.get("RUN", "c13")
        results = os.environ.get("RESULTS_DIR", f"{root}/results/cislnet")
        out = os.environ.get("QAT_OUTPUT_DIR", f"{results}/quant/{run}/qat_{q.bits}")
        return cls(
            quant=q,
            source_ckpt=os.environ.get("SOURCE_CKPT", f"{results}/{run}/latest_net_G.pth"),
            opt_dir=os.environ.get("OPT_DIR", f"{results}/{run}"),
            run=run,
            output_dir=out,
            dataroot=os.environ.get("DATAROOT", ""),
            patch_size=int(os.environ.get("PATCH_SIZE", "1024")),
            batch_size=int(os.environ.get("QAT_BATCH_SIZE", os.environ.get("BATCH_SIZE", "4"))),
            grad_accum_steps=int(os.environ.get("GRAD_ACCUM_STEPS", "1")),
            gpu_ids=os.environ.get("GPU_IDS", "0"),
            save_epoch_freq=int(os.environ.get("QAT_SAVE_EPOCH_FREQ", "1")),
            print_freq=int(os.environ.get("QAT_PRINT_FREQ", "100")),
            early_stop_patience=int(os.environ.get("EARLY_STOP_PATIENCE", "0")),
        )
