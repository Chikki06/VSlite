"""Optional OPAL + local-LLM wrapping for CISLNet generator updates.

Enabled when OPAL_ENABLE=1. Does not change training unless that env is set.
The LLM runs in a separate process (opal_local backend); residency defaults to sleep.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional


def _opal_local_root() -> Path:
    return Path(os.environ.get("OPAL_LOCAL_ROOT", "/projects/illinois/eng/bioe/rxb/akshat/opal_local"))


def enable_opal(trainer, opt, *, total_g_updates: int, run_dir: str) -> Optional[Any]:
    """Wrap trainer.optimizer_G with OPAL + LocalOpalAgent. Returns the agent or None."""
    if os.environ.get("OPAL_ENABLE", "0") != "1":
        return None
    if trainer.optimizer_G is None:
        raise RuntimeError("OPAL_ENABLE=1 but optimizer_G is None (freeze_netG?)")

    root = _opal_local_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import opal
    from backend.inference_server import load_config
    from backend.local_agent import LocalOpalAgent

    cfg_path = os.environ.get("OPAL_AGENT_CONFIG", str(root / "config" / "local_agent.yaml"))
    overrides = [
        o.strip()
        for o in os.environ.get("OPAL_AGENT_OVERRIDE", "").split(";")
        if o.strip()
    ]
    # Defaults for production co-residency on one A100 80 GB.
    if not any(o.startswith("server.residency=") for o in overrides):
        overrides.append("server.residency=sleep")
    if not any(o.startswith("server.model_key=") for o in overrides):
        overrides.append("server.model_key=awq4")
    if not any(o.startswith("server.gpu_memory_utilization=") for o in overrides):
        overrides.append("server.gpu_memory_utilization=0.30")
    if not any(o.startswith("agent.on_failure=") for o in overrides):
        overrides.append("agent.on_failure=continue")

    cfg = load_config(cfg_path, overrides)
    state_dir = Path(run_dir) / "opal_agent"
    state_dir.mkdir(parents=True, exist_ok=True)
    agent = LocalOpalAgent(cfg, state_dir)

    model = trainer.pix2pix_model_on_one_gpu.netG
    goal = os.environ.get(
        "OPAL_GOAL",
        "Train a virtual-staining generator (IR→H&E) with reconstruction + NCE; "
        "minimize validation L1 / ConvNeXt-TCGA LPIPS on held-out tissue.",
    )
    decisions = int(os.environ.get("OPAL_DECISIONS", "50"))
    schedule = os.environ.get("OPAL_SCHEDULE", "front")
    every = int(os.environ.get("OPAL_EVERY", "100"))
    trial_steps = int(os.environ.get("OPAL_TRIAL_STEPS", "20"))
    prompt = os.environ.get("OPAL_PROMPT", "neutral")

    wrapped = opal.wrap(
        trainer.optimizer_G,
        model,
        total_steps=int(total_g_updates),
        goal=goal,
        scheduler=None,
        backend=agent,
        decisions=decisions,
        schedule=schedule,
        every=every,
        trial_steps=trial_steps,
        blocking=True,
        summarize=False,
        run_dir=str(Path(run_dir) / "opal_run"),
        prompt=prompt,
        batch_size=int(opt.batch_size),
        epochs=int(opt.niter) + int(getattr(opt, "niter_decay", 0) or 0),
    )
    trainer.optimizer_G = wrapped
    trainer._opal_agent = agent
    trainer._opal_enabled = True
    try:
        agent.attach(model, [wrapped], schedulers=None, total_steps=int(total_g_updates))
    except Exception as e:  # noqa: BLE001
        print(f"[opal] attach warning: {e}")
    print(
        f"[opal] wrapped optimizer_G: total_steps={total_g_updates} decisions={decisions} "
        f"schedule={schedule} residency={cfg['server'].get('residency')} "
        f"run_dir={run_dir}/opal_run"
    )
    return agent


def close_opal(trainer) -> None:
    opt = getattr(trainer, "optimizer_G", None)
    if opt is not None and hasattr(opt, "close"):
        try:
            opt.close()
        except Exception as e:  # noqa: BLE001
            print(f"[opal] close warning: {e}")
    agent = getattr(trainer, "_opal_agent", None)
    if agent is not None and hasattr(agent, "close"):
        try:
            agent.close()
        except Exception as e:  # noqa: BLE001
            print(f"[opal] agent close warning: {e}")
