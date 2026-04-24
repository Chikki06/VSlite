## VSlite Agent Notes (cluster + repo conventions)

This repo mixes 4 “families” of code:

- **Baseline pix2pix / contrastive_pix2pix**: training + patch-eval lives under `scripts/`.
- **SwinGAN**: lives under `SwinGAN/` (plus Slurm entrypoints under `/u/akshahi2/jobs/swingan/...`).
- **Restormer**: lives under `Restormer/` (plus Slurm entrypoints under `/u/akshahi2/jobs/restormer/...`).
- **NAFNet**: lives under `NAFNet/` (plus Slurm entrypoints under `/u/akshahi2/jobs/nafnet/...`).

### Canonical evaluation flow (what “test” means)

All test pipelines should do the same 3 steps:

1. **WSI inference** (optional / slow)
2. **Patch inference for 75,000 samples** (must not OOM)
3. **Metrics computation** (PSNR / SSIM / FID) and write `metrics.json`/`metrics.yaml`

For fast iteration while debugging patch/FID, run with **`SKIP_WSI=1`** when supported.

### Output layout (required)

All results should land under:

`/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/<model>/test/<run>/`

Where `<run>` is typically a short tag like `sgb_10`, `sga_13`, `restormer_attn_13`, `vs13`, etc.

- `<run>/patch_eval/` for patch-level eval artifacts, containing:
  - `input/` `target/` `pred/` (paired PNGs)
  - `metrics.json` / `metrics.yaml`
- `<run>/wsi_pred/` for WSI PNG outputs (if WSI inference is run)

### Weight / epoch policy (requested defaults)

- **SwinGAN / NAFNet**: default to `ckpt_latest.pth`.
- **Restormer**: tests should use `ckpt_epoch_30.pth` (epoch 30); you can still override via `RESTORMER_CKPT_PATH`.
- **Baseline vs13**: use `which_epoch=latest` by default.

### Where the entrypoints live (important)

The repo-root shims `train.py`, `test.py`, `vs_infer_and_metrics.py` were removed.
Use these directly:

- `scripts/train.py` (baseline training)
- `scripts/test.py` (baseline “html-style” inference, mostly legacy)
- `scripts/vs_infer_and_metrics.py` (**baseline patch inference + PSNR/SSIM/FID**)
- `SwinGAN/test_i2i.py` (SwinGAN patch eval; supports `--out_dir`, `--fid_max_items`, `--fid_batch_size`)
- `Restormer/test_i2i_restormer.py` (Restormer patch eval; supports `--out_dir`, `--fid_max_items`)
- `NAFNet/test_i2i_nafnet.py` (NAFNet patch eval; supports `--out_dir`, `--fid_max_items`)

### Eval cache + “already completed” behavior

- Baseline FID evaluation logic historically used `evaluation/cache`. If a run “completes” instantly, clear:
  - `evaluation/cache/`
  - and the model’s `results/<model>/test/...` directory for that run

### Common gotchas fixed recently

- **Do not** create `VSlite/images/` at import-time. Only create it under debug branches.
- Patch-level FID must **not** keep all Inception activations in RAM. Use online mean/cov accumulation.
- Ensure tests do not “succeed” without producing `metrics.json` / `metrics.yaml`.

### Slurm job scripts you likely queue (home directory)

These are the authoritative queue scripts; the repo’s `jobs/` folder contains symlinks in many cases.

#### Baseline (contrastive pix2pix)

- **Train**:
  - `/u/akshahi2/jobs/baseline/train/baseline_10.sh` → calls `scripts/train.py`
  - `/u/akshahi2/jobs/baseline/train/baseline_13.sh` → calls `scripts/train.py`
  - `/u/akshahi2/jobs/baseline/train/train_job.sh` → calls `scripts/train.py` twice (10ch then 13ch)
- **Patch eval + metrics**:
  - `/u/akshahi2/jobs/baseline/test/test_10.sh` → calls `scripts/vs_infer_and_metrics.py`
  - `/u/akshahi2/jobs/baseline/test/test_13.sh` → calls `scripts/vs_infer_and_metrics.py`

#### SwinGAN Attention (often “sga”)

- `/u/akshahi2/jobs/swingan/attention/test/swingan_attn_test_10.sh`
- `/u/akshahi2/jobs/swingan/attention/test/swingan_attn_test_13.sh`

These were adjusted to support standardized patch eval output (defaulting to `latest`).

#### SwinGAN Baseline (often “sgb”)

- `/u/akshahi2/jobs/swingan/baseline/test/swingan_base_test_10.sh`  (sgb_t10)
- `/u/akshahi2/jobs/swingan/baseline/test/swingan_base_test_13.sh`  (sgb_t13)

These were updated to:

- write patch eval outputs under `results/swingan_i2i/test/.../patch_eval`
- default to `ckpt_latest.pth`
- default `SKIP_WSI=1` path to skip slow WSI steps while debugging patch metrics

#### Restormer Attention

- `/u/akshahi2/jobs/restormer/attention/test/restormer_attn_test_10.sh`
- `/u/akshahi2/jobs/restormer/attention/test/restormer_attn_test_13.sh`

`restormer_attn_test_10.sh` was updated to run patch eval + metrics via `Restormer/test_i2i_restormer.py`
and to write into the standardized results layout.

`restormer_attn_test_13.sh` is currently WSI-metrics oriented; if you want it aligned to the new test flow,
mirror the same patch-eval call pattern used in `restormer_attn_test_10.sh`.

#### NAFNet Attention

- `/u/akshahi2/jobs/nafnet/attention/test/nafnet_a_test_10.sh`
- `/u/akshahi2/jobs/nafnet/attention/test/nafnet_a_test_13.sh`

These were updated to run patch eval + metrics first and default to `SKIP_WSI=1`.

