## VSlite Agent Notes (cluster + repo conventions)

This repo mixes 4 “families” of code:

- **Baseline pix2pix / contrastive_pix2pix**: training + patch-eval lives under `scripts/`.
- **SwinGAN**: lives under `SwinGAN/` (plus Slurm entrypoints under `/u/akshahi2/jobs/swingan/...`).
- **Restormer**: lives under `Restormer/` (plus Slurm entrypoints under `/u/akshahi2/jobs/restormer/...`).
- **NAFNet**: lives under `NAFNet/` (plus Slurm entrypoints under `/u/akshahi2/jobs/nafnet/...`).

### Effective batch size (training, 1 GPU)

\(B_{\mathrm{eff}} = B_{\mathrm{micro}} \times K\) per optimizer step. Defaults below match current Slurm train scripts:

| Stack | \(B_{\mathrm{micro}} \times K \to B_{\mathrm{eff}}\) | Overrides |
|-------|------------------------------------------------------|-----------|
| Restormer i2i (all train jobs under `jobs/restormer/**/train/`) | **3 × 8 → 24** | `RESTORMER_BATCH_SIZE`, `RESTORMER_GRAD_ACCUM_STEPS` |
| VQGAN HE (`vqgan_he_train.sh`) | **8 × 4 → 32** | `VQGAN_BS`, `VQGAN_ACCUM` (e.g. `ACCUM=2` → \(B_{\mathrm{eff}}=16\)) |
| SwinGAN i2i | **32 × 1 → 32** | `batch_size` in `SwinGAN/config_i2i_*.yaml` |

**Restormer attn 13 train** (`restormer_attn_13.sh`): `scavenger` + `gres=gpu:H100:1`, `--time=1-00:00:00`. Other Restormer train scripts: `IllinoisComputes-GPU` unless edited.

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
- **Restormer**: default test/WSI jobs use `ckpt_latest.pth`; override with `RESTORMER_CKPT_PATH` if needed. Training also writes numbered `ckpt_epoch{N}.pth` on `save_epoch_freq`.
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
- **WSI inference must be lazy/tile-windowed.** Earlier versions of
  `SwinGAN/infer_wsi_png.py` (and the Restormer/NAFNet copies that import its
  loader) did `np.array(image.load())` to materialize the entire ENVI cube
  and then `data.vis.to(device)` to move a full `~10 GB` VIS tensor to GPU.
  On 30k×30k slides this OOM-killed the slurm step well before any tile was
  produced. The fixed loader returns `spectral.open_memmap()` for IR and
  `PIL.Image.open(...)` (no `.load()`) for VIS TIFFs; tile reads go through
  the shared helpers `read_ir_tile_to_device` / `read_vis_tile_to_device`
  exported from `SwinGAN/infer_wsi_png.py`, mirroring
  `scripts/infer_wsi_png_baseline.py`. `_prep_vis` (full-image resample) was
  removed: like baseline, the VIS source is assumed to already be at the
  target 0.69 µm/px (true for `input/VIS/*_VIS_REG.tif`), so the output
  geometry uses VIS native dims and no full-WSI resample happens. If you
  ever add an ENVI VIS source whose `pixel_size != 0.69`, switch to
  per-tile resampling — never go back to full-WSI resample.

### Inference resolution vs. training resolution (per-backbone rules)

Baseline can be inferred at a different resolution than it was trained on because the
generator is a pure FCN. The other backbones cannot, or can only with very specific
caveats. Treat this section as authoritative when someone proposes "just run at 1024".

**Where the train resolution is set:**

- Baseline: `data/VS_dataset.py:472-482` always `cv2.resize`s patches to `opt.image_size`
  (default **1024**). It **ignores** `--load_size` / `--crop_size` from `BaseOptions`,
  so flags like `--load_size 1024 --crop_size 512` in `jobs/baseline/train/*.sh` are
  effectively no-ops; baseline is `train@1024 / test@1024` in practice.
- SwinGAN: `SwinGAN/config_i2i_{baseline,attention}.yaml` `img_size: 512`. Threaded
  into `ProjectedSwinUnet` → `SwinTransformerSys(img_size=config.img_size, ...)` at
  construction time.
- VQGAN: `jobs/vqgan/train/vqgan_he_train.sh` passes `--image_size 512
  --latent_grid_size 16` → `n_down = 5` in `third_party/VQGAN-pytorch/encoder.py`.
- Restormer / NAFNet: see their own training scripts; both are FCN-style (conv +
  Restormer's MDTA which is channel-attention) so they also handle different test
  resolutions naturally, similar to baseline.

**Baseline (`AttUnetGenerator`, `models/networks/generator.py:251-295`)**

Fully convolutional ResNet50 encoder + ConvTranspose decoder + CBAM (CBAM uses
`AdaptiveAvgPool2d`/`AdaptiveMaxPool2d`, which are scale-agnostic). Any input that
is a multiple of 32 works. Safe to mix-and-match train/test resolutions.

**SwinGAN (`ProjectedSwinUnet` → `SwinTransformerSys`)**

Hard "no" for cross-resolution inference. Multiple architecture-level constraints
in `SwinGAN/Networks/swin_transformer_unet_skip_expand_decoder_sys.py`:

- `PatchEmbed.forward` (line 595-597) has `assert H == self.img_size[0] and W == self.img_size[1]`. A 1024 input dies on the first forward step against a 512-built model.
- Every `SwinTransformerBlock` precomputes `attn_mask` as a registered buffer
  whose shape depends on the constructor-time `input_resolution` (lines 249-272),
  and `forward` asserts `L == H * W` against that resolution.
- `PatchMerging`, `PatchExpand`, `FinalPatchExpand_X4` each carry their own
  `input_resolution` and identical asserts.

If you *must* run SwinGAN at 1024, the only correct way is to rebuild a fresh
`SwinTransformerSys(img_size=1024, ...)` and load the 512 checkpoint with
`strict=False`. What transfers cleanly: the 1×1 input projection, `PatchEmbed.proj`
(stride-4 conv, kernel-only), every `nn.Linear`/MLP in every Swin block, and the
`relative_position_bias_table` (sized by `window_size`, not `input_resolution`,
lines 82-84). What does **not** transfer and gets recomputed for the new resolution:
all `attn_mask` buffers. Even after this surgery, **Swin v1 does not extrapolate
to 2× resolution** without finetuning (SwinV2 / log-CPB exists precisely because of
this) — empirically expect FID/PSNR to regress.

**The intended path for SwinGAN at WSI scale is tiled inference**, which is what
`SwinGAN/infer_wsi_png.py:288-323` already does: `--tile_size` defaults to
`cfg.img_size` (= 512), and the WSI is processed in 512-px tiles. Don't try to
feed a full 1024 patch into a 512-trained SwinGAN.

**VQGAN (`third_party/VQGAN-pytorch/{encoder,decoder,codebook,helper}.py`)**

Conditionally yes — encoder/decoder are fully conv, and `NonLocalBlock`
(`helper.py:65-97`) uses 1×1 q/k/v + dynamic `reshape(b, c, h*w)` (no precomputed
sizes, no asserts). `Codebook` (`codebook.py:15-32`) is per-vector and
spatial-size-agnostic.

The trap is the constructor-time `n_down`:

```
n_down = 0
s = image_size
while s > latent_grid_size:
    s //= 2
    n_down += 1
```

Training is `image_size=512, latent_grid_size=16` → **n_down = 5**, channels
`[128,128,128,256,256,512,512]`. The current default in
`scripts/test_vqgan_vs.py:36-37` is `--image_size 1024 --latent_grid_size 16` →
**n_down = 6**, which adds an extra `DownSampleBlock` + extra residuals. The
state-dict keys then don't match a 512-trained checkpoint, and the load at
`scripts/test_vqgan_vs.py:79` is `strict=True`, so it **crashes**.

Correct inference at 1024 from a 512 checkpoint: pass `--image_size 1024
--latent_grid_size 32`. That keeps `n_down=5`, the channel schedule is identical,
all conv shapes match, the bottleneck `NonLocalBlock` just operates on a
32×32 = 1024-token grid (≈ 2 MB attention matrix in bf16 — trivial). Strict
load succeeds. Expect mild quality drift (encoder/decoder were never asked to
bottleneck a 1024 patch) but no crash.

**Restormer / NAFNet (FCN-style)**

Like baseline, they handle multiple-of-stride input sizes natively. Use whatever
resolution matches the eval distribution; no special handling needed.

Summary table:

| Stack | Train res | Naïve 1024 inference from 512 ckpt? | Correct path |
|---|---|---|---|
| Baseline (`AttUnetGenerator`) | nominally 512 via flags, actually 1024 via `VS_dataset.image_size` | Works (FCN) | Already what you do |
| SwinGAN (`ProjectedSwinUnet`) | 512 (`img_size: 512` in YAML) | Crashes on `PatchEmbed` assert; even after rebuild Swin v1 doesn't extrapolate | Tile in `SwinGAN/infer_wsi_png.py` at `tile_size=512`; patch eval stays at 512 |
| VQGAN (third-party) | 512, `latent_grid_size 16`, n_down=5 | Crashes on `load_state_dict` with `test_vqgan_vs.py` defaults | Pass `--image_size 1024 --latent_grid_size 32` |
| Restormer / NAFNet | per their train scripts | Works (FCN) | Match eval distribution; no special handling |

### Slurm job scripts you likely queue (home directory)

These are the authoritative queue scripts; the repo’s `jobs/` folder contains symlinks in many cases.

#### Baseline (contrastive pix2pix)

- **Train** (`jobs/baseline/train/`): `baseline_3.sh`, `baseline_6.sh`, `baseline_10.sh`, `baseline_13.sh` → `scripts/train.py` (same flags shape; channel count / IR indices differ per `NAME`).
- **Patch eval + metrics** (`jobs/baseline/test/`): `test_3.sh`, `test_6.sh`, `test_10.sh`, `test_13.sh` → `scripts/vs_infer_and_metrics.py` (aligned: scavenger test partition, ConvNeXt FID env, 75k cap, train+test splits).
- **WSI** (`jobs/baseline/wsi/`): `baseline_wsi_{3,6,10,13}_latest.sh` → `scripts/infer_wsi_png_baseline.py`.
- **Multi-stage train** (optional): `train_job.sh` → two `scripts/train.py` runs (10ch then 13ch).

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

#### Restormer train (attention + baseline)

- `/u/akshahi2/jobs/restormer/attention/train/restormer_attn_10.sh`
- `/u/akshahi2/jobs/restormer/attention/train/restormer_attn_13.sh`
- `/u/akshahi2/jobs/restormer/baseline/train/restormer_base_10.sh`
- `/u/akshahi2/jobs/restormer/baseline/train/restormer_base_13.sh`

#### VQGAN train

- `/u/akshahi2/jobs/vqgan/train/vqgan_he_train.sh`

#### Restormer Attention (test)

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

### Canonical WSI inference scripts (aligned)

All WSI slurm scripts share the same shape: `chdir` to repo root, source `~/.bashrc`,
activate `torchenv`, point `--parent_folder` at `${VSLITE_ROOT}/input/IR`, and write
PNGs to `results/<model>/test/<run>/wsi_pred/`. Resources are unified at
`--gpus=1 --cpus-per-task=8 --mem=64G --time=24:00:00` on `IllinoisComputes-GPU`.

**Restormer WSI tile size:** jobs only pass `--tile_size` when `WSI_TILE_SIZE` is set; otherwise `Restormer/infer_wsi_png.py` uses `img_size` from the `--config` YAML (512 vs 256, etc.), matching the trained patch size.

| Stack | Slurm script | Inference entrypoint | Output dir |
|---|---|---|---|
| Baseline 3ch | `jobs/baseline/wsi/baseline_wsi_3_latest.sh` | `scripts/infer_wsi_png_baseline.py` | `results/baseline/test/vs3/wsi_pred/` |
| Baseline 6ch | `jobs/baseline/wsi/baseline_wsi_6_latest.sh` | `scripts/infer_wsi_png_baseline.py` | `results/baseline/test/vs6/wsi_pred/` |
| Baseline 10ch | `jobs/baseline/wsi/baseline_wsi_10_latest.sh` | `scripts/infer_wsi_png_baseline.py` | `results/baseline/test/vs10/wsi_pred/` |
| Baseline 13ch | `jobs/baseline/wsi/baseline_wsi_13_latest.sh` | `scripts/infer_wsi_png_baseline.py` | `results/baseline/test/vs13/wsi_pred/` |
| SwinGAN attention | `jobs/swingan/attention/wsi/swingan_attn_wsi_13_latest.sh` | `SwinGAN/infer_wsi_png.py` | `results/swingan_i2i/test/a13/wsi_pred/` |
| SwinGAN baseline | `jobs/swingan/baseline/wsi/swingan_base_wsi_13_latest.sh` | `SwinGAN/infer_wsi_png.py` | `results/swingan_i2i/test/b13/wsi_pred/` |
| Restormer attention | `jobs/restormer/attention/wsi/restormer_attn_wsi_13_latest.sh` | `Restormer/infer_wsi_png.py` | `results/restormer_i2i/test/a13/wsi_pred/` |
| Restormer attention (256px legacy run) | `jobs/restormer/attention/wsi/restormer_attn_wsi_13_legacy256.sh` | `Restormer/infer_wsi_png.py` | `results/restormer_i2i/test/a13_legacy256/wsi_pred/` |
| VQGAN H&E AE | `jobs/vqgan/wsi/vqgan_wsi_latest.sh` | `scripts/infer_wsi_png_vqgan.py` | `results/vqgan_vs/test/vq_b/wsi_pred/` |

Notes specific to VQGAN WSI inference:

- VQGAN is an H&E → H&E autoencoder, so it ignores IR and only consumes the paired
  VIS slide. `scripts/infer_wsi_png_vqgan.py` reuses `SwinGAN.infer_wsi_png.VSLiteFolderLoader`
  with `model_type=1` so the input contract (`--parent_folder input/IR`) is identical
  to the other WSI scripts; VIS is resolved via the same sibling/TIFF lookup.
- The VIS slide is resampled to 0.69 µm/px (same `_prep_vis` as SwinGAN/Restormer/NAFNet)
  before tiling. Default `--tile_size 512` matches the training resolution
  (`image_size=512, latent_grid_size=16`, `n_down=5`); see the resolution rules
  above before tiling at 1024.
- The script accepts both checkpoint formats produced by `scripts/train_vqgan_vs.py`:
  the dict payload at `<run>/checkpoints/ckpt_latest.pt` (key `vqgan`) and the bare
  `state_dict` at `<run>/checkpoints/vqgan_epoch_XXX.pt`. Default checkpoint is
  `ckpt_latest.pt`; override with `VQGAN_CKPT`.

