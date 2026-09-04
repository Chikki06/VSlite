# VSlite ablation environment guide

All Slurm train jobs should `source` [`vslite_env.sh`](vslite_env.sh), which optionally loads
[`ablation.env`](ablation.env) if you create it from [`ablation.env.example`](ablation.env.example).

Copy the example once:

```bash
cp jobs/_lib/ablation.env.example jobs/_lib/ablation.env
# edit jobs/_lib/ablation.env for your sweep
```

Or export variables inline / in your sbatch preamble:

```bash
export NUM_CHANNELS=13 PATCH_SIZE=512 BATCH_SIZE=8 EPOCHS=50
sbatch jobs/restormer/train/train.sh
```

## Required variables (every model)

| Variable | Values | Maps to | Notes |
|----------|--------|---------|-------|
| `NUM_CHANNELS` | `3`, `6`, `10`, `13` | `input_nc` / `input_mode` | **Required.** CISLNet: `--input_nc`. i2i: `ir10` or `irvis13`. |
| `PATCH_SIZE` | e.g. `256`, `512`, `1024` | `--image_size` / `img_size` | CISLNet default `1024`; Restormer/SwinGAN default `512`. |
| `BATCH_SIZE` | positive int | micro-batch per GPU step | |
| `GRAD_ACCUM_STEPS` | ≥ 1 | effective batch = `BATCH_SIZE × GRAD_ACCUM` | Restormer jobs often use `3` and `8`. |
| `EPOCHS` | ≥ 1 | total training epochs | CISLNet: `--total_epochs` when set. |

**Run identity**

| Variable | Example | Purpose |
|----------|---------|---------|
| `RUN` | `a13`, `c13`, `b10` | Short tag; becomes output subdir and unified TB run name. |
| `OUTPUT_DIR` | `$VSLITE_ROOT/results/restormer/a13` | Override checkpoint root (i2i stacks). |
| `CHECKPOINTS_DIR` | `$VSLITE_ROOT/results/cislnet` | CISLNet checkpoints root. |
| `NAME` | `vs13` | CISLNet run folder name under `CHECKPOINTS_DIR`. |
| `DATAROOT` / `VS_DATAROOT` | `dataset/data_frozen` | Patch `.npy` root (`ir/`, `he/`, `vis/`). |

## Regularization & data

| Variable | Default | Effect |
|----------|---------|--------|
| `WEIGHT_DECAY` | `0` | Adam(W) weight decay. |
| `DROPOUT` | `0` | NAFNet `drop_out_rate` when > 0. |
| `NO_AUG` | `0` | Set `1` to disable augmentation. |
| `AUG_STRENGTH` | `1.0` | Scale aug intensity (stack-specific). |
| `HE_COLOR_SPACE` | `srgb` | `linear` trains in linear light on H&E targets. |

## Loss weights

| Variable | Stacks | Effect |
|----------|--------|--------|
| `LAMBDA_L1` | CISLNet, i2i | L1 weight. |
| `LAMBDA_NCE` | CISLNet | Contrastive / NCE weight. |
| `LAMBDA_PERC` | SwinGAN, Restormer (attn) | ConvNeXt perceptual loss (not VGG). |
| `LAMBDA_GAN` | CISLNet, SwinGAN | GAN weight. |
| `NO_GAN` | `0` | Set `1` to disable GAN term where supported. |

## Optimizer & schedule

| Variable | Values | Effect |
|----------|--------|--------|
| `LR` | e.g. `2e-4` | Base learning rate. |
| `OPTIMIZER` | `adam`, `adamw` | CISLNet supports both via CLI. |
| `LR_SCHEDULE` | `fixed`, `cosine`, `plateau` | Documented; wire per stack in YAML/CLI. |
| `WARMUP_EPOCHS` | `0` | Warmup length when schedule supports it. |
| `EMA_DECAY` | `0` | Off when 0; I2SB/BBDM use their own EMA when set. |

## Early stopping (validation L1)

| Variable | Default | Effect |
|----------|---------|--------|
| `EARLY_STOP_PATIENCE` | `0` | `0` = disabled. Stop after N epochs without val L1 improvement. |
| `EARLY_STOP_MIN_DELTA` | `1e-4` | Minimum improvement to count as better. |

Best checkpoint: `ckpt_best.pth` (i2i) or `best_net_G.pth` (CISLNet).

## Progressive patch training

| Variable | Format | Example |
|----------|--------|---------|
| `PROGRESSIVE_SCHEDULE` | `PATCH:EPOCHS,...` | `256:10,512:40` |

Restormer reads this in training; rebuilds phases when set.

## Preview & TensorBoard

| Variable | Effect |
|----------|--------|
| `PREVIEW_TILE_LIST` | JSON file listing 8 val patch ids — lock tiles across models. |
| `UNIFIED_TB_ROOT` | Override `results/unified/` root. |
| `UNIFIED_TB_MIRROR_TRAIN` | `1` (default): mirror step losses to unified TB. |
| `UNIFIED_EVAL_MAX_BATCHES_*` | Cap epoch-end L1 eval batches (not FID). |

TensorBoard:

```bash
tensorboard --logdir /projects/illinois/eng/bioe/rxb/akshat/VSlite/results/unified
```

Compare `l1/val` across `cislnet/c13`, `restormer/a13`, etc. Image slider: tag `preview/fixed_val/grid` (epoch axis).

## Test-only metrics (never set during training)

| Variable | Default | Purpose |
|----------|---------|---------|
| `NUM_TEST` / `VSLITE_FID_MAX_ITEMS` | `75000` | Patch count for FID/PSNR/SSIM. |
| `VSLITE_FID_BATCH_SIZE` | `16`–`64` | FID feature batch size. |
| `VSLITE_CONVNEXT_FID_CKPT` | TCGA ConvNeXt path | FID backbone (test only). |
| `SAVE_TRIPLETS` | `0` | Set `1` on CISLNet test job to fill `input/`, `target/`, `pred/`. |
| `SKIP_WSI` | `1` | Skip WSI block in combined test jobs. |

## Example sweeps

**Patch size ablation (Restormer):**

```bash
for ps in 256 512; do
  export RUN=a13_ps${ps} PATCH_SIZE=$ps EPOCHS=20
  sbatch jobs/restormer/train/train.sh
done
```

**CISLNet longer training:**

```bash
export NUM_CHANNELS=13 RUN=c13 EPOCHS=50 EARLY_STOP_PATIENCE=8
sbatch jobs/cislnet/train/train.sh
```

**Linear RGB:**

```bash
export HE_COLOR_SPACE=linear RUN=a13_lin
sbatch jobs/restormer/attention/train/restormer_attn_13.sh
```

## Stack-specific overrides

Some jobs still read legacy env names; these are mapped in training code:

| Legacy env | Canonical |
|------------|-----------|
| `RESTORMER_BATCH_SIZE` | `BATCH_SIZE` |
| `RESTORMER_GRAD_ACCUM_STEPS` | `GRAD_ACCUM_STEPS` |
| `RESTORMER_OUTPUT_DIR` | `OUTPUT_DIR` |
| `SWINGAN_OUTPUT_DIR` | `OUTPUT_DIR` |
| `INPUT_NC` | `NUM_CHANNELS` (CISLNet) |

See [`FRAMEWORK.md`](../../FRAMEWORK.md) for output paths and the new-model checklist.
