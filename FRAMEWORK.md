# VSlite Benchmarking Framework

Authoritative spec for agents adding models, jobs, or ablations. See also [`AGENTS.md`](AGENTS.md) for cluster conventions.

## Principles

1. **One benchmark** — no `_i2i` / `_vs` suffixes on paths or TensorBoard tags.
2. **FID / PSNR / SSIM at test time only** — never during training loops.
3. **Every model** implements [`cislnet/train_interface.py`](cislnet/train_interface.py) and uses shared preview / TensorBoard / early-stopping utilities.
4. **Required ablations** — `num_channels`, `patch_size`, `batch_size`, `epochs` must be configurable via env or CLI for every stack.

## Model registry

| `model` tag | Train dir | Test dir | Entrypoint |
|-------------|-----------|----------|------------|
| `cislnet` | `results/cislnet/<run>/` | `results/cislnet/test/<run>/` | `scripts/train.py` |
| `swingan` | `results/swingan/<run>/` | `results/swingan/test/<run>/` | `SwinGAN/train.py` |
| `restormer` | `results/restormer/<run>/` | `results/restormer/test/<run>/` | `Restormer/train.py` |
| `nafnet` | `results/nafnet/<run>/` | `results/nafnet/test/<run>/` | `NAFNet/train.py` |
| `vqgan` | `results/vqgan/<run>/` | `results/vqgan/test/<run>/` | `scripts/train_vqgan.py` |
| `bbdm` | `results/bbdm/<run>/` | `results/bbdm/test/<run>/` | `scripts/train_bbdm.py` |
| `i2sb` | `results/i2sb/<run>/` | `results/i2sb/test/<run>/` | `scripts/train_i2sb.py` |

Unified TensorBoard (cross-model comparison):

```
results/unified/<model>/<run>/
```

Legacy symlinks (one release): `results/baseline` → `cislnet`, `results/*_i2i` → canonical names.

### Run tags

`{variant}{channels}` — variant `c` (CISLNet) | `a` (attention) | `b` (base) | `vq` | `i` (I2SB); channels `3|6|10|13`.

Optional suffix: `c13_ps512_ep30_wd1e4` (filesystem-safe).

## Jobs layout

```
jobs/
  _lib/
    vslite_env.sh           # VSLITE_ROOT, conda, paths
    slurm_defaults.sh       # partition, mem, GPU
    ablation.env.example    # all env knobs
  <model>/
    train/train.sh          # parameterized; sources _lib
    test/test.sh
    wsi/wsi.sh              # if applicable
```

SwinGAN/Restormer use `base/` (not `baseline/`) for the no-attention variant.

## Output layout (training)

```
results/<model>/<run>/
  ckpt_latest.pth | latest_net_G.pth
  ckpt_best.pth                 # early-stop winner
  samples/                      # epoch*_step* PNG grids
  preview_tiles.json            # optional locked val tiles
  loss_log.txt, opt.txt, iter.txt
  tensorboard/                  # per-run (optional)
  curves/                       # post-hoc from unified TB
```

## Output layout (test)

```
results/<model>/test/<run>/patch_eval/
  metrics.json, metrics.yaml
  input/ target/ pred/         # when --save_triplets
```

## Mandatory training interface

New models **must**:

1. Build [`VSliteTrainConfig`](cislnet/train_interface.py) and call `validate_train_config()`.
2. Log `meta/num_channels`, `meta/patch_size`, `meta/batch_size_effective` to unified TB.
3. Use [`util/preview_manager.py`](util/preview_manager.py) on **val** tiles (8 fixed).
4. Use [`util/tb_logger.py`](util/tb_logger.py) for scalars + `preview/fixed_val/grid` images (epoch step = slider).
5. Use [`util/early_stopping.py`](util/early_stopping.py) when `EARLY_STOP_PATIENCE > 0`.
6. Expose all ablation env vars from `jobs/_lib/ablation.env.example`.

Register in [`cislnet/registry.py`](cislnet/registry.py).

### New model checklist

- [ ] `VSliteTrainConfig` + validation
- [ ] `jobs/<model>/train/train.sh` sourcing `_lib/vslite_env.sh`
- [ ] Row in model registry table above
- [ ] 3-epoch smoke train → `results/unified/<model>/<run>/` has `l1/val` + preview images
- [ ] Test script writes `metrics.json` (FID test-only)

## Ablation environment variables

**Full guide:** [`jobs/_lib/ABLATIONS.md`](jobs/_lib/ABLATIONS.md)  
**Quick template:** [`jobs/_lib/ablation.env.example`](jobs/_lib/ablation.env.example)

All stacks must honor:

| Variable | Purpose |
|----------|---------|
| `NUM_CHANNELS` / `INPUT_NC` | 3, 6, 10, 13 |
| `PATCH_SIZE` / `IMG_SIZE` | Patch spatial size |
| `BATCH_SIZE` | Micro-batch |
| `GRAD_ACCUM_STEPS` | Effective batch multiplier |
| `EPOCHS` | Total epochs |
| `WEIGHT_DECAY`, `DROPOUT` | Regularization |
| `NO_AUG`, `AUG_STRENGTH` | Augmentation |
| `HE_COLOR_SPACE` | `srgb` or `linear` |
| `LAMBDA_L1`, `LAMBDA_NCE`, `LAMBDA_PERC`, `LAMBDA_GAN`, `NO_GAN` | Loss weights |
| `LR`, `LR_SCHEDULE`, `WARMUP_EPOCHS` | Optimizer schedule |
| `OPTIMIZER` | `adam` or `adamw` |
| `EMA_DECAY` | EMA when supported |
| `PROGRESSIVE_SCHEDULE` | e.g. `256:10,512:40` |
| `EARLY_STOP_PATIENCE`, `EARLY_STOP_MIN_DELTA` | Val L1 early stop |
| `PREVIEW_TILE_LIST` | JSON list of val patch ids |

**Excluded:** train-time FID; VGG perceptual (ConvNeXt only); encoder-init ablations.

## Perceptual loss naming

Use `convnext_*` flags only (`convnext_normalize_input`, `no_convnext_loss`). Legacy `vgg_*` CLI aliases may exist one release.

## Diffusion plug-in

BBDM and I2SB are reference diffusion integrations under `diffusion_vs/`. New diffusion stacks register in `cislnet/registry.py` and wrap runners with the same preview/TB/early-stop hooks as `scripts/train_bbdm.py`.
