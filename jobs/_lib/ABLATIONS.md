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
| `LAMBDA_L2` | CISLNet | L2 (MSE) pixel weight (default `1.0`; term off unless `USE_L2=1`). |
| `USE_L1` | CISLNet | `1` enable / `0` force-off pixel L1 (codebase default: **off**, `no_L1_loss=true`). |
| `USE_L2` | CISLNet | `1` enable / `0` force-off pixel L2 (MSE) (codebase default: **off**). |
| `USE_NCE` | CISLNet | `1` / `0` toggle contrastive NCE loss (codebase default: **on**, `lambda_NCE=10`). |
| `LAMBDA_NCE` | CISLNet | Contrastive / NCE weight. |
| `LAMBDA_PERC` | SwinGAN, Restormer (attn) | ConvNeXt perceptual loss (not VGG). |
| `LAMBDA_GAN` | CISLNet, SwinGAN | GAN weight (LSGAN on logits — least-squares / L2-form, `lambda_gan=1`). |
| `NO_GAN` | `0` | Set `1` to disable GAN term where supported. |
| `ATTENTION_TYPE` | CISLNet | Skip-attention for AttUnet: `none` = attention-free U-Net (default `cbam`). |

## Contrastive feature extractor (netF) — UNI ablation

The NCE/contrastive encoder (`--netF`) defaults to `vgg19`, which is the frozen
**ConvNeXt-Tiny TCGA** checkpoint. A gated **UNI v1** (MahmoodLab/UNI, ViT-L/16,
1024-d) alternative exists for CISLNet:

| Variable | Default | Effect |
|----------|---------|--------|
| `NETF` | *(unset)* | `uni` → `--netF uni` (UNI replaces ConvNeXt-TCGA as the frozen NCE feature extractor; maps from transformer blocks 2/5/8/11). |
| `UNI_CKPT` | `$VSLITE_ROOT/foundation_models/UNI/pytorch_model.bin` | Gated UNI v1 checkpoint path (`--uni_feature_ckpt`; also env `VSLITE_UNI_CKPT`). |
| `UNI_INPUT_SIZE` | `1024` | UNI operating resolution (multiple of 16). Position embeddings are interpolated from the canonical 224 grid at runtime; 1024 = full-res 64×64 tokens. |
| `UNI_GRAD_CHECKPOINT` | `true` | Checkpoint UNI blocks (needed for the 4096-token query path). |
| `VSLITE_UNI_LPIPS` | `0` | `1` → test jobs also compute the **`lpips_uni`** metric (Zhang lin0 on frozen UNI, blocks 2/5/8/11) alongside `lpips_convnext_tcga`. Explicit ckpt override: `VSLITE_UNI_LPIPS_CKPT`. |
| `UNI_BLOCKS` | `2,5,8,11` | Comma-separated ViT-L/16 block indices for netF=uni (`--uni_blocks`). Mid/late e.g. `4,10,16,22` for stronger structure/texture. |
| `NCE_T` | `0.07` | PatchNCE temperature (`--nce_t`). For UNI, try `0.10` / `0.15` (or `0.05` if sims are compressed). |
| `NUM_PATCHES` | `1024` | Patches sampled per NCE layer (`--num_patches`). At 1024² UNI (64×64=4096 tokens), 512–1024 is typical. |
| `NCE_DIAG` | `0` | `1` → log post-`mlp_sample` sim_pos / sim_neg (mean+hard) / margins / PatchNCE top-1 to `results/cislnet/<run>/nce_diag.jsonl` and print a summary. |
| `MAX_DATASET_SIZE` | *(unset)* | Cap train set size for short sanity runs (`--max_dataset_size`). |
| `PRINT_FREQ` | `100` | Train print/TB interval (samples). |

Canonical queue for the UNI run (train → patch eval → WSI → DZI → campus upload):

```bash
bash jobs/cislnet/submit_c13_uni.sh    # RUN=c13_uni, EPOCHS=20, NETF=uni
```

Diagnose PatchNCE separation, then queue the targeted `nce_t` / `UNI_BLOCKS` / `NUM_PATCHES` tune arms (afterok on diagnose):

```bash
bash jobs/cislnet/submit_c13_uni_nce_diag_then_tune.sh
# or stepwise:
#   bash jobs/cislnet/submit_c13_uni_nce_diag.sh
#   DIAG_JID=<jid> bash jobs/cislnet/submit_c13_uni_nce_tune.sh
```

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

Compare `l1/val` across `cislnet/c13`, `restormer/a13`, etc. Image slider: tags `preview/fixed_test/pXX` (epoch axis; left=pred, right=real).

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

## CISLNet architecture ablation groups (Exp 2)

Jobs under `jobs/cislnet/experiments/` write `TRACK_GROUP` into
`results/cislnet/experiments/tracking.csv`. Canonical meaning:

| Group | Meaning | Runs |
|-------|---------|------|
| **A** | Skip-connection **attention** | a1 CoordAtt, a2 Triplet, a3 SCSE, a4 GCBlock, **b4 ECA (=A5)**, **b5 SimAM (=A6)** |
| **B** | **Structure / capacity** | b1 half decoder, b2 SCConv, b3 ConvNeXt encoder, b6 depthwise, b7 PixelShuffle |
| **C** | Color space | c1 linear HE |
| **IO** | Data-path only (not a D knob) | fp16 on-disk IR |

Legacy run tags `b4_eca_*` / `b5_simam_*` stay for checkpoint continuity but are
**group A**. D-combine selection (`scripts/select_pareto_d1.py`) ranks by
**ConvNeXt-LPIPS** then latency and writes the top-K×top-K grid under
`results/cislnet/experiments/d_combos/`. Queue with:

```bash
bash jobs/cislnet/experiments/submit_d_combos.sh   # default TOP_K=2 → 4 jobs
```
