# Organ-agnostic vs organ-specific CISLNet (c13)

**As of 2026-08-25.** Patch-level holdout evals. Figures: [`results_figures_organ/`](./results_figures_organ/).

Related: [`results_c13.md`](./results_c13.md) (frozen-split epoch curves), [`results_inference.md`](./results_inference.md) (Vis+ / latency).

Compute is matched on **training iters** (optimizer steps: samples / batch 10). LPIPS (ConvNeXt-TCGA, Zhang lin0) is part of every holdout `metrics.json` on the matched-iter grid.

## Status

All five matched-iter trains and their holdout FID/PSNR/SSIM/LPIPS jobs are **done**.

| Run | Train length | Train | Holdout tests |
|-----|-------------:|-------|---------------|
| all samples | 30 | 9929216 → 9929217 (TIMEOUT then resume) | 9929218–9929236 |
| P+K+U | 59 | 9916408 → 9916409 | 9916433–9916492 (matched grid) |
| prostate only | 93 | 9929122 → 9929123 | 9929147–9929152 |
| kidney only | 282 | 9929124 → 9929125 (TIMEOUT then resume) | 9929153–9929158 |
| uterus only | 415 | 9929126 → 9929128 (TIMEOUT then resume) | 9929159–9929164 |

**Caveat — P+K+U at 106k / 133k iters:** `39_net_G.pth` and `49_net_G.pth` were never written. Those six test jobs still ran and scored a randomly initialized G (PSNR≈9, FID>100). They are **excluded** from figures and from the table below. P+K+U curves therefore have a gap between ~80k and ~159k iters.

Refresh figures with:

```bash
python scripts/plot_organ_exp_results.py
```

## Setup

CISLNet `contrastive_pix2pix`, 13-channel IR+VIS, AttUNet G, ConvNeXt NCE, 1024² patches, batch 10, lr 2e-4. Same architecture every run; only the **training pool** changes.

**Budget:** 30 passes on all samples = **1 594 800** samples = **159 480** training iters (`drop_last`, batch 10, 5316 iters/pass). Every other run trains until it has seen that many samples. Figure x-axes are **iters** (optimizer steps), so 159k is the shared budget.

| Training set | Train patches | Train length | Samples seen | Holdout organ | Holdout patches |
|--------------|--------------:|-------------:|-------------:|---------------|----------------:|
| all samples | 53 164 | 30 | 1 594 800 | prostate | 2379 |
| all samples | 53 164 | 30 | 1 594 800 | kidney | 807 |
| all samples | 53 164 | 30 | 1 594 800 | uterus | 538 |
| prostate, kidney, uterus | 26 944 | 59 | 1 589 600 | prostate | 2379 |
| prostate, kidney, uterus | 26 944 | 59 | 1 589 600 | kidney | 807 |
| prostate, kidney, uterus | 26 944 | 59 | 1 589 600 | uterus | 538 |
| prostate only | 17 183 | 93 | 1 597 770 | prostate | 2379 |
| kidney only | 5 653 | 282 | 1 594 260 | kidney | 807 |
| uterus only | 3 840 | 415 | 1 593 600 | uterus | 538 |

Holdout checkpoints sit on the same sample counts as all-samples 27k / 53k / 80k / 106k / 133k / 159k iters:

| iters | all samples | P+K+U | prostate only | kidney only | uterus only |
|------:|------------:|------:|--------------:|------------:|------------:|
| 27k | 5 | 10 | 15 | 47 | 69 |
| 53k | 10 | 20 | 31 | 94 | 138 |
| 80k | 15 | 30 | 46 | 141 | 208 |
| 106k | 20 | 39 ✗ | 62 | 188 | 277 |
| 133k | 25 | 49 ✗ | 77 | 235 | 346 |
| 159k | 30 | 59 | 93 | 282 | 415 |

✗ = checkpoint missing (not plotted). Numbers under each run are the saved `{N}_net_G.pth` index.

<img src="./results_figures_organ/fig_iter_budget.png" alt="Train length vs shared iter budget (~159k training iters)." />

Holdout **N** is the same column as in the metrics table below (prostate 2379, kidney 807, uterus 538).

**This holdout is not the canonical `dataset/data_frozen/test.txt`.** Those frozen-split IDs (`test.txt`: 4466, 10445, 4520; plus 10442, 3840) are **excluded from the entire organ-exp**. Holdouts are a **new** sample-level subset of the remaining Prostate / Kidney / Uterus slides, hardcoded in `HOLDOUTS` in `scripts/build_organ_exp_lists.py`. Seed 42 was only an offline search to pick which remaining samples hit ~11.6% patches (test/train ≈ 0.131). It is not used at train or test time.

FID is ConvNeXt-Tiny on TCGA/H&E features (not Inception). **LPIPS** is the same ConvNeXt-TCGA backbone with Zhang lin0 (equal-weight stages); it is a test metric only, not a training loss.

<img src="./results_figures_organ/fig_data_split.png" alt="Training pool sizes vs shared per-organ holdouts (2379 / 807 / 538). Every model is scored on each organ list separately, not the 3724-patch union." />

## Holdout results

X-axis on every curve is **training iters** (one optimizer step per batch of 10). Tick 159k is the shared sample budget, not 30 passes over a small organ list.

<img src="./results_figures_organ/fig_holdout_fid.png" alt="Holdout FID vs iters" />

<img src="./results_figures_organ/fig_holdout_lpips.png" alt="Holdout LPIPS ConvNeXt-TCGA vs iters" />

<img src="./results_figures_organ/fig_holdout_psnr_ssim.png" alt="Holdout PSNR and SSIM vs iters" />

### How to read the figures

1. **Compare runs at the same x**, not after the same number of passes over a small list. Uterus-only at 159k iters is ckpt 415, not ckpt 30.
2. **FID and LPIPS** are the metrics that should move. Pixel PSNR/SSIM often saturate early.
3. Organ-specific curves only appear on that organ’s panel (prostate-only on prostate, etc.).
4. P+K+U has no points at 106k / 133k iters (missing `39` / `49` checkpoints).

### Final budget (159k iters)

| Training set | Organ | Ckpt | PSNR | SSIM | FID ↓ | LPIPS ↓ |
|--------------|-------|-----:|-----:|-----:|------:|--------:|
| all samples | prostate | 30 | 18.085 | 0.4337 | 4.81 | 0.00190 |
| P+K+U | prostate | 59 | 18.036 | 0.4297 | 4.70 | 0.00190 |
| prostate only | prostate | 93 | **18.355** | **0.4522** | **3.92** | **0.00182** |
| all samples | kidney | 30 | 16.341 | 0.3463 | 12.24 | 0.00238 |
| P+K+U | kidney | 59 | 16.387 | 0.3374 | 12.03 | 0.00237 |
| kidney only | kidney | 282 | **16.466** | 0.3445 | **7.47** | 0.00239 |
| all samples | uterus | 30 | **16.817** | **0.3603** | 25.35 | **0.00224** |
| P+K+U | uterus | 59 | 16.406 | 0.3398 | 36.37 | 0.00233 |
| uterus only | uterus | 415 | 16.534 | 0.3268 | **10.50** | 0.00230 |

At matched iters, **organ-specific training wins FID on every organ holdout**. The gap is largest on uterus (OS 10.5 vs all-samples 25.4) and kidney (7.5 vs 12.2). Prostate is closer but still favors prostate-only (3.92 vs 4.81). LPIPS is nearly flat across protocols; all-samples is slightly better on uterus LPIPS despite much worse FID.

### Full holdout table

Matched-iter grid only (rows with a real `{N}_net_G.pth`). `grid` is the all-samples pass index (5…30); see the mapping table for training iters. Source: `results_figures_organ/holdout_metrics.csv`.

| Training set | Organ | grid | Ckpt | N | PSNR | SSIM | FID | LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all samples | kidney | 5.0 | 5 | 807 | 15.800 | 0.3249 | 12.76 | 0.0024 |
| all samples | kidney | 10.0 | 10 | 807 | 16.348 | 0.3503 | 6.64 | 0.0023 |
| all samples | kidney | 15.0 | 15 | 807 | 16.552 | 0.3500 | 7.70 | 0.0023 |
| all samples | kidney | 20.0 | 20 | 807 | 16.457 | 0.3572 | 11.90 | 0.0023 |
| all samples | kidney | 25.0 | 25 | 807 | 16.230 | 0.3407 | 12.48 | 0.0024 |
| all samples | kidney | 30.0 | 30 | 807 | 16.341 | 0.3463 | 12.24 | 0.0024 |
| all samples | prostate | 5.0 | 5 | 2379 | 17.240 | 0.4108 | 10.41 | 0.0020 |
| all samples | prostate | 10.0 | 10 | 2379 | 18.257 | 0.4456 | 4.04 | 0.0019 |
| all samples | prostate | 15.0 | 15 | 2379 | 18.360 | 0.4370 | 4.47 | 0.0018 |
| all samples | prostate | 20.0 | 20 | 2379 | 18.165 | 0.4464 | 4.20 | 0.0019 |
| all samples | prostate | 25.0 | 25 | 2379 | 17.986 | 0.4293 | 5.05 | 0.0019 |
| all samples | prostate | 30.0 | 30 | 2379 | 18.085 | 0.4337 | 4.81 | 0.0019 |
| all samples | uterus | 5.0 | 5 | 538 | 16.364 | 0.3550 | 57.94 | 0.0024 |
| all samples | uterus | 10.0 | 10 | 538 | 16.846 | 0.3657 | 24.25 | 0.0022 |
| all samples | uterus | 15.0 | 15 | 538 | 16.918 | 0.3614 | 28.84 | 0.0022 |
| all samples | uterus | 20.0 | 20 | 538 | 16.963 | 0.3745 | 28.45 | 0.0022 |
| all samples | uterus | 25.0 | 25 | 538 | 16.757 | 0.3547 | 24.27 | 0.0022 |
| all samples | uterus | 30.0 | 30 | 538 | 16.817 | 0.3603 | 25.35 | 0.0022 |
| P+K+U | kidney | 5.1 | 10 | 807 | 15.927 | 0.3212 | 16.57 | 0.0024 |
| P+K+U | kidney | 10.1 | 20 | 807 | 16.443 | 0.3513 | 9.84 | 0.0024 |
| P+K+U | kidney | 15.2 | 30 | 807 | 16.197 | 0.3370 | 9.09 | 0.0024 |
| P+K+U | kidney | 29.9 | 59 | 807 | 16.387 | 0.3374 | 12.03 | 0.0024 |
| P+K+U | prostate | 5.1 | 10 | 2379 | 17.806 | 0.4110 | 6.13 | 0.0019 |
| P+K+U | prostate | 10.1 | 20 | 2379 | 18.345 | 0.4522 | 5.75 | 0.0019 |
| P+K+U | prostate | 15.2 | 30 | 2379 | 18.000 | 0.4302 | 3.60 | 0.0019 |
| P+K+U | prostate | 29.9 | 59 | 2379 | 18.036 | 0.4297 | 4.70 | 0.0019 |
| P+K+U | uterus | 5.1 | 10 | 538 | 16.528 | 0.3421 | 32.68 | 0.0023 |
| P+K+U | uterus | 10.1 | 20 | 538 | 16.799 | 0.3579 | 33.11 | 0.0023 |
| P+K+U | uterus | 15.2 | 30 | 538 | 16.475 | 0.3423 | 19.15 | 0.0023 |
| P+K+U | uterus | 29.9 | 59 | 538 | 16.406 | 0.3398 | 36.37 | 0.0023 |
| prostate only | prostate | 4.8 | 15 | 2379 | 17.816 | 0.4203 | 6.28 | 0.0019 |
| prostate only | prostate | 10.0 | 31 | 2379 | 18.028 | 0.4336 | 6.01 | 0.0019 |
| prostate only | prostate | 14.9 | 46 | 2379 | 18.374 | 0.4400 | 5.41 | 0.0018 |
| prostate only | prostate | 20.0 | 62 | 2379 | 18.239 | 0.4440 | 4.92 | 0.0019 |
| prostate only | prostate | 24.9 | 77 | 2379 | 18.325 | 0.4418 | 3.64 | 0.0018 |
| prostate only | prostate | 30.1 | 93 | 2379 | 18.355 | 0.4522 | 3.92 | 0.0018 |
| kidney only | kidney | 5.0 | 47 | 807 | 16.108 | 0.3284 | 7.32 | 0.0024 |
| kidney only | kidney | 10.0 | 94 | 807 | 16.490 | 0.3565 | 6.04 | 0.0023 |
| kidney only | kidney | 15.0 | 141 | 807 | 16.720 | 0.3590 | 9.15 | 0.0023 |
| kidney only | kidney | 20.0 | 188 | 807 | 16.650 | 0.3547 | 7.94 | 0.0023 |
| kidney only | kidney | 25.0 | 235 | 807 | 16.620 | 0.3576 | 9.63 | 0.0023 |
| kidney only | kidney | 30.0 | 282 | 807 | 16.466 | 0.3445 | 7.47 | 0.0024 |
| uterus only | uterus | 5.0 | 69 | 538 | 16.253 | 0.3045 | 12.83 | 0.0024 |
| uterus only | uterus | 10.0 | 138 | 538 | 16.396 | 0.3088 | 9.74 | 0.0024 |
| uterus only | uterus | 15.0 | 208 | 538 | 16.288 | 0.3040 | 9.82 | 0.0024 |
| uterus only | uterus | 20.0 | 277 | 538 | 16.584 | 0.3200 | 8.17 | 0.0023 |
| uterus only | uterus | 25.0 | 346 | 538 | 16.523 | 0.3255 | 9.08 | 0.0023 |
| uterus only | uterus | 30.0 | 415 | 538 | 16.534 | 0.3268 | 10.50 | 0.0023 |

## Training dynamics

<img src="./results_figures_organ/fig_train_losses.png" alt="Training NCE loss vs iters" />

NCE is logged every `print_freq` samples. X-axis is training iters (optimizer steps). Curves are stitched across resume hops with a short moving average so the run-to-run shape is still visible.

<img src="./results_figures_organ/fig_epoch_l1.png" alt="Train/holdout L1 and LPIPS vs iters" />

TensorBoard `l1/*` and `lpips_convnext_tcga/*` are short reconstruction probes (not the holdout FID job). L1 is logged for the full run. LPIPS TB tags were only written after that logger was enabled, so those two panels begin later than L1 (P+K+U and prostate-only especially).

## Interpretation

With **matched iters**, organ-specific models beat both all-samples and P+K+U on **holdout FID** for their own organ. Pooling does not look beneficial once under-training of small pools is removed.

- **Uterus** is the clearest case: uterus-only FID ≈ 10.5 vs all-samples ≈ 25.4 at the same sample budget (n=538 still noisy, but the gap is large and stable across the OS curve).
- **Kidney** shows a similar pattern (OS ≈ 7.5 vs all ≈ 12.2).
- **Prostate** is closest; prostate-only still edges FID and LPIPS at the budget.
- **P+K+U** tracks all-samples more than organ-specific, and is worst on uterus FID at the final point. Missing mid-budget checkpoints limit the curve.
- **LPIPS** barely separates protocols; use FID (and visual checks) for model selection on this holdout.

Do not pick checkpoints with PSNR/SSIM alone.

**Caveats**

- Uterus n=538 makes FID noisier than prostate (n=2379).
- FID and LPIPS here are ConvNeXt/TCGA, not comparable to Inception-FID or official LPIPS-Alex/VGG elsewhere.
- No WSI-level scores in this writeup.
- P+K+U ckpts 39 and 49 were scheduled but never saved; those evals are invalid and omitted.
- Several trains hit the 3-day walltime and finished on an `afterany` resume hop.

## Reproduce

```bash
# lists (optional if dataset/organ_exp/ already exists)
python scripts/build_organ_exp_lists.py

# schedule used by submit + plots
python scripts/organ_exp_schedule.py

# trains (hop 2 = afterany self-resume) + holdout tests afterok hop 2
BUILD_LISTS=0 CONTINUE_TRAIN=true TRAIN_TIME=3-00:00:00 \
  ./jobs/cislnet/submit_organ_exp.sh

python scripts/plot_organ_exp_results.py
```
