# CISLNet ablations: attention-free U-Net & loss objectives

**As of 2026-11-10.** One experiment family — **Exp 4: no-attn + objective sweep** — on the **canonical frozen test** (`dataset/data_frozen`, **n=6929**). This is part of the inference-ablation series in `[results_inference.md](./results_inference.md)` (Exp 1 = Vis+ channels, Exp 2 = architecture A/B/D, Exp 3 = VIS-only + capacity ladder); those docs' figures live in `results_figures_inference/`, this doc's in `[results_figures/](./results_figures/)`.

Two questions this family answers:

1. **Does the U-Net need skip attention at all?** Exp 2 swapped attention *types* (A1–A6) but never tested the zero-attention endpoint. `c13_noattn` = attention-free U-Net (`ATTENTION_TYPE=none`), objective unchanged.
2. **What is each loss term actually doing?** The CISLNet objective is LSGAN + contrastive NCE @10 (pixel L1/L2 are **off** by default). The ablations isolate each term and try pixel losses in its place: GAN-only, NCE-only, L2-only, and a GAN+L2 weight ladder.

Primary quality metric: **LPIPS** (ConvNeXt-TCGA, Zhang lin0).

Related: `[results_organ.md](./results_organ.md)` (organ holdouts), `[results_c13.md](./results_c13.md)` (c13 epoch curves).

---

## Setup

Chain: `jobs/cislnet/submit_c13_noattn_objective_ablation.sh` — train hop1→hop2 (`afterany`, `CONTINUE_TRAIN=true`) → patch eval (`afterok`, NUM_TEST=75000) → WSI → DZI → campus-viewer upload. All runs are 13ch, 20 epochs, batch 10, 1024², frozen split; eval = `latest` epoch (20). Reference = canonical `c13` (best-LPIPS epoch 24 promoted to `latest`, LPIPS 0.001819 / FID 5.84).

| Run | Change (env) | Objective | Status |
| --- | --- | --- | --- |
| **c13** (ref) | — | LSGAN + NCE @10 | ✓ (prior, promoted ckpt) |
| c13_noattn | `ATTENTION_TYPE=none` | LSGAN + NCE @10 (unchanged) | ✓ |
| c13_gan | `USE_NCE=0` | LSGAN only | ✓ |
| c13_ganl2_01 | `USE_NCE=0, USE_L2=1, LAMBDA_L2=0.1` | LSGAN + L2 @0.1 | ✓ |
| c13_ganl2_10 | `USE_NCE=0, USE_L2=1, LAMBDA_L2=10` | LSGAN + L2 @10 | ✓ |
| c13_nce | `NO_GAN=1` | NCE only @10 | ✓ |
| c13_l2 | `USE_NCE=0, USE_L2=1, LAMBDA_L2=1.0, NO_GAN=1` | L2 only @1.0 | ✓ |

All six trains completed 20/20 epochs; patch eval n=6929 on the frozen test for every run; WSI legs finished (10/10 slides each, uploaded to the `campus_ablations_bestlpips` viewer as `<run>-latest`).

---

## Results (frozen test, n=6929)

| Run | Objective | LPIPS (CN) ↓ | Δ vs c13 | FID ↓ | PSNR ↑ | SSIM ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| **c13** | LSGAN + NCE @10 | 0.001819 | — | **5.84** | 18.09 | 0.466 |
| c13_noattn | LSGAN + NCE @10, no attention | 0.001911 | +5.0% | 7.36 | 17.89 | 0.455 |
| c13_gan | LSGAN only | 0.002446 | +34.5% | 14.57 | 15.40 | 0.341 |
| c13_ganl2_01 | LSGAN + L2 @0.1 | 0.002231 | +22.7% | 13.38 | 16.38 | 0.380 |
| c13_ganl2_10 | LSGAN + L2 @10 | 0.002025 | +11.4% | 12.00 | 17.17 | 0.455 |
| c13_nce | NCE only @10 | 0.001917 | +5.4% | 98.55 | 16.12 | 0.235 |
| c13_l2 | L2 only @1 | **0.001763** | **−3.0%** | 82.06 | **18.84** | **0.554** |

Source: `results/cislnet/test/{c13,c13_noattn,c13_gan,c13_ganl2_01,c13_ganl2_10,c13_nce,c13_l2}/patch_eval_test/metrics.json` (CSV: `results_figures/objective_metrics.csv`).

![LPIPS — no-attn / objective ablations](./results_figures/fig_obj_lpips.png)

## Inference time

Method: identical to Exp 2/3 (`util/exp_metrics.py`: CUDA-synced `perf_counter`, **10 warmup + 100 timed forwards**, dummy `randn(1,13,1024,1024)`, batch=1, fp32, A100).

- The five objective ablations are **weight-for-weight identical to c13** (only the loss changed) — there is nothing to time: their latency **is** c13's by construction, and the prior `vs13` tracking row gives **46.59 ms ± 0.06**.
- **`c13_noattn` is the only architecture change in this family** and the only run that needs a fresh timing. Job 12316621 (`jobs/cislnet/experiments/latency_obj.sh`, completed 2026-11-11) timed **c13 (same-lane baseline) + c13_noattn** on one node. The fresh c13 number (46.63 ± 0.06) confirms the prior lane (46.59 ± 0.06); it lands in `results/cislnet/experiments/latency_obj/test/<run>/patch_eval_test/metrics.json` — deliberately *not* appended to `tracking.csv`, so `plot_capacity_results.py`'s 46.6 ms baseline stays untouched.

| Run | Latency (ms / 1024² patch, batch=1) | Speedup vs c13 |
| --- | --- | --- |
| c13 (ref) | 46.59 (prior lane); fresh same-lane 46.63 ± 0.06 | 1.00× |
| c13_noattn | **24.47 ± 0.05** | **1.91×** |
| c13_gan / c13_ganl2_01 / c13_ganl2_10 / c13_nce / c13_l2 | 46.59 (same arch as c13, not re-timed) | 1.00× |

Removing CBAM cuts per-patch inference time by **47%** (46.63 → 24.47 ms).

## Qualitative results

Qualitative assessment is on the **full-size WSI outputs**, already visible in the campus viewer (no same-patch patch-ladder figure needed). Each run's WSI leg finished and uploaded **10 held-out test slides** (`infer → DZI → upload`, `10/10 ok`) into project `campus_ablations_bestlpips` (display name *Campus_ablations_best_LPIPS_WSI*), one output type per run:

| Run | Viewer output type |
| --- | --- |
| c13_noattn | `c13_noattn-latest` |
| c13_gan | `c13_gan-latest` |
| c13_ganl2_01 | `c13_ganl2_01-latest` |
| c13_ganl2_10 | `c13_ganl2_10-latest` |
| c13_nce | `c13_nce-latest` |
| c13_l2 | `c13_l2-latest` |

Local artifacts: `results/cislnet/test/<run>/selected_epoch.txt` (=`latest`); WSI PNGs went straight to the remote DZI viewer (`results/cislnet/remote_scratch/campus_ablations_bestlpips/` scratch, no local `wsi_pred/`). Expect from the metrics: `c13_noattn` ≈ c13 visually; `c13_gan*` color/texture artifacts; `c13_nce`/`c13_l2` realistic-tissue loss (FID 98.5 / 82) despite fine pixel metrics.

---

## Takeaways

1. **Attention-free U-Net is a real Pareto point, not a free win.** Removing CBAM entirely (`c13_noattn`) costs +5.0% LPIPS (0.001911 vs 0.001819) and +1.5 FID — a visible but modest toll, in line with the group-A attention swaps — and buys a **1.91× speedup** (46.63 → 24.47 ms/1024² patch, −47% latency). It is the family's only architectural lever.
2. **The objective matters far more than any single architecture knob tested here.** GAN-only is catastrophic (LPIPS +34%, FID 14.6, PSNR −2.7 dB): the adversarial term alone cannot hold structure or color.
3. **The contrastive NCE term does most of the perceptual work — and nothing tested here replaces it.** NCE-only nearly matches c13 LPIPS (0.001917 vs 0.001819) but collapses on FID (98.5) and SSIM (0.235): structure-like texture, wrong tone. The GAN+L2 ladder heals monotonically (L2 @0.1 → @10: LPIPS 0.002231 → 0.002025, FID 13.4 → 12.0) but even L2 @10 stays 11% worse than c13 — pixel losses do not substitute for NCE at these weights.
4. **L2-only is the pixel-metric trap.** Best LPIPS of the entire family (0.001763, −3.0% vs c13), best PSNR/SSIM by a wide margin — and FID 82: regression to the mean buys pixel metrics and destroys realism. Do not quote pixel-only wins from this row.
5. **Net:** keep the full objective (LSGAN + NCE @10). The family's practical output is the attention-free variant — the latency job confirmed a 1.91× speedup at +5% LPIPS, making it a genuine deployment Pareto point when throughput matters.

---

## Reproduce

```bash
# train + patch eval + WSI chains (6 runs)
bash jobs/cislnet/submit_c13_noattn_objective_ablation.sh
RUN_FILTER=c13_noattn,c13_l2 bash jobs/cislnet/submit_c13_noattn_objective_ablation.sh  # subset

# latency re-timing (7 runs, one node)
sbatch jobs/cislnet/experiments/latency_obj.sh

# figures
python scripts/plot_noattn_objective_results.py
```

Plain c13 patch leaf: `NUM_CHANNELS=13 RUN=c13 NAME=c13 sbatch jobs/cislnet/test/test.sh`.
