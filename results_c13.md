# Default CISLNet on the frozen all-organ split (c13_e35)

**As of 2026-08-13.** Patch-level eval on the canonical frozen test set. Figures: [`results_figures_c13/`](./results_figures_c13/).

## Status

**0 / 18** planned even-epoch evals have `metrics.json` so far (epochs none).

Train job writes numbered `{epoch}_net_G.pth` every 2 epochs (and the final epoch 35). Eval is **post-training**: one sweep job scores each saved checkpoint on `dataset/data_frozen/test.txt`.

## Setup

CISLNet `contrastive_pix2pix`, 13-channel IR+VIS, AttUNet G, ConvNeXt NCE, 1024² patches, **35 epochs**, lr 2e-4, batch 10, `save_epoch_freq=2`. Training pool is the default frozen split (no organ-exp allowlist).

| Split | Source | Patches |
|-------|--------|--------:|
| train | all `ir/*.npy` except `test.txt` | 60,047 |
| **test** | `test.txt` (`4466`, `10445`, `4520`, `10442`, `3840`) | **6,929** |

There is **no val split**. Former `val.txt` samples (`10442`, `3840`) were moved into `test.txt` because classic CISLNet never used val for checkpoint selection. `val.txt` is kept present-but-empty so the dataset loader does not hash-split train into a fake val set.

This **is** the canonical `dataset/data_frozen` split, not the organ-exp holdouts in [`results.md`](./results.md). FID and LPIPS both use ConvNeXt-Tiny on TCGA/H&E features (FID = Fréchet distance; LPIPS = Zhang lin0 equal-weight stages — not AlexNet LPIPS, not Inception-FID).

<img src="./results_figures_c13/fig_data_split.png" alt="Frozen split patch counts: 60047 train, 6929 test." />

## Test results

<img src="./results_figures_c13/fig_fid.png" alt="Canonical test FID vs training epoch" />

FID is the metric that moves. Pixel metrics are flatter across checkpoints. LPIPS is the ConvNeXt-TCGA perceptual distance logged in `metrics.json` as `lpips_convnext_tcga_mean`.

<img src="./results_figures_c13/fig_psnr_ssim.png" alt="Canonical test PSNR, SSIM, and LPIPS vs epoch" />

### Full epoch table

_No `metrics.json` yet — train/eval still running._

**Best FID over tested epochs**

_Pending evals._

## Training dynamics

_Training NCE figure not available yet (`log.json` missing)._

NCE is the training signal logged every `print_freq` steps. Epoch axis uses train-set size 60,047 patches.

_Epoch-end L1 figure not available yet (TensorBoard scalars missing)._

TensorBoard `l1/train` and `l1/test` are short reconstruction probes at each epoch end (not the FID job). Pixel L1 does **not** automatically pick the best FID checkpoint.

_Epoch-end LPIPS figure not available yet (TensorBoard scalars missing)._

TensorBoard `lpips_convnext_tcga/train` and `lpips_convnext_tcga/test` are the same capped epoch-end probes using the frozen ConvNeXt-TCGA Zhang-lin0 metric. Lower is better. These are diagnostics, not a training loss.

## Interpretation

Interpretation will be filled in once the post-train epoch sweep writes `metrics.json`.

**Caveats**

- Eval N is 6,929 patches (`test.txt` only). FID and LPIPS here are ConvNeXt/TCGA, not comparable to Inception-FID or official AlexNet LPIPS.
- Numbered checkpoints include G, D, and F_q (`{epoch}_net_{G,D,F_q}.pth`); eval loads G only.
- No WSI-level scores in this writeup.

## Reproduce

```bash
# queue 35-epoch train + post-train even-epoch eval (already submitted once)
./jobs/cislnet/submit_c13_epoch_eval.sh

# regenerate figures + this file after metrics exist
python scripts/plot_c13_epoch_eval.py --run c13_e35
```
