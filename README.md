# VSlite: Virtual Staining Repository

VSlite is a specialized repository designed for translating multi-channel Infrared (IR) and optionally Visual (VIS) imaging modalities into synthetic H&E stains. The repository houses three distinct model architectures representing different generations and approaches to the virtual staining task:

1. **SwinGAN (Attention / Baseline)**
2. **Restormer (Transformers)**
3. **Contrastive Pix2Pix (Baseline)**

## Directory Structure

The repository is modularized based on architecture, dependencies, and functionality. All inactive, deprecated, or experimental profiling files have been moved to `legacy/`.

| Directory / File | Purpose |
|------------------|---------|
| **`SwinGAN/`** | Contains the full implementation of the SwinGAN architecture (Generator, Discriminator, configs, and training/testing loops: `train_i2i.py`, `test_i2i.py`, `loss_i2i.py`). |
| **`Restormer/`** | Contains the full PyTorch implementation of the Restormer architecture (`train_i2i_restormer.py`, `config_i2i_restormer.yaml`, `demo.py`). |
| **`NAFNet/`** | Contains the NAFNet i2i implementation (`train_i2i_nafnet.py`, `test_i2i_nafnet.py`, `config_i2i_nafnet_*.yaml`). |
| **`models/`** | Contains the PyTorch network definitions for the older `contrastive_pix2pix` and standard `pix2pix` models. |
| **`trainers/`** | Houses the training loops (`contrastive_pix2pix_trainer.py`) specifically for the Pix2Pix models. |
| **`data/`** | Dataset classes (`VS_dataset.py`, `base_dataset.py`) and augmentations for the Pix2Pix dataloaders. |
| **`dataset/`** | Local directory hosting the actual `.npy` data arrays and subsets (e.g., `data_frozen/`). |
| **`evaluation/`** | Contains calculation scripts for standard Image Translation metrics (`fid`, `psnr_ssim`, `group_evaluator`). |
| **`options/`** | `argparse` configuration classes (`train_options.py`, `test_options.py`) for the Pix2Pix baseline. |
| **`util/`** | Miscellaneous logging, HTML visualizers, and tensor transformation utilities. |
| **`jobs/`** | Repo-local job scripts (some workflows also use authoritative Slurm scripts under `/u/akshahi2/jobs/...`). |
| **`tests/`** | The automated Pytest suite validating dataset integrity, CLI configurations, and architecture block functionality. |
| **`tcga_tl_checkpoints/`** | Houses the locally downloaded, finetuned ConvNeXt weights used dynamically as a Perceptual Feature Extractor by SwinGAN and Restormer. |
| **`legacy/`** | **DEPRECATED**. Archived profiling scripts, old `final_code` clones, jupyter notebooks, and unused experiments (`DRB.py`, `memory_optimized` scripts). Files here are entirely disconnected from active training. |
| `scripts/train.py` | Training execution entrypoint for the **Pix2Pix** baseline. |
| `scripts/test.py` | Inference execution entrypoint for the **Pix2Pix** baseline. |
| `scripts/train_vqgan_vs.py` | Train VQGAN autoencoder on H&E targets (used by diffusion stacks). |
| `environment.yml` / `requirements.txt` | Python package dependency declarations. |
| `pytest.ini` | Pytest environment configuration settings. |

## Slurm job scripts (authoritative)

For cluster runs, the authoritative job scripts live under:

- **Home**: `/u/akshahi2/jobs/<family>/<variant>/<train|test>/*.sh`

These scripts call the repo entrypoints (below) and set environment variables (GPU ID, dataroot, output dir, and eval caps).

## Dataset layout + deterministic splitting (train/val/test)

All model families are intended to use the same dataset root:

- `dataset/data_frozen/ir/*.npy`
- `dataset/data_frozen/vis/*.npy` (optional; only needed for 13ch IR+VIS)
- `dataset/data_frozen/he/*.npy`

Deterministic split is defined by **sample IDs / prefixes**:

- `dataset/data_frozen/test.txt` (authoritative test split)
- `dataset/data_frozen/val.txt` (authoritative val split)

### Who reads `val.txt` / `test.txt`?

| Model family | How split is applied | Uses `val.txt` / `test.txt`? |
|---|---|---|
| **Baseline (contrastive_pix2pix)** | `data/VS_dataset.py` uses `opt.phase in {train,val,test}` and filters by `val.txt`/`test.txt` if present | **Yes** |
| **VQGAN / diffusion datasets** | `diffusion_vs/data.py` wraps `VS_dataset` and sets `opt.phase = split` | **Yes** |
| **Restormer / SwinGAN / NAFNet (i2i YAML)** | `SwinGAN/utils/dataset_vslite_i2i.py` auto-detects `val.txt`/`test.txt` when `split` is enabled in YAML | **Yes** |

### Important: enabling split for YAML-based i2i trainers

For Restormer/SwinGAN/NAFNet i2i trainers, splitting is enabled by setting a truthy `split` in the YAML config (e.g. `split: "train"`). Current configs are set to enable splitting:

- `Restormer/config_i2i_restormer.yaml`, `Restormer/config_i2i_restormer_perceptual.yaml`
- `SwinGAN/config_i2i_baseline.yaml`, `SwinGAN/config_i2i_attention.yaml`
- `NAFNet/config_i2i_nafnet_base.yaml`, `NAFNet/config_i2i_nafnet_attention.yaml`

## Centralized eval caps (consistent defaults across trainers)

Eval caps are centralized in `util/eval_caps.py` and can be overridden via environment variables.

| Cap type | Env var(s) | Used by | Default |
|---|---|---|---|
| **Items** | `UNIFIED_EVAL_MAX_ITEMS_TRAIN/VAL/TEST` | Restormer / SwinGAN / NAFNet i2i trainers | 512 / 512 / 512 |
| **Batches** | `UNIFIED_EVAL_MAX_BATCHES_TRAIN/VAL/TEST` | Baseline + VQGAN training | 20 / 50 / 50 |

If you want *full* evaluation, set caps to `0` (where supported) or to a very large number; otherwise, these caps limit epoch-end evaluation overhead.

## Train entrypoints (what jobs call)

Each architecture uses its own isolated scripts mapped via YAML configurations (or `argparse` for Pix2Pix).

### SwinGAN (YAML)
```bash
python SwinGAN/train_i2i.py --config SwinGAN/config_i2i_attention.yaml
```

### Restormer (YAML)
```bash
python Restormer/train_i2i_restormer.py --config Restormer/config_i2i_restormer_perceptual.yaml
```

### NAFNet (YAML)
```bash
python NAFNet/train_i2i_nafnet.py --config NAFNet/config_i2i_nafnet_base.yaml
```

### Pix2Pix Baseline (argparse)
```bash
python scripts/train.py --input_nc 10 --num_ir_channels 10 --name vs10
python scripts/train.py --input_nc 13 --num_ir_channels 10 --name vs13
```

### VQGAN (argparse, resumable)
```bash
python scripts/train_vqgan_vs.py --name vq_b --dataroot dataset/data_frozen --image_size 512 --amp --resume auto
```

## Train job “head-to-head” (default parameters)

These reflect the defaults when using the **training job scripts** in `/u/akshahi2/jobs/...` together with the repo configs.

| Model | Train job script | Input mode | Patch size used in training | Batch size | Grad accum | Effective batch | AMP | Perceptual / feature loss |
|---|---|---|---:|---:|---:|---:|---|---|
| **Baseline** | `/u/akshahi2/jobs/baseline/train/baseline_13.sh` | `irvis13` | `load_size=1024`, `crop_size=512` | 10 | 1 | 10 | (no explicit AMP) | ConvNeXt “VGG” is **off by default** unless `--no_vgg_loss false` |
| **SwinGAN baseline** | `/u/akshahi2/jobs/swingan/baseline/train/swingan_base_13.sh` | `irvis13` | `img_size=512` | 32 | 1 | 32 | (FP32) | none |
| **SwinGAN attention** | `/u/akshahi2/jobs/swingan/attention/train/swingan_attn_13.sh` | `irvis13` | `img_size=512` | 32 | 1 | 32 | (FP32) | ConvNeXt perceptual (`lambda_perceptual=10`) |
| **Restormer attention** | `/u/akshahi2/jobs/restormer/attention/train/restormer_attn_13.sh` | `irvis13` | `img_size=512` | 1 | `RESTORMER_GRAD_ACCUM_STEPS` (default 1) | 1 | **Yes** | ConvNeXt perceptual (`lambda_perceptual=10`) |
| **NAFNet base** | `/u/akshahi2/jobs/nafnet/base/train/nafnet_b_13.sh` | `irvis13` | `img_size=256` | 6 | 1 | 6 | **Yes** | none (`lambda_perceptual=0`) |
| **VQGAN (HE)** | `/u/akshahi2/jobs/vqgan/train/vqgan_he_train.sh` | targets only (HE) | `--image_size 512` | `VQGAN_BS` (default 1) | `VQGAN_ACCUM` (default 12) | 12 | **Yes** | LPIPS if enabled (`VQGAN_LPIPS`, default 1.0) |

## Test entrypoints (patch eval + metrics)

Preferred test scripts:

| Model family | Patch eval entrypoint |
|---|---|
| Baseline | `scripts/vs_infer_and_metrics.py` |
| SwinGAN | `SwinGAN/test_i2i.py` |
| Restormer | `Restormer/test_i2i_restormer.py` |
| NAFNet | `NAFNet/test_i2i_nafnet.py` |

## Results hygiene

- Old runs may be moved under `results_old/` when split logic or training regime changes, to avoid mixing leaky/non-leaky training artifacts.
- Unified TensorBoard event files live under `results/unified/<model>/<run>/events.out.tfevents.*` and may be deleted to reduce storage when runs are archived.

## Legacy Files Notice
All redundant dataloaders, old memory profiling sweeps, duplicate python scripts, and the previous `final_code/` workspace clone have been systematically retired to `/legacy/`. Nothing outside of `/legacy/` routes to it.
