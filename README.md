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
| **`models/`** | Contains the PyTorch network definitions for the older `contrastive_pix2pix` and standard `pix2pix` models. |
| **`trainers/`** | Houses the training loops (`contrastive_pix2pix_trainer.py`) specifically for the Pix2Pix models. |
| **`data/`** | Dataset classes (`VS_dataset.py`, `base_dataset.py`) and augmentations for the Pix2Pix dataloaders. |
| **`dataset/`** | Local directory hosting the actual `.npy` data arrays and subsets (e.g., `data_frozen/`). |
| **`evaluation/`** | Contains calculation scripts for standard Image Translation metrics (`fid`, `psnr_ssim`, `group_evaluator`). |
| **`options/`** | `argparse` configuration classes (`train_options.py`, `test_options.py`) for the Pix2Pix baseline. |
| **`util/`** | Miscellaneous logging, HTML visualizers, and tensor transformation utilities. |
| **`jobs/`** | Includes Slurm environment configs and remote execution logic (`remote_db_upsert.py`). Standard job scripts from the user's home directory are symlinked here. |
| **`tests/`** | The automated Pytest suite validating dataset integrity, CLI configurations, and architecture block functionality. |
| **`tcga_tl_checkpoints/`** | Houses the locally downloaded, finetuned ConvNeXt weights used dynamically as a Perceptual Feature Extractor by SwinGAN and Restormer. |
| **`legacy/`** | **DEPRECATED**. Archived profiling scripts, old `final_code` clones, jupyter notebooks, and unused experiments (`DRB.py`, `memory_optimized` scripts). Files here are entirely disconnected from active training. |
| `scripts/train.py` | Training execution entrypoint for the **Pix2Pix** baseline. |
| `scripts/test.py` | Inference execution entrypoint for the **Pix2Pix** baseline. |
| `environment.yml` / `requirements.txt` | Python package dependency declarations. |
| `pytest.ini` | Pytest environment configuration settings. |

## Quick Start & Usage

Each architecture uses its own isolated scripts mapped via YAML configurations (or `argparse` for Pix2Pix).

### SwinGAN
```bash
python SwinGAN/train_i2i.py --config SwinGAN/config_i2i_attention.yaml
```

### Restormer
```bash
python Restormer/train_i2i_restormer.py --config Restormer/config_i2i_restormer_perceptual.yaml
```

### Pix2Pix Baseline (10 and 13 channel)
```bash
python scripts/train.py --input_nc 10 --num_ir_channels 10 --name vs10
python scripts/train.py --input_nc 13 --num_ir_channels 10 --name vs13
```

## Legacy Files Notice
All redundant dataloaders, old memory profiling sweeps, duplicate python scripts, and the previous `final_code/` workspace clone have been systematically retired to `/legacy/`. Nothing outside of `/legacy/` routes to it.
