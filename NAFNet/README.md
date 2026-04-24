## NAFNet i2i (VSlite)

This folder integrates a **NAFNet-style** backbone for VSlite patch i2i (**10ch IR** or **13ch IR+VIS** → **3ch HE**), following the same runner conventions as `Restormer/` and `SwinGAN/`.

### Variants
- **Base (`b`)**: `config_i2i_nafnet_base.yaml` (`use_cbam: false`)
- **Attention (`a`)**: `config_i2i_nafnet_attention.yaml` (`use_cbam: true` + `SwinGAN/loss_i2i.py:ConvNeXtFeatureExtractor` when `lambda_perceptual > 0`)

### Train / test / WSI infer
Run from this directory (Slurm jobs set `--chdir` here):

```bash
conda activate torchenv
export NAFNET_DATAROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite/dataset/data_frozen"
export NAFNET_OUTPUT_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/nafnet_i2i/b13"
export NAFNET_INPUT_MODE="irvis13"   # or ir10
python train_i2i_nafnet.py --config config_i2i_nafnet_base.yaml
```

```bash
python test_i2i_nafnet.py --config config_i2i_nafnet_base.yaml --ckpt "$NAFNET_OUTPUT_DIR/ckpt_latest.pth"
```

```bash
python infer_wsi_png.py --ckpt "$NAFNET_OUTPUT_DIR/ckpt_latest.pth" --config config_i2i_nafnet_base.yaml --model 1 --parent_folder /path/to/IR/parent --output_dir /path/to/out
```

### Slurm jobs
Canonical scripts live under `/u/akshahi2/jobs/nafnet/...` with convenience symlinks in `VSlite/jobs/`.

### Upstream note
Core blocks are adapted from `megvii-research/NAFNet` (`NAFBlock` + LayerNorm2d). The original restoration residual `output + input` is **not** used here because input/output channels differ for VSlite i2i.
