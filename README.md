Python 3.6

## VSlite: virtual staining baseline (this repo)

This targets **IR (+ optional VIS) → H&E** with `dataset_mode VS` and `model contrastive_pix2pix`.


| Path                                                                         | Purpose                                                                                                                            |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `[train.py](train.py)`, `[test.py](test.py)`                                 | Training and evaluation entry points                                                                                               |
| `[options/](options/)`                                                       | CLI defaults (`--dataroot` default `dataset/data_1024`, schedules, crops)                                                          |
| `[data/VS_dataset.py](data/VS_dataset.py)`                                   | Loads `dataroot/ir/*.npy` with paired `vis/` and `he/`; **10ch IR-only** vs **13ch IR+VIS** via `--input_nc` / `--num_ir_channels` |
| `[models/contrastive_pix2pix_model.py](models/contrastive_pix2pix_model.py)` | Losses and forward                                                                                                                 |
| `[models/networks/](models/networks/)`                                       | Generator (uses `--input_nc`), discriminator, encoders, NCE                                                                        |
| `[trainers/](trainers/)`                                                     | Contrastive pix2pix training steps                                                                                                 |
| `[evaluation/](evaluation/)`                                                 | `fid`, `psnr_ssim` (comma-separated `--evaluation_metrics`)                                                                        |
| `checkpoints/<name>/`                                                        | Weights; filenames use `util.checkpoint_filename` (legacy `*_net_*.pth` still load)                                                |
| `results/<name>/...`                                                         | PNGs from `test.py` plus `metrics.json`                                                                                            |
| `[legacy/](legacy/)`                                                         | TensorRT scripts, old `final_code` snapshot, unused dataset loaders, memory experiments                                            |


**Data layout:** `dataset/data_1024/` (or `--dataroot`) with subfolders `ir/`, `vis/`, `he/` and matching `.npy` names.

**Cluster jobs:** keep your own `train_job.sh` (or Slurm script) outside the repo; set `VS_DATAROOT` or `--dataroot` and run two baselines, e.g. `--input_nc 10` and `--input_nc 13 --num_ir_channels 10`.

**Evaluation:** e.g. `--evaluation_metrics fid,psnr_ssim` with `--reconstruction_max_num_samples` (see `--help`). Training debug `.npy` dumps use `train.py --debug` and `--display_freq`.

---

# Contrastive Feature Loss for Image Prediction

We provide a PyTorch implementation of our contrastive feature loss presented in:

[Contrastive Feature Loss for Image Prediction](https://arxiv.org/abs/2111.06934)
[Alex Andonian](https://www.alexandonian.com), [Taesung Park](https://taesung.me/), [Bryan Russell](https://bryanrussell.org/), [Phillip Isola](http://web.mit.edu/phillipi/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Richard Zhang](https://richzhang.github.io/)

Presented in AIM Workshop at ICCV 2021

## Prerequisites

- Linux or macOS
- Python 3.6+
- CPU or NVIDIA GPU + CUDA CuDNN

**Table of Contents:**

1. [Setup](#setup)
2. [Dataset Preprocessing](#dataset-preparation)
3. [Training](#training-models)
4. [Evaluating and Visualizing](#evaluatating-and-visualizing)

### Setup

- Clone this repo:

```bash
git clone https://github.com/alexandonian/contrastive-feature-loss.git
cd contrastive-feature-loss
```

- Create python virtual environment
- Install a recent version of [PyTorch](https://pytorch.org/get-started/locally/) and other dependencies specified below.

We highly recommend that you install additional dependencies in an isolated python virtual environment (of your choosing). For Conda+pip users, you can create a new conda environment and then pip install dependencies with the following snippet:

```bash
ENV_NAME=contrastive-feature-loss
conda create --name $ENV_NAME python=3.8
conda activate $ENV_NAME
pip install -r requirements.txt
```

Alternatively, you can create a new Conda environment in one command using `conda env create -f environment.yml`, followed by `conda activate contrastive-feature-loss` to activate the environment.

This code also requires the Synchronized-BatchNorm-PyTorch rep.

```bash
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

## Dataset Preparation

For COCO-Stuff, Cityscapes or ADE20K, the datasets must be downloaded beforehand. Please download them on the respective webpages. In the case of COCO-stuff, we put a few sample images in this code repo.

**Preparing COCO-Stuff Dataset**. The dataset can be downloaded [here](https://github.com/nightrome/cocostuff). In particular, you will need to download train2017.zip, val2017.zip, stuffthingmaps_trainval2017.zip, and annotations_trainval2017.zip. The images, labels, and instance maps should be arranged in the same directory structure as in `datasets/coco_stuff/`. In particular, we used an instance map that combines both the boundaries of "things instance map" and "stuff label map". To do this, we used a simple script `datasets/coco_generate_instance_map.py`. Please install `pycocotools` using `pip install pycocotools` and refer to the script to generate instance maps.

**Preparing ADE20K Dataset**. The dataset can be downloaded [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), which is from [MIT Scene Parsing BenchMark](http://sceneparsing.csail.mit.edu/). After unzipping the datgaset, put the jpg image files `ADEChallengeData2016/images/` and png label files `ADEChallengeData2016/annotatoins/` in the same directory.

There are different modes to load images by specifying `--preprocess_mode` along with `--load_size`. `--crop_size`. There are options such as `resize_and_crop`, which resizes the images into square images of side length `load_size` and randomly crops to `crop_size`. `scale_shortside_and_crop` scales the image to have a short side of length `load_size` and crops to `crop_size` x `crop_size` square. To see all modes, please use `python train.py --help` and take a look at `data/base_dataset.py`. By default at the training phase, the images are randomly flipped horizontally. To prevent this use `--no_flip`.

## Training Models

Models can be trained with the following steps.

1. Ensure you have prepared the dataset of interest using the instruction above. To train on the one of the datasets listed above, you can download the datasets and use `--dataset_mode` option, which will choose which subclass of `BaseDataset` is loaded. For custom datasets, the easiest way is to use `./data/custom_dataset.py` by specifying the option `--dataset_mode custom`, along with `--label_dir [path_to_labels] --image_dir [path_to_images]`. You also need to specify options such as `--label_nc` for the number of label classes in the dataset, `--contain_dontcare_label` to specify whether it has an unknown label, or `--no_instance` to denote the dataset doesn't have instance maps.
2. Run the training script with the following command:

```bash
# To train on the COCO dataset, for example.
python train.py --name [experiment_name] --dataset_mode coco --dataroot [path_to_coco_dataset]

# To train on your own custom dataset
python train.py --name [experiment_name] --dataset_mode custom --label_dir [path_to_labels] --image_dir [path_to_images] --label_nc [num_labels]
```

There are many options you can specify. Please use `python train.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `--gpu_ids`. If you want to use the second and third GPUs for example, use `--gpu_ids 1,2`.

The training logs are stored in json format in `[checkpoints_dir]/[experiment_name]/log.json`, with sample generations populated in `[checkpoint_dir]/[experiment_name]/web`.
We provide support for both Tensorboard (with `--tf_log`) and Weights & Biases (with `--use_wandb`) experiment tracking.

## Evaluating and Visualizing

In order to evaluate and visualize the generations of a trained model, run `test.py` in a similar manner, specifying the name of the experiment, the dataset and its path:

```bash
python test.py --name [name_of_experiment] --dataset_mode [dataset_mode] --dataroot [path_to_dataset]
```

where `[name_of_experiment]` is the directory name of the checkpoint created during training. If you are running on CPU mode, append `--gpu_ids -1`.

Use `--results_dir` to specify the output directory. `--num_test` will specify the maximum number of images to generate. By default, it loads the latest checkpoint. It can be changed using `--which_epoch`.

## Code Structure

- `train.py`, `test.py`: the entry point for training and testing.
- `trainers/contrastive_pix2pix_trainer.py`: harnesses and reports the progress of training of contrastive model.
- `models/contrastive_pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `models/networks/loss`: contains proposed PatchNCE loss
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading images and label maps.

## Options

This code repo contains many options. Some options belong to only one specific model, and some options have different default values depending on other options. To address this, the `BaseOption` class dynamically loads and sets options depending on what model, network, and datasets are used. This is done by calling the static method `modify_commandline_options` of various classes. It takes in the`parser` of `argparse` package and modifies the list of options. 



