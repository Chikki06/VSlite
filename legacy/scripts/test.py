"""General-purpose test script for image-to-image translation.

This file contains the original repo-root `test.py` implementation, moved under `scripts/`.
"""

import json
import os
from collections import OrderedDict

import torch

import evaluation.group_evaluator as group_evaluator
import util.util as util
from data import create_dataloader
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import Visualizer

try:
    import yaml
except Exception:
    yaml = None

DATA_ROOT = os.getenv("DATA_ROOT", "data")
DATAROOTS = {
    "cityscapes": os.getenv("CITYSCAPES_DIR", os.path.join(DATA_ROOT, "Cityscapes")),
    "coco": os.getenv("COCO_DIR", os.path.join(DATA_ROOT, "CocoStuff")),
    "ade20k": os.getenv("ADE20K_DIR", os.path.join(DATA_ROOT, "ADEChallengeData2016")),
    "nyudepth": os.getenv("NYUDEPTH_DIR", os.path.join(DATA_ROOT, "NYUDepth")),
    "maps": os.getenv("MAPS_DIR", os.path.join(DATA_ROOT, "pix2pix_maps")),
    "gta": os.getenv("GTA_DIR", os.path.join(DATA_ROOT, "GTA")),
}


def evaluate(model, opt, train_dataset, test_dataset, web_dir):
    evaluator = group_evaluator.GroupEvaluator(opt)
    metrics = evaluator.evaluate(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        fn_model_forward=model.generate_visuals_for_evaluation,
    )
    mode = "EVAL" if opt.use_eval_mode else "TRAIN"
    message = "MODE: {} | ".format(mode)
    for k in sorted(list(metrics.keys())):
        v = metrics[k]
        message += "%s: %.3f " % (k, v)
    print(message)
    metric_log_path = os.path.join(web_dir, "metrics.txt")
    with open(metric_log_path, "a") as f:
        f.write(message + "\n")
    metric_log_path = os.path.join(web_dir, "metrics.json")
    with open(metric_log_path, "w") as f:
        json.dump(metrics, f)
    if yaml is not None:
        metric_log_path = os.path.join(web_dir, "metrics.yaml")
        with open(metric_log_path, "w") as f:
            yaml.safe_dump(metrics, f)
    metric_log_path = os.path.join(web_dir, "metrics_all.jsonl")
    metrics["mode"] = mode.lower()
    with open(metric_log_path, "a") as f:
        f.write(json.dumps(metrics, ensure_ascii=True) + "\n")
    metrics.pop("mode")
    return metrics


def test(
    name,
    opt=None,
    which_epoch="latest",
    num_test=float("inf"),
    phase="test",
    load_from_opt_file=True,
    evaluation_metrics="fid",
    results_dir="results",
    gpu_ids=[0],
    num_threads=1,
    batch_size=1,
    serial_batches=True,
    no_flip=True,
    no_flip_vert=True,
    no_rotate=True,
    fid_max_num_samples=50000,
    use_wandb=True,
    use_eval_mode=True,
    inception_weights="fid_inception",
    **kwargs,
):
    opt.wandb_job_type = "eval"
    opt.phase = phase
    opt.gpu_ids = gpu_ids
    opt.num_test = num_test
    opt.num_threads = num_threads
    opt.use_eval_mode = use_eval_mode
    opt.inception_weights = inception_weights
    opt.batch_size = batch_size
    opt.load_from_opt_file = load_from_opt_file
    opt.evaluation_metrics = evaluation_metrics
    opt.serial_batches = serial_batches
    opt.FID_max_num_samples = fid_max_num_samples
    opt.cache_dir = "evaluation/cache"
    opt.cityscapes_FCN_dataroot = os.getenv("CITYSCAPES_DIR", "/data/datasets/Cityscapes/")
    opt.results_dir = results_dir
    opt.use_wandb = use_wandb
    opt.no_flip = no_flip
    opt.no_flip_vert = no_flip_vert
    opt.no_rotate = no_rotate
    opt.which_epoch = which_epoch

    dataset = create_dataloader(opt)
    train_dataset = create_dataloader(util.copyconf(opt, phase="train"))

    model = create_model(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.which_epoch))

    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir,
        "Experiment = {}, Phase = {}, Epoch = {}".format(opt.name, opt.phase, opt.which_epoch),
    )

    if opt.use_eval_mode:
        model.eval()

    visualizer = Visualizer(opt)
    with torch.no_grad():
        metrics = evaluate(model, opt, train_dataset, dataset, web_dir)
        if visualizer.wandb_run is not None:
            for name, value in metrics.items():
                visualizer.wandb_run.summary[name] = value

        model.eval()
        for i, data_i in enumerate(dataset):
            if i * opt.batch_size >= opt.num_test:
                break

            device = (
                next(model.parameters()).device
                if hasattr(model, "parameters")
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            data_i = {k: v.to(device) if hasattr(v, "to") else v for k, v in data_i.items()}
            generated = model(data_i, mode="inference")
            visuals = OrderedDict(
                [
                    ("input_label", data_i["label"]),
                    ("synthesized_image", generated),
                    ("real_image", data_i["image"]),
                ]
            )
            visualizer.display_current_results(visuals, 0, i)

            img_path = data_i["path"]
            for b in range(generated.shape[0]):
                print("process image... %s" % img_path[b])
                visuals = OrderedDict(
                    [
                        ("input_label", data_i["label"][b]),
                        ("synthesized_image", generated[b]),
                        ("real_image", data_i["image"][b]),
                    ]
                )
                visualizer.save_images(webpage, visuals, img_path[b : b + 1])

        webpage.save()
        if visualizer.wandb_run is not None:
            visualizer.wandb_run.finish()


if __name__ == "__main__":
    opt = TestOptions().parse()
    test(opt.name, opt)

