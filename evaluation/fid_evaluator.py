###################################################
# Functions related to computing FID were borrowed from
# https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
###################################################
import json
import numpy as np
import torch
import os
from scipy import linalg
from .base_evaluator import BaseEvaluator
from util.i2i_metrics import _ConvNeXtTinyFID
from collections import OrderedDict

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def compute_mean_and_cov(acts):
    return acts.mean(axis=0), np.cov(acts, rowvar=False)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), 'Training and test mean vectors have different lengths'
    assert (
        sigma1.shape == sigma2.shape
    ), 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            'fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates'
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_activations(model, x):
    x = (x + 1.0) * 0.5  # to range [0, 1]
    pred = model(x)
    return pred.detach().cpu().numpy().reshape(pred.size(0), -1)


class FIDEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--FID_max_num_samples", default=75000, type=int)
        parser.add_argument(
            '--cache_dir', type=str, default=os.path.join(DIR_PATH, 'cache')
        )
        parser.add_argument(
            '--skip_fid_if_cached', action='store_true',
            help='Skip FID computation if cached metrics exist for this run (see evaluation/cache).',
        )
        parser.add_argument(
            '--skip_f_if_cached',
            dest='skip_fid_if_cached',
            action='store_true',
            help='Alias for --skip_fid_if_cached.',
        )
        parser.add_argument(
            '--fid_include_traintest',
            action='store_true',
            help='Also compute a combined FID over train+test activations ("traintest"). '
            'Disabled by default to keep evaluation to exactly train and test.',
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.all_acts = {}

    def is_target_phase(self, phase):
        return phase == "test"

    def cache_path(self, phase, dataset_mode=None):
        if dataset_mode is None:
            dataset_mode = self.opt.dataset_mode
        target_dir = os.path.expanduser(os.path.join(self.opt.cache_dir, dataset_mode))
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(
            target_dir,
            "stats_for_fid_{}_{}_{}.npz".format(
                phase, getattr(self, "fid_feature_name", "convnext_tiny_tcga"), self.opt.FID_max_num_samples
            ),
        )

    def fid_metrics_json_path(self, phase, dataset_mode=None):
        """Path to cached final FID numbers (used by --skip_fid_if_cached)."""
        if dataset_mode is None:
            dataset_mode = self.opt.dataset_mode
        target_dir = os.path.expanduser(os.path.join(self.opt.cache_dir, dataset_mode))
        os.makedirs(target_dir, exist_ok=True)
        name = getattr(self.opt, 'name', 'unknown')
        w_ep = getattr(self.opt, 'which_epoch', 'latest')
        return os.path.join(
            target_dir,
            'fid_metrics__{}__{}__{}__{}__{}__{}.json'.format(
                name,
                w_ep,
                dataset_mode,
                phase,
                getattr(self, "fid_feature_name", "convnext_tiny_tcga"),
                self.opt.FID_max_num_samples,
            ),
        )

    def find_cached_activations(self, phase):
        cache_path = self.cache_path(phase)
        if os.path.exists(cache_path):
            print("Finding cached activations at %s" % cache_path)
            cached_acts = np.load(cache_path)
            return cached_acts["mean"], cached_acts["cov"]
            # return cached_acts
        return None

    def cache_activations(self, mean, cov, phase):
        cache_path = self.cache_path(phase)
        np.savez(cache_path, mean=mean, cov=cov)
        print("Cache saved at %s for FID" % cache_path)

    def prepare_evaluation(self, phase, dataloader, fn_model_forward, name):
        self.current_phase = phase
        self._skip_fid = False
        self._cached_fid_metrics = None
        if getattr(self.opt, 'skip_fid_if_cached', False):
            cache_json = self.fid_metrics_json_path(phase)
            if os.path.isfile(cache_json):
                with open(cache_json, 'r') as f:
                    self._cached_fid_metrics = json.load(f)
                print(
                    'Skipping FID (loaded cached metrics from %s)' % cache_json
                )
                self._skip_fid = True
                return

        print("preparing for FID")
        print('Loading ConvNeXt feature network (for FID)')
        self.fid_feature_name = "convnext_tiny_tcga"
        self.model = _ConvNeXtTinyFID()
        self.model.cuda()
        self.model.eval()
        print('Loaded ConvNeXt feature network')
        self.need_real_stat = self.find_cached_activations("traintest") is None
        self.nsamples = min(len(dataloader), self.opt.FID_max_num_samples)
        feat_dim = 768
        if self.need_real_stat:
            self.real_acts = np.random.normal(size=(self.nsamples, feat_dim)) * 100
        self.fake_acts = np.random.normal(size=(self.nsamples, feat_dim)) + 100
        self.cur_idx = 0

        self.batch_size = dataloader.batch_size
        self.cur_target = None

    def evaluate_current_batch_bs1(self, data, fn_model_forward, name):
        AtoB = self.opt.direction == "AtoB"
        target = data["B" if AtoB else "A"].cuda()
        source = data["A" if AtoB else "B"].cuda()
        source_path = data["A_paths" if AtoB else "B_paths"]
        input = {
            "A": source,
            "B": source,
            "A_paths": source_path,
            "B_paths": source_path,
        }
        fake = fn_model_forward(input, "forward")["fake_B"]
        bs, C, H, W = target.size()
        begin = self.cur_idx
        end = self.cur_idx + 1
        self.fake_acts[begin:end] = get_activations(self.model, fake)[: end - begin]
        self.real_acts[begin:end] = get_activations(self.model, target)[: end - begin]
        self.cur_idx = self.cur_idx + 1

    def _evaluate_current_batch(self, data, fn_model_forward, name):
        AtoB = self.opt.direction == "AtoB"
        target = data["B" if AtoB else "A"]
        bs, C, H, W = target.size()

        if self.cur_target is None:
            self.cur_target = torch.zeros(self.batch_size, C, H, W, dtype=torch.float32)
            self.cur_source = torch.zeros(self.batch_size, C, H, W, dtype=torch.float32)

        cur_begin = self.cur_idx % self.batch_size
        cur_end = cur_begin + bs
        self.cur_target[cur_begin:cur_end] = target.cpu()
        self.cur_source[cur_begin:cur_end] = data["A" if AtoB else "B"].cpu()
        self.cur_idx = min(self.cur_idx + bs, self.nsamples)
        needs_activation = cur_end == self.batch_size or self.cur_idx == self.nsamples

        if needs_activation:
            begin = self.cur_idx - self.batch_size
            end = self.cur_idx

            source = self.cur_source.cuda()
            source_path = data["A_paths" if AtoB else "B_paths"]
            input = {
                "A": source,
                "B": source,
                "A_paths": source_path,
                "B_paths": source_path,
            }
            fake = fn_model_forward(input, "forward")["fake_B"]
            self.fake_acts[begin:end] = get_activations(self.model, fake)[: end - begin]
            if self.need_real_stat:
                cur_target = self.cur_target.cuda()
                cur_real_acts = get_activations(self.model, cur_target)[: end - begin]
                self.real_acts[begin:end] = cur_real_acts

    def evaluate_current_batch(self, data, fn_model_forward, name):
        if getattr(self, '_skip_fid', False):
            return
        # AtoB = self.opt.direction == "AtoB"
        target = data['image']
        bs, C, H, W = target.size()

        if self.cur_target is None:
            self.cur_target = torch.zeros(self.batch_size, C, H, W, dtype=torch.float32)
            self.cur_source = torch.zeros(self.batch_size, C, H, W, dtype=torch.float32)

        cur_begin = self.cur_idx % self.batch_size
        cur_end = cur_begin + bs
        self.cur_target[cur_begin:cur_end] = target.cpu()
        src = data["label"].cpu()
        # FID expects RGB-like tensors (C==3). When the model input has >3 channels
        # (e.g. 10 IR or 13 IR+VIS), down-project to 3 channels for bookkeeping.
        # This keeps FID evaluation from crashing; it does not change the model forward path.
        if src.dim() == 3:
            src = src.unsqueeze(0)
        if src.size(1) != self.cur_source.size(1):
            if src.size(1) >= self.cur_source.size(1):
                src = src[:, : self.cur_source.size(1)]
            else:
                pad = self.cur_source.size(1) - src.size(1)
                src = torch.cat([src, src.new_zeros(src.size(0), pad, src.size(2), src.size(3))], dim=1)
        self.cur_source[cur_begin:cur_end] = src
        self.cur_idx = min(self.cur_idx + bs, self.nsamples)
        needs_activation = cur_end == self.batch_size or self.cur_idx == self.nsamples

        if needs_activation:
            begin = self.cur_idx - self.batch_size
            end = self.cur_idx

            # source = self.cur_source.cuda()
            # source_path = data["A_paths" if AtoB else "B_paths"]
            # Ensure model forward runs on GPU when available.
            if torch.cuda.is_available():
                data_cuda = {}
                for k, v in data.items():
                    if torch.is_tensor(v):
                        data_cuda[k] = v.cuda(non_blocking=True)
                    else:
                        data_cuda[k] = v
            else:
                data_cuda = data

            fake = fn_model_forward(data_cuda, "forward")["fake_B"]
            self.fake_acts[begin:end] = get_activations(self.model, fake)[: end - begin]
            if self.need_real_stat:
                cur_target = self.cur_target.cuda()
                cur_real_acts = get_activations(self.model, cur_target)[: end - begin]
                self.real_acts[begin:end] = cur_real_acts

    def should_stop_evaluation(self, num_cum_samples):
        if getattr(self, '_skip_fid', False):
            return True
        return self.cur_idx >= self.nsamples

    def remember_current_activations(self):
        if self.need_real_stat:
            real_acts = self.real_acts[: self.cur_idx]
            real_mean, real_cov = compute_mean_and_cov(real_acts)
            self.all_acts[self.current_phase + "real"] = real_acts
            self.cache_activations(real_mean, real_cov, self.current_phase)

            if "trainreal" in self.all_acts and "testreal" in self.all_acts:
                all_real_acts = np.concatenate(
                    (self.all_acts["trainreal"], self.all_acts["testreal"]), axis=0
                )
                real_mean, real_cov = compute_mean_and_cov(all_real_acts)
                self.cache_activations(real_mean, real_cov, "traintest")

        fake_acts = self.fake_acts[: self.cur_idx]
        self.all_acts[self.current_phase + "fake"] = fake_acts

    def finish_evaluation(self, dataloader, fn_model_forward, name):
        if getattr(self, '_skip_fid', False) and self._cached_fid_metrics is not None:
            return OrderedDict(self._cached_fid_metrics)

        self.remember_current_activations()

        fids = OrderedDict()

        fake_mean, fake_cov = compute_mean_and_cov(
            self.all_acts[self.current_phase + "fake"]
        )
        real_mean, real_cov = self.find_cached_activations(self.current_phase)
        fid = calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
        fids['FID%s%d' % (self.current_phase, self.cur_idx)] = fid

        if (
            getattr(self.opt, 'fid_include_traintest', False)
            and "trainfake" in self.all_acts
            and "testfake" in self.all_acts
        ):
            all_fake_acts = np.concatenate(
                (self.all_acts["trainfake"], self.all_acts["testfake"]), axis=0
            )
            fake_mean, fake_cov = compute_mean_and_cov(all_fake_acts)
            real_mean, real_cov = self.find_cached_activations("traintest")
            fid = calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
            fids['FID%s' % "traintest"] = fid

        cache_json = self.fid_metrics_json_path(self.current_phase)
        try:
            with open(cache_json, 'w') as f:
                json.dump(dict(fids), f, indent=2)
            print('Wrote FID metrics cache to %s' % cache_json)
        except OSError as e:
            print('WARNING: could not write FID metrics cache: %s' % e)

        return fids
