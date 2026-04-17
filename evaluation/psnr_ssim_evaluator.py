"""Patch-level PSNR and SSIM vs ground-truth (test phase)."""

import numpy as np
import torch
from collections import OrderedDict
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .base_evaluator import BaseEvaluator


def _to_hwc_01(t):
    """Tensor (C,H,W) in approx [-1, 1] -> float array (H,W,C) in [0, 1]."""
    x = t.detach().cpu().float().numpy().transpose(1, 2, 0)
    x = (x + 1.0) * 0.5
    return np.clip(x, 0.0, 1.0)


class PsnrSsimEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            '--reconstruction_max_num_samples',
            type=int,
            default=500,
            help='Max patches to average for PSNR/SSIM (test set).',
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.psnrs = []
        self.ssims = []
        self.cur_idx = 0
        self.nsamples = 0

    def is_target_phase(self, phase):
        return phase == 'test'

    def prepare_evaluation(self, phase, dataloader, fn_model_forward, name):
        self.current_phase = phase
        self.psnrs = []
        self.ssims = []
        self.cur_idx = 0
        n = len(dataloader.dataset)
        cap = getattr(self.opt, 'reconstruction_max_num_samples', 500)
        self.nsamples = min(n, cap)
        print(
            'PSNR/SSIM: accumulating up to %d samples (dataset size %d)'
            % (self.nsamples, n)
        )

    def evaluate_current_batch(self, data, fn_model_forward, name):
        if self.cur_idx >= self.nsamples:
            return
        with torch.no_grad():
            out = fn_model_forward(data, 'forward')
            fake = out['fake_B']
        real = data.get('image', data['target'])
        if fake.device != real.device:
            real = real.to(fake.device)
        bs = min(fake.size(0), self.nsamples - self.cur_idx)
        for bi in range(bs):
            f = fake[bi]
            r = real[bi]
            fn = _to_hwc_01(f)
            rn = _to_hwc_01(r)
            psnr = peak_signal_noise_ratio(rn, fn, data_range=1.0)
            try:
                ssim = structural_similarity(
                    rn, fn, channel_axis=2, data_range=1.0
                )
            except TypeError:
                ssim = structural_similarity(
                    rn, fn, multichannel=True, data_range=1.0
                )
            self.psnrs.append(float(psnr))
            self.ssims.append(float(ssim))
        self.cur_idx += bs

    def should_stop_evaluation(self, num_cum_samples):
        return self.cur_idx >= self.nsamples

    def finish_evaluation(self, dataloader, fn_model_forward, name):
        out = OrderedDict()
        phase = getattr(self, 'current_phase', 'test')
        if not self.psnrs:
            return out
        out['PSNR_%s' % phase] = float(np.mean(self.psnrs))
        out['SSIM_%s' % phase] = float(np.mean(self.ssims))
        return out
