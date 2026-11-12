"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from models import create_model
from models.networks.sync_batchnorm import DataParallelWithCallback
from util.batch_device import move_to_device


class Pix2PixTrainer:
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = create_model(opt)
        n_gpu = len(opt.gpu_ids)
        self._use_dp = n_gpu > 1
        if self._use_dp:
            self.pix2pix_model = DataParallelWithCallback(
                self.pix2pix_model, device_ids=opt.gpu_ids
            )
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
            self.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
        elif n_gpu == 1:
            self.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
            self.pix2pix_model = self.pix2pix_model.to(self.device)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model
        else:
            self.device = torch.device("cpu")
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            (
                self.optimizer_G,
                self.optimizer_D,
            ) = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def _prepare_data(self, data):
        # DataParallel scatters CPU (pinned) batches; only single-GPU needs explicit H2D.
        if self._use_dp or self.device.type == "cpu":
            return data
        return move_to_device(data, self.device, non_blocking=True)

    def run_generator_one_step(self, data, iter=None):
        data = self._prepare_data(data)
        self.optimizer_G.zero_grad(set_to_none=True)
        g_losses, generated = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data, iter=None):
        self.d_losses = {}
        if self.optimizer_D is not None:
            data = self._prepare_data(data)
            if not self._use_dp and self.generated is not None:
                data = {**data, 'fake_image': self.generated}
            self.optimizer_D.zero_grad(set_to_none=True)
            d_losses = self.pix2pix_model(data, mode='discriminator')
            if d_losses:
                d_loss = sum(d_losses.values()).mean()
                d_loss.backward()
                self.optimizer_D.step()
            self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    @property
    def model(self):
        return self.pix2pix_model_on_one_gpu

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
