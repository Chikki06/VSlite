import torch
from models import create_model
from models.networks.sync_batchnorm import DataParallelWithCallback
from torch.cuda.amp import autocast, GradScaler

class ContrastivePix2PixTrainerMP:
    """
    A trainer class for the pix2pix model that enables mixed precision training.
    """
    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = create_model(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(
                self.pix2pix_model, device_ids=opt.gpu_ids
            )
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            (
                self.optimizer_G,
                self.optimizer_D,
                self.optimizer_F,
            ) = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
            # 1. Initialize the gradient scaler
            self.scaler = GradScaler()

    def run_generator_one_step(self, data, iter=0):
        if self.optimizer_G is not None:
            self.optimizer_G.zero_grad()
        if self.optimizer_F is not None:
            self.optimizer_F.zero_grad()

        # 2. Use autocast for the forward pass (Generator)
        with autocast():
            g_losses, generated = self.pix2pix_model(data, mode='generator')
            g_loss = sum(g_losses.values()).mean()

        # 3. Scale the loss and call backward
        self.scaler.scale(g_loss).backward()

        # 4. Step optimizers with the scaler
        if iter % self.opt.num_F_steps_per_G == 0 and self.optimizer_G is not None:
            self.scaler.step(self.optimizer_G)
        if self.optimizer_F is not None:
            self.scaler.step(self.optimizer_F)
        
        # 5. Update the scaler
        self.scaler.update()

        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.d_losses = {}
        if self.optimizer_D is not None:
            self.optimizer_D.zero_grad()

            # 2. Use autocast for the forward pass (Discriminator)
            with autocast():
                d_losses = self.pix2pix_model(data, mode='discriminator')

            if d_losses:
                d_loss = sum(d_losses.values()).mean()
                # 3. Scale the loss and call backward
                self.scaler.scale(d_loss).backward()
                # 4. Step optimizer with the scaler
                self.scaler.step(self.optimizer_D)
                # 5. Update the scaler
                self.scaler.update()

            self.d_losses = d_losses

    def get_features(self, data):
        return self.model(data, mode='features')

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
