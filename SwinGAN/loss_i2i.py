import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """
    Non-saturating GAN loss using BCEWithLogits.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def d_loss(self, pred_real, pred_fake):
        real_t = torch.ones_like(pred_real)
        fake_t = torch.zeros_like(pred_fake)
        return 0.5 * (self.bce(pred_real, real_t) + self.bce(pred_fake, fake_t))

    def g_loss(self, pred_fake):
        real_t = torch.ones_like(pred_fake)
        return self.bce(pred_fake, real_t)


class ConvNeXtFeatureExtractor(nn.Module):
    """
    Standardized perceptual loss module.

    Despite the historical name ("FeatureExtractor"/"VGG"), this module is used as a
    **perceptual loss** between prediction and target by comparing ConvNeXt features.

    This implementation intentionally matches the repo's canonical ConvNeXt-based
    perceptual loss used by the pix2pix codepath:
    - loads the ConvNeXt backbone via `models.networks.architecture.ConvNeXtFeatureExtractor`
      (including optional ImageNet normalization controlled by `opt.vgg_normalize_input`)
    - computes a weighted L1 distance across multi-scale feature maps (and optionally the input)
    """
    def __init__(self, opt=None, pretrained=True, freeze_weights=True, before_relu=False):
        super().__init__()
        self.opt = opt
        # Import here to avoid SwinGAN import-time side effects in some environments.
        from models.networks.architecture import ConvNeXtFeatureExtractor as _Backbone

        self.backbone = _Backbone(
            opt=opt,
            pretrained=pretrained,
            freeze_weights=freeze_weights,
            before_relu=before_relu,
        )
        self.criterion = nn.L1Loss()

        # Match `models/networks/loss.py` defaults: [input, s1, s2, s3, s4, s5]
        # (weights are normalized to sum to 1).
        default_weights = torch.tensor(
            [1.0, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0],
            dtype=torch.float32,
        )
        self.register_buffer("weights", default_weights / default_weights.sum())
        self.start_idx = 0 if bool(
            getattr(opt, "convnext_loss_include_input", getattr(opt, "vgg_loss_include_input", False))
        ) else 1

    def forward(self, x, y):
        # Expected: x,y are 3ch images in [-1, 1] (same convention as the pix2pix codepath).
        x_feats = self.backbone(x)
        y_feats = self.backbone(y)
        x_feats.insert(0, x)
        y_feats.insert(0, y)

        loss = x.new_zeros(())
        for i in range(self.start_idx, len(x_feats)):
            loss = loss + self.weights[i] * self.criterion(x_feats[i], y_feats[i].detach())
        return loss
