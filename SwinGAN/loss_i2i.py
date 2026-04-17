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
    In-house feature extractor (ConvNeXt) loaded from TCGA checkpoint.
    Replaces the previous torchvision VGG19.
    """
    def __init__(self, opt=None, pretrained=True, freeze_weights=True, before_relu=False):
        super().__init__()
        self.opt = opt

        import timm
        pretrained_model = timm.create_model('timm/convnext_tiny.in12k_ft_in1k')
        pretrained_model.head.fc = nn.Identity()

        # Load finetuned tcga weights if opt is provided
        if self.opt is not None:
            import os
            raw = getattr(self.opt, 'tcga_feature_ckpt', None)
            ckpt_path = str(raw).strip() if raw is not None else ''
            if not ckpt_path:
                raise ValueError('The feature encoder requires a TCGA checkpoint path (--tcga_feature_ckpt).')
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f'TCGA checkpoint not found: {ckpt_path}')
            
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if isinstance(ckpt, dict):
                for k in ('state_dict', 'model', 'net', 'weights'):
                    if k in ckpt and isinstance(ckpt[k], dict):
                        ckpt = ckpt[k]
                        break
            if isinstance(ckpt, dict):
                cleaned = {}
                for k, v in ckpt.items():
                    kk = k[7:] if k.startswith('module.') else k
                    cleaned[kk] = v
                ckpt = cleaned
            pretrained_model.load_state_dict(ckpt, strict=False)

        self.slice1 = torch.nn.Sequential(pretrained_model.stem)
        self.slice2 = torch.nn.Sequential(pretrained_model.stages[0])
        self.slice3 = torch.nn.Sequential(pretrained_model.stages[1])
        self.slice4 = torch.nn.Sequential(pretrained_model.stages[2])
        self.slice5 = torch.nn.Sequential(pretrained_model.stages[3])
        
        self.slices = nn.ModuleList([self.slice1, self.slice2, self.slice3, self.slice4, self.slice5])
        
        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False
                
        self.layer_weights = [1.0] * len(self.slices)

    def forward(self, x, y):
        # x,y in [-1,1] or [0,1] with 3 channels.
        loss = 0.0
        # Resize to 224x224 as ConvNeXt was likely trained on that resolution
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=True)
        
        for w, s in zip(self.layer_weights, self.slices):
            x = s(x)
            y = s(y)
            loss = loss + float(w) * F.l1_loss(x, y)
        return loss
