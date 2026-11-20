"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
from models.networks.normalization import SPADE
from torch.nn import Parameter as P

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        return x_s + dx

    def shortcut(self, x, seg):
        return self.conv_s(self.norm_s(x, seg)) if self.learned_shortcut else x

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
        )

    def forward(self, x):
        y = self.conv_block(x)
        return x + y


# # VGG architecter, used for the perceptual loss using a pretrained VGG network
# class ConvNeXtFeatureExtractor(torch.nn.Module):
#     def __init__(self, opt=None, pretrained=True, freeze_weights=True, before_relu=False):
#         super().__init__()
#         self.opt = opt
#         vgg_pretrained_features = torchvision.models.vgg19(pretrained=pretrained).features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         offset = 1 if before_relu else 0
#         for x in range(2 - offset):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7 - offset):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12 - offset):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21 - offset):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30 - offset):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if freeze_weights:
#             for param in self.parameters():
#                 param.requires_grad = False
#
#         self.mean = P(
#             torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1), requires_grad=False
#         )
#         self.std = P(
#             torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1), requires_grad=False
#         )
#
#     def forward(self, x):
#         if self.opt.vgg_normalize_input:
#             x = (x - self.mean) / self.std
#         if (x.shape[2] != 224 or x.shape[3] != 224) and self.opt.vgg_resize_input:
#             x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
#         h_relu1 = self.slice1(x)
#         h_relu2 = self.slice2(h_relu1)
#         h_relu3 = self.slice3(h_relu2)
#         h_relu4 = self.slice4(h_relu3)
#         h_relu5 = self.slice5(h_relu4)
#         return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


# In-house feature extractor; by default, always pretrained & after relu
class ConvNeXtFeatureExtractor(torch.nn.Module):
    def __init__(self, opt=None, pretrained=True, freeze_weights=True, before_relu=False):
        super().__init__()
        self.opt = opt

        import timm
        pretrained_model = timm.create_model('timm/convnext_tiny.in12k_ft_in1k')
        pretrained_model.head.fc = nn.Identity()

        # Feature encoder (ConvNeXt) is defined by the TCGA checkpoint; training with this module
        # always loads it when |opt| is provided (perceptual loss / vgg19 netF).
        if self.opt is not None:
            import os

            raw = getattr(self.opt, 'tcga_feature_ckpt', None)
            ckpt_path = str(raw).strip() if raw is not None else ''
            if not ckpt_path:
                raise ValueError(
                    'The feature encoder requires a TCGA checkpoint path (--tcga_feature_ckpt). '
                    'Empty or missing paths are not allowed when using this encoder.'
                )
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f'TCGA feature checkpoint not found (required for feature encoder): {ckpt_path}'
                )
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # Accept common formats: raw state_dict or dict wrappers.
            if isinstance(ckpt, dict):
                for k in ('state_dict', 'model', 'net', 'weights'):
                    if k in ckpt and isinstance(ckpt[k], dict):
                        ckpt = ckpt[k]
                        break
            # Strip prefixes if present (e.g., 'module.')
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

        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

        self.mean = P(
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1), requires_grad=False
        )
        self.std = P(
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1), requires_grad=False
        )

    def forward(self, x_org):
        if x_org.shape[-1] >= 5012:
            pad_height = ((x_org.shape[-2] + 223) // 224) * 224 - x_org.shape[-2]
            pad_width = ((x_org.shape[-1] + 223) // 224) * 224 - x_org.shape[-1]
            x_org_padded = torch.nn.functional.pad(x_org, (0, pad_width, 0, pad_height))
            patch_list = []
            i = 0
            while i + 224 <= x_org_padded.shape[-2]:
                j = 0
                while j + 224 <= x_org_padded.shape[-1]:
                    patch_list += [x_org_padded[:, :, i:i+224, j:j+224]]
                    j += 224
                i += 224
            # Add the downsampled x_org to the patch_list
            x_resized = F.interpolate(x_org, size=(224, 224), mode='bilinear', align_corners=True)
            patch_list += [x_resized]
            x = torch.cat(patch_list, dim=0)
        else:
            x = x_org



        
        if self.opt.vgg_normalize_input:
            x = x/2 + .5
            x = (x - self.mean) / self.std
       # if (x.shape[2] != 224 or x.shape[3] != 224) and self.opt.vgg_resize_input:
        #    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


_UNI_DEFAULT_CKPT_NAME = "UNI"
_UNI_DEFAULT_CKPT = "foundation_models/UNI/pytorch_model.bin"


def resolve_uni_feature_ckpt(ckpt_path=None, opt=None) -> str:
    """Resolve UNI v1 weights from an explicit path, ``opt.uni_feature_ckpt``,
    ``VSLITE_UNI_CKPT``, or the repo default ``foundation_models/UNI/pytorch_model.bin``."""
    import os
    from pathlib import Path

    candidates: list[str] = []
    if ckpt_path is not None and str(ckpt_path).strip():
        candidates.append(str(ckpt_path).strip())
    if opt is not None:
        raw = getattr(opt, "uni_feature_ckpt", None)
        if raw is not None and str(raw).strip():
            candidates.append(str(raw).strip())
    env = os.environ.get("VSLITE_UNI_CKPT", "").strip()
    if env:
        candidates.append(env)
    root = os.environ.get("VSLITE_ROOT", "").strip() or str(
        Path(__file__).resolve().parents[2]
    )
    candidates.append(os.path.join(root, _UNI_DEFAULT_CKPT))

    tried: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        if raw in seen:
            continue
        seen.add(raw)
        for p in (Path(raw),):
            tried.append(str(p))
            if p.is_file():
                return str(p)
            if p.is_dir():
                files = sorted(x for x in p.iterdir() if x.is_file())
                if len(files) == 1:
                    return str(files[0])
                for name in ("pytorch_model.bin", "model.safetensors"):
                    cand = p / name
                    if cand.is_file():
                        return str(cand)
    raise FileNotFoundError(
        "UNI feature checkpoint not found. Provide --uni_feature_ckpt or set "
        f"VSLITE_UNI_CKPT (expected {candidates[0] if candidates else _UNI_DEFAULT_CKPT}). "
        f"Tried: {tried}"
    )


class UNIFeatureExtractor(torch.nn.Module):
    """UNI v1 (MahmoodLab/UNI, ViT-L/16, 1024-d) patch-token feature maps.

    Drop-in counterpart of ``ConvNeXtFeatureExtractor`` for ``--netF uni``:
    returns a list of BCHW feature maps from intermediate transformer blocks
    (default blocks 2/5/8/11 — four maps, matching the four ConvNeXt stage maps
    produced by ``nce_layers='1,2,3,4,5'`` -> ``out[1:]``). CLS token is dropped;
    patch tokens are reshaped to ``(B, C, H', W')`` where ``H' = W' = size / 16``.

    The backbone is built at the canonical 224px grid and loads the checkpoint
    with ``strict=True``; timm's ``dynamic_img_size`` interpolates the learned
    position embeddings to the runtime grid (e.g. 1024 -> 64x64 tokens).

    Inputs arrive in ``[-1, 1]`` and are mapped to ``[0, 1]`` then ImageNet
    mean/std normalized (same contract as the ConvNeXt extractor).
    """

    UNI_BLOCKS = (2, 5, 8, 11)

    def __init__(
        self,
        opt=None,
        pretrained=True,
        freeze_weights=True,
        uni_input_size=None,
        uni_ckpt_path=None,
        grad_checkpoint=True,
        blocks=None,
    ):
        super().__init__()
        self.opt = opt
        import timm
        from torch.utils.checkpoint import checkpoint as _ckpt

        self._ckpt_fn = _ckpt

        if uni_input_size is not None:
            size = int(uni_input_size)
        elif opt is not None:
            size = int(getattr(opt, "uni_input_size", 1024))
        else:
            size = 1024
        if size % 16 != 0:
            raise ValueError(
                f"UNI feature extractor input size must be a multiple of 16, got {size}"
            )
        self.input_size = size

        self.take_blocks = tuple(
            blocks if blocks is not None else self.UNI_BLOCKS
        )
        n_blocks = 24  # ViT-L/16 depth
        if any(b < 0 or b >= n_blocks for b in self.take_blocks):
            raise ValueError(
                f"UNI blocks must be in [0, {n_blocks - 1}], got {self.take_blocks}"
            )

        # Canonical 224 construction: matches the UNI checkpoint exactly so the
        # state_dict loads with strict=True; runtime inputs at other sizes get
        # their position embeddings interpolated by timm (dynamic_img_size).
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        if pretrained:
            ckpt_path = resolve_uni_feature_ckpt(uni_ckpt_path, opt)
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict):
                for k in ("state_dict", "model", "net", "weights"):
                    if k in state and isinstance(state[k], dict):
                        state = state[k]
                        break
            if isinstance(state, dict):
                state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
            else:
                raise RuntimeError(f"Unexpected UNI checkpoint format: {type(state)}")
            model.load_state_dict(state, strict=True)
            self.ckpt_path = ckpt_path
        else:
            self.ckpt_path = None

        self.model = model
        self.grad_checkpoint = bool(grad_checkpoint)

        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
            persistent=False,
        )

    def forward(self, x_org):
        vit = self.model
        x = x_org.float()
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
            )
        x = x / 2 + 0.5
        x = (x - self.mean) / self.std

        x = vit.patch_embed(x)
        x = vit._pos_embed(x)
        x = vit.pos_drop(x)

        use_ckpt = self.grad_checkpoint and torch.is_grad_enabled() and x.requires_grad
        max_idx = max(self.take_blocks)
        feats = []
        for i, blk in enumerate(vit.blocks):
            if i > max_idx:
                break
            if use_ckpt:
                x = self._ckpt_fn(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i in self.take_blocks:
                B, N, C = x.shape
                grid = (self.input_size // 16) ** 2
                # ViT-L/16 carries a single CLS prefix token (no register tokens).
                tokens = x[:, 1:, :] if N - 1 == grid else x[:, -grid:, :]
                H = W = int(tokens.shape[1] ** 0.5)
                feats.append(
                    tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()
                )
        if len(feats) != len(self.take_blocks):
            raise RuntimeError(
                f"UNI feature extractor expected {len(self.take_blocks)} maps, got {len(feats)}"
            )
        return feats