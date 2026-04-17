# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


def load_swin_unet_pretrained(swin_unet: nn.Module, pretrained_path: str) -> dict:
    """
    Load a Swin(-Unet) checkpoint into SwinTransformerSys with best-effort key mapping.

    - Accepts checkpoints that are either:
      - a dict with key 'model' (common in Swin repos), or
      - a flat state_dict whose keys may be prefixed (e.g. 'module.' or 'swin_unet.').
    - Removes incompatible tensors by shape.
    - Returns the load_state_dict message dict (missing/unexpected keys).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(pretrained_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Strip common prefixes
    cleaned = {}
    for k, v in state.items():
        kk = k
        for pref in ("module.", "swin_unet.", "model."):
            if kk.startswith(pref):
                kk = kk[len(pref) :]
        cleaned[kk] = v

    model_dict = swin_unet.state_dict()
    full_dict = copy.deepcopy(cleaned)

    # Swin-Unet trick: mirror encoder layers -> decoder layers_up.*
    for k, v in cleaned.items():
        if "layers." in k:
            # key format: layers.{i_layer}.*
            try:
                i_layer = int(k.split(".", 2)[1])
            except Exception:
                continue
            current_layer_num = 3 - i_layer
            current_k = "layers_up." + str(current_layer_num) + k.split(".", 2)[2]
            full_dict[current_k] = v

    # Drop mismatched shapes
    for k in list(full_dict.keys()):
        if k in model_dict and hasattr(full_dict[k], "shape"):
            if full_dict[k].shape != model_dict[k].shape:
                del full_dict[k]

    msg = swin_unet.load_state_dict(full_dict, strict=False)
    logger.info(
        "Loaded pretrained Swin weights from %s (missing=%d unexpected=%d)",
        pretrained_path,
        len(msg.missing_keys),
        len(msg.unexpected_keys),
    )
    return {"missing_keys": msg.missing_keys, "unexpected_keys": msg.unexpected_keys}


class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,in_chans=1):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.img_size,
                                patch_size=config.patch_size,
                                in_chans=in_chans,
                                num_classes=self.num_classes,
                                embed_dim=config.embed_dim,
                                depths=config.depths,
                                num_heads=config.num_heads,
                                window_size=config.window_size,
                                mlp_ratio=config.mlp_ratio,
                                qkv_bias=config.qkv_bias,
                                qk_scale=config.qk_scale,
                                drop_rate=config.drop_rate,
                                drop_path_rate=config.drop_path_rate,
                                ape=config.ape,
                                patch_norm=config.patch_norm,
                                use_checkpoint=config.use_checkpoint)

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


class ProjectedSwinUnet(nn.Module):
    """
    Image-to-image SwinUnet with a learnable 1x1 projection to 3 channels before patch embedding.

    This enables:
    - input channels: 10 (IR) or 13 (IR+VIS)
    - keep Swin patch embedding unchanged (expects 3 channels)
    - load pretrained Swin weights for the backbone while projection stays randomly initialized
    """

    def __init__(self, config, in_chans: int, out_chans: int = 3, use_cbam: bool = False):
        super().__init__()
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.use_cbam = bool(use_cbam)

        self.proj = nn.Conv2d(self.in_chans, 3, kernel_size=1, stride=1, padding=0, bias=True)

        if self.use_cbam:
            # Lazy import to avoid circular deps; CBAM implementation will live in SwinGAN/loss_i2i.py or a small module.
            from .cbam import CBAM  # type: ignore

            self.cbam = CBAM(3)
        else:
            self.cbam = None

        self.swin_unet = SwinTransformerSys(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=3,
            num_classes=self.out_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
        )

    def forward(self, x):
        x = self.proj(x)
        if self.cbam is not None:
            x = self.cbam(x)
        y = self.swin_unet(x)
        return y

    def load_pretrained_backbone(self, pretrained_path: str) -> dict:
        """
        Load pretrained Swin weights into the backbone only.
        The 1x1 projection remains randomly initialized.
        """
        return load_swin_unet_pretrained(self.swin_unet, pretrained_path)

 