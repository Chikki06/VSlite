"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
    
    
    
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import collections
import pdb

from typing import Dict
from collections import OrderedDict

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models import resnet50


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return x, out
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def _copy_trunc_conv(dst: nn.Conv2d, src: nn.Conv2d) -> None:
    """Copy the leading (out, in) filter block of a pretrained conv into a narrower one."""
    with torch.no_grad():
        dst.weight.copy_(src.weight.data[:dst.out_channels, :dst.in_channels])
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias.data[:dst.out_channels])


def _copy_trunc_bn(dst, src) -> None:
    with torch.no_grad():
        n = int(dst.num_features)
        if getattr(dst, 'affine', False) and getattr(src, 'affine', False):
            dst.weight.copy_(src.weight.data[:n])
            dst.bias.copy_(src.bias.data[:n])
        if getattr(dst, 'track_running_stats', False) and getattr(src, 'track_running_stats', False):
            dst.running_mean.copy_(src.running_mean[:n])
            dst.running_var.copy_(src.running_var[:n])


def _slim_resnet50_width(backbone, mult: float) -> None:
    """Scale every resnet50 backbone kernel by `mult` (e.g. 0.5 = half kernels).

    Rebuilds layer1..4 Bottlenecks with scaled `planes`, truncating the
    ImageNet-pretrained weights to the leading filter block (same init trick as
    the convnext stem above). Residual shapes stay consistent because both the
    main branch and the downsample are scaled together. Operates in place;
    child names (conv1/layerN) are preserved for IntermediateLayerGetter.
    """
    from torchvision.models.resnet import Bottleneck

    def _scaled(n: int) -> int:
        return max(1, int(round(n * mult)))

    # conv1 was already rebuilt with scaled outputs in _build_encoder; narrow
    # its companion bn1 to match (otherwise 32ch in vs 64ch stats at forward).
    if int(backbone.bn1.num_features) != int(backbone.conv1.out_channels):
        new_bn1 = nn.BatchNorm2d(int(backbone.conv1.out_channels))
        _copy_trunc_bn(new_bn1, backbone.bn1)
        backbone.bn1 = new_bn1

    inplanes = int(backbone.conv1.out_channels)
    for lname in ('layer1', 'layer2', 'layer3', 'layer4'):
        layer = getattr(backbone, lname)
        new_blocks = []
        for block in layer:
            assert isinstance(block, Bottleneck), f'{lname}: expected Bottleneck, got {type(block)}'
            planes_s = _scaled(int(block.conv1.out_channels))
            stride = block.conv2.stride
            _dil = block.conv2.dilation
            dilation = int(_dil[0] if isinstance(_dil, (tuple, list)) else _dil)
            _str = stride
            stride = int(_str[0] if isinstance(_str, (tuple, list)) else _str)
            nb = Bottleneck(
                inplanes, planes_s, stride=stride, downsample=None,
                groups=int(block.conv2.groups), base_width=64,
                dilation=dilation, norm_layer=nn.BatchNorm2d,
            )
            _copy_trunc_conv(nb.conv1, block.conv1)
            _copy_trunc_bn(nb.bn1, block.bn1)
            _copy_trunc_conv(nb.conv2, block.conv2)
            _copy_trunc_bn(nb.bn2, block.bn2)
            _copy_trunc_conv(nb.conv3, block.conv3)
            _copy_trunc_bn(nb.bn3, block.bn3)
            if block.downsample is not None:
                ds_conv_src, ds_bn_src = block.downsample[0], block.downsample[1]
                ds_conv = nn.Conv2d(
                    inplanes, planes_s * int(block.expansion), kernel_size=1,
                    stride=stride, bias=False,
                )
                ds_bn = nn.BatchNorm2d(planes_s * int(block.expansion))
                _copy_trunc_conv(ds_conv, ds_conv_src)
                _copy_trunc_bn(ds_bn, ds_bn_src)
                nb.downsample = nn.Sequential(ds_conv, ds_bn)
            new_blocks.append(nb)
            inplanes = planes_s * int(block.expansion)
        setattr(backbone, lname, nn.Sequential(*new_blocks))


class AttUnetGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            '--attention_type',
            type=str,
            default='cbam',
            choices=['none', 'cbam', 'coordatt', 'triplet', 'scse', 'gcblock', 'eca', 'simam'],
            help='Skip-connection attention for AttUnet (default cbam = baseline). '
                 'none = attention-free U-Net (same topology, no attention gates). '
                 'eca/simam match build_skip_attention (B4/B5 ablations).',
        )
        parser.add_argument(
            '--encoder_backbone',
            type=str,
            default='resnet50',
            choices=['resnet50', 'convnext_tiny'],
            help='Encoder backbone for AttUnet (default resnet50 = baseline).',
        )
        parser.add_argument(
            '--decoder_conv',
            type=str,
            default='standard',
            choices=['standard', 'scconv', 'depthwise'],
            help='Decoder conv type inside Up.conv (default standard = baseline). '
                 'depthwise = depthwise 3x3 + pointwise 1x1 in place of each 3x3.',
        )
        parser.add_argument(
            '--decoder_width_mult',
            type=float,
            default=1.0,
            help='Scale decoder/attention channels relative to encoder skips (1.0 = baseline).',
        )
        parser.add_argument(
            '--encoder_width_mult',
            type=float,
            default=1.0,
            help='Scale resnet50 encoder kernels (all convs incl. skips/bottleneck). '
                 '1.0 = baseline full-width encoder; 0.5 = half kernels in the encoder too.',
        )
        parser.add_argument(
            '--upsample_mode',
            type=str,
            default='convtranspose',
            choices=['convtranspose', 'pixelshuffle'],
            help='Upsample op for AttUnet final_upsample only (default convtranspose = baseline). '
                 'pixelshuffle = Conv2d + PixelShuffle(2); interior Up blocks stay ConvTranspose2d.',
        )
        return parser

    def __init__(self, opt, norm_layer=nn.BatchNorm2d, fpn_feature='decoder'):
        super(AttUnetGenerator, self).__init__()
        from models.networks.attention_blocks import build_skip_attention

        assert fpn_feature == 'encoder' or fpn_feature == 'decoder'
        self.fpn_feature = fpn_feature
        self.opt = opt
        self.output_nc = 3
        self.attention_type = str(getattr(opt, 'attention_type', 'cbam')).lower()
        self.encoder_backbone = str(getattr(opt, 'encoder_backbone', 'resnet50')).lower()
        self.decoder_conv = str(getattr(opt, 'decoder_conv', 'standard')).lower()
        self.upsample_mode = str(getattr(opt, 'upsample_mode', 'convtranspose')).lower()
        width_mult = float(getattr(opt, 'decoder_width_mult', 1.0) or 1.0)
        if width_mult <= 0:
            raise ValueError(f'decoder_width_mult must be > 0, got {width_mult}')
        self.decoder_width_mult = width_mult
        enc_mult = float(getattr(opt, 'encoder_width_mult', 1.0) or 1.0)
        if enc_mult <= 0:
            raise ValueError(f'encoder_width_mult must be > 0, got {enc_mult}')
        self.encoder_width_mult = enc_mult
        if self.upsample_mode not in ('convtranspose', 'pixelshuffle'):
            raise ValueError(
                f"Unknown upsample_mode={self.upsample_mode!r}. "
                "Expected convtranspose|pixelshuffle."
            )

        input_nc = getattr(opt, 'input_nc', 13)
        self.skip_keys, self.enc_channels = self._build_encoder(input_nc)
        # enc_channels: [bottleneck, skip_deep, ..., skip_shallow]
        self.n_downsampling = len(self.enc_channels)
        self.ngf = int(round(self.enc_channels[-1] * width_mult))

        # Optional 1x1 projectors when decoder width != encoder skip width.
        self.use_proj = abs(width_mult - 1.0) > 1e-8
        if self.use_proj:
            for i, c_enc in enumerate(self.enc_channels):
                c_dec = max(1, int(round(c_enc * width_mult)))
                setattr(self, f'proj_{i}', nn.Conv2d(c_enc, c_dec, kernel_size=1, bias=False))

        def _dec_ch(enc_c: int) -> int:
            return max(1, int(round(enc_c * width_mult)))

        # Attention on bottleneck + each skip (same count as enc_channels).
        for i, c_enc in enumerate(self.enc_channels):
            c_attn = _dec_ch(c_enc)
            attn = build_skip_attention(self.attention_type, c_attn, cbam_cls=CBAM)
            name = f'cbam_{i}' if self.attention_type == 'cbam' else f'attn_{i}'
            setattr(self, name, attn)

        # Upsample blocks: n_downsampling - 1
        for i in range(self.n_downsampling - 1):
            in_c = _dec_ch(self.enc_channels[i])
            # Last Up lands on shallowest encoder channel (scaled).
            if i == self.n_downsampling - 2:
                out_c = _dec_ch(self.enc_channels[-1])
            else:
                out_c = _dec_ch(self.enc_channels[i + 1])
            setattr(
                self,
                f'upsample_{i}',
                Up(in_c, out_c, norm_layer=norm_layer, decoder_conv=self.decoder_conv),
            )

        self.final_upsample = self._build_final_upsample(norm_layer)

    def _build_final_upsample(self, norm_layer):
        """Restore full resolution from post-decoder map.

        ResNet AttUnet: bottleneck H/32, 4 Ups → H/2, one final ×2 → H.
        ConvNeXt (4 maps): bottleneck H/32, 3 Ups → H/4, need two final ×2 → H.
        """
        n_ups = self.n_downsampling - 1
        # Assume encoder bottleneck stride 32 (ResNet50 / ConvNeXt features_only).
        remaining_doubles = max(1, 5 - n_ups)
        mid = self.ngf
        layers = []
        for i in range(remaining_doubles):
            out_c = max(1, mid // 2) if i == remaining_doubles - 1 else mid
            if self.upsample_mode == 'pixelshuffle':
                # Sub-pixel ×2: Conv to 4·C then PixelShuffle(2). No output_padding ambiguity.
                layers += [
                    nn.Conv2d(mid, out_c * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    norm_layer(out_c),
                    nn.ReLU(True),
                ]
            else:
                layers += [
                    nn.ConvTranspose2d(
                        mid, out_c, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    norm_layer(out_c),
                    nn.ReLU(True),
                ]
            mid = out_c
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(mid, self.output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        return nn.Sequential(*layers)

    def _build_encoder(self, input_nc: int):
        if self.encoder_backbone in ('resnet50', 'resnet'):
            enc_mult = float(getattr(self, 'encoder_width_mult', 1.0))
            backbone = resnet50(pretrained=True)
            c1_out = max(1, int(round(64 * enc_mult)))
            new_conv1 = nn.Conv2d(input_nc, c1_out, kernel_size=7, stride=2, padding=3, bias=False)
            backbone.conv1 = new_conv1
            if abs(enc_mult - 1.0) > 1e-8:
                _slim_resnet50_width(backbone, enc_mult)
            self.encoder = IntermediateLayerGetter(
                backbone,
                return_layers={
                    'conv1': 'feat',
                    'layer1': 'feat0',
                    'layer2': 'feat1',
                    'layer3': 'feat2',
                    'layer4': 'feat3',
                },
            )
            # Bottleneck + skips deep→shallow (matches historical CBAM/Up schedule).
            skip_keys = ['feat2', 'feat1', 'feat0', 'feat']
            enc_channels = [max(1, int(round(c * enc_mult))) for c in (2048, 1024, 512, 256, 64)]
            return skip_keys, enc_channels

        if self.encoder_backbone in ('convnext_tiny', 'convnext'):
            if abs(float(getattr(self, 'encoder_width_mult', 1.0)) - 1.0) > 1e-8:
                raise ValueError('encoder_width_mult != 1.0 is only implemented for resnet50 backbone')
            import timm

            backbone = timm.create_model(
                'convnext_tiny.in12k_ft_in1k',
                pretrained=True,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            # Replace the first Conv2d (stem) for multi-channel IR(+VIS) input.
            first_name, first_mod = None, None
            for name, mod in backbone.named_modules():
                if isinstance(mod, nn.Conv2d):
                    first_name, first_mod = name, mod
                    break
            if first_name is None or first_mod is None:
                raise RuntimeError('convnext_tiny: could not find stem Conv2d')
            new_conv = nn.Conv2d(
                input_nc,
                first_mod.out_channels,
                kernel_size=first_mod.kernel_size,
                stride=first_mod.stride,
                padding=first_mod.padding,
                bias=first_mod.bias is not None,
            )
            with torch.no_grad():
                if first_mod.weight.shape[1] == 3 and input_nc != 3:
                    w = first_mod.weight.data.mean(dim=1, keepdim=True).repeat(1, input_nc, 1, 1)
                    new_conv.weight.copy_(w * (3.0 / float(input_nc)))
                elif first_mod.weight.shape[1] == input_nc:
                    new_conv.weight.copy_(first_mod.weight.data)
                if first_mod.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(first_mod.bias.data)
            parent = backbone
            parts = first_name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)

            chans = list(backbone.feature_info.channels())
            if len(chans) != 4:
                raise RuntimeError(f'convnext_tiny features_only expected 4 maps, got {chans}')
            self.encoder = _ConvNeXtFeaturesEncoder(backbone)
            # bottleneck + skips deep→shallow: [768, 384, 192, 96]
            enc_channels = list(reversed(chans))
            # Ups consume feat1 (384), feat0 (192), feat (96)
            skip_keys = ['feat1', 'feat0', 'feat']
            return skip_keys, enc_channels

        raise ValueError(
            f"Unknown encoder_backbone={self.encoder_backbone!r}. "
            "Expected resnet50|convnext_tiny."
        )

    def _get_attn(self, i: int):
        if self.attention_type == 'cbam':
            return getattr(self, f'cbam_{i}')
        return getattr(self, f'attn_{i}')

    def _project(self, i: int, x: torch.Tensor) -> torch.Tensor:
        if self.use_proj:
            return getattr(self, f'proj_{i}')(x)
        return x

    def forward(self, input):
        x, features = self.encoder(input)
        x = self._get_attn(0)(self._project(0, x))
        for i in range(self.n_downsampling - 1):
            model = getattr(self, f'upsample_{i}')
            feature_key = self.skip_keys[i]
            skip = features[feature_key]
            skip = self._get_attn(i + 1)(self._project(i + 1, skip))
            x = model(x, skip)
            if self.fpn_feature == 'decoder':
                features[feature_key] = x
            elif self.fpn_feature == 'encoder':
                features[feature_key] = skip
        x = self.final_upsample(x)
        if 'feat' in features:
            del features['feat']
        return x


class _ConvNeXtFeaturesEncoder(nn.Module):
    """Wrap timm features_only ConvNeXt to AttUnet (x, features_dict) API."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        feats = self.backbone(x)
        if len(feats) != 4:
            raise RuntimeError(f'Expected 4 ConvNeXt feature maps, got {len(feats)}')
        # feats: [96, 192, 384, 768] shallow→deep
        features = {
            'feat': feats[0],   # 96
            'feat0': feats[1],  # 192
            'feat1': feats[2],  # 384
        }
        return feats[3], features


# Define CBAM block
class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.mlp(avg_values) + self.mlp(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.bn(self.conv(out))
        scale = x * torch.sigmoid(out)
        return scale


# Define an Attention block in Attention U-Net
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=nn.BatchNorm2d,
        decoder_conv='standard',
    ):
        super(Up, self).__init__()
        activation = nn.ReLU(True)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            norm_layer(out_channels),
            activation,
        )
        decoder_conv = str(decoder_conv or 'standard').lower()
        if decoder_conv == 'scconv':
            from models.networks.scconv import ScConv

            self.conv = nn.Sequential(
                ScConv(out_channels * 2, out_channels, kernel_size=3),
                norm_layer(out_channels),
                activation,
                ScConv(out_channels, out_channels, kernel_size=3),
                norm_layer(out_channels),
                activation,
            )
        elif decoder_conv == 'depthwise':
            # Depthwise-separable: DW 3×3 + PW 1×1 replaces each standard 3×3.
            # FLOPs ~ Cin·K² + Cin·Cout instead of Cin·Cout·K² per layer.
            cin = out_channels * 2
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cin, kernel_size=3, padding=1, groups=cin, bias=False),
                nn.Conv2d(cin, out_channels, kernel_size=1, bias=True),
                norm_layer(out_channels),
                activation,
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1,
                    groups=out_channels, bias=False,
                ),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
                norm_layer(out_channels),
                activation,
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                norm_layer(out_channels),
                activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                norm_layer(out_channels),
                activation,
            )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = self.conv(torch.cat([x2, x1], dim=1))
        return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


