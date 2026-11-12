"""Vendored skip-attention blocks for CISLNet AttUnet experiments.

Sources (adapted; do not import clone trees at runtime):
  - CoordAtt: houqb/CoordAttention (MIT)
  - TripletAttention: landskape-ai/triplet-attention (MIT)
  - SCSEBlock: nyoki-mtl/pytorch-segmentation scse.py (MIT)
  - ContextBlock2d: dl19940602/GCnet-pytorch (MIT); mmcv inits replaced with nn.init
  - ECA: BangguWu/ECANet eca_module.py (MIT)
  - SimAM: ZjjConan/SimAM networks/attentions/simam_module.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --- CoordAtt (houqb/CoordAttention) -------------------------------------------------


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


# --- Triplet Attention (landskape-ai) -------------------------------------------------


class _BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class _ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1,
        )


class _AttentionGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = _ZPool()
        self.conv = _BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super().__init__()
        self.cw = _AttentionGate()
        self.hc = _AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = _AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            return (x_out + x_out11 + x_out21) / 3.0
        return (x_out11 + x_out21) / 2.0


# --- SCSE (nyoki-mtl/pytorch-segmentation) -------------------------------------------


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(
            nn.Linear(channel, int(channel // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel),
        )
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        chn_se = self.avg_pool(x).view(b, c)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(b, c, 1, 1))
        chn_se = x * chn_se
        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = x * spa_se
        return chn_se + spa_se


# --- GCNet ContextBlock2d (dl19940602/GCnet-pytorch; mmcv-free) -----------------------


def _last_zero_init(module: nn.Module) -> None:
    if isinstance(module, nn.Sequential):
        m = module[-1]
    else:
        m = module
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ContextBlock2d(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        pool="att",
        fusions=("channel_add", "channel_mul"),
        ratio=8,
    ):
        super().__init__()
        assert pool in ("avg", "att")
        assert all(f in ("channel_add", "channel_mul") for f in fusions)
        assert len(fusions) > 0
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = list(fusions)
        if pool == "att":
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if "channel_add" in self.fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1),
            )
        else:
            self.channel_add_conv = None
        if "channel_mul" in self.fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1),
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == "att":
            nn.init.kaiming_normal_(self.conv_mask.weight, mode="fan_in", nonlinearity="relu")
            if self.conv_mask.bias is not None:
                nn.init.zeros_(self.conv_mask.bias)
        if self.channel_add_conv is not None:
            _last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            _last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == "att":
            input_x = x.view(batch, channel, height * width).unsqueeze(1)
            context_mask = self.conv_mask(x).view(batch, 1, height * width)
            context_mask = self.softmax(context_mask).unsqueeze(3)
            context = torch.matmul(input_x, context_mask).view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        if self.channel_mul_conv is not None:
            out = x * torch.sigmoid(self.channel_mul_conv(context))
        else:
            out = x
        if self.channel_add_conv is not None:
            out = out + self.channel_add_conv(context)
        return out


# --- ECA (BangguWu/ECANet) ------------------------------------------------------------


class ECALayer(nn.Module):
    """Efficient Channel Attention (ECA). Same-C BCHW skip drop-in.

    Vendored from BangguWu/ECANet ``eca_module.eca_layer`` (default k_size=3).
    """

    def __init__(self, channel: int, k_size: int = 3):
        super().__init__()
        # ``channel`` kept for factory API parity with other skip attns.
        del channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# --- SimAM (ZjjConan/SimAM) -----------------------------------------------------------


class SimAM(nn.Module):
    """Simple Parameter-Free Attention Module. Same-C BCHW skip drop-in."""

    def __init__(self, channels=None, e_lambda: float = 1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5
        return x * self.activation(y)


# --- Factory ------------------------------------------------------------------------


def build_skip_attention(attention_type: str, channels: int, cbam_cls=None):
    """Build a same-C BCHW skip attention module.

    ``cbam_cls`` should be the in-tree ``CBAM`` class when ``attention_type=cbam``
    so interpret hooks keep working on the default path.
    """
    key = str(attention_type or "cbam").lower().strip()
    if key in ("none", "identity", "off", "noattn"):
        # Attention-free baseline: pass-through gates keep the AttUnet topology
        # but remove every skip/bottleneck attention module.
        return nn.Identity()
    if key in ("cbam", ""):
        if cbam_cls is None:
            raise ValueError("cbam_cls is required for attention_type=cbam")
        return cbam_cls(channels)
    if key == "coordatt":
        return CoordAtt(channels, channels, reduction=32)
    if key == "triplet":
        return TripletAttention(no_spatial=False)
    if key == "scse":
        return SCSEBlock(channels, reduction=16)
    if key in ("gcblock", "gc"):
        return ContextBlock2d(
            channels,
            channels,
            pool="att",
            fusions=("channel_add", "channel_mul"),
            ratio=8,
        )
    if key in ("eca", "ecanet"):
        return ECALayer(channels)
    if key == "simam":
        return SimAM(channels)
    raise ValueError(
        f"Unknown attention_type={attention_type!r}. "
        "Expected none|cbam|coordatt|triplet|scse|gcblock|eca|simam."
    )
