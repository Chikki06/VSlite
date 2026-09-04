"""Reference int8 conv/GEMM for FPGA validation (im2col + int32 accum)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def int8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (M, K) int8, b: (K, N) int8 -> (M, N) int32
    Reference kernel — swap with FPGA/DPU GEMM in deployment.
    """
    return torch.matmul(a.to(torch.int32), b.to(torch.int32))


def int8_conv2d_reference(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int],
    dilation: int | Tuple[int, int] = 1,
    groups: int = 1,
    act_scale: float = 1.0,
) -> torch.Tensor:
    """
    Symmetric int8 conv: quantize input on-the-fly, im2col GEMM, rescale output.
    x: (N,C,H,W) fp32/fp16; q_weight: same shape as nn.Conv2d.weight in int8.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = ( dilation, dilation)

    n, c_in, h, w = x.shape
    c_out, _, kh, kw = q_weight.shape
    x_qmax = 127.0
    x_scale = max(float(act_scale), 1e-8) / x_qmax

    x_q = torch.round(x / x_scale).clamp(-x_qmax, x_qmax).to(torch.int8)
    cols = F.unfold(
        x_q.float(),
        kernel_size=(kh, kw),
        dilation=dilation,
        padding=padding,
        stride=stride,
    ).to(torch.int8)  # (N, C*kh*kw, L)

    out_h = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    out_w = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    l = cols.shape[2]

    w_flat = q_weight.reshape(c_out, -1).t()  # (C*kh*kw, c_out)
    out = torch.empty(n, c_out, l, dtype=torch.int32, device=x.device)
    for bi in range(n):
        out[bi] = int8_gemm(cols[bi].t(), w_flat)

    out = out.float().reshape(n, c_out, out_h, out_w)
    scale = x_scale * weight_scale.view(1, -1, 1, 1).float()
    out = out * scale
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1).float()
    return out
