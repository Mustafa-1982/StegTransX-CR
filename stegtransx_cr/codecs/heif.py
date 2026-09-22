"""DiffHEIF: differentiable HEIF / HEVC-intra simulator (specification 3.4.3).

YCbCr processing -> blend of 8 x 8 and 16 x 16 DCT representations selected by
an adaptive pooling gate (a stand-in for variable-size coding tree units) ->
flat HEVC quantisation with step 2^((QP - 4) / 6) and GAF rounding -> learned
in-loop deblocking filter -> FiLM-conditioned learned distortion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaf import gaf_round, straight_through_binary
from .modules import LearnedDeblockingFilter, LearnedDistortion, as_quality_tensor
from .quant import hevc_quant_step
from .transforms import (
    block_transform,
    dct_matrix,
    inverse_block_transform,
    pad_to_multiple,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


class CTUPartitionGate(nn.Module):
    """Predicts a per-region blend between the 8 x 8 and 16 x 16 transforms."""

    def __init__(self, in_channels: int = 3, hidden: int = 16, pool: int = 16) -> None:
        super().__init__()
        self.pool = pool
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        # Positive bias: an uncalibrated simulator prefers the 8 x 8 transform,
        # matching HEVC's most common intra transform unit size.
        nn.init.constant_(self.net[-1].bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        coarse = F.adaptive_avg_pool2d(x, (max(1, h // self.pool), max(1, w // self.pool)))
        logits = self.net(coarse)
        gate = torch.sigmoid(logits)
        gate = F.interpolate(gate, size=(h, w), mode="bilinear", align_corners=False)
        return straight_through_binary(gate)


class DiffHEIF(nn.Module):
    """HEVC-style intra coding simulator."""

    def __init__(self, distortion_channels: int = 64, distortion_blocks: int = 3) -> None:
        super().__init__()
        self.gate = CTUPartitionGate()
        self.deblock = LearnedDeblockingFilter(channels=3, block_size=8)
        self.distortion = LearnedDistortion(distortion_channels, distortion_blocks)

    @staticmethod
    def _transform_quantise(x: torch.Tensor, size: int, step: torch.Tensor) -> torch.Tensor:
        """Flat-matrix transform coding at a given block size."""
        height, width = x.shape[-2:]
        padded, _, _ = pad_to_multiple(x, size)
        matrix = dct_matrix(size, x.device, x.dtype)
        coeffs = block_transform(padded, matrix, size)
        dequantised = gaf_round(coeffs / step) * step
        restored = inverse_block_transform(dequantised, matrix)
        return restored[..., :height, :width]

    def forward(self, x: torch.Tensor, quality) -> torch.Tensor:
        """``x``: (B, 3, H, W) in [0, 1]."""
        q = as_quality_tensor(quality, x.shape[0], x.device, x.dtype)
        step = hevc_quant_step(q).to(dtype=x.dtype)

        ycbcr = rgb_to_ycbcr(x * 255.0) - 128.0

        small = self._transform_quantise(ycbcr, 8, step)
        large = self._transform_quantise(ycbcr, 16, step)

        gate = self.gate(x).to(dtype=small.dtype)
        blended = gate * small + (1.0 - gate) * large

        rgb = ycbcr_to_rgb(blended + 128.0)
        out = (rgb / 255.0).clamp(0.0, 1.0)

        out = self.deblock(out)
        return self.distortion(out, q).clamp(0.0, 1.0)
