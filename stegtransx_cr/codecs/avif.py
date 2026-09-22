"""DiffAVIF: differentiable AVIF / AV1-intra simulator (specification 3.4.4).

Pipeline: multi-scale transform coding that mixes DCT and ADST bases through a
learned block-size selection gate -> GAF quantisation with an AV1 step size ->
CDEF-like directional filtering steered by a local variance estimator ->
Wiener-style loop restoration with a symmetric separable 7 x 7 filter ->
FiLM-conditioned learned distortion.

The directional and restoration stages are initialised as the identity, so an
uncalibrated simulator reduces to its analytical transform coding core.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaf import gaf_round, straight_through_gate
from .modules import LearnedDistortion, as_quality_tensor
from .quant import av1_quant_step
from .transforms import (
    adst_matrix,
    block_transform,
    dct_matrix,
    inverse_block_transform,
    inverse_separable_block_transform,
    pad_to_multiple,
    rgb_to_ycbcr,
    separable_block_transform,
    ycbcr_to_rgb,
)

CDEF_ANGLES = (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)


def _directional_kernel(size: int, angle_deg: float, sigma: float = 0.6) -> torch.Tensor:
    """A normalised line-shaped smoothing kernel oriented at ``angle_deg``."""
    radius = size // 2
    coords = torch.arange(size, dtype=torch.float32) - radius
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    angle = math.radians(angle_deg)
    # Perpendicular distance from the line through the origin at `angle`.
    distance = (xx * math.sin(angle) - yy * math.cos(angle)).abs()
    kernel = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


class CDEFFilter(nn.Module):
    """Constrained Directional Enhancement Filter approximation."""

    def __init__(self, channels: int = 3, kernel_size: int = 5, hidden: int = 16) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        kernels = torch.stack([_directional_kernel(kernel_size, a) for a in CDEF_ANGLES])
        self.kernels = nn.Parameter(kernels.unsqueeze(1))          # (8, 1, k, k)
        self.selector = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, len(CDEF_ANGLES), kernel_size=1),
        )
        # Strength starts at zero: the filter is the identity until calibrated.
        self.strength = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _local_variance(x: torch.Tensor, window: int = 5) -> torch.Tensor:
        pad = window // 2
        mean = F.avg_pool2d(x, window, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(x * x, window, stride=1, padding=pad)
        variance = (mean_sq - mean * mean).clamp_min(0.0)
        return variance.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        variance = self._local_variance(x)
        logits = self.selector(torch.cat([x, variance], dim=1))
        weights = torch.softmax(logits, dim=1).unsqueeze(2)        # (B, 8, 1, H, W)

        filtered = []
        for index in range(self.kernels.shape[0]):
            weight = self.kernels[index].to(dtype=x.dtype).repeat(self.channels, 1, 1, 1)
            filtered.append(F.conv2d(x, weight, padding=pad, groups=self.channels))
        stacked = torch.stack(filtered, dim=1)                     # (B, 8, C, H, W)

        mixed = (stacked * weights).sum(dim=1)
        return x + self.strength * (mixed - x)


class LoopRestoration(nn.Module):
    """Symmetric separable 7 x 7 filter approximating AV1 Wiener restoration."""

    def __init__(self, channels: int = 3, taps: int = 7) -> None:
        super().__init__()
        if taps % 2 == 0:
            raise ValueError("taps must be odd")
        self.channels = channels
        self.taps = taps
        half = taps // 2
        # Half-kernel parameters; initialised to a delta so the filter starts
        # as the identity mapping.
        init = torch.zeros(half + 1)
        init[0] = 1.0
        self.half_kernel = nn.Parameter(init)

    def _kernel(self, dtype: torch.dtype) -> torch.Tensor:
        half = self.half_kernel.to(dtype=dtype)
        kernel = torch.cat([half.flip(0)[:-1], half])
        return kernel / kernel.sum().clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._kernel(x.dtype)
        pad = self.taps // 2
        row = kernel.view(1, 1, 1, self.taps).repeat(self.channels, 1, 1, 1)
        col = kernel.view(1, 1, self.taps, 1).repeat(self.channels, 1, 1, 1)
        out = F.conv2d(x, row, padding=(0, pad), groups=self.channels)
        return F.conv2d(out, col, padding=(pad, 0), groups=self.channels)


class DiffAVIF(nn.Module):
    """AV1-intra style simulator."""

    def __init__(self, distortion_channels: int = 64, distortion_blocks: int = 3) -> None:
        super().__init__()
        # Selects between DCT 8, ADST 8 and DCT 16 representations.
        self.block_gate = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
        )
        nn.init.normal_(self.block_gate[-1].weight, std=0.01)
        # Prefer the 8 x 8 DCT before calibration; ADST and the 16 x 16
        # transform are selected where the gate learns they fit better.
        nn.init.constant_(self.block_gate[-1].bias, 0.0)
        with torch.no_grad():
            self.block_gate[-1].bias[0] = 2.0
        self.cdef = CDEFFilter()
        self.restoration = LoopRestoration()
        self.distortion = LearnedDistortion(distortion_channels, distortion_blocks)

    @staticmethod
    def _code_dct(x: torch.Tensor, size: int, step: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        padded, _, _ = pad_to_multiple(x, size)
        matrix = dct_matrix(size, x.device, x.dtype)
        coeffs = block_transform(padded, matrix, size)
        dequantised = gaf_round(coeffs / step) * step
        return inverse_block_transform(dequantised, matrix)[..., :height, :width]

    @staticmethod
    def _code_adst(x: torch.Tensor, size: int, step: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        padded, _, _ = pad_to_multiple(x, size)
        row = adst_matrix(size, x.device, x.dtype)
        col = dct_matrix(size, x.device, x.dtype)
        coeffs = separable_block_transform(padded, row, col, size)
        dequantised = gaf_round(coeffs / step) * step
        restored = inverse_separable_block_transform(dequantised, row, col)
        return restored[..., :height, :width]

    def forward(self, x: torch.Tensor, quality) -> torch.Tensor:
        """``x``: (B, 3, H, W) in [0, 1]."""
        height, width = x.shape[-2:]
        q = as_quality_tensor(quality, x.shape[0], x.device, x.dtype)
        step = av1_quant_step(q).to(dtype=x.dtype)

        ycbcr = rgb_to_ycbcr(x * 255.0) - 128.0

        candidates = torch.stack(
            [
                self._code_dct(ycbcr, 8, step),
                self._code_adst(ycbcr, 8, step),
                self._code_dct(ycbcr, 16, step),
            ],
            dim=1,
        )                                                          # (B, 3, C, H, W)

        logits = self.block_gate(x)
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        weights = straight_through_gate(torch.softmax(logits, dim=1), dim=1)
        weights = weights.unsqueeze(2).to(dtype=candidates.dtype)
        blended = (candidates * weights).sum(dim=1)

        rgb = ycbcr_to_rgb(blended + 128.0)
        out = (rgb / 255.0).clamp(0.0, 1.0)

        out = self.cdef(out)
        out = self.restoration(out)
        return self.distortion(out, q).clamp(0.0, 1.0)
