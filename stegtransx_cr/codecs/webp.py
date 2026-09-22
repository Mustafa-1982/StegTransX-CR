"""DiffWebP: differentiable WebP simulator (specification section 3.4.2).

WebP (VP8 intra) codes prediction residuals rather than raw pixels, so the
simulator is: soft-selected spatial intra prediction -> 4 x 4 DCT of the
residual -> GAF quantisation -> reconstruction -> FiLM-conditioned learned
distortion.

The four prediction modes are evaluated on 4 x 4 blocks using the neighbouring
row and column of the input image. A real encoder predicts from already
reconstructed neighbours, which is inherently sequential; predicting from the
input keeps the whole operation parallel and differentiable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaf import gaf_round
from .modules import LearnedDistortion, as_quality_tensor
from .quant import webp_quant_step
from .transforms import (
    block_merge,
    block_split,
    dct_matrix,
    pad_to_multiple,
)


class SoftIntraPredictor(nn.Module):
    """Blends DC / Horizontal / Vertical / TrueMotion predictions per image."""

    num_modes = 4

    def __init__(self, block: int = 4, in_channels: int = 3) -> None:
        super().__init__()
        self.block = block
        self.gate = nn.Linear(in_channels, self.num_modes)

    def _neighbours(self, x: torch.Tensor):
        """Above row, left column and above-left corner for every block."""
        b, c, h, w = x.shape
        n = self.block
        gh, gw = h // n, w // n
        padded = F.pad(x, (1, 0, 1, 0), mode="replicate")

        row_idx = torch.arange(gh, device=x.device) * n
        col_idx = torch.arange(gw, device=x.device) * n

        above = padded[:, :, row_idx, :][..., 1:]            # (B, C, gh, W)
        above = above.reshape(b, c, gh, gw, n)

        left = padded[:, :, 1:, :][..., col_idx]             # (B, C, H, gw)
        left = left.reshape(b, c, gh, n, gw).permute(0, 1, 2, 4, 3).contiguous()

        corner = padded[:, :, row_idx, :][..., col_idx]      # (B, C, gh, gw)
        return above, left, corner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the prediction as blocks: (B, C, gh, gw, n, n)."""
        n = self.block
        above, left, corner = self._neighbours(x)

        dc = ((above.sum(dim=-1) + left.sum(dim=-1)) / (2 * n))
        dc = dc[..., None, None].expand(*dc.shape, n, n)
        vertical = above[..., None, :].expand(*above.shape[:-1], n, n)
        horizontal = left[..., :, None].expand(*left.shape[:-1], n, n)
        true_motion = left[..., :, None] + above[..., None, :] - corner[..., None, None]

        modes = torch.stack([dc, vertical, horizontal, true_motion], dim=0)

        logits = self.gate(x.mean(dim=(2, 3)))               # (B, num_modes)
        alpha = torch.softmax(logits, dim=1).to(dtype=modes.dtype)
        alpha = alpha.permute(1, 0).view(self.num_modes, -1, 1, 1, 1, 1, 1)
        return (modes * alpha).sum(dim=0)


class DiffWebP(nn.Module):
    """Predictive coding + 4 x 4 transform coding + learned WebP artifacts."""

    block = 4

    def __init__(self, distortion_channels: int = 64, distortion_blocks: int = 3) -> None:
        super().__init__()
        self.predictor = SoftIntraPredictor(block=self.block)
        self.distortion = LearnedDistortion(distortion_channels, distortion_blocks)

    def forward(self, x: torch.Tensor, quality) -> torch.Tensor:
        """``x``: (B, 3, H, W) in [0, 1]."""
        height, width = x.shape[-2:]
        q = as_quality_tensor(quality, x.shape[0], x.device, x.dtype)

        scaled = x * 255.0
        padded, _, _ = pad_to_multiple(scaled, self.block)

        prediction = self.predictor(padded)
        residual = block_split(padded, self.block) - prediction

        matrix = dct_matrix(self.block, x.device, x.dtype)
        coeffs = matrix @ residual @ matrix.transpose(-1, -2)

        step = webp_quant_step(q).to(dtype=coeffs.dtype)
        dequantised = gaf_round(coeffs / step) * step

        residual_hat = matrix.transpose(-1, -2) @ dequantised @ matrix
        reconstructed = block_merge(prediction + residual_hat)
        reconstructed = reconstructed[..., :height, :width]

        out = (reconstructed / 255.0).clamp(0.0, 1.0)
        return self.distortion(out, q).clamp(0.0, 1.0)
