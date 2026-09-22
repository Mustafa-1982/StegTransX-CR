"""DiffJPEG: differentiable JPEG simulator (specification section 3.4.1).

RGB -> YCbCr -> 4:2:0 chroma subsampling -> 8 x 8 DCT -> quantisation with the
standard tables scaled by the quality factor -> GAF rounding -> dequantisation
-> inverse DCT -> chroma upsampling -> RGB.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gaf import gaf_round
from .modules import as_quality_tensor
from .quant import JPEG_CHROMA_TABLE, JPEG_LUMA_TABLE, jpeg_scaled_table
from .transforms import (
    block_transform,
    chroma_downsample,
    chroma_upsample,
    dct_matrix,
    inverse_block_transform,
    pad_to_multiple,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


class DiffJPEG(nn.Module):
    """Analytical JPEG pipeline with GAF rounding. No learned parameters."""

    block = 8

    def __init__(self, chroma_subsampling: bool = True) -> None:
        super().__init__()
        self.chroma_subsampling = chroma_subsampling
        self.register_buffer("luma_table", JPEG_LUMA_TABLE.clone(), persistent=False)
        self.register_buffer("chroma_table", JPEG_CHROMA_TABLE.clone(), persistent=False)

    def _component(self, component: torch.Tensor, table: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        """Quantise a single (B, 1, H, W) component on the 0-255 scale."""
        height, width = component.shape[-2:]
        padded, _, _ = pad_to_multiple(component, self.block)
        shifted = padded - 128.0

        matrix = dct_matrix(self.block, component.device, component.dtype)
        coeffs = block_transform(shifted, matrix, self.block)

        step = jpeg_scaled_table(table.to(component.dtype), quality)
        quantised = gaf_round(coeffs / step)
        dequantised = quantised * step

        restored = inverse_block_transform(dequantised, matrix) + 128.0
        return restored[..., :height, :width]

    def forward(self, x: torch.Tensor, quality) -> torch.Tensor:
        """``x``: (B, 3, H, W) in [0, 1]. Returns the compressed image in [0, 1]."""
        height, width = x.shape[-2:]
        q = as_quality_tensor(quality, x.shape[0], x.device, x.dtype)

        ycbcr = rgb_to_ycbcr(x * 255.0)
        y = ycbcr[:, 0:1]
        cb = ycbcr[:, 1:2]
        cr = ycbcr[:, 2:3]

        if self.chroma_subsampling:
            cb_small = chroma_downsample(cb)
            cr_small = chroma_downsample(cr)
        else:
            cb_small, cr_small = cb, cr

        y = self._component(y, self.luma_table, q)
        cb_small = self._component(cb_small, self.chroma_table, q)
        cr_small = self._component(cr_small, self.chroma_table, q)

        if self.chroma_subsampling:
            cb = chroma_upsample(cb_small, (height, width))
            cr = chroma_upsample(cr_small, (height, width))
        else:
            cb, cr = cb_small, cr_small

        rgb = ycbcr_to_rgb(torch.cat([y, cb, cr], dim=1))
        return (rgb / 255.0).clamp(0.0, 1.0)
