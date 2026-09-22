"""Block transforms and colour conversions shared by the codec simulators.

Everything here is differentiable and batched over (B, C, H, W) tensors.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Transform matrices
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=16)
def _dct_matrix_cpu(size: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix of shape (size, size)."""
    n = torch.arange(size, dtype=torch.float64)
    k = n.view(-1, 1)
    matrix = torch.cos(math.pi * (2 * n + 1) * k / (2 * size))
    matrix *= math.sqrt(2.0 / size)
    matrix[0] *= 1.0 / math.sqrt(2.0)
    return matrix.float()


@lru_cache(maxsize=16)
def _adst_matrix_cpu(size: int) -> torch.Tensor:
    """Asymmetric Discrete Sine Transform basis used by VP9/AV1."""
    n = torch.arange(size, dtype=torch.float64)
    k = n.view(-1, 1)
    matrix = torch.sin(math.pi * (2 * n + 1) * (k + 1) / (2 * size + 1))
    matrix *= math.sqrt(4.0 / (2 * size + 1))
    return matrix.float()


def dct_matrix(size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return _dct_matrix_cpu(size).to(device=device, dtype=dtype)


def adst_matrix(size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return _adst_matrix_cpu(size).to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Block splitting / merging
# --------------------------------------------------------------------------- #
def pad_to_multiple(x: torch.Tensor, size: int) -> Tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    pad_h = (size - h % size) % size
    pad_w = (size - w % size) % size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, pad_h, pad_w


def block_split(x: torch.Tensor, size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, H/size, W/size, size, size)."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // size, size, w // size, size)
    return x.permute(0, 1, 2, 4, 3, 5).contiguous()


def block_merge(blocks: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`block_split`."""
    b, c, gh, gw, size, _ = blocks.shape
    x = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    return x.view(b, c, gh * size, gw * size)


def block_transform(x: torch.Tensor, matrix: torch.Tensor, size: int) -> torch.Tensor:
    """Apply ``matrix @ block @ matrix^T`` to every ``size x size`` block."""
    blocks = block_split(x, size)
    matrix = matrix.to(dtype=blocks.dtype)
    return matrix @ blocks @ matrix.transpose(-1, -2)


def inverse_block_transform(coeffs: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply ``matrix^T @ block @ matrix`` and merge blocks back into an image."""
    matrix = matrix.to(dtype=coeffs.dtype)
    blocks = matrix.transpose(-1, -2) @ coeffs @ matrix
    return block_merge(blocks)


def separable_block_transform(
    x: torch.Tensor,
    row_matrix: torch.Tensor,
    col_matrix: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Row/column transforms may differ (AV1 mixes DCT and ADST per axis)."""
    blocks = block_split(x, size)
    row_matrix = row_matrix.to(dtype=blocks.dtype)
    col_matrix = col_matrix.to(dtype=blocks.dtype)
    return row_matrix @ blocks @ col_matrix.transpose(-1, -2)


def inverse_separable_block_transform(
    coeffs: torch.Tensor,
    row_matrix: torch.Tensor,
    col_matrix: torch.Tensor,
) -> torch.Tensor:
    row_matrix = row_matrix.to(dtype=coeffs.dtype)
    col_matrix = col_matrix.to(dtype=coeffs.dtype)
    blocks = row_matrix.transpose(-1, -2) @ coeffs @ col_matrix
    return block_merge(blocks)


# --------------------------------------------------------------------------- #
# Colour conversion (JPEG full-range YCbCr, values on the 0-255 scale)
# --------------------------------------------------------------------------- #
_RGB_TO_YCBCR = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.168735892, -0.331264108, 0.5],
        [0.5, -0.418687589, -0.081312411],
    ]
)
_YCBCR_OFFSET = torch.tensor([0.0, 128.0, 128.0])
_YCBCR_TO_RGB = torch.tensor(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136286, -0.714136286],
        [1.0, 1.772, 0.0],
    ]
)


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """``x``: (B, 3, H, W) on the 0-255 scale."""
    matrix = _RGB_TO_YCBCR.to(device=x.device, dtype=x.dtype)
    offset = _YCBCR_OFFSET.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    out = torch.einsum("ij,bjhw->bihw", matrix, x)
    return out + offset


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """``x``: (B, 3, H, W) YCbCr on the 0-255 scale."""
    matrix = _YCBCR_TO_RGB.to(device=x.device, dtype=x.dtype)
    offset = _YCBCR_OFFSET.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return torch.einsum("ij,bjhw->bihw", matrix, x - offset)


def chroma_downsample(x: torch.Tensor) -> torch.Tensor:
    """4:2:0 chroma subsampling by 2 x 2 average pooling."""
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def chroma_upsample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
