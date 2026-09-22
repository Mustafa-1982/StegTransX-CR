"""Quantisation tables and quality-factor mappings for the codec simulators.

The JPEG tables and scaling rule are the standard ones. The WebP, HEIF and AVIF
step sizes are analytical approximations of each codec's rate control; they set
the starting point that the learned distortion modules refine during
calibration against real encoders.
"""

from __future__ import annotations

import torch

# Standard JPEG Annex K quantisation tables.
JPEG_LUMA_TABLE = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)

JPEG_CHROMA_TABLE = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def jpeg_quality_scale(quality: torch.Tensor) -> torch.Tensor:
    """Standard JPEG scaling factor s(q) / 100 for a (B, 1) quality tensor."""
    q = quality.clamp(1.0, 100.0)
    scale = torch.where(q < 50.0, 5000.0 / q, 200.0 - 2.0 * q)
    return scale / 100.0


def jpeg_scaled_table(table: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """Scale a base table by the quality factor -> (B, 1, 1, 1, 8, 8)."""
    scale = jpeg_quality_scale(quality).view(-1, 1, 1)
    scaled = (table.unsqueeze(0) * scale).clamp(1.0, 255.0)
    return scaled.view(-1, 1, 1, 1, table.shape[-2], table.shape[-1])


# --------------------------------------------------------------------------- #
# WebP / HEIF / AVIF step sizes.
#
# The literal quantiser formulas of VP8, HEVC and AV1 are defined against those
# codecs' own integer transforms, whose scaling differs from the orthonormal
# DCT used here, so transplanting them directly produces a simulator that
# barely distorts anything. The constants below are fitted instead, so that an
# uncalibrated simulator sits in the same distortion range as the real encoder
# at the same nominal quality factor. The learned distortion modules then
# refine the artifact *shape* during calibration; these constants set the
# artifact *magnitude*.
# --------------------------------------------------------------------------- #
def webp_quant_step(quality: torch.Tensor) -> torch.Tensor:
    """WebP residual quantisation step -> (B, 1, 1, 1, 1, 1)."""
    q = quality.clamp(1.0, 100.0)
    step = 18.8 + 0.52 * (100.0 - q)
    return step.view(-1, 1, 1, 1, 1, 1)


def hevc_qp(quality: torch.Tensor) -> torch.Tensor:
    """Effective HEVC-style quantisation parameter.

    Specification section 3.4.3 quotes QP(q) ~ 51 - 0.51 q, which is HEVC's
    internal parameter. The refitted line below preserves that linear form
    while matching the distortion of real HEIF output.
    """
    q = quality.clamp(1.0, 100.0)
    return (47.8 - 0.238 * q).clamp(0.0, 51.0)


def hevc_quant_step(quality: torch.Tensor) -> torch.Tensor:
    """HEVC step size Qstep = 2^((QP - 4) / 6) -> (B, 1, 1, 1, 1, 1)."""
    step = torch.pow(2.0, (hevc_qp(quality) - 4.0) / 6.0)
    return step.view(-1, 1, 1, 1, 1, 1)


def av1_qp(quality: torch.Tensor) -> torch.Tensor:
    """AV1 is more efficient than HEVC at equal quality, so its curve sits one
    QP step lower, which is roughly 1 dB less distortion."""
    q = quality.clamp(1.0, 100.0)
    return (46.8 - 0.238 * q).clamp(0.0, 51.0)


def av1_quant_step(quality: torch.Tensor) -> torch.Tensor:
    step = torch.pow(2.0, (av1_qp(quality) - 4.0) / 6.0)
    return step.view(-1, 1, 1, 1, 1, 1)
