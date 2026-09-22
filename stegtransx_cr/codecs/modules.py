"""Learned components shared by the non-JPEG codec simulators.

Specification section 3.4.2 / paper section 3.3.3: each codec simulator adds a
quality-conditioned learned distortion module using FiLM-style conditioning,

    I~ = I_hat + Conv_out( ResBlocks(Conv_in(I_hat)) * (1 + gamma(q)) + beta(q) )

with (gamma(q), beta(q)) = MLP(q) and lightweight residual blocks (64 channels,
3 blocks).

``Conv_out`` is zero-initialised, so an untrained simulator is exactly its
analytical transform coding pipeline. The learned part only becomes active
after calibration against a real encoder (see ``stegtransx_cr.calibrate``),
which prevents random weights from injecting meaningless artifacts.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def as_quality_tensor(quality, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalise a scalar or per-sample quality factor to a (B, 1) tensor."""
    if isinstance(quality, torch.Tensor):
        q = quality.to(device=device, dtype=dtype).reshape(-1)
        if q.numel() == 1:
            q = q.expand(batch)
    else:
        q = torch.full((batch,), float(quality), device=device, dtype=dtype)
    return q.view(batch, 1)


class QualityConditioning(nn.Module):
    """Maps a quality factor to FiLM scale and shift vectors."""

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, quality: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``quality``: (B, 1) with values in [0, 100]."""
        gamma, beta = self.net(quality / 100.0).chunk(2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class LearnedDistortion(nn.Module):
    """FiLM-conditioned residual module modelling codec-specific artifacts."""

    def __init__(self, channels: int = 64, num_blocks: int = 3, in_channels: int = 3) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.film = QualityConditioning(channels)
        self.conv_out = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, quality) -> torch.Tensor:
        q = as_quality_tensor(quality, x.shape[0], x.device, x.dtype)
        gamma, beta = self.film(q)
        features = self.blocks(self.conv_in(x))
        features = features * (1.0 + gamma) + beta
        return x + self.conv_out(features)


class LearnedDeblockingFilter(nn.Module):
    """Differentiable in-loop deblocking filter conditioned on block edges.

    Specification section 3.4.3: depthwise separable convolutions driven by an
    edge boundary map, so smoothing concentrates on transform block borders.
    """

    def __init__(self, channels: int = 3, block_size: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.block_size = block_size
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.strength = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # Start as a genuine 3 x 3 blur applied per channel, so that once the
        # filter is switched on it smooths rather than injects noise.
        blur = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
        ) / 16.0
        with torch.no_grad():
            self.depthwise.weight.copy_(blur.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
            self.depthwise.bias.zero_()
            self.pointwise.weight.copy_(torch.eye(channels).view(channels, channels, 1, 1))
            self.pointwise.bias.zero_()
        # Global gate, zero at initialisation: the untrained filter is exactly
        # the identity and only becomes active after calibration.
        self.scale = nn.Parameter(torch.zeros(1))

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        rows = torch.arange(h, device=x.device)
        cols = torch.arange(w, device=x.device)
        row_edge = ((rows % self.block_size) == 0).float().view(1, 1, h, 1)
        col_edge = ((cols % self.block_size) == 0).float().view(1, 1, 1, w)
        edge = torch.clamp(row_edge + col_edge, max=1.0)
        return edge.expand(x.shape[0], 1, h, w).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = self._edge_map(x)
        weight = self.strength(torch.cat([x, edge], dim=1)) * edge * self.scale
        smoothed = self.pointwise(self.depthwise(x))
        return x + weight * (smoothed - x)
