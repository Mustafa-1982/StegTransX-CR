"""Reveal network R_phi (specification section 3.3).

The network takes only the received image. It is never told which codec was
applied, which matches paper Algorithm 4 and real deployment, where the
receiver has no side channel telling it "this arrived as WebP q = 80".
Paper Algorithm 1 line 22 passes k to R_phi during training; that would create
a train/test mismatch, so the specification's codec-free form is used instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .blocks import TransformerBlock


class RevealNetwork(nn.Module):
    """Extracts the secret image from a (possibly compressed) stego image."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        channels = cfg.reveal_channels

        self.stem = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        # Residual Transformer blocks at native resolution: no downsampling, so
        # high-frequency spatial detail of the payload is preserved.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(channels, cfg.num_heads, cfg.window_size, cfg.ffn_expansion)
                for _ in range(cfg.reveal_blocks)
            ]
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(cfg.leaky_slope, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(cfg.leaky_slope, inplace=True),
        )
        self.to_rgb = nn.Conv2d(channels, 3, kernel_size=1)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """``stego``: (B, 3, H, W) in [0, 1]. Returns the secret in [0, 1]."""
        x = self.stem(stego)
        for block in self.blocks:
            x = block(x)
        x = self.refine(x)
        return torch.sigmoid(self.to_rgb(x))
