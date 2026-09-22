"""Hiding network H_theta (specification section 3.1).

Asymmetric U-Net with depthwise-separable convolutions, hybrid CNN/Transformer
blocks and the frequency-adaptive attention bottleneck. The stego image is a
clamped residual on top of the cover image:

    I_stego = clamp(I_cover + Conv3x3(X_dec), 0, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .blocks import ConvBlock, Downsample, HybridBlock, TransformerBlock, Upsample
from .faa import FrequencyAdaptiveAttention


class HidingNetwork(nn.Module):
    """Embeds a full-resolution secret image inside a cover image."""

    def __init__(self, cfg: ModelConfig | None = None, num_codecs: int = 4) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        c1, c2, c3 = cfg.stem_channels, cfg.stage2_channels, cfg.stage3_channels

        # Stem: 6-channel [cover || secret] -> C1
        self.stem = nn.Sequential(
            nn.Conv2d(6, c1, kernel_size=3, padding=1),
            nn.LeakyReLU(cfg.leaky_slope, inplace=True),
        )

        # Encoder stage 1 at H x W
        self.enc1 = nn.Sequential(
            ConvBlock(c1, cfg.leaky_slope),
            ConvBlock(c1, cfg.leaky_slope),
        )

        # Encoder stage 2 at H/2 x W/2
        self.down1 = Downsample(c1, c2, cfg.leaky_slope)
        self.enc2 = nn.ModuleList(
            [
                HybridBlock(c2, cfg.num_heads, cfg.window_size, cfg.ffn_expansion, cfg.leaky_slope)
                for _ in range(2)
            ]
        )

        # Encoder stage 3 at H/4 x W/4
        self.down2 = Downsample(c2, c3, cfg.leaky_slope)
        self.enc3 = nn.ModuleList(
            [
                TransformerBlock(c3, cfg.num_heads, cfg.window_size, cfg.ffn_expansion)
                for _ in range(2)
            ]
        )

        # Bottleneck
        self.faa = FrequencyAdaptiveAttention(
            channels=c3,
            num_bands=cfg.freq_bands,
            num_codecs=num_codecs,
            codec_dim=cfg.codec_embed_dim,
            slope=cfg.leaky_slope,
        )

        # Decoder stage 2
        self.up2 = Upsample(c3, c2)
        self.fuse2 = nn.Conv2d(c2 * 2, c2, kernel_size=1)
        self.dec2 = nn.ModuleList(
            [
                HybridBlock(c2, cfg.num_heads, cfg.window_size, cfg.ffn_expansion, cfg.leaky_slope)
                for _ in range(2)
            ]
        )

        # Decoder stage 1
        self.up1 = Upsample(c2, c1)
        self.fuse1 = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.dec1 = nn.Sequential(
            ConvBlock(c1, cfg.leaky_slope),
            ConvBlock(c1, cfg.leaky_slope),
        )

        # Output projection. Small-scale init keeps the initial stego image
        # close to the cover without the dead gradient path that an exactly
        # zero-initialised residual branch would create.
        self.to_rgb = nn.Conv2d(c1, 3, kernel_size=3, padding=1)
        nn.init.normal_(self.to_rgb.weight, std=0.01)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(
        self,
        cover: torch.Tensor,
        secret: torch.Tensor,
        codec: torch.Tensor,
        clamp_output: bool = False,
    ) -> torch.Tensor:
        """``cover``/``secret``: (B, 3, H, W) in [0, 1]. ``codec``: (B,) long.

        The raw, unclamped residual sum is returned by default. Specification
        equation (1) clamps the stego image, but the restriction loss
        L_restrict(I_stego) of section 4.1.3 can only do its job if it sees
        values outside [0, 1]; clamping first makes that term identically zero
        and blocks gradients for the very pixels it is supposed to pull back.
        Callers clamp before compression and before saving.
        """
        x = torch.cat([cover, secret], dim=1)
        x = self.stem(x)

        skip1 = self.enc1(x)

        x = self.down1(skip1)
        for block in self.enc2:
            x = block(x)
        skip2 = x

        x = self.down2(x)
        for block in self.enc3:
            x = block(x)

        x = self.faa(x, codec)

        x = self.up2(x, size=skip2.shape[-2:])
        x = self.fuse2(torch.cat([x, skip2], dim=1))
        for block in self.dec2:
            x = block(x)

        x = self.up1(x, size=skip1.shape[-2:])
        x = self.fuse1(torch.cat([x, skip1], dim=1))
        x = self.dec1(x)

        stego = cover + self.to_rgb(x)
        # The range-restriction loss keeps values inside [0, 1]; the clamp is a
        # hard guarantee for the compression stage and for saved images.
        return stego.clamp(0.0, 1.0) if clamp_output else stego
