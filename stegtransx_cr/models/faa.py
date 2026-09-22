"""Frequency-Adaptive Attention (FAA) bottleneck module.

Implements specification section 3.2 / paper section 3.5 and Algorithm 3:

    X_b   = DWConv_{2b+3}(X_res),  X_res <- X_res - X_b        (band split)
    e     = MLP( concat_b GAP(X_b) )                           (band energies)
    c_k   = Embedding(k)                                       (codec condition)
    r     = sigma( W2 * LeakyReLU(W1 [e || c_k] + b1) + b2 )   (resilience)
    g     = sigma( FC(GAP(X)) )                                (channel gate)
    w     = g * (0.7 + 0.3 * r),   X_hat = w * X               (gating)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrequencyAdaptiveAttention(nn.Module):
    """Codec-conditioned frequency-aware channel gating."""

    def __init__(
        self,
        channels: int,
        num_bands: int = 4,
        num_codecs: int = 4,
        codec_dim: int = 32,
        slope: float = 0.2,
        resilience_mix: float = 0.3,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands
        self.resilience_mix = resilience_mix

        # Progressively larger depthwise kernels: K_b = 2b + 3 -> {3, 5, 7, 9}.
        self.band_filters = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=2 * b + 3,
                    padding=b + 1,
                    groups=channels,
                    bias=False,
                )
                for b in range(num_bands)
            ]
        )

        self.energy_mlp = nn.Sequential(
            nn.Linear(channels * num_bands, channels),
            nn.LeakyReLU(slope, inplace=True),
            nn.Linear(channels, channels),
        )

        self.codec_embedding = nn.Embedding(num_codecs, codec_dim)

        self.resilience = nn.Sequential(
            nn.Linear(channels + codec_dim, channels),
            nn.LeakyReLU(slope, inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

        self.channel_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))

    def decompose(self, x: torch.Tensor) -> list:
        """Progressive band-pass decomposition into ``num_bands`` sub-bands."""
        residual = x
        bands = []
        for band_filter in self.band_filters:
            band = band_filter(residual)
            residual = residual - band
            bands.append(band)
        return bands

    def forward(self, x: torch.Tensor, codec: torch.Tensor) -> torch.Tensor:
        """``x``: (B, C, H, W) bottleneck features. ``codec``: (B,) long tensor."""
        if codec.dim() == 0:
            codec = codec.view(1).expand(x.shape[0])
        codec = codec.to(device=x.device, dtype=torch.long)

        bands = self.decompose(x)
        energies = torch.cat([self._gap(band) for band in bands], dim=1)
        e = self.energy_mlp(energies)

        c_k = self.codec_embedding(codec).to(dtype=e.dtype)
        r = self.resilience(torch.cat([e, c_k], dim=1))

        g = self.channel_gate(self._gap(x))
        w = g * ((1.0 - self.resilience_mix) + self.resilience_mix * r)
        return x * w.unsqueeze(-1).unsqueeze(-1)
