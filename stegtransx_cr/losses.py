"""Loss functions for StegTransX-CR (specification section 4).

    L_total = lambda_1 L_H(I_stego, I_cover)
            + lambda_2 L_R(I_rec, I_secret)
            + lambda_f L_F(I_stego, I_stego_compressed)

with

    L_H = L_LP + beta_1 L_Charbonnier + beta_2 L_restrict

and the same form for L_R. Paper equations (14)-(15) list only L_LP and
L_restrict, but paper section 3.6.1 states that both losses combine the
Laplacian pyramid, Charbonnier and restriction terms; the specification writes
that combination out explicitly and it is what is implemented here.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


# --------------------------------------------------------------------------- #
# Laplacian pyramid loss
# --------------------------------------------------------------------------- #
def _gaussian_kernel_2d(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


class LaplacianPyramidLoss(nn.Module):
    """Weighted L1 distance between Laplacian pyramids (spec section 4.1.1)."""

    def __init__(self, levels: int = 5, kernel_size: int = 5, sigma: float = 1.0) -> None:
        super().__init__()
        self.levels = levels
        self.kernel_size = kernel_size
        self.register_buffer("kernel", _gaussian_kernel_2d(kernel_size, sigma), persistent=False)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        weight = self.kernel.to(dtype=x.dtype).expand(channels, 1, self.kernel_size, self.kernel_size)
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, weight, groups=channels)

    def pyramid(self, x: torch.Tensor):
        """Returns the Laplacian levels L_0 .. L_{levels-1}."""
        gaussians = [x]
        current = x
        for _ in range(self.levels - 1):
            if min(current.shape[-2:]) < 2 * self.kernel_size:
                break
            current = F.avg_pool2d(self._blur(current), kernel_size=2)
            gaussians.append(current)

        laplacians = []
        for index in range(len(gaussians) - 1):
            coarse = F.interpolate(
                gaussians[index + 1],
                size=gaussians[index].shape[-2:],
                mode="nearest",
            )
            laplacians.append(gaussians[index] - self._blur(coarse))
        laplacians.append(gaussians[-1])
        return laplacians

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_levels = self.pyramid(prediction)
        target_levels = self.pyramid(target)
        loss = prediction.new_zeros(())
        for index, (p, t) in enumerate(zip(pred_levels, target_levels)):
            loss = loss + (2 ** index) * (p - t).abs().mean()
        return loss


# --------------------------------------------------------------------------- #
# Pixel-domain terms
# --------------------------------------------------------------------------- #
class CharbonnierLoss(nn.Module):
    """Smooth L1 approximation sqrt((x - y)^2 + eps^2) (spec section 4.1.2)."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        return torch.sqrt(diff * diff + self.eps ** 2).mean()


class RestrictLoss(nn.Module):
    """Penalises pixels outside [0, 1] without clipping gradients (spec 4.1.3)."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        over = torch.clamp(image - 1.0, min=0.0)
        under = torch.clamp(-image, min=0.0)
        return (over ** 2 + under ** 2).mean()


class ReconstructionLoss(nn.Module):
    """The shared form of L_H and L_R."""

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.pyramid = LaplacianPyramidLoss(levels=cfg.pyramid_levels)
        self.charbonnier = CharbonnierLoss(cfg.charbonnier_eps)
        self.restrict = RestrictLoss()
        self.beta_charbonnier = cfg.beta_charbonnier
        self.beta_restrict = cfg.beta_restrict

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        content, restrict = self.components(prediction, target)
        return content + restrict

    def components(self, prediction: torch.Tensor, target: torch.Tensor):
        """Split the spec L_H / L_R so the trainer can keep restriction on
        while ramping the perceptual terms."""
        content = self.pyramid(prediction, target) + self.beta_charbonnier * self.charbonnier(
            prediction, target
        )
        restrict = self.beta_restrict * self.restrict(prediction)
        return content, restrict


# --------------------------------------------------------------------------- #
# Frequency consistency loss
# --------------------------------------------------------------------------- #
class FrequencyConsistencyLoss(nn.Module):
    """Spectral distance between the stego image before and after compression.

    Specification section 4.3. The magnitude spectrum is split into B
    concentric radial bands and each band contributes its mean squared
    magnitude difference weighted by (1 + 0.5 b), so high-frequency bands, the
    ones compression attacks hardest, are penalised most.

    The FFT uses ``norm="ortho"``. The unnormalised transform of paper equation
    (23) grows with resolution and would swamp the other terms, while a full
    1 / (H W) scaling shrinks the band differences to around 1e-8 and makes the
    term inert. The orthonormal scaling keeps L_F on the same order as L_H and
    L_R at any resolution.
    """

    def __init__(self, num_bands: int = 4) -> None:
        super().__init__()
        self.num_bands = num_bands
        self._cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    def _masks(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, device)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        rows = torch.arange(height, device=device, dtype=torch.float32) - height / 2.0
        cols = torch.arange(width, device=device, dtype=torch.float32) - width / 2.0
        radius = torch.sqrt(rows.view(-1, 1) ** 2 + cols.view(1, -1) ** 2)
        r_max = 0.5 * float((height ** 2 + width ** 2) ** 0.5)

        masks = []
        for band in range(self.num_bands):
            lower = band / self.num_bands * r_max
            upper = (band + 1) / self.num_bands * r_max
            mask = ((radius >= lower) & (radius < upper)).float()
            masks.append(mask)
        stacked = torch.stack(masks, dim=0)
        self._cache[key] = stacked
        return stacked

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        # FFT is computed in float32: half precision complex support is patchy
        # and the magnitudes here are small.
        before = before.float()
        after = after.float()
        height, width = before.shape[-2:]

        spectrum_before = torch.fft.fftshift(
            torch.fft.fft2(before, norm="ortho"), dim=(-2, -1)
        ).abs()
        spectrum_after = torch.fft.fftshift(
            torch.fft.fft2(after, norm="ortho"), dim=(-2, -1)
        ).abs()
        difference = (spectrum_before - spectrum_after) ** 2

        masks = self._masks(height, width, before.device)
        loss = before.new_zeros(())
        for band in range(self.num_bands):
            mask = masks[band]
            count = mask.sum().clamp_min(1.0)
            band_loss = (difference * mask).sum(dim=(-2, -1)) / count
            loss = loss + (1.0 + 0.5 * band) * band_loss.mean()
        return loss / self.num_bands


# --------------------------------------------------------------------------- #
# Total objective
# --------------------------------------------------------------------------- #
class StegTransXCRLoss(nn.Module):
    """Bundles L_H, L_R and L_F with their weights."""

    def __init__(self, cfg: Optional[LossConfig] = None) -> None:
        super().__init__()
        cfg = cfg or LossConfig()
        self.cfg = cfg
        self.hiding = ReconstructionLoss(cfg)
        self.reveal = ReconstructionLoss(cfg)
        self.frequency = FrequencyConsistencyLoss(cfg.freq_bands)
        # Mutable so the trainer can ramp lambda_1 during warm-up.
        self.lambda_hide = cfg.lambda_hide

    def forward(
        self,
        stego: torch.Tensor,
        cover: torch.Tensor,
        recovered: torch.Tensor,
        secret: torch.Tensor,
        compressed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hide_content, hide_restrict = self.hiding.components(stego, cover)
        loss_hide = hide_content + hide_restrict
        loss_reveal = self.reveal(recovered, secret)
        if compressed is None:
            loss_freq = stego.new_zeros(())
        else:
            loss_freq = self.frequency(stego.clamp(0.0, 1.0), compressed)

        # Restriction is never ramped to zero: otherwise warmup with
        # lambda_hide = 0 lets the residual explode, the clamp kills the
        # gradient, and the payload never embeds.
        total = (
            self.lambda_hide * hide_content
            + hide_restrict
            + self.cfg.lambda_reveal * loss_reveal
            + self.cfg.lambda_freq * loss_freq
        )
        return {
            "loss": total,
            "loss_hide": loss_hide,
            "loss_reveal": loss_reveal,
            "loss_freq": loss_freq,
        }


__all__ = [
    "CharbonnierLoss",
    "FrequencyConsistencyLoss",
    "LaplacianPyramidLoss",
    "ReconstructionLoss",
    "RestrictLoss",
    "StegTransXCRLoss",
]
