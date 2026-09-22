"""Reproducibility helpers, image quality metrics and checkpoint I/O."""

from __future__ import annotations

import math
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Reproducibility (specification section 5.1)
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def autocast(device: torch.device, enabled: bool = True):
    """AMP context that works on PyTorch 2.1+ (Colab) and on CPU."""
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    from contextlib import nullcontext

    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    """GradScaler compatible with both ``torch.amp`` (2.4+) and ``torch.cuda.amp`` (2.1)."""
    enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            kind = "cuda" if device.type == "cuda" else "cpu"
            return torch.amp.GradScaler(kind, enabled=enabled)
        except Exception:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


# --------------------------------------------------------------------------- #
# Metrics. Tensors are (B, 3, H, W) in [0, 1].
# PSNR uses data_range = 1. MAE and RMSE are reported on the 0-255 scale so the
# numbers are comparable with Table 1 of the paper (RMSE targets 1.5 - 8.0).
# --------------------------------------------------------------------------- #
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-image PSNR in dB, returned as a (B,) tensor."""
    pred = pred.clamp(0.0, 1.0).float()
    target = target.clamp(0.0, 1.0).float()
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error on the 0-255 scale, per image."""
    diff = (pred.clamp(0, 1) - target.clamp(0, 1)).abs() * 255.0
    return diff.flatten(1).mean(dim=1)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error on the 0-255 scale, per image."""
    diff = (pred.clamp(0, 1) - target.clamp(0, 1)) * 255.0
    return torch.sqrt((diff ** 2).flatten(1).mean(dim=1))


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Gaussian-window SSIM averaged over channels, returned per image."""
    pred = pred.clamp(0.0, 1.0).float()
    target = target.clamp(0.0, 1.0).float()
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, padding=pad, groups=channels)

    mu_x, mu_y = filt(pred), filt(target)
    mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = filt(pred * pred) - mu_x_sq
    sigma_y = filt(target * target) - mu_y_sq
    sigma_xy = filt(pred * target) - mu_xy

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).flatten(1).mean(dim=1)


def image_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """All four full-reference metrics, each a (B,) tensor."""
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
    }


# --------------------------------------------------------------------------- #
# Running statistics
# --------------------------------------------------------------------------- #
class AverageMeter:
    """Tracks a running mean over a training epoch or evaluation pass."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().mean().item())
        if math.isnan(value) or math.isinf(value):
            return
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class MeterDict:
    """A named collection of AverageMeters."""

    def __init__(self) -> None:
        self._meters: Dict[str, AverageMeter] = {}

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            self._meters.setdefault(key, AverageMeter()).update(value, n)

    def averages(self) -> Dict[str, float]:
        return {key: meter.avg for key, meter in self._meters.items()}

    def reset(self) -> None:
        for meter in self._meters.values():
            meter.reset()


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def __str__(self) -> str:
        return format_seconds(self.elapsed())


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict:
    kwargs = {"map_location": map_location or "cpu"}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
