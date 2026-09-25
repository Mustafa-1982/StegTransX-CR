"""StegTransX baseline (Duan et al., Information Sciences 716, 122264, 2025).

The network is imported unchanged from the authors' repository
(https://github.com/QQ-Stars/StegTransX, Apache-2.0, commit 8b40375, file
StegTransX-V1.py) through a three-symbol mmcv shim. The loss functions below
are ported from the authors' train.py (Apache-2.0): Charbonnier (eps 1e-6) +
range restriction + 5-level Laplacian pyramid with Charbonnier per level, for
both the concealing and the revealing branch, with lambda_1 = lambda_2 = 1.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import STX_DIR

STX_COMMIT = "8b403756439cb3e5dc9573f98f9abbdc27cb6b69"
_SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shims")


def load_stx_module():
    if _SHIM not in sys.path:
        sys.path.insert(0, _SHIM)
    path = os.path.join(STX_DIR, "StegTransX-V1.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"StegTransX source not found at {path}")
    spec = importlib.util.spec_from_file_location("stegtransx_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_stx_pair():
    mod = load_stx_module()
    hiding = mod.stegTransX(in_channels=6, out_channels=3)
    reveal = mod.stegTransX(in_channels=3, out_channels=3)
    return hiding, reveal


class L1CharbonnierLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = x - y
        return torch.sqrt(diff * diff + self.eps).mean()


class STXLaplacianPyramidLoss(nn.Module):
    def __init__(self, num_levels=5, kernel_size=5, sigma=1.0):
        super().__init__()
        self.num_levels = num_levels
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid([ax, ax], indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / torch.sum(kernel)
        self.register_buffer("kernel", kernel.view(1, 1, kernel_size, kernel_size), persistent=False)
        self.padding = kernel_size // 2
        self.charbonnier = L1CharbonnierLoss()

    def blur(self, img):
        c = img.shape[1]
        return F.conv2d(img, self.kernel.to(img.dtype).repeat(c, 1, 1, 1), padding=self.padding, groups=c)

    def pyramid(self, img):
        gauss = [img]
        for _ in range(self.num_levels):
            img = self.blur(img)
            img = F.interpolate(img, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=True)
            gauss.append(img)
        lap = []
        for i in range(self.num_levels):
            up = F.interpolate(gauss[i + 1], size=gauss[i].shape[2:], mode="bilinear", align_corners=False)
            lap.append(gauss[i] - self.blur(up))
        return lap

    def forward(self, pred, target):
        loss = pred.new_zeros(())
        for a, b in zip(self.pyramid(pred), self.pyramid(target)):
            loss = loss + self.charbonnier(a, b)
        return loss


def restrict_loss(x):
    return ((torch.relu(x - 1) + torch.relu(-x)) ** 2).mean()


class STXLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.charb = L1CharbonnierLoss()
        self.lp = STXLaplacianPyramidLoss(num_levels=5)

    def forward(self, stego, cover, recovered, secret):
        h = self.charb(stego, cover) + restrict_loss(stego) + self.lp(stego, cover)
        r = self.charb(recovered, secret) + restrict_loss(recovered) + self.lp(recovered, secret)
        return {"loss": h + r, "loss_hide": h, "loss_reveal": r, "loss_freq": stego.new_zeros(())}
