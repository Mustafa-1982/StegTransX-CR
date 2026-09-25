"""Deep steganalysis with SRNet (Boroumand, Chen and Fridrich, IEEE TIFS 2019).

A detector is trained from scratch for every model under test on pairs of
cover / stego images and tested on the held-out coco_test covers and the
stego images the model makes from the coco_test pairs. Two channels:
``png`` (the 8-bit stego as saved) and ``jpeg80`` (cover and stego both
passed through real JPEG q = 80, i.e. what a warden sees after upload).
Colour input (3 channels); architecture as in the paper (12 layers: 2 x type 1,
5 x type 2, 4 x type 3, 1 x type 4, then a linear layer).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SMOKE, log, pairs_to_tensors, quantize, save_json
from .systems import System, codec_tensor


class _T1(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _T2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(_T1(c, c), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return x + self.body(x)


class _T3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(_T1(cin, cout), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(cout), nn.AvgPool2d(3, stride=2, padding=1))
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=2, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.body(x) + self.skip(x)


class SRNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            _T1(in_channels, 64), _T1(64, 16),
            *[_T2(16) for _ in range(5)],
            _T3(16, 16), _T3(16, 64), _T3(64, 128), _T3(128, 256),
            _T1(256, 512), nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512),
        )
        self.fc = nn.Linear(512, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.mean(dim=(2, 3)))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pe(scores: np.ndarray, labels: np.ndarray) -> float:
    thr = np.unique(scores)
    best = 0.5
    pos = labels == 1
    for t in np.concatenate([thr, [thr.max() + 1]]):
        pfa = float((scores[~pos] >= t).mean())
        pmd = float((scores[pos] < t).mean())
        best = min(best, 0.5 * (pfa + pmd))
    return best


@torch.no_grad()
def make_images(system: System, cache: np.ndarray, pairs, device, channel: str, batch: int = 50):
    """Returns (covers_u8, stegos_u8) as (N, H, W, 3) arrays for the channel."""
    from .evaluate import _to_u8, run_chain_local

    system.eval()
    if getattr(system, "variant", "") == "stx":
        batch = min(batch, 8)  # the baseline's global attention is O(B N^2)
    covs, stes = [], []
    for b0 in range(0, len(pairs), batch):
        cov, sec = pairs_to_tensors(cache, pairs[b0 : b0 + batch], device)
        st = quantize(system.hide(cov, sec, codec_tensor("JPEG", cov.shape[0], device)).float())
        covs.append(_to_u8(cov))
        stes.append(_to_u8(st))
    covs, stes = np.concatenate(covs), np.concatenate(stes)
    if channel == "jpeg80":
        covs, _ = run_chain_local(covs, [("JPEG", 80)])
        stes, _ = run_chain_local(stes, [("JPEG", 80)])
    return covs, stes


def _dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    if k & 1:
        x = x.flip(-1)
    if k & 2:
        x = x.flip(-2)
    if k & 4:
        x = x.transpose(-1, -2)
    return x


def train_detector(tr_cov, tr_ste, te_cov, te_ste, device, steps: int = 1500, seed: int = 0,
                   batch_pairs: int = 16) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SRNet(3).to(device)
    decay, no_decay = [], []
    for name, p in net.named_parameters():
        (decay if p.dim() == 4 else no_decay).append(p)
    opt = torch.optim.Adamax([{"params": decay, "weight_decay": 2e-4}, {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(steps * 0.7)], gamma=0.1)
    tc = torch.from_numpy(tr_cov.transpose(0, 3, 1, 2).copy()).to(device)
    ts = torch.from_numpy(tr_ste.transpose(0, 3, 1, 2).copy()).to(device)
    n = tc.shape[0]
    net.train()
    t0 = time.time()
    use_amp = device.type == "cuda"
    for step in range(steps):
        idx = torch.from_numpy(rng.choice(n, size=min(batch_pairs, n), replace=False)).to(device)
        k = int(rng.integers(0, 8))
        x = torch.cat([_dihedral(tc[idx], k), _dihedral(ts[idx], k)]).float().div_(255.0) - 0.5
        y = torch.cat([torch.zeros(len(idx)), torch.ones(len(idx))]).long().to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = net(x)
        loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    scores, labels = [], []
    with torch.no_grad():
        for arr, lab in ((te_cov, 0), (te_ste, 1)):
            for b0 in range(0, len(arr), 100):
                x = torch.from_numpy(arr[b0 : b0 + 100].transpose(0, 3, 1, 2).copy()).to(device).float() / 255.0 - 0.5
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    p = torch.softmax(net(x).float(), dim=1)[:, 1]
                scores.append(p.cpu().numpy())
                labels.append(np.full(len(p), lab))
    scores, labels = np.concatenate(scores), np.concatenate(labels)
    acc = float(((scores >= 0.5).astype(int) == labels).mean())
    return {"accuracy": acc, "auc": _auc(scores, labels), "pe": _pe(scores, labels),
            "train_pairs": int(n), "test_images": int(len(labels)), "steps": steps,
            "seconds": time.time() - t0, "final_train_loss": float(loss.detach())}


def run_steganalysis(models: Dict[str, System], cache: np.ndarray, device, out_dir: str,
                     channels: Sequence[str] = ("png", "jpeg80"), n_sweep: Dict[str, List[int]] = None,
                     steps: int = 1500) -> Dict:
    if SMOKE:
        train_pairs = [(i, 32 + i) for i in range(12)]
        test_pairs = [(16 + i, 44 + i) for i in range(8)]
    else:
        train_pairs = [(200 + i, 3000 + i) for i in range(800)]
        test_pairs = [(1000 + i, 2000 + i) for i in range(1000)]
    n_sweep = n_sweep or {}
    results: Dict[str, Dict] = {}
    for name, system in models.items():
        for ch in channels:
            tr_c, tr_s = make_images(system, cache, train_pairs, device, ch)
            te_c, te_s = make_images(system, cache, test_pairs, device, ch)
            for n in n_sweep.get(name, [len(train_pairs)]):
                n = min(n, len(train_pairs))
                res = train_detector(tr_c[:n], tr_s[:n], te_c, te_s, device, steps=steps)
                key = f"{name}|{ch}|n{n}"
                results[key] = res
                log(f"steganalysis {key}: acc {res['accuracy']:.4f} auc {res['auc']:.4f} pe {res['pe']:.4f}")
                save_json(os.path.join(out_dir, "steganalysis.json"), results)
    return results
