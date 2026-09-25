"""Steps-based training of StegTransX-CR v2 variants and the StegTransX baseline.

Same networks, losses and distortion channel as the v1 release; what changes
is the training protocol: a fixed optimisation budget with cosine decay and no
early stopping, random-resized DIV2K crops generated on the GPU, bf16 autocast,
and the final checkpoint as the reported model (no checkpoint selection).
"""
from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .common import (
    CODECS,
    OUT_DIR,
    GpuCropSampler,
    env_info,
    eval_sets,
    load_train_images,
    log,
    pairs_to_tensors,
    push_results,
    quantize,
    save_json,
    setup_package,
    torch_save,
)
from .systems import System, codec_tensor

HISTORY_FIELDS = [
    "step", "lr", "lambda_hide", "train_loss", "train_loss_hide", "train_loss_reveal",
    "train_loss_freq", "train_residual", "val_cover_psnr", "val_cover_ssim", "val_cover_psnr_u8",
    "val_secret_psnr", "val_secret_ssim", "val_secret_psnr_realjpeg80", "val_residual",
    "sec_per_step", "elapsed_h",
]


@dataclass
class RunConfig:
    name: str
    regime: str
    variant: str
    seed: int
    total_steps: int = 40000
    batch: int = 32
    micro_batch: int = 0        # 0 = one forward per step; else gradient accumulation
    lr: float = 2e-4
    weight_decay: float = 1e-2
    beta1: float = 0.5
    beta2: float = 0.999
    adam_eps: float = 1e-8
    eta_min: float = 2e-6
    grad_clip: float = 1.0
    hold_frac: float = 0.02
    ramp_frac: float = 0.10
    lambda_freq: float = 0.1
    val_every: int = 1000
    push_every_min: float = 30.0
    amp: str = "bf16"
    compile: bool = False
    image_size: int = 256
    max_hours: float = 0.0

    @staticmethod
    def from_name(name: str, **overrides) -> "RunConfig":
        regime, variant, seed = name.split("-")
        cfg = RunConfig(name=name, regime=regime, variant=variant, seed=int(seed.lstrip("s")))
        if variant == "stx":  # the authors' optimiser settings (config.py / train.py)
            cfg.weight_decay = 1e-5
            cfg.adam_eps = 1e-6
            cfg.eta_min = 1e-7
            cfg.grad_clip = 0.0
            cfg.hold_frac = 0.0
            cfg.ramp_frac = 0.0
            # the authors' global attention is O(B N^2); 32 x 256 x 256 does not
            # fit in 80 GB, so the batch is accumulated in four forward passes.
            cfg.micro_batch = 8
        if variant == "nofreq":
            cfg.lambda_freq = 0.0
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, type(getattr(cfg, key))(value))
        return cfg


def _autocast(device, amp: str):
    if device.type != "cuda" or amp == "off":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class Runner:
    def __init__(self, cfg: RunConfig, device: torch.device, sim_state: Dict, out_dir: Optional[str] = None):
        setup_package()
        from stegtransx_cr.codecs.simulator import CompressionSampler
        from stegtransx_cr.config import LossConfig
        from stegtransx_cr.losses import StegTransXCRLoss

        self.cfg = cfg
        self.device = device
        self.out_dir = out_dir or os.path.join(OUT_DIR, cfg.name)
        os.makedirs(self.out_dir, exist_ok=True)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.system = System(cfg.variant, cfg.regime).to(device)
        self.identity_flags = self.system.load_simulator(sim_state)
        self.system.to(device)
        if cfg.variant == "stx":
            from .stx_baseline import STXLoss

            self.criterion = STXLoss().to(device)
        else:
            self.criterion = StegTransXCRLoss(LossConfig(lambda_freq=cfg.lambda_freq)).to(device)
        self.params = self.system.trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            self.params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps, weight_decay=cfg.weight_decay
        )
        total = max(1, cfg.total_steps)
        floor = cfg.eta_min / cfg.lr

        def cosine(step: int) -> float:
            t = min(step, total) / total
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, cosine)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp == "fp16" and device.type == "cuda"))
        self.sampler = CompressionSampler(
            mode=cfg.regime, codecs=list(CODECS), fixed_codec="JPEG", fixed_quality=80,
            quality_range=(50, 95), cascade_range=(1, 3),
            generator=torch.Generator().manual_seed(1234 + cfg.seed),
        )
        self.step = 0
        self.history: List[Dict] = []
        self.elapsed = 0.0
        self._hide = self.system.hide
        self._recover = self.system.recover
        if cfg.compile and device.type == "cuda":
            self._hide = torch.compile(self.system.hide)
            self._recover = torch.compile(self.system.recover)

    # ------------------------------------------------------------------ #
    def lambda_hide(self, step: int) -> float:
        hold = int(round(self.cfg.hold_frac * self.cfg.total_steps))
        ramp = int(round(self.cfg.ramp_frac * self.cfg.total_steps))
        if step < hold:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(1.0, (step - hold + 1) / float(ramp))

    @property
    def last_path(self) -> str:
        return os.path.join(self.out_dir, "last.pth")

    def save_last(self) -> None:
        torch_save(self.last_path, {
            "hiding": self.system.hiding.state_dict(), "reveal": self.system.reveal.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(), "step": self.step, "history": self.history,
            "elapsed": self.elapsed, "sampler_gen": self.sampler.generator.get_state(),
            "crops": self.crops.state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "cfg": asdict(self.cfg),
        })

    def maybe_resume(self) -> bool:
        if not os.path.exists(self.last_path):
            return False
        p = torch.load(self.last_path, map_location="cpu", weights_only=False)
        self.system.hiding.load_state_dict(p["hiding"])
        self.system.reveal.load_state_dict(p["reveal"])
        self.optimizer.load_state_dict(p["optimizer"])
        self.scheduler.load_state_dict(p["scheduler"])
        self.scaler.load_state_dict(p["scaler"])
        self.step = int(p["step"])
        self.history = p["history"]
        self.elapsed = float(p.get("elapsed", 0.0))
        self.sampler.generator.set_state(p["sampler_gen"])
        self.crops.load_state(p["crops"])
        torch.set_rng_state(p["torch_rng"])
        if p.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(p["cuda_rng"])
        log(f"{self.cfg.name}: resumed at step {self.step}")
        return True

    def write_history(self) -> None:
        path = os.path.join(self.out_dir, "history.csv")
        with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
            w.writeheader()
            for row in self.history:
                w.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})
        os.replace(path + ".tmp", path)

    # ------------------------------------------------------------------ #
    def forward(self, cover, secret, sampler, codec=None):
        if codec is None:
            codec = sampler.sample_codec(cover.shape[0]).to(self.device)
        with _autocast(self.device, self.cfg.amp):
            stego = self._hide(cover, secret, codec)
        stego = stego.float()
        compressed = sampler.apply(self.system.simulator, stego.clamp(0.0, 1.0), codec)
        with _autocast(self.device, self.cfg.amp):
            recovered = self._recover(compressed)
        recovered = recovered.float()
        if self.cfg.variant == "stx":
            losses = self.criterion(stego, cover, recovered, secret)
        else:
            losses = self.criterion(stego, cover, recovered, secret, compressed)
        return stego, compressed, recovered, losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        from stegtransx_cr.codecs.simulator import CompressionSampler
        from stegtransx_cr.utils import image_metrics

        from .evaluate import run_chain_local

        self.system.eval()
        if self.cfg.variant != "stx":
            keep = self.criterion.lambda_hide
            self.criterion.lambda_hide = 1.0
        sampler = CompressionSampler(
            mode=self.cfg.regime, codecs=list(CODECS), fixed_codec="JPEG", fixed_quality=80,
            quality_range=(50, 95), cascade_range=(1, 3), generator=torch.Generator().manual_seed(2024),
        )
        acc: Dict[str, List[float]] = {}
        n_total = 0
        # the baseline's global attention cannot hold 50 images at once
        vb = self.cfg.micro_batch or 50
        for b0 in range(0, self.val_cover.shape[0], vb):
            cover = self.val_cover[b0 : b0 + vb]
            secret = self.val_secret[b0 : b0 + vb]
            stego, _, recovered, _ = self.forward(cover, secret, sampler)
            st = stego.clamp(0, 1)
            cm = image_metrics(st, cover)
            cq = image_metrics(quantize(st), cover)
            sm = image_metrics(recovered, secret)
            # real JPEG q = 80 on the 8-bit stego, hiding label JPEG
            lab = codec_tensor("JPEG", cover.shape[0], self.device)
            with _autocast(self.device, self.cfg.amp):
                st_j = quantize(self._hide(cover, secret, lab).float())
            arr = (st_j * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            dec, _ = run_chain_local(arr, [("JPEG", 80)])
            dec_t = torch.from_numpy(dec.transpose(0, 3, 1, 2).copy()).to(self.device).float() / 255.0
            with _autocast(self.device, self.cfg.amp):
                rj = self._recover(dec_t).float()
            rjm = image_metrics(rj, secret)
            n = cover.shape[0]
            n_total += n
            for key, val in (("val_cover_psnr", cm["psnr"]), ("val_cover_ssim", cm["ssim"]),
                             ("val_cover_psnr_u8", cq["psnr"]), ("val_secret_psnr", sm["psnr"]),
                             ("val_secret_ssim", sm["ssim"]), ("val_secret_psnr_realjpeg80", rjm["psnr"])):
                acc.setdefault(key, []).append(float(val.sum()))
            acc.setdefault("val_residual", []).append(float((st - cover).abs().mean()) * 255.0 * n)
        if self.cfg.variant != "stx":
            self.criterion.lambda_hide = keep
        self.system.train()
        return {k: sum(v) / n_total for k, v in acc.items()}

    # ------------------------------------------------------------------ #
    def run(self) -> Dict:
        cfg = self.cfg
        log(f"{cfg.name}: {cfg.total_steps} steps, batch {cfg.batch}, amp {cfg.amp}, compile {cfg.compile}")
        images = load_train_images(self.device)
        self.crops = GpuCropSampler(images, cfg.image_size, seed=cfg.seed)
        sets = eval_sets(cfg.image_size)
        self.val_cover, self.val_secret = pairs_to_tensors(
            sets["coco_val"]["cache"], sets["coco_val"]["pairs"], self.device
        )
        self.maybe_resume()
        save_json(os.path.join(self.out_dir, "config.json"),
                  {"cfg": asdict(cfg), "env": env_info(), "identity_flags": self.identity_flags})
        self.system.train()
        last_push = time.time()
        t_block = time.time()
        steps_block = 0
        sums_t = None
        run_start = time.time()
        while self.step < cfg.total_steps:
            if cfg.variant != "stx":
                self.criterion.lambda_hide = self.lambda_hide(self.step)
            cover, secret = self.crops.batch(cfg.batch)
            codec = self.sampler.sample_codec(cfg.batch).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            micro = cfg.micro_batch or cfg.batch
            vals = None
            for m0 in range(0, cfg.batch, micro):
                cov_m, sec_m = cover[m0 : m0 + micro], secret[m0 : m0 + micro]
                share = cov_m.shape[0] / cfg.batch
                stego, _, _, losses = self.forward(cov_m, sec_m, self.sampler, codec[m0 : m0 + micro])
                self.scaler.scale(losses["loss"] * share).backward()
                part = torch.stack([losses["loss"].detach(), losses["loss_hide"].detach(),
                                    losses["loss_reveal"].detach(), losses["loss_freq"].detach(),
                                    (stego.detach().clamp(0, 1) - cov_m).abs().mean() * 255.0]) * share
                vals = part if vals is None else vals + part
            if cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.step += 1
            steps_block += 1
            sums_t = vals if sums_t is None else sums_t + vals

            if self.step % cfg.val_every == 0 or self.step == cfg.total_steps:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t_block
                self.elapsed += dt
                means = (sums_t / steps_block).tolist()
                if not all(math.isfinite(m) for m in means):
                    raise FloatingPointError(f"non-finite training loss before step {self.step}")
                val = self.validate()
                row = {"step": self.step, "lr": self.optimizer.param_groups[0]["lr"],
                       "lambda_hide": self.lambda_hide(self.step - 1) if cfg.variant != "stx" else 1.0,
                       "train_loss": means[0], "train_loss_hide": means[1], "train_loss_reveal": means[2],
                       "train_loss_freq": means[3], "train_residual": means[4],
                       "sec_per_step": dt / steps_block, "elapsed_h": self.elapsed / 3600.0, **val}
                self.history.append(row)
                self.write_history()
                self.save_last()
                log(f"{cfg.name} step {self.step}/{cfg.total_steps} | lam {row['lambda_hide']:.2f} "
                    f"| loss {means[0]:.4f} | cover {val['val_cover_psnr']:.2f} dB | secret(sim) "
                    f"{val['val_secret_psnr']:.2f} | secret(realJPEG80) {val['val_secret_psnr_realjpeg80']:.2f} "
                    f"| {dt / steps_block:.3f} s/step")
                sums_t = None
                steps_block = 0
                t_block = time.time()
                if time.time() - last_push > cfg.push_every_min * 60:
                    push_results(cfg.name, self.out_dir, exclude_ext=(".tmp", "last.pth"))
                    last_push = time.time()
                if cfg.max_hours and (time.time() - run_start) > cfg.max_hours * 3600:
                    log(f"{cfg.name}: max_hours reached at step {self.step}")
                    break
        meta = {"cfg": asdict(cfg), "steps_done": self.step, "train_hours": self.elapsed / 3600.0, "env": env_info()}
        torch_save(os.path.join(self.out_dir, "weights_final.pth"), self.system.weights_payload(meta))
        save_json(os.path.join(self.out_dir, "train_done.json"), meta)
        return meta


def load_sim_state(path: str) -> Dict:
    p = torch.load(path, map_location="cpu", weights_only=False)
    return p["simulator"] if "simulator" in p else p
