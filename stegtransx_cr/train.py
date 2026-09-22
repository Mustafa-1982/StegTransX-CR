"""Training loop for the three compression regimes.

Follows paper Algorithm 1 with the agreed Colab protocol: at most 500 epochs
per experiment and early stopping with patience 9 on validation
secret-recovery PSNR, instead of the paper's fixed 8,000 epochs.

Every epoch appends one row to ``history_<name>.csv`` and rewrites
``last_<name>.pth``, so a disconnected Colab session can resume without losing
work.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .codecs.simulator import CompressionSampler, MultiCodecSimulator
from .config import ExperimentConfig, LossConfig, ModelConfig, Paths
from .losses import StegTransXCRLoss
from .models import HidingNetwork, RevealNetwork
from .utils import (
    MeterDict,
    Timer,
    autocast,
    format_seconds,
    image_metrics,
    load_checkpoint,
    make_grad_scaler,
    save_checkpoint,
)

HISTORY_FIELDS = [
    "epoch",
    "lr",
    "lambda_hide",
    "train_loss",
    "train_loss_hide",
    "train_loss_reveal",
    "train_loss_freq",
    "val_loss",
    "val_cover_psnr",
    "val_cover_ssim",
    "val_secret_psnr",
    "val_secret_ssim",
    "train_residual",
    "val_residual",
    "epoch_seconds",
    "is_best",
]


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """Stops training after ``patience`` epochs without a meaningful gain."""

    def __init__(self, patience: int = 9, min_delta: float = 1e-3) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Records a monitored value and returns True when it is a new best."""
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> Dict:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: Dict) -> None:
        self.best = state.get("best", float("-inf"))
        self.bad_epochs = state.get("bad_epochs", 0)


# --------------------------------------------------------------------------- #
# The system under training
# --------------------------------------------------------------------------- #
class StegTransXCR(nn.Module):
    """Hiding network, reveal network and the frozen distortion channel."""

    def __init__(
        self,
        model_cfg: Optional[ModelConfig] = None,
        simulator: Optional[MultiCodecSimulator] = None,
    ) -> None:
        super().__init__()
        model_cfg = model_cfg or ModelConfig()
        self.hiding = HidingNetwork(model_cfg)
        self.reveal = RevealNetwork(model_cfg)
        self.simulator = simulator or MultiCodecSimulator()
        self.simulator.freeze()

    def reset_stego_networks(self) -> "StegTransXCR":
        """Re-initialise hiding and reveal, keeping the calibrated simulator.

        Experiments A, B and C must be independent. Reusing the same
        ``StegTransXCR`` object without this reset would train B from A's
        weights and C from B's, which is not what the paper reports.
        """
        device = next(self.hiding.parameters()).device
        hide_cfg = self.hiding.cfg
        reveal_cfg = self.reveal.cfg
        self.hiding = HidingNetwork(hide_cfg).to(device)
        self.reveal = RevealNetwork(reveal_cfg).to(device)
        return self

    def trainable_parameters(self):
        """Only the steganography networks are optimised."""
        return list(self.hiding.parameters()) + list(self.reveal.parameters())

    def train(self, mode: bool = True) -> "StegTransXCR":
        super().train(mode)
        # The distortion channel is a fixed environment, never a trained module.
        self.simulator.eval()
        return self


@dataclass
class TrainState:
    epoch: int = 0
    history: List[Dict] = field(default_factory=list)
    best_metric: float = float("-inf")
    elapsed: float = 0.0
    stopped_early: bool = False


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Runs one experiment (single-codec, multi-codec or cascading)."""

    def __init__(
        self,
        system: StegTransXCR,
        experiment: ExperimentConfig,
        paths: Paths,
        device: torch.device,
        loss_cfg: Optional[LossConfig] = None,
        max_hours: Optional[float] = None,
    ) -> None:
        self.system = system.to(device)
        self.experiment = experiment
        self.paths = paths.ensure()
        self.device = device
        self.max_hours = max_hours
        self.criterion = StegTransXCRLoss(loss_cfg or LossConfig()).to(device)

        self.optimizer = torch.optim.AdamW(
            self.system.trainable_parameters(),
            lr=experiment.lr,
            betas=experiment.betas,
            weight_decay=experiment.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=experiment.epochs, eta_min=experiment.lr * 0.01
        )
        self.use_amp = experiment.amp and device.type == "cuda"
        self.scaler = make_grad_scaler(device, self.use_amp)

        self.sampler = CompressionSampler(
            mode=experiment.mode,
            codecs=experiment.codecs,
            fixed_codec=experiment.fixed_codec,
            fixed_quality=experiment.fixed_quality,
            quality_range=experiment.quality_range,
            cascade_range=experiment.cascade_range,
            generator=torch.Generator().manual_seed(1234),
        )
        self.stopper = EarlyStopping(experiment.patience, experiment.min_delta)
        self.state = TrainState()

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    @property
    def best_path(self) -> str:
        return os.path.join(self.paths.checkpoints, f"best_{self.experiment.name}.pth")

    @property
    def last_path(self) -> str:
        return os.path.join(self.paths.checkpoints, f"last_{self.experiment.name}.pth")

    @property
    def history_path(self) -> str:
        return os.path.join(self.paths.history, f"history_{self.experiment.name}.csv")

    # ------------------------------------------------------------------ #
    # One optimisation step
    # ------------------------------------------------------------------ #
    def lambda_hide_at(self, epoch: int) -> float:
        """Hold lambda_1 low, then ramp it to the specified value.

        Epochs ``0 .. warmup_epochs-1`` stay at the start value. The next
        ``ramp_epochs`` epochs rise linearly and the last of those is exactly
        the specified weight, so early stopping can start on the real objective.
        """
        final = self.criterion.cfg.lambda_hide
        start = self.experiment.lambda_hide_start * final
        hold = self.experiment.warmup_epochs
        ramp = self.experiment.ramp_epochs
        if epoch < hold:
            return start
        if ramp <= 0:
            return final
        progress = min(1.0, (epoch - hold + 1) / float(ramp))
        return start + (final - start) * progress

    def _forward(self, cover: torch.Tensor, secret: torch.Tensor, sampler: CompressionSampler):
        codec = sampler.sample_codec(cover.shape[0]).to(self.device)
        with autocast(self.device, enabled=self.use_amp):
            stego = self.system.hiding(cover, secret, codec)
        stego = stego.float()
        # The codec and the metrics see a valid image; the losses see the raw
        # output so the restriction term keeps it inside [0, 1] by itself.
        stego_clamped = stego.clamp(0.0, 1.0)
        compressed = sampler.apply(self.system.simulator, stego_clamped, codec)
        with autocast(self.device, enabled=self.use_amp):
            recovered = self.system.reveal(compressed)
        losses = self.criterion(stego, cover, recovered.float(), secret, compressed)
        return stego_clamped, compressed, recovered, losses

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.system.train()
        self.criterion.lambda_hide = self.lambda_hide_at(epoch)
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)

        meters = MeterDict()
        for iteration, (cover, secret) in enumerate(loader):
            cover = cover.to(self.device, non_blocking=True)
            secret = secret.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            stego, _, _, losses = self._forward(cover, secret, self.sampler)

            self.scaler.scale(losses["loss"]).backward()
            if self.experiment.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.system.trainable_parameters(), self.experiment.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            stats = {key: float(value.detach()) for key, value in losses.items()}
            # Mean payload strength on the 0-255 scale. If this falls to zero
            # the hiding network has stopped embedding anything.
            stats["residual"] = float((stego.detach() - cover).abs().mean()) * 255.0
            meters.update(stats, cover.shape[0])
            if self.experiment.log_every and iteration % self.experiment.log_every == 0:
                print(
                    f"    iter {iteration:>4} | loss {float(losses['loss']):.4f} "
                    f"| hide {float(losses['loss_hide']):.4f} "
                    f"| reveal {float(losses['loss_reveal']):.4f} "
                    f"| freq {float(losses['loss_freq']):.4f}",
                    flush=True,
                )
        return meters.averages()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validation uses a fixed compression seed so epochs are comparable."""
        self.system.eval()
        # Report the validation loss under the specified weights, not the
        # warm-up ones, so the curve is comparable across all epochs.
        warmup_lambda = self.criterion.lambda_hide
        self.criterion.lambda_hide = self.criterion.cfg.lambda_hide
        sampler = CompressionSampler(
            mode=self.experiment.mode,
            codecs=self.experiment.codecs,
            fixed_codec=self.experiment.fixed_codec,
            fixed_quality=self.experiment.fixed_quality,
            quality_range=self.experiment.quality_range,
            cascade_range=self.experiment.cascade_range,
            generator=torch.Generator().manual_seed(2024),
        )

        meters = MeterDict()
        for cover, secret in loader:
            cover = cover.to(self.device, non_blocking=True)
            secret = secret.to(self.device, non_blocking=True)
            stego, compressed, recovered, losses = self._forward(cover, secret, sampler)

            cover_metrics = image_metrics(stego.float(), cover)
            secret_metrics = image_metrics(recovered.float(), secret)
            meters.update(
                {
                    "val_loss": float(losses["loss"]),
                    "val_cover_psnr": float(cover_metrics["psnr"].mean()),
                    "val_cover_ssim": float(cover_metrics["ssim"].mean()),
                    "val_secret_psnr": float(secret_metrics["psnr"].mean()),
                    "val_secret_ssim": float(secret_metrics["ssim"].mean()),
                    "val_residual": float((stego - cover).abs().mean()) * 255.0,
                },
                cover.shape[0],
            )
        self.criterion.lambda_hide = warmup_lambda
        if not meters.averages():
            raise RuntimeError("validation loader produced no batches")
        return meters.averages()

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def _payload(self) -> Dict:
        return {
            "experiment": self.experiment.name,
            "hiding": self.system.hiding.state_dict(),
            "reveal": self.system.reveal.state_dict(),
            "simulator": self.system.simulator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "stopper": self.stopper.state_dict(),
            "epoch": self.state.epoch,
            "history": self.state.history,
            "best_metric": self.state.best_metric,
            "elapsed": self.state.elapsed,
            "stopped_early": self.state.stopped_early,
        }

    def maybe_resume(self) -> bool:
        if not os.path.exists(self.last_path):
            return False
        payload = load_checkpoint(self.last_path, map_location=str(self.device))
        self.system.hiding.load_state_dict(payload["hiding"])
        self.system.reveal.load_state_dict(payload["reveal"])
        self.system.simulator.load_state_dict(payload["simulator"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.stopper.load_state_dict(payload["stopper"])
        self.state.epoch = payload["epoch"]
        self.state.history = payload["history"]
        self.state.best_metric = payload["best_metric"]
        self.state.elapsed = payload.get("elapsed", 0.0)
        self.state.stopped_early = payload.get("stopped_early", False)
        print(f"  resumed {self.experiment.name} from epoch {self.state.epoch}")
        return True

    def _warn_if_collapsed(self, row: Dict, epoch: int) -> None:
        """Flag the trivial solution early instead of after a wasted session."""
        if epoch < self.experiment.warmup_epochs:
            return
        residual = row.get("val_residual", float("inf"))
        secret_psnr = row.get("val_secret_psnr", float("inf"))
        if residual < self.experiment.collapse_residual and secret_psnr < self.experiment.collapse_psnr:
            print(
                "  WARNING: the stego residual has collapsed "
                f"({residual:.3f}/255) while secret recovery is still "
                f"{secret_psnr:.2f} dB. The hiding network is embedding nothing. "
                "Extend warmup_epochs / ramp_epochs, or lower lambda_hide, "
                "and restart this experiment.",
                flush=True,
            )

    def _rewrite_history(self) -> None:
        """Write the in-memory log from scratch so resume cannot duplicate rows."""
        with open(self.history_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            for row in self.state.history:
                writer.writerow({key: row.get(key, "") for key in HISTORY_FIELDS})

    def _append_history(self, row: Dict) -> None:
        exists = os.path.exists(self.history_path)
        with open(self.history_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in HISTORY_FIELDS})

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume: bool = True,
    ) -> List[Dict]:
        if len(train_loader) == 0:
            raise RuntimeError(
                "train loader produced no batches; lower batch_size or check the DIV2K cache"
            )
        if len(val_loader) == 0:
            raise RuntimeError("validation loader is empty; check the COCO cache and splits")

        resumed = False
        if resume:
            resumed = self.maybe_resume()
            if resumed:
                self._rewrite_history()
                if self.state.stopped_early or self.state.epoch >= self.experiment.epochs:
                    print(
                        f"  {self.experiment.name} already finished "
                        f"(epoch {self.state.epoch}, best secret PSNR {self.state.best_metric:.2f} dB)"
                    )
                    return self.state.history
        if not resumed and os.path.exists(self.history_path):
            os.remove(self.history_path)

        print(
            f"\n=== Experiment {self.experiment.name} "
            f"(mode={self.experiment.mode}, max {self.experiment.epochs} epochs, "
            f"patience {self.experiment.patience}) ==="
        )
        timer = Timer()
        start_epoch = self.state.epoch
        freeze_stop = self.experiment.warmup_epochs + self.experiment.ramp_epochs

        for epoch in range(start_epoch, self.experiment.epochs):
            epoch_timer = Timer()
            train_stats = self.train_epoch(train_loader, epoch)
            if not train_stats:
                raise RuntimeError(
                    "train loader produced no batches; lower batch_size or check the DIV2K cache"
                )

            if (epoch + 1) % self.experiment.val_every == 0:
                val_stats = self.validate(val_loader)
            else:
                val_stats = self.state.history[-1] if self.state.history else {}

            monitored = val_stats.get("val_secret_psnr", float("-inf"))
            # Do not elect a "best" checkpoint until lambda_1 has reached its
            # specified value. A warmup model that ignores hiding loss is not
            # the result of this experiment.
            can_stop = epoch + 1 >= freeze_stop
            is_best = False
            if can_stop:
                is_best = self.stopper.step(monitored)
                if is_best:
                    self.state.best_metric = monitored

            self.scheduler.step()
            self.state.epoch = epoch + 1
            self.state.elapsed += epoch_timer.elapsed()

            row = {
                "epoch": epoch + 1,
                "lr": self.optimizer.param_groups[0]["lr"],
                "lambda_hide": self.lambda_hide_at(epoch),
                "train_loss": train_stats.get("loss", 0.0),
                "train_loss_hide": train_stats.get("loss_hide", 0.0),
                "train_loss_reveal": train_stats.get("loss_reveal", 0.0),
                "train_loss_freq": train_stats.get("loss_freq", 0.0),
                "train_residual": train_stats.get("residual", 0.0),
                "epoch_seconds": epoch_timer.elapsed(),
                "is_best": int(is_best),
                **val_stats,
            }
            self.state.history.append(row)
            self._append_history(row)
            if is_best:
                save_checkpoint(self.best_path, self._payload())
            save_checkpoint(self.last_path, self._payload())

            if can_stop:
                progress = "best" if is_best else f"no gain x{self.stopper.bad_epochs}"
            else:
                progress = "warmup"
            print(
                f"  epoch {epoch + 1:>3}/{self.experiment.epochs} "
                f"| lam1 {row['lambda_hide']:.2f} "
                f"| train {row['train_loss']:.4f} "
                f"| cover {row.get('val_cover_psnr', 0.0):.2f} dB "
                f"| secret {row.get('val_secret_psnr', 0.0):.2f} dB "
                f"| residual {row.get('val_residual', 0.0):.2f}/255 "
                f"| {progress} "
                f"| {epoch_timer}",
                flush=True,
            )
            self._warn_if_collapsed(row, epoch)

            if can_stop and self.stopper.should_stop:
                self.state.stopped_early = True
                save_checkpoint(self.last_path, self._payload())
                print(
                    f"  early stop at epoch {epoch + 1}: no improvement for "
                    f"{self.experiment.patience} epochs "
                    f"(best secret PSNR {self.stopper.best:.2f} dB)"
                )
                break

            if self.max_hours and timer.elapsed() > self.max_hours * 3600:
                print(f"  time budget of {self.max_hours} h reached; stopping")
                break

        print(
            f"  finished {self.experiment.name} in {format_seconds(timer.elapsed())} "
            f"| best secret PSNR {self.state.best_metric:.2f} dB "
            f"| checkpoint {os.path.basename(self.best_path)}"
        )
        return self.state.history


def learning_sanity_check(
    system: StegTransXCR,
    loader: DataLoader,
    device: torch.device,
    steps: int = 300,
    lr: float = 2e-4,
    lambda_hide: float = 0.0,
    loss_cfg: Optional[LossConfig] = None,
    log_every: int = 50,
) -> Dict[str, float]:
    """Short run that answers one question: is the payload being learned?

    Trains a throwaway copy of the weights for a few hundred steps with the
    hiding pressure switched off, so the only thing being optimised is
    embed-then-extract. Secret PSNR should climb clearly above its starting
    value. If it stays flat, no amount of full-length training will help and
    something upstream is wrong.

    Run this before committing a Colab session to the full experiments.
    """
    import copy

    probe = copy.deepcopy(system).to(device)
    probe.train()
    criterion = StegTransXCRLoss(loss_cfg or LossConfig()).to(device)
    criterion.lambda_hide = lambda_hide
    optimizer = torch.optim.AdamW(probe.trainable_parameters(), lr=lr, betas=(0.5, 0.999))
    simulator = probe.simulator

    first_psnr = None
    last_psnr = 0.0
    step = 0
    print(f"learning sanity check: {steps} steps, lambda_hide = {lambda_hide}")
    while step < steps:
        progressed = False
        for cover, secret in loader:
            if step >= steps:
                break
            progressed = True
            cover = cover.to(device)
            secret = secret.to(device)
            codec = torch.zeros(cover.shape[0], dtype=torch.long, device=device)

            stego = probe.hiding(cover, secret, codec)
            compressed = simulator(stego.clamp(0, 1), codec, 80)
            recovered = probe.reveal(compressed)
            losses = criterion(stego, cover, recovered, secret, compressed)

            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(probe.trainable_parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                from .utils import psnr as _psnr

                last_psnr = float(_psnr(recovered, secret).mean())
            if first_psnr is None:
                first_psnr = last_psnr
            if log_every and step % log_every == 0:
                print(
                    f"  step {step:>4} | L_R {float(losses['loss_reveal']):.4f} "
                    f"| secret {last_psnr:.2f} dB"
                )
            step += 1
        if not progressed:
            raise RuntimeError("sanity check ran zero steps; the train loader is empty")

    gain = last_psnr - (first_psnr or 0.0)
    verdict = "learning" if gain > 1.0 else "NOT learning - investigate before the long runs"
    print(f"  secret PSNR {first_psnr:.2f} -> {last_psnr:.2f} dB (gain {gain:+.2f} dB): {verdict}")
    return {"start_psnr": first_psnr or 0.0, "end_psnr": last_psnr, "gain": gain}


def load_best(system: StegTransXCR, path: str, device: torch.device) -> StegTransXCR:
    """Load a saved experiment checkpoint into a system for evaluation."""
    payload = load_checkpoint(path, map_location=str(device))
    system.hiding.load_state_dict(payload["hiding"])
    system.reveal.load_state_dict(payload["reveal"])
    system.simulator.load_state_dict(payload["simulator"])
    return system.to(device).eval()


__all__ = [
    "EarlyStopping",
    "StegTransXCR",
    "Trainer",
    "TrainState",
    "learning_sanity_check",
    "load_best",
]
