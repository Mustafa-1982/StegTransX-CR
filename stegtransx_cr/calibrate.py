"""Optional calibration of the learned simulator modules against real encoders.

The analytical part of each simulator sets the magnitude of the distortion; the
learned FiLM / deblocking / CDEF modules are meant to capture the artifact
*shape* that the analytical transform cannot express. They start as exact
identities, so calibration is what gives them meaning.

Calibration is a supervised regression: for a batch of images, a codec k and a
quality q, minimise the distance between S(I, k, q) and the real encoder output
C_real(I, k, q). It only touches simulator parameters, never the steganography
networks, and it needs the corresponding Pillow plugin to be installed.

Running this before the three experiments narrows the simulator-to-real gap
that the evaluation reports.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .codecs import real as real_codecs
from .codecs.simulator import MultiCodecSimulator
from .config import CODECS, codec_index
from .utils import load_checkpoint, psnr, save_checkpoint


def _sample_batch(cache: np.ndarray, batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    index = rng.integers(0, cache.shape[0], size=batch_size)
    batch = cache[index].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return torch.from_numpy(batch).to(device)


@torch.no_grad()
def measure_gap(
    simulator: MultiCodecSimulator,
    cache: np.ndarray,
    codecs: Sequence[str] = CODECS,
    qualities: Sequence[int] = (50, 65, 80, 95),
    batch_size: int = 8,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """PSNR between the simulator output and the real encoder output."""
    if device is None:
        try:
            device = next(simulator.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for codec in codecs:
        if not real_codecs.is_available(codec):
            rows.append({"codec": codec, "quality": None, "status": "unavailable"})
            continue
        for quality in qualities:
            images = _sample_batch(cache, batch_size, rng, device)
            simulated = simulator(images, codec, quality)
            actual = real_codecs.encode_decode(images, codec, quality)
            rows.append(
                {
                    "codec": codec,
                    "quality": quality,
                    "status": "ok",
                    "sim_vs_real_psnr": float(psnr(simulated, actual).mean()),
                    "sim_psnr": float(psnr(simulated, images).mean()),
                    "real_psnr": float(psnr(actual, images).mean()),
                }
            )
    return rows


def format_gap(rows: Sequence[Dict]) -> str:
    lines = [f"{'codec':<6}{'q':>5}{'sim dB':>10}{'real dB':>10}{'sim vs real':>13}"]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"{row['codec']:<6}{'-':>5}{'unavailable':>33}")
            continue
        lines.append(
            f"{row['codec']:<6}{row['quality']:>5}"
            f"{row['sim_psnr']:>10.2f}{row['real_psnr']:>10.2f}{row['sim_vs_real_psnr']:>13.2f}"
        )
    return "\n".join(lines)


def calibrate_simulator(
    simulator: MultiCodecSimulator,
    cache: np.ndarray,
    device: torch.device,
    codecs: Optional[Sequence[str]] = None,
    steps: int = 400,
    batch_size: int = 8,
    lr: float = 1e-3,
    quality_range: Sequence[int] = (50, 95),
    seed: int = 7,
    log_every: int = 50,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Fit the learned simulator modules to real encoder output.

    JPEG is skipped: DiffJPEG has no trainable weights, so its loss has no
    ``grad_fn`` and ``backward()`` raises. WebP / HEIF / AVIF keep their
    learned FiLM / deblock / CDEF modules and are the ones being fit.
    """
    codecs = [c for c in (codecs or CODECS) if real_codecs.is_available(c)]
    if not codecs:
        print("calibration skipped: no real codecs available in this environment")
        return {}

    simulator = simulator.to(device)
    simulator.unfreeze()
    simulator.train()

    skipped = [
        name
        for name in codecs
        if not any(p.requires_grad for p in simulator.simulators[name].parameters())
    ]
    codecs = [name for name in codecs if name not in skipped]
    if skipped:
        print(f"  skipping analytical codecs (nothing to train): {', '.join(skipped)}")
    if not codecs:
        print("calibration skipped: no learned simulator modules to fit")
        simulator.freeze()
        return {}

    print(f"calibrating simulators for: {', '.join(codecs)}")
    trainable = [p for p in simulator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    history: Dict[str, List[float]] = {codec: [] for codec in codecs}

    for step in range(steps):
        codec = codecs[int(rng.integers(len(codecs)))]
        quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
        images = _sample_batch(cache, batch_size, rng, device)

        with torch.no_grad():
            target = real_codecs.encode_decode(images, codec, quality)

        simulated = simulator(images, codec, quality)
        # L1 keeps the fit robust to the occasional strong ringing artifact,
        # the spectral term matches the frequency signature of the codec.
        spatial = F.l1_loss(simulated, target)
        spectral = F.l1_loss(
            torch.fft.fft2(simulated.float(), norm="forward").abs(),
            torch.fft.fft2(target.float(), norm="forward").abs(),
        )
        loss = spatial + 0.5 * spectral

        if loss.requires_grad:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        history[codec].append(float(loss.detach()))
        if log_every and step % log_every == 0:
            with torch.no_grad():
                match = float(psnr(simulated.detach(), target).mean())
            print(
                f"  step {step:>4} | {codec:<5} q={quality:<3} | "
                f"loss {float(loss.detach()):.5f} | match {match:.2f} dB"
            )

    simulator.freeze()
    if checkpoint_path:
        save_checkpoint(checkpoint_path, {"simulator": simulator.state_dict(), "codecs": list(codecs)})
        print(f"  saved calibrated simulator to {os.path.basename(checkpoint_path)}")
    return history


def load_calibrated_simulator(
    simulator: MultiCodecSimulator,
    checkpoint_path: str,
    device: torch.device,
) -> bool:
    """Load a previously calibrated simulator. Returns True if the file existed."""
    if not os.path.exists(checkpoint_path):
        return False
    payload = load_checkpoint(checkpoint_path, map_location=str(device))
    simulator.load_state_dict(payload["simulator"])
    simulator.to(device)
    simulator.freeze()
    print(f"  loaded calibrated simulator from {os.path.basename(checkpoint_path)}")
    return True


__all__ = ["calibrate_simulator", "load_calibrated_simulator", "measure_gap", "format_gap"]
