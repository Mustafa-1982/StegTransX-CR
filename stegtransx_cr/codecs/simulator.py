"""Unified differentiable multi-codec simulator S(I, k, q).

Specification section 3.4.4 / paper section 3.3.4 and Algorithm 2. A single
interface dispatches to DiffJPEG, DiffWebP, DiffHEIF or DiffAVIF, supports
per-sample codec and quality selection, and chains simulators to model the
cascading recompression of paper section 3.4.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from ..config import CODECS, codec_index
from ..utils import autocast
from .avif import DiffAVIF
from .gaf import straight_through
from .heif import DiffHEIF
from .jpeg import DiffJPEG
from .modules import as_quality_tensor
from .webp import DiffWebP

CodecArg = Union[str, int, torch.Tensor]
QualityArg = Union[int, float, torch.Tensor]


class MultiCodecSimulator(nn.Module):
    """``S(I, k, q) = D_k(I; q)`` with a shared, differentiable interface."""

    def __init__(self, codecs: Sequence[str] = CODECS) -> None:
        super().__init__()
        self.codec_names: Tuple[str, ...] = tuple(codecs)
        self.simulators = nn.ModuleDict(
            {
                "JPEG": DiffJPEG(),
                "WebP": DiffWebP(),
                "HEIF": DiffHEIF(),
                "AVIF": DiffAVIF(),
            }
        )

    # ------------------------------------------------------------------ #
    # Argument handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def as_codec_tensor(codec: CodecArg, batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(codec, torch.Tensor):
            out = codec.to(device=device, dtype=torch.long).reshape(-1)
        elif isinstance(codec, str):
            out = torch.full((batch,), codec_index(codec), device=device, dtype=torch.long)
        else:
            out = torch.full((batch,), int(codec), device=device, dtype=torch.long)
        if out.numel() == 1 and batch > 1:
            out = out.expand(batch)
        return out

    # ------------------------------------------------------------------ #
    # Core interface
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, codec: CodecArg, quality: QualityArg) -> torch.Tensor:
        """Compress ``x`` (B, 3, H, W) in [0, 1] with codec ``k`` at quality ``q``.

        Transform coefficients reach a few thousand while quantisation steps are
        of order ten, a dynamic range that half precision cannot represent
        reliably, so the codec math always runs in float32 even under autocast.
        """
        with autocast(x.device, enabled=False):
            x = x.float()
            batch = x.shape[0]
            codec_ids = self.as_codec_tensor(codec, batch, x.device)
            quality_t = as_quality_tensor(quality, batch, x.device, x.dtype).view(-1)

            unique = torch.unique(codec_ids)
            if unique.numel() == 1:
                name = CODECS[int(unique.item())]
                return self.simulators[name](x, quality_t)

            out = x.new_zeros(x.shape)
            for value in unique.tolist():
                index = (codec_ids == value).nonzero(as_tuple=True)[0]
                name = CODECS[int(value)]
                compressed = self.simulators[name](
                    x.index_select(0, index), quality_t.index_select(0, index)
                )
                out = out.index_copy(0, index, compressed.to(dtype=out.dtype))
            return out

    def cascade(
        self,
        x: torch.Tensor,
        chain: Iterable[Tuple[CodecArg, QualityArg]],
    ) -> torch.Tensor:
        """Apply a sequence of compressions: ``S_kN,qN o ... o S_k1,q1(x)``."""
        out = x
        for codec, quality in chain:
            out = self.forward(out, codec, quality).clamp(0.0, 1.0)
        return out

    # ------------------------------------------------------------------ #
    # Parameter control
    # ------------------------------------------------------------------ #
    def freeze(self) -> "MultiCodecSimulator":
        """Freeze the learned distortion parameters.

        The simulator is a fixed distortion channel during steganography
        training. If its weights were optimised by the steganography loss the
        joint optimum would be a simulator that does nothing, which is exactly
        the degenerate solution the compression stage is meant to prevent.
        """
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self) -> "MultiCodecSimulator":
        """Re-enable gradients, used by the calibration stage."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self

    def learned_parameters(self) -> List[nn.Parameter]:
        """Parameters that calibration against real encoders may update."""
        return [p for p in self.parameters()]


class CompressionSampler:
    """Draws codecs, qualities and cascade chains for a training regime.

    Mirrors paper Algorithm 1: the codec k conditioning the hiding network is
    sampled first, then the distortion channel is applied according to the
    experiment mode.
    """

    def __init__(
        self,
        mode: str,
        codecs: Sequence[str] = CODECS,
        fixed_codec: str = "JPEG",
        fixed_quality: int = 80,
        quality_range: Tuple[int, int] = (50, 95),
        cascade_range: Tuple[int, int] = (1, 3),
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if mode not in {"single", "multi", "cascade"}:
            raise ValueError(f"unknown compression mode {mode!r}")
        self.mode = mode
        self.codec_ids = [codec_index(name) for name in codecs]
        self.fixed_codec_id = codec_index(fixed_codec)
        self.fixed_quality = fixed_quality
        self.quality_range = quality_range
        self.cascade_range = cascade_range
        self.generator = generator

    # -- sampling primitives ------------------------------------------- #
    def _randint(self, low: int, high: int, size: Tuple[int, ...]) -> torch.Tensor:
        return torch.randint(low, high + 1, size, generator=self.generator)

    def sample_codec(self, batch: int) -> torch.Tensor:
        """Codec index per sample, used both for FAA and for compression."""
        if self.mode == "single":
            return torch.full((batch,), self.fixed_codec_id, dtype=torch.long)
        choices = torch.tensor(self.codec_ids, dtype=torch.long)
        picks = torch.randint(0, len(self.codec_ids), (batch,), generator=self.generator)
        return choices[picks]

    def sample_quality(self, batch: int) -> torch.Tensor:
        if self.mode == "single":
            return torch.full((batch,), float(self.fixed_quality))
        low, high = self.quality_range
        return self._randint(low, high, (batch,)).float()

    def sample_chain_length(self) -> int:
        low, high = self.cascade_range
        return int(self._randint(low, high, (1,)).item())

    # -- full distortion channel --------------------------------------- #
    def apply(
        self,
        simulator: MultiCodecSimulator,
        stego: torch.Tensor,
        hiding_codec: torch.Tensor,
    ) -> torch.Tensor:
        """Compress the stego image according to the regime."""
        batch = stego.shape[0]
        device = stego.device

        hiding_codec = hiding_codec.to(device)
        if self.mode in {"single", "multi"}:
            quality = self.sample_quality(batch).to(device)
            return simulator(stego, hiding_codec, quality).clamp(0.0, 1.0)

        # First stage uses the same codec the hiding network was conditioned on.
        # Later stages are extra unknown recompressions, as in spec section 5.5.
        n_stages = self.sample_chain_length()
        chain = [(hiding_codec, self.sample_quality(batch).to(device))]
        for _ in range(n_stages - 1):
            chain.append(
                (
                    self.sample_codec(batch).to(device),
                    self.sample_quality(batch).to(device),
                )
            )
        return simulator.cascade(stego, chain)


__all__ = [
    "CompressionSampler",
    "MultiCodecSimulator",
    "straight_through",
]
