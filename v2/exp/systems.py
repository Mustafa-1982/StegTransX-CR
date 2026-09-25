"""Model variants shared by training and evaluation."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .common import CODECS, IdentityFAA, refresh_identity_flags, setup_package

VARIANTS = ("cond", "nocond", "nofaa", "nofreq", "stx")
REGIMES = ("single", "multi", "cascade")


class System(nn.Module):
    """Hiding network, reveal network and a frozen distortion channel."""

    def __init__(self, variant: str, regime: str) -> None:
        super().__init__()
        setup_package()
        from stegtransx_cr.codecs.simulator import MultiCodecSimulator
        from stegtransx_cr.config import ModelConfig
        from stegtransx_cr.models import HidingNetwork, RevealNetwork

        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        if regime not in REGIMES:
            raise ValueError(f"unknown regime {regime}")
        self.variant = variant
        self.regime = regime
        if variant == "stx":
            from .stx_baseline import build_stx_pair

            self.hiding, self.reveal = build_stx_pair()
        else:
            cfg = ModelConfig()
            self.hiding = HidingNetwork(cfg)
            self.reveal = RevealNetwork(cfg)
            if variant == "nofaa":
                self.hiding.faa = IdentityFAA()
        self.simulator = MultiCodecSimulator()
        self.simulator.freeze()

    # ------------------------------------------------------------------ #
    @property
    def uses_label(self) -> bool:
        return self.variant in ("cond", "nofreq")

    def label_policy(self) -> str:
        """Which codec label the sender feeds the hiding network at test time."""
        if not self.uses_label:
            return "const"
        if self.regime == "single":
            return "fixed:JPEG"
        return "first"

    def hide(self, cover: torch.Tensor, secret: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.variant == "stx":
            return self.hiding(torch.cat([cover, secret], dim=1))
        if not self.uses_label:
            label = torch.zeros_like(label)
        return self.hiding(cover, secret, label)

    def recover(self, received: torch.Tensor) -> torch.Tensor:
        return self.reveal(received)

    def trainable_parameters(self):
        return list(self.hiding.parameters()) + list(self.reveal.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        self.simulator.eval()
        return self

    # ------------------------------------------------------------------ #
    def load_simulator(self, state: Dict) -> Dict[str, bool]:
        self.simulator.load_state_dict(state)
        self.simulator.freeze()
        return refresh_identity_flags(self.simulator)

    def weights_payload(self, meta: Optional[Dict] = None) -> Dict:
        return {
            "format": "stegtransx_cr.weights.v2",
            "variant": self.variant,
            "regime": self.regime,
            "hiding": self.hiding.state_dict(),
            "reveal": self.reveal.state_dict(),
            "simulator": self.simulator.state_dict(),
            "meta": meta or {},
        }


def load_system(path: str, device, variant: Optional[str] = None, regime: Optional[str] = None) -> System:
    """Load a v2 weights file, or a v1 release file (full or weights-only)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") == "stegtransx_cr.weights.v2":
        variant = payload["variant"]
        regime = payload["regime"]
    else:  # v1 release: stegtransx_cr.weights.v1 or a full training checkpoint
        variant = variant or "cond"
        regime = regime or (payload.get("meta", {}) or {}).get("regime") or payload.get("experiment")
    system = System(variant, regime)
    system.hiding.load_state_dict(payload["hiding"])
    system.reveal.load_state_dict(payload["reveal"])
    system.load_simulator(payload["simulator"])
    return system.to(device).eval()


def codec_tensor(name: str, n: int, device) -> torch.Tensor:
    return torch.full((n,), CODECS.index(name), dtype=torch.long, device=device)
