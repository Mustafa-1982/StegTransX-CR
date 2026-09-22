"""Configuration objects for StegTransX-CR.

All hyperparameters follow the technical specification (Table 1) except the
epoch budget and early stopping, which are the agreed Colab protocol.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Sequence, Tuple

# Codec vocabulary. The integer index is the value fed to the codec embedding
# in the frequency-adaptive attention module, so the order must stay fixed.
CODECS: Tuple[str, ...] = ("JPEG", "WebP", "HEIF", "AVIF")
CODEC_TO_INDEX = {name: i for i, name in enumerate(CODECS)}


def codec_index(name: str) -> int:
    """Map a codec name to its embedding index (case-insensitive)."""
    for key, value in CODEC_TO_INDEX.items():
        if key.lower() == name.lower():
            return value
    raise KeyError(f"unknown codec {name!r}; expected one of {CODECS}")


@dataclass
class ModelConfig:
    """Architecture sizes from specification section 3."""

    stem_channels: int = 64          # C1
    stage2_channels: int = 128       # C2
    stage3_channels: int = 256       # C3
    window_size: int = 8             # attention window (spec: 8 x 8)
    num_heads: int = 4
    ffn_expansion: int = 2
    codec_embed_dim: int = 32
    freq_bands: int = 4              # B = 4, kernels {3, 5, 7, 9}
    reveal_channels: int = 64
    reveal_blocks: int = 4
    leaky_slope: float = 0.2


@dataclass
class LossConfig:
    """Loss weights from specification section 4."""

    lambda_hide: float = 1.0         # lambda_1
    lambda_reveal: float = 1.0       # lambda_2
    lambda_freq: float = 0.1         # lambda_f
    beta_charbonnier: float = 1.0    # beta_1
    beta_restrict: float = 10.0      # beta_2
    pyramid_levels: int = 5          # L = 5
    charbonnier_eps: float = 1e-3
    freq_bands: int = 4              # B = 4 radial bands for L_F


@dataclass
class DataConfig:
    """Dataset construction (specification section 5.2)."""

    image_size: int = 256
    num_train_images: int = 800      # DIV2K train HR only
    eval_dataset: str = "COCO"       # paper lists ImageNet and COCO; we use COCO only
    num_val_pairs: int = 50
    num_test_pairs: int = 200
    num_val_images: int = 80         # disjoint COCO images for val pairs
    num_test_images: int = 220       # disjoint COCO images for test pairs
    num_workers: int = 0             # RAM cache: workers add copies and break set_epoch
    seed: int = 42


@dataclass
class ExperimentConfig:
    """One training regime (specification sections 5.3 - 5.5)."""

    name: str                        # "single" | "multi" | "cascade"
    mode: str                        # compression sampling mode
    epochs: int = 250                # Colab protocol (paper: 8000)
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.5, 0.999)
    patience: int = 9                # early stopping on val secret PSNR
    min_delta: float = 1e-3          # dB improvement that counts as progress
    amp: bool = True
    grad_clip: float = 1.0
    # Hiding-loss schedule. Measured at initialisation, the gradient L_H sends
    # into the hiding network is about 120x the one arriving from L_R through
    # the reveal network. Optimised jointly from step one, the pair therefore
    # collapses onto the trivial solution I_stego = I_cover: the cover PSNR
    # looks excellent, no payload is embedded, and L_R never receives a signal
    # to learn from.
    #
    # lambda_1 is held at `lambda_hide_start` for `warmup_epochs` so embedding
    # and extraction are established first, then ramped linearly to its
    # specified value of 1.0 across `ramp_epochs`. The objective that the model
    # spends most of training on, and converges under, is the one in the
    # specification.
    warmup_epochs: int = 30
    ramp_epochs: int = 120
    lambda_hide_start: float = 0.0
    # Warn when the stego residual has collapsed while recovery is still poor.
    collapse_residual: float = 0.5   # mean |stego - cover| on the 0-255 scale
    collapse_psnr: float = 20.0      # dB
    # Compression sampling
    fixed_codec: str = "JPEG"
    fixed_quality: int = 80
    quality_range: Tuple[int, int] = (50, 95)
    cascade_range: Tuple[int, int] = (1, 3)
    codecs: Sequence[str] = field(default_factory=lambda: list(CODECS))
    # Bookkeeping
    val_every: int = 1
    log_every: int = 50              # iterations between console updates


def experiment_a() -> ExperimentConfig:
    """Experiment A: single-codec regime, DiffJPEG at q = 80."""
    return ExperimentConfig(name="single", mode="single", fixed_codec="JPEG", fixed_quality=80)


def experiment_b() -> ExperimentConfig:
    """Experiment B: multi-codec universal regime, k ~ K, q ~ U(50, 95)."""
    return ExperimentConfig(name="multi", mode="multi")


def experiment_c() -> ExperimentConfig:
    """Experiment C: cascading regime, N ~ U(1, 3) sequential recompressions."""
    return ExperimentConfig(name="cascade", mode="cascade")


def all_experiments() -> List[ExperimentConfig]:
    return [experiment_a(), experiment_b(), experiment_c()]


@dataclass
class Paths:
    """Filesystem layout. On Colab, point `root` at a Google Drive folder."""

    root: str = "."

    @property
    def data(self) -> str:
        return os.path.join(self.root, "data")

    @property
    def outputs(self) -> str:
        return self.root

    @property
    def checkpoints(self) -> str:
        return os.path.join(self.root, "checkpoints")

    @property
    def history(self) -> str:
        return os.path.join(self.root, "history")

    @property
    def figures(self) -> str:
        return os.path.join(self.root, "figures")

    @property
    def tables(self) -> str:
        return os.path.join(self.root, "tables")

    @property
    def samples(self) -> str:
        return os.path.join(self.root, "samples")

    def ensure(self) -> "Paths":
        for directory in (
            self.data,
            self.checkpoints,
            self.history,
            self.figures,
            self.tables,
            self.samples,
        ):
            os.makedirs(directory, exist_ok=True)
        return self


@dataclass
class Config:
    """Top-level configuration bundle."""

    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiments: List[ExperimentConfig] = field(default_factory=all_experiments)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, default=str)
