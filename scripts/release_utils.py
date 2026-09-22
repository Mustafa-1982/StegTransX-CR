"""Helpers shared by the release scripts.

These helpers are not part of the ``stegtransx_cr`` training package, which is
shipped exactly as it was used for the reported runs.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stegtransx_cr.codecs import real as real_codecs  # noqa: E402
from stegtransx_cr.config import CODECS, ModelConfig  # noqa: E402
from stegtransx_cr.data import load_and_resize  # noqa: E402
from stegtransx_cr.train import StegTransXCR  # noqa: E402

WEIGHTS_FORMAT = "stegtransx_cr.weights.v1"
REGIMES = ("single", "multi", "cascade")
PARTS = ("hiding", "reveal", "simulator")
IMAGE_SIZE = 256


def sha256(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path) -> Dict:
    """Load a full or weights-only checkpoint with the pickle-safe loader.

    Both formats contain only dictionaries, lists, numbers, strings and
    tensors, so ``weights_only=True`` is sufficient and no arbitrary code is
    executed while loading.
    """
    return torch.load(str(path), map_location="cpu", weights_only=True)


def describe(payload: Dict) -> Dict:
    """Short description of a checkpoint in either format."""
    if "meta" in payload:
        return dict(payload["meta"])
    return {
        "regime": payload.get("experiment"),
        "best_epoch": payload.get("epoch"),
        "best_val_secret_psnr_db": payload.get("best_metric"),
    }


def load_system(path, device: str = "cpu") -> Tuple[StegTransXCR, Dict]:
    """Build the network and load hiding, reveal and simulator weights strictly."""
    payload = read_checkpoint(path)
    system = StegTransXCR(ModelConfig())
    for part in PARTS:
        getattr(system, part).load_state_dict(payload[part], strict=True)
    system = system.to(device).eval()
    return system, describe(payload)


def load_image(path, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Centre-crop and bicubic-resize, as in training; returns (1, 3, H, W) in [0, 1]."""
    array = load_and_resize(str(path), size)
    return image_array_to_tensor(array)


def load_image_as_is(path) -> torch.Tensor:
    """Read a received image without any geometric change."""
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return image_array_to_tensor(array)


def image_array_to_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(255.0)
    return tensor.unsqueeze(0)


def save_png(tensor: torch.Tensor, path) -> None:
    """Save a (1, 3, H, W) or (3, H, W) tensor in [0, 1] as a lossless 8-bit PNG."""
    image = tensor[0] if tensor.dim() == 4 else tensor
    real_codecs.tensor_to_pil(image).save(str(path), format="PNG")


def parse_chain(text: str):
    """'JPEG:80,WebP:80' -> [('JPEG', 80), ('WebP', 80)]; 'none' -> []."""
    if text.strip().lower() in {"", "none"}:
        return []
    chain = []
    for item in text.split(","):
        name, _, quality = item.strip().partition(":")
        match = [codec for codec in CODECS if codec.lower() == name.strip().lower()]
        if not match:
            raise ValueError(f"unknown codec {name!r}; expected one of {CODECS}")
        chain.append((match[0], int(quality or 80)))
    return chain
