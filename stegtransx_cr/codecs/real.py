"""Real encoder round-trips for evaluation.

The differentiable simulator S is what makes training possible, but it is an
approximation. These helpers push tensors through the genuine libjpeg /
libwebp / libheif / libavif encoders so the reported numbers correspond to
actual .jpg, .webp, .heic and .avif files.

HEIF and AVIF need optional plugins:

    pip install pillow-heif pillow-avif-plugin

If a plugin is missing the codec is reported as unavailable rather than
silently falling back to something else.
"""

from __future__ import annotations

import io
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# Optional plugin registration.
try:  # pragma: no cover - depends on the runtime environment
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    try:
        pillow_heif.register_avif_opener()
    except Exception:
        pass
except Exception:  # pragma: no cover
    pillow_heif = None

try:  # pragma: no cover
    import pillow_avif  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    pillow_avif = None


PIL_FORMAT = {
    "JPEG": "JPEG",
    "WebP": "WEBP",
    "HEIF": "HEIF",
    "AVIF": "AVIF",
}

# pillow-heif sometimes registers as HEIC rather than HEIF.
_FORMAT_CANDIDATES = {
    "JPEG": ("JPEG",),
    "WebP": ("WEBP",),
    "HEIF": ("HEIF", "HEIC"),
    "AVIF": ("AVIF",),
}

FILE_SUFFIX = {
    "JPEG": ".jpg",
    "WebP": ".webp",
    "HEIF": ".heic",
    "AVIF": ".avif",
}

_WORKING_FORMAT: Dict[str, str] = {}


def _probe(codec: str) -> bool:
    """Try a tiny encode/decode round trip to see whether the codec works."""
    image = Image.new("RGB", (16, 16), (127, 127, 127))
    for fmt in _FORMAT_CANDIDATES[codec]:
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=fmt, quality=80)
            buffer.seek(0)
            Image.open(buffer).convert("RGB")
            _WORKING_FORMAT[codec] = fmt
            return True
        except Exception:
            continue
    return False


_AVAILABILITY: Dict[str, bool] = {}


def available_codecs(refresh: bool = False) -> Dict[str, bool]:
    """Which real codecs this environment can actually encode and decode."""
    global _AVAILABILITY
    if refresh or not _AVAILABILITY:
        _AVAILABILITY = {name: _probe(name) for name in PIL_FORMAT}
    return dict(_AVAILABILITY)


def is_available(codec: str) -> bool:
    return available_codecs().get(codec, False)


def availability_report(refresh: bool = False) -> str:
    lines = ["Real codec availability:"]
    for name, ok in available_codecs(refresh=refresh).items():
        status = "available" if ok else "MISSING (install pillow-heif / pillow-avif-plugin)"
        lines.append(f"  {name:<5} {status}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tensor <-> PIL
# --------------------------------------------------------------------------- #
def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """``image``: (3, H, W) in [0, 1]."""
    array = (image.detach().clamp(0, 1).cpu().float().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array.transpose(1, 2, 0), mode="RGB")


def pil_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return tensor.to(device=device, dtype=dtype) / 255.0


def encode_bytes(image: torch.Tensor, codec: str, quality: int) -> bytes:
    """Encode a single (3, H, W) image and return the raw file bytes."""
    buffer = io.BytesIO()
    pil = tensor_to_pil(image)
    save_kwargs = {"quality": int(quality)}
    if codec == "JPEG":
        save_kwargs["subsampling"] = "4:2:0"
    fmt = _WORKING_FORMAT.get(codec) or PIL_FORMAT[codec]
    pil.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def decode_bytes(payload: bytes, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    with io.BytesIO(payload) as buffer:
        image = Image.open(buffer)
        image.load()
        return pil_to_tensor(image, device, dtype)


@torch.no_grad()
def encode_decode(batch: torch.Tensor, codec: str, quality: int) -> torch.Tensor:
    """Round-trip a whole batch (B, 3, H, W) through a real encoder."""
    if not is_available(codec):
        raise RuntimeError(
            f"real {codec} codec is unavailable in this environment; "
            "install pillow-heif and/or pillow-avif-plugin"
        )
    outputs = [
        decode_bytes(encode_bytes(image, codec, quality), batch.device, batch.dtype)
        for image in batch
    ]
    return torch.stack(outputs, dim=0)


@torch.no_grad()
def encode_decode_cascade(batch: torch.Tensor, chain: Sequence[Tuple[str, int]]) -> torch.Tensor:
    """Apply a sequence of real recompressions, e.g. JPEG -> WebP -> JPEG."""
    out = batch
    for codec, quality in chain:
        out = encode_decode(out, codec, quality)
    return out


@torch.no_grad()
def file_sizes(batch: torch.Tensor, codec: str, quality: int) -> List[int]:
    """Encoded size in bytes for each image, useful as a sanity check."""
    return [len(encode_bytes(image, codec, quality)) for image in batch]


def filter_available(chain: Iterable[Tuple[str, int]]) -> bool:
    """True when every codec in a cascade chain can actually be used."""
    return all(is_available(codec) for codec, _ in chain)
