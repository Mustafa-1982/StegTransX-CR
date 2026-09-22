#!/usr/bin/env python3
"""Check a StegTransX-CR release folder.

1. Every file listed in SHA256SUMS is present and unchanged.
2. Every checkpoint loads with torch.load(weights_only=True) and strict=True.
3. Each weights-only file holds exactly the hiding, reveal and simulator tensors
   of its full training checkpoint and records that checkpoint's SHA-256.
4. Full and weights-only models give bit-identical outputs (hiding, simulated
   and real-codec reveal) on fixed synthetic inputs.

    python scripts/verify_release.py
    python scripts/verify_release.py --skip-checksums
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from release_utils import PARTS, REGIMES, ROOT, load_system, read_checkpoint, sha256
from stegtransx_cr.codecs import real as real_codecs
from stegtransx_cr.config import codec_index
from stegtransx_cr.utils import count_parameters


def check_checksums() -> bool:
    sums = ROOT / "SHA256SUMS"
    if not sums.exists():
        print("SHA256SUMS not found")
        return False
    ok, count = True, 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        path = ROOT / name
        count += 1
        if not path.is_file():
            print(f"  missing: {name}")
            ok = False
        elif sha256(path) != digest:
            print(f"  changed: {name}")
            ok = False
    print(f"checksums: {count} files listed, {'all match' if ok else 'PROBLEMS FOUND'}")
    return ok


def synthetic_batch(seed: int, count: int = 1, size: int = 128) -> torch.Tensor:
    """Smooth, fixed test images (only used to compare two models; 128 px keeps the CPU check fast)."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(count, 3, 8, 8, generator=generator)
    fine = torch.rand(count, 3, 32, 32, generator=generator)
    image = 0.7 * F.interpolate(coarse, size=size, mode="bicubic", align_corners=False)
    image = image + 0.3 * F.interpolate(fine, size=size, mode="bilinear", align_corners=False)
    return image.clamp(0.0, 1.0)


@torch.no_grad()
def model_outputs(system, cover: torch.Tensor, secret: torch.Tensor):
    outputs = {}
    for condition in ("JPEG", "WebP", "HEIF", "AVIF"):
        codec = torch.full((cover.shape[0],), codec_index(condition), dtype=torch.long)
        stego = system.hiding(cover, secret, codec).clamp(0.0, 1.0)
        outputs[f"stego[{condition}]"] = stego
        simulated = system.simulator(stego, codec, 80).clamp(0.0, 1.0)
        outputs[f"reveal(sim {condition} q80)"] = system.reveal(simulated)
    real = real_codecs.encode_decode(outputs["stego[JPEG]"], "JPEG", 80)
    outputs["reveal(real JPEG q80)"] = system.reveal(real)
    return outputs


def check_regime(regime: str) -> bool:
    full_path = ROOT / "checkpoints" / "full" / f"best_{regime}.pth"
    weights_path = ROOT / "checkpoints" / "weights" / f"stegtransx_cr_{regime}.pth"
    full = read_checkpoint(full_path)
    weights = read_checkpoint(weights_path)

    same_tensors = all(
        list(full[part].keys()) == list(weights[part].keys())
        and all(torch.equal(full[part][key], weights[part][key]) for key in full[part])
        for part in PARTS
    )
    source_ok = weights["meta"]["source_sha256"] == sha256(full_path)

    system_full, _ = load_system(full_path)
    system_weights, info = load_system(weights_path)
    cover, secret = synthetic_batch(1), synthetic_batch(2)
    out_full = model_outputs(system_full, cover, secret)
    out_weights = model_outputs(system_weights, cover, secret)
    identical = all(torch.equal(out_full[key], out_weights[key]) for key in out_full)

    hiding = count_parameters(system_weights.hiding) / 1e6
    reveal = count_parameters(system_weights.reveal) / 1e6
    ok = same_tensors and source_ok and identical
    print(
        f"{regime:8s} {'OK ' if ok else 'FAIL'} tensors identical={same_tensors} "
        f"source hash matches={source_ok} outputs identical={identical} | "
        f"best epoch {info['best_epoch']}, best val secret PSNR {info['best_val_secret_psnr_db']:.2f} dB, "
        f"params hiding {hiding:.2f} M + reveal {reveal:.2f} M"
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=REGIMES)
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    ok = True
    if not args.skip_checksums:
        ok &= check_checksums()
    print("real codecs:", ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in real_codecs.available_codecs().items()))
    for regime in args.regimes:
        ok &= check_regime(regime)
    print("release check passed" if ok else "release check FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
