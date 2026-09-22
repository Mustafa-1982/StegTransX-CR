#!/usr/bin/env python3
"""Export weights-only StegTransX-CR checkpoints.

The training checkpoints in ``checkpoints/full/`` also hold the optimiser,
scheduler, AMP scaler and early-stopping state and the per-epoch history. For
inference only three state dictionaries are needed: the hiding network, the
reveal network and the frozen codec simulator. This script copies exactly those
tensors (unchanged, float32) into ``checkpoints/weights/`` with a small
metadata block, then reloads the result with ``torch.load(weights_only=True)``
and ``strict=True`` and checks that every tensor matches the source.

    python scripts/export_weights.py
    python scripts/export_weights.py --regimes multi --out /tmp/weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from release_utils import (
    PARTS,
    REGIMES,
    ROOT,
    WEIGHTS_FORMAT,
    load_system,
    read_checkpoint,
    sha256,
)
from stegtransx_cr import __version__
from stegtransx_cr.config import CODECS


def export(regime: str, source_dir: Path, target_dir: Path) -> Path:
    source = source_dir / f"best_{regime}.pth"
    full = read_checkpoint(source)
    if full.get("experiment") != regime:
        raise ValueError(f"{source} holds experiment {full.get('experiment')!r}, expected {regime!r}")

    payload = {"format": WEIGHTS_FORMAT}
    for part in PARTS:
        payload[part] = full[part]
    payload["meta"] = {
        "regime": regime,
        "package_version": __version__,
        "codecs": list(CODECS),
        "image_size": 256,
        "best_epoch": int(full["epoch"]),
        "best_val_secret_psnr_db": float(full["best_metric"]),
        "source_checkpoint": source.name,
        "source_sha256": sha256(source),
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"stegtransx_cr_{regime}.pth"
    torch.save(payload, target)

    # Reload through the safe loader and compare tensor by tensor.
    loaded = read_checkpoint(target)
    for part in PARTS:
        if list(loaded[part].keys()) != list(full[part].keys()):
            raise RuntimeError(f"{target.name}: {part} keys differ from the source")
        for key, value in full[part].items():
            if not torch.equal(loaded[part][key], value):
                raise RuntimeError(f"{target.name}: tensor {part}.{key} differs from the source")
    load_system(target)  # strict load into a fresh network
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=REGIMES)
    parser.add_argument("--source", type=Path, default=ROOT / "checkpoints" / "full")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "weights")
    args = parser.parse_args()

    for regime in args.regimes:
        target = export(regime, args.source, args.out)
        size_mb = target.stat().st_size / 1e6
        print(f"{regime:8s} -> {target}  ({size_mb:.1f} MB, sha256 {sha256(target)[:16]}...)")


if __name__ == "__main__":
    main()
