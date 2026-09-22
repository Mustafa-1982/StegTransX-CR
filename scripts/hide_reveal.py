#!/usr/bin/env python3
"""Hide a secret image in a cover image and recover it after real compression.

Three sub-commands:

  demo    hide -> real codec chain -> reveal, with PSNR/SSIM for both images
  hide    sender side: write the stego PNG and the compressed file to share
  reveal  receiver side: recover the secret from any received image file

Examples (run from the release folder):

    python scripts/hide_reveal.py demo --cover cover.jpg --secret secret.png \
        --chain JPEG:80,WebP:80 --out demo_out

    python scripts/hide_reveal.py hide --cover cover.jpg --secret secret.png \
        --codec WebP --quality 80 --out sent

    python scripts/hide_reveal.py reveal --image sent/stego_WebP_q80.webp \
        --out recovered.png

Cover and secret are centre-cropped and resized to 256 x 256, the resolution
the released models were trained and evaluated at. The hiding network is
conditioned on the codec the stego image will first go through (JPEG when no
compression is applied), as in the evaluation code. The reveal network uses
the received image only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from release_utils import (
    ROOT,
    load_image,
    load_image_as_is,
    load_system,
    parse_chain,
    save_png,
)
from stegtransx_cr.codecs import real as real_codecs
from stegtransx_cr.config import CODECS, codec_index
from stegtransx_cr.utils import image_metrics

DEFAULT_WEIGHTS = ROOT / "checkpoints" / "weights" / "stegtransx_cr_multi.pth"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _require_codecs(chain) -> None:
    missing = [codec for codec, _ in chain if not real_codecs.is_available(codec)]
    if missing:
        raise SystemExit(
            f"real encoder unavailable for {sorted(set(missing))}; "
            "install pillow-heif and pillow-avif-plugin (see requirements.txt)"
        )


@torch.no_grad()
def hide(system, cover, secret, condition: str) -> torch.Tensor:
    codec = torch.full((cover.shape[0],), codec_index(condition), dtype=torch.long, device=cover.device)
    return system.hiding(cover, secret, codec).clamp(0.0, 1.0)


@torch.no_grad()
def reveal(system, received) -> torch.Tensor:
    return system.reveal(received.clamp(0.0, 1.0))


def run_chain(stego: torch.Tensor, chain, out_dir: Path, stem: str = "stego") -> torch.Tensor:
    """Pass the stego image through real encoders, writing every file produced."""
    current = stego
    for stage, (codec, quality) in enumerate(chain, start=1):
        payload = real_codecs.encode_bytes(current[0], codec, quality)
        suffix = real_codecs.FILE_SUFFIX[codec]
        name = f"{stem}_{codec}_q{quality}{suffix}" if len(chain) == 1 else f"{stem}_stage{stage}_{codec}_q{quality}{suffix}"
        (out_dir / name).write_bytes(payload)
        current = real_codecs.decode_bytes(payload, current.device, current.dtype).unsqueeze(0)
        print(f"  stage {stage}: {codec} q={quality} -> {name} ({len(payload)} bytes)")
    return current


def _report(label: str, pred: torch.Tensor, target: torch.Tensor) -> None:
    metrics = image_metrics(pred, target)
    print(f"  {label}: PSNR {float(metrics['psnr'][0]):.2f} dB, SSIM {float(metrics['ssim'][0]):.4f}")


def cmd_demo(args) -> None:
    device = _device(args.device)
    chain = parse_chain(args.chain)
    _require_codecs(chain)
    system, info = load_system(args.weights, device)
    print(f"model: {args.weights} ({info.get('regime')})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cover = load_image(args.cover, args.size).to(device)
    secret = load_image(args.secret, args.size).to(device)
    condition = chain[0][0] if chain else "JPEG"

    stego = hide(system, cover, secret, condition)
    save_png(cover, out / "cover.png")
    save_png(secret, out / "secret.png")
    save_png(stego, out / "stego.png")
    received = run_chain(stego, chain, out) if chain else stego
    recovered = reveal(system, received)
    save_png(recovered, out / "recovered.png")

    print(f"hiding conditioned on {condition}; chain: {args.chain}")
    _report("cover vs stego     ", stego, cover)
    _report("secret vs recovered", recovered, secret)
    print(f"images written to {out}")


def cmd_hide(args) -> None:
    device = _device(args.device)
    chain = [] if args.codec.lower() == "none" else parse_chain(f"{args.codec}:{args.quality}")
    _require_codecs(chain)
    system, info = load_system(args.weights, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cover = load_image(args.cover, args.size).to(device)
    secret = load_image(args.secret, args.size).to(device)
    condition = chain[0][0] if chain else "JPEG"

    stego = hide(system, cover, secret, condition)
    save_png(stego, out / "stego.png")
    print(f"model: {args.weights} ({info.get('regime')}); hiding conditioned on {condition}")
    print(f"  wrote {out / 'stego.png'} (lossless)")
    if chain:
        run_chain(stego, chain, out)
    _report("cover vs stego", stego, cover)


def cmd_reveal(args) -> None:
    device = _device(args.device)
    system, info = load_system(args.weights, device)
    received = load_image_as_is(args.image).to(device)
    recovered = reveal(system, received)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_png(recovered, out)
    print(f"model: {args.weights} ({info.get('regime')}); recovered secret written to {out}")
    if args.secret:
        size = received.shape[-1]
        secret = load_image(args.secret, size).to(device)
        _report("secret vs recovered", recovered, secret)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="weights-only or full checkpoint (default: multi-codec model)")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="hide, compress with real encoders, reveal, report metrics")
    demo.add_argument("--cover", required=True)
    demo.add_argument("--secret", required=True)
    demo.add_argument("--chain", default="JPEG:80",
                      help="comma-separated CODEC:QUALITY stages, e.g. JPEG:80,WebP:80; 'none' for no compression")
    demo.add_argument("--size", type=int, default=256)
    demo.add_argument("--out", default="demo_out")
    demo.set_defaults(func=cmd_demo)

    hid = sub.add_parser("hide", help="sender side")
    hid.add_argument("--cover", required=True)
    hid.add_argument("--secret", required=True)
    hid.add_argument("--codec", default="JPEG", help=f"one of {', '.join(CODECS)} or none")
    hid.add_argument("--quality", type=int, default=80)
    hid.add_argument("--size", type=int, default=256)
    hid.add_argument("--out", default="sent")
    hid.set_defaults(func=cmd_hide)

    rev = sub.add_parser("reveal", help="receiver side")
    rev.add_argument("--image", required=True, help="received stego image (any format Pillow can read)")
    rev.add_argument("--out", default="recovered.png")
    rev.add_argument("--secret", help="optional original secret, to report PSNR/SSIM")
    rev.set_defaults(func=cmd_reveal)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
