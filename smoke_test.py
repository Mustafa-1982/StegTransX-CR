"""Fast end-to-end check of the StegTransX-CR pipeline.

Runs the whole flow at a tiny scale: synthetic images, 64 x 64 crops, a couple
of epochs per regime, then the benchmark, figures and LaTeX export. It takes a
few minutes on CPU and under a minute on a GPU.

Run this once after uploading the project to Colab. If it finishes, the long
experiments will not fail on a missing dependency, a shape bug or a path
problem several hours in.

    python smoke_test.py --root /content/drive/MyDrive/stegtransx_cr_smoke
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch

from stegtransx_cr import PROTOCOL_NOTES
from stegtransx_cr.calibrate import format_gap, measure_gap
from stegtransx_cr.codecs import real as real_codecs
from stegtransx_cr.config import (
    Config,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    Paths,
)
from stegtransx_cr.data import FixedPairDataset, TrainPairDataset, build_pairs, make_synthetic_cache
from stegtransx_cr.eval import (
    export_tables,
    plot_convergence,
    plot_experiment_comparison,
    plot_quality_sweep,
    qualitative_grid,
    run_benchmark,
    save_sample_images,
)
from stegtransx_cr.train import StegTransXCR, Trainer, learning_sanity_check
from stegtransx_cr.utils import count_parameters, get_device, seed_everything


def build_tiny_bundle(size: int = 64, count: int = 24, seed: int = 42):
    cache = make_synthetic_cache(count, size, seed)
    eval_cache = make_synthetic_cache(12, size, seed + 1)
    pairs = build_pairs(eval_cache.shape[0], 8, seed)
    return (
        TrainPairDataset(cache, seed=seed),
        FixedPairDataset(eval_cache, pairs[:4]),
        FixedPairDataset(eval_cache, pairs[4:]),
        cache,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="smoke_run", help="output directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sanity-steps", type=int, default=20)
    parser.add_argument("--keep", action="store_true", help="keep the output directory")
    args = parser.parse_args()

    seed_everything(42)
    device = get_device()
    print(f"device: {device}")
    print(real_codecs.availability_report())
    print("\nprotocol:")
    for note in PROTOCOL_NOTES:
        print(f"  - {note}")

    paths = Paths(root=args.root).ensure()
    train_ds, val_ds, test_ds, cache = build_tiny_bundle(args.image_size)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    system = StegTransXCR(ModelConfig())
    print(
        f"\nparameters: hiding {count_parameters(system.hiding) / 1e6:.2f}M, "
        f"reveal {count_parameters(system.reveal) / 1e6:.2f}M, "
        f"total trainable {sum(p.numel() for p in system.trainable_parameters()) / 1e6:.2f}M"
    )

    print("\nsimulator vs real encoders before training:")
    print(format_gap(measure_gap(system.simulator.to(device), cache, batch_size=4, device=device)))

    print()
    learning_sanity_check(system, train_loader, device, steps=args.sanity_steps, log_every=0)

    histories = {}
    for name, mode in (("single", "single"), ("multi", "multi"), ("cascade", "cascade")):
        experiment = ExperimentConfig(
            name=name,
            mode=mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=9,
            log_every=0,
            warmup_epochs=1,
            ramp_epochs=1,
        )
        system.reset_stego_networks()
        trainer = Trainer(system, experiment, paths, device)
        trainer.fit(train_loader, val_loader, resume=False)
        histories[name] = trainer.history_path
        plot_convergence(trainer.history_path, paths.figures, name)

    plot_experiment_comparison(histories, paths.figures)

    print("\nbenchmark (cascade checkpoint):")
    rows = run_benchmark(
        system,
        test_loader,
        device,
        experiment="cascade",
        quality_sweep=(80, 50),
    )
    export_tables(rows, paths.tables, "cascade")
    plot_quality_sweep(rows, paths.figures, "cascade")
    qualitative_grid(system, test_ds, device, paths.figures, "cascade", indices=(0, 1))
    save_sample_images(system, test_ds, device, paths.samples, "cascade", indices=(0,))

    print("\nartifacts:")
    for directory in (paths.history, paths.figures, paths.tables, paths.samples, paths.checkpoints):
        names = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
        print(f"  {os.path.relpath(directory, args.root)}: {len(names)} files")
        for name in names[:6]:
            print(f"    {name}")

    if not args.keep:
        shutil.rmtree(args.root, ignore_errors=True)
        print(f"\nremoved {args.root} (pass --keep to inspect the outputs)")
    print("\nsmoke test passed")


if __name__ == "__main__":
    main()
