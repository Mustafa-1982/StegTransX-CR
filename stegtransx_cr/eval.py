"""Benchmarking, figures and LaTeX export.

Every scenario is measured twice:

  * ``sim``  - through the differentiable simulator S, which is the protocol
               the paper reports.
  * ``real`` - through genuine libjpeg / libwebp / libheif / libavif files,
               which is what an image actually goes through on a platform.

Reporting both makes the simulator-to-real gap explicit instead of hiding it.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .codecs import real as real_codecs  # noqa: E402
from .config import CODECS, codec_index  # noqa: E402
from .train import StegTransXCR  # noqa: E402
from .utils import image_metrics  # noqa: E402

METRIC_KEYS = ("psnr", "ssim", "mae", "rmse")

# Paper Table 1: every codec at Q = 80, plus the uncompressed reference.
SINGLE_SCENARIOS: List[Tuple[str, Optional[str], int]] = [
    ("None", None, 0),
    ("JPEG", "JPEG", 80),
    ("WebP", "WebP", 80),
    ("HEIF", "HEIF", 80),
    ("AVIF", "AVIF", 80),
]

# Paper Table 2 cascading scenarios. The paper only fixes the quality of the
# single-codec baseline, so every stage here uses Q = 80.
CASCADE_SCENARIOS: List[Tuple[str, List[Tuple[str, int]]]] = [
    ("Single (baseline)", [("JPEG", 80)]),
    ("Double same", [("JPEG", 80), ("JPEG", 80)]),
    ("Double cross", [("JPEG", 80), ("WebP", 80)]),
    ("Triple mixed", [("JPEG", 80), ("WebP", 80), ("JPEG", 80)]),
    ("Platform simulate", [("JPEG", 80), ("HEIF", 80), ("WebP", 80)]),
    ("Full chain", [("JPEG", 80), ("WebP", 80), ("HEIF", 80), ("AVIF", 80)]),
]

# Quality sweep for the robustness curve.
QUALITY_SWEEP = (95, 90, 80, 65, 50)


def chain_label(chain: Sequence[Tuple[str, int]]) -> str:
    return " -> ".join(codec for codec, _ in chain)


# --------------------------------------------------------------------------- #
# Core measurement
# --------------------------------------------------------------------------- #
def _compress(
    system: StegTransXCR,
    stego: torch.Tensor,
    chain: Sequence[Tuple[str, int]],
    engine: str,
) -> torch.Tensor:
    if not chain:
        return stego
    if engine == "sim":
        return system.simulator.cascade(stego, list(chain))
    if engine == "real":
        return real_codecs.encode_decode_cascade(stego, list(chain))
    raise ValueError(f"unknown engine {engine!r}")


@torch.no_grad()
def evaluate_chain(
    system: StegTransXCR,
    loader: DataLoader,
    chain: Sequence[Tuple[str, int]],
    engine: str,
    device: torch.device,
    hiding_codec: Optional[str] = None,
) -> Dict[str, float]:
    """Average metrics over a loader for one compression chain."""
    system.eval()
    # The sender knows the platform it is posting to, so the hiding network is
    # conditioned on the first codec of the chain.
    condition = hiding_codec or (chain[0][0] if chain else "JPEG")
    condition_index = codec_index(condition)

    totals: Dict[str, float] = {}
    count = 0
    for cover, secret in loader:
        cover = cover.to(device)
        secret = secret.to(device)
        codec_tensor = torch.full((cover.shape[0],), condition_index, dtype=torch.long, device=device)

        stego = system.hiding(cover, secret, codec_tensor).clamp(0.0, 1.0)
        compressed = _compress(system, stego, chain, engine).clamp(0.0, 1.0)
        recovered = system.reveal(compressed)

        cover_metrics = image_metrics(stego, cover)
        secret_metrics = image_metrics(recovered, secret)
        batch = cover.shape[0]
        for key in METRIC_KEYS:
            totals[f"cs_{key}"] = totals.get(f"cs_{key}", 0.0) + float(cover_metrics[key].sum())
            totals[f"sr_{key}"] = totals.get(f"sr_{key}", 0.0) + float(secret_metrics[key].sum())
        count += batch

    if count == 0:
        raise RuntimeError("evaluation loader is empty")
    return {key: value / count for key, value in totals.items()}


def _chain_supported(chain: Sequence[Tuple[str, int]], engine: str) -> bool:
    if engine == "sim":
        return True
    return all(real_codecs.is_available(codec) for codec, _ in chain)


@torch.no_grad()
def run_benchmark(
    system: StegTransXCR,
    loader: DataLoader,
    device: torch.device,
    experiment: str,
    engines: Sequence[str] = ("sim", "real"),
    single_scenarios: Sequence[Tuple[str, Optional[str], int]] = SINGLE_SCENARIOS,
    cascade_scenarios: Sequence[Tuple[str, List[Tuple[str, int]]]] = CASCADE_SCENARIOS,
    quality_sweep: Sequence[int] = QUALITY_SWEEP,
    verbose: bool = True,
) -> List[Dict]:
    """Run every scenario for one trained checkpoint."""
    rows: List[Dict] = []

    def record(group: str, name: str, chain: Sequence[Tuple[str, int]], engine: str) -> None:
        if not _chain_supported(chain, engine):
            if verbose:
                print(f"  [{engine}] {name}: skipped (codec unavailable)")
            return
        metrics = evaluate_chain(system, loader, chain, engine, device)
        row = {
            "experiment": experiment,
            "group": group,
            "scenario": name,
            "chain": chain_label(chain) if chain else "None",
            "quality": chain[0][1] if chain else 0,
            "engine": engine,
            **metrics,
        }
        rows.append(row)
        if verbose:
            print(
                f"  [{engine}] {name:<20} cover {row['cs_psnr']:6.2f} dB "
                f"| secret {row['sr_psnr']:6.2f} dB / SSIM {row['sr_ssim']:.4f}"
            )

    for engine in engines:
        if verbose:
            print(f"\n{experiment}: single-codec scenarios ({engine})")
        for name, codec, quality in single_scenarios:
            chain: List[Tuple[str, int]] = [] if codec is None else [(codec, quality)]
            record("single", name, chain, engine)

        if verbose:
            print(f"\n{experiment}: cascading scenarios ({engine})")
        for name, chain in cascade_scenarios:
            record("cascade", name, chain, engine)

        if verbose:
            print(f"\n{experiment}: quality sweep ({engine})")
        for codec in CODECS:
            for quality in quality_sweep:
                record("sweep", f"{codec} q={quality}", [(codec, quality)], engine)

    return rows


# --------------------------------------------------------------------------- #
# CSV and LaTeX export
# --------------------------------------------------------------------------- #
def write_csv(rows: Sequence[Dict], path: str) -> str:
    if not rows:
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else str(value)


def write_latex_table(
    rows: Sequence[Dict],
    path: str,
    caption: str,
    label: str,
    row_key: str = "scenario",
    row_header: str = "Scenario",
) -> str:
    """Write a booktabs table with cover/stego and secret/recovery columns."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "% Generated by stegtransx_cr.eval -- \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        f"& \\multicolumn{{3}}{{c}}{{Cover / Stego}} & \\multicolumn{{3}}{{c}}{{Secret / Recovery}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        f"{row_header} & PSNR $\\uparrow$ & SSIM $\\uparrow$ & RMSE $\\downarrow$ "
        "& PSNR $\\uparrow$ & SSIM $\\uparrow$ & RMSE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        name = str(row.get(row_key, "")).replace("_", "\\_").replace("->", "$\\rightarrow$")
        lines.append(
            f"{name} & {_fmt(row['cs_psnr'])} & {_fmt(row['cs_ssim'], 4)} & {_fmt(row['cs_rmse'])} "
            f"& {_fmt(row['sr_psnr'])} & {_fmt(row['sr_ssim'], 4)} & {_fmt(row['sr_rmse'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def write_comparison_table(
    rows: Sequence[Dict],
    path: str,
    caption: str,
    label: str,
) -> str:
    """Side-by-side simulator vs real secret-recovery PSNR for each scenario."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    by_scenario: Dict[str, Dict[str, Dict]] = {}
    order: List[str] = []
    for row in rows:
        name = row["scenario"]
        if name not in by_scenario:
            by_scenario[name] = {}
            order.append(name)
        by_scenario[name][row["engine"]] = row

    lines = [
        "% Generated by stegtransx_cr.eval -- \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Scenario & Simulator PSNR $\\uparrow$ & Real codec PSNR $\\uparrow$ & Gap (dB) \\\\",
        "\\midrule",
    ]
    for name in order:
        entry = by_scenario[name]
        sim = entry.get("sim")
        real = entry.get("real")
        label_text = name.replace("_", "\\_").replace("->", "$\\rightarrow$")
        sim_text = _fmt(sim["sr_psnr"]) if sim else "--"
        real_text = _fmt(real["sr_psnr"]) if real else "n/a"
        gap_text = _fmt(sim["sr_psnr"] - real["sr_psnr"]) if sim and real else "--"
        lines.append(f"{label_text} & {sim_text} & {real_text} & {gap_text} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def export_tables(rows: Sequence[Dict], tables_dir: str, experiment: str) -> List[str]:
    """Write the CSV plus the LaTeX tables for one experiment."""
    written = [write_csv(rows, os.path.join(tables_dir, f"results_{experiment}.csv"))]

    for engine in ("sim", "real"):
        for group, caption in (
            ("single", "single-codec compression"),
            ("cascade", "cascading compression chains"),
        ):
            subset = [r for r in rows if r["engine"] == engine and r["group"] == group]
            if not subset:
                continue
            engine_name = "simulator" if engine == "sim" else "real codec files"
            written.append(
                write_latex_table(
                    subset,
                    os.path.join(tables_dir, f"table_{experiment}_{group}_{engine}.tex"),
                    caption=(
                        f"StegTransX-CR ({experiment} regime) under {caption}, "
                        f"measured with {engine_name}."
                    ),
                    label=f"tab:{experiment}-{group}-{engine}",
                )
            )

    for group in ("single", "cascade"):
        subset = [r for r in rows if r["group"] == group]
        if subset:
            written.append(
                write_comparison_table(
                    subset,
                    os.path.join(tables_dir, f"table_{experiment}_{group}_gap.tex"),
                    caption=(
                        f"Simulator versus real-encoder secret recovery for the "
                        f"{experiment} regime ({group} scenarios)."
                    ),
                    label=f"tab:{experiment}-{group}-gap",
                )
            )
    return written


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def read_history(path: str) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                try:
                    columns.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    columns.setdefault(key, []).append(float("nan"))
    return columns


def _save(fig: plt.Figure, path_without_ext: str) -> List[str]:
    os.makedirs(os.path.dirname(os.path.abspath(path_without_ext)), exist_ok=True)
    outputs = []
    for extension in ("pdf", "png"):
        path = f"{path_without_ext}.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=180)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_convergence(history_path: str, figures_dir: str, experiment: str) -> List[str]:
    """Loss curves and validation PSNR for one experiment."""
    history = read_history(history_path)
    epochs = history.get("epoch", [])
    if not epochs:
        return []

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    panels = [
        ("train_loss", "Total loss $L$"),
        ("train_loss_hide", "Hiding loss $L_H$"),
        ("train_loss_reveal", "Reveal loss $L_R$"),
        ("train_loss_freq", "Frequency loss $L_F$"),
    ]
    for axis, (key, title) in zip(axes, panels):
        if key in history:
            axis.plot(epochs, history[key], linewidth=1.4)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    fig.suptitle(f"StegTransX-CR convergence: {experiment} regime")
    paths = _save(fig, os.path.join(figures_dir, f"fig_convergence_{experiment}"))

    fig, axis = plt.subplots(figsize=(6.2, 4))
    if "train_loss" in history:
        axis.plot(epochs, history["train_loss"], label="Train", linewidth=1.4)
    if "val_loss" in history:
        axis.plot(epochs, history["val_loss"], label="Validation", linewidth=1.4)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss $L$")
    axis.set_title(f"Train / validation loss: {experiment} regime")
    axis.legend()
    axis.grid(alpha=0.3)
    paths += _save(fig, os.path.join(figures_dir, f"fig_loss_train_val_{experiment}"))

    fig, axis = plt.subplots(figsize=(6, 4))
    if "val_cover_psnr" in history:
        axis.plot(epochs, history["val_cover_psnr"], label="Cover / stego", linewidth=1.4)
    if "val_secret_psnr" in history:
        axis.plot(epochs, history["val_secret_psnr"], label="Secret / recovery", linewidth=1.4)
    axis.axhline(30.0, linestyle="--", linewidth=1.0, color="grey", label="30 dB target")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title(f"Validation PSNR: {experiment} regime")
    axis.legend()
    axis.grid(alpha=0.3)
    paths += _save(fig, os.path.join(figures_dir, f"fig_val_psnr_{experiment}"))
    return paths


def plot_experiment_comparison(history_paths: Dict[str, str], figures_dir: str) -> List[str]:
    """Overlay the three regimes on shared axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for experiment, path in history_paths.items():
        if not os.path.exists(path):
            continue
        history = read_history(path)
        epochs = history.get("epoch", [])
        if not epochs:
            continue
        axes[0].plot(epochs, history.get("train_loss", []), label=experiment, linewidth=1.4)
        axes[1].plot(epochs, history.get("val_secret_psnr", []), label=experiment, linewidth=1.4)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[1].set_title("Validation secret-recovery PSNR")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("StegTransX-CR: single vs multi vs cascading regimes")
    paths = _save(fig, os.path.join(figures_dir, "fig_convergence_all"))

    available = [(name, path) for name, path in history_paths.items() if os.path.exists(path)]
    if not available:
        return paths
    fig, axes = plt.subplots(1, len(available), figsize=(5.4 * len(available), 4), squeeze=False)
    for axis, (experiment, path) in zip(axes[0], available):
        history = read_history(path)
        epochs = history.get("epoch", [])
        if "train_loss" in history:
            axis.plot(epochs, history["train_loss"], label="Train", linewidth=1.4)
        if "val_loss" in history:
            axis.plot(epochs, history["val_loss"], label="Validation", linewidth=1.4)
        axis.set_title(experiment)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss $L$")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("StegTransX-CR train and validation loss")
    paths += _save(fig, os.path.join(figures_dir, "fig_loss_train_val_all"))
    return paths


def plot_quality_sweep(rows: Sequence[Dict], figures_dir: str, experiment: str) -> List[str]:
    """Secret-recovery PSNR against quality factor for each codec."""
    sweep = [r for r in rows if r["group"] == "sweep"]
    if not sweep:
        return []
    engines = sorted({r["engine"] for r in sweep})
    if not engines:
        return []
    fig, axes = plt.subplots(1, len(engines), figsize=(6 * len(engines), 4), squeeze=False)
    for axis, engine in zip(axes[0], engines):
        for codec in CODECS:
            points = sorted(
                [(r["quality"], r["sr_psnr"]) for r in sweep if r["engine"] == engine and r["chain"] == codec]
            )
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=codec)
        axis.axhline(30.0, linestyle="--", color="grey", linewidth=1.0)
        axis.set_xlabel("Quality factor $q$")
        axis.set_ylabel("Secret PSNR (dB)")
        axis.set_title("Simulator" if engine == "sim" else "Real codec files")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(f"Robustness against quality factor: {experiment} regime")
    return _save(fig, os.path.join(figures_dir, f"fig_quality_sweep_{experiment}"))


@torch.no_grad()
def qualitative_grid(
    system: StegTransXCR,
    dataset,
    device: torch.device,
    figures_dir: str,
    experiment: str,
    chain: Sequence[Tuple[str, int]] = (("JPEG", 80),),
    engine: str = "sim",
    indices: Sequence[int] = (0, 1, 2),
    amplify: float = 10.0,
) -> List[str]:
    """Cover, secret, stego, residual, compressed, recovery and error panels."""
    if engine == "real" and not _chain_supported(list(chain), engine):
        print(f"  qualitative {engine} skipped: codec unavailable")
        return []
    system.eval()
    titles = [
        "Cover",
        "Secret",
        "Stego",
        f"|Cover - Stego| x{int(amplify)}",
        f"Compressed ({chain_label(chain)})",
        "Recovered",
        f"|Secret - Recovered| x{int(amplify)}",
    ]
    rows = len(indices)
    fig, axes = plt.subplots(rows, len(titles), figsize=(2.2 * len(titles), 2.3 * rows), squeeze=False)

    condition = codec_index(chain[0][0] if chain else "JPEG")
    for row_index, sample_index in enumerate(indices):
        cover, secret = dataset[sample_index]
        cover = cover.unsqueeze(0).to(device)
        secret = secret.unsqueeze(0).to(device)
        codec_tensor = torch.full((1,), condition, dtype=torch.long, device=device)

        stego = system.hiding(cover, secret, codec_tensor).clamp(0.0, 1.0)
        compressed = _compress(system, stego, list(chain), engine).clamp(0.0, 1.0)
        recovered = system.reveal(compressed)

        panels = [
            cover,
            secret,
            stego,
            ((cover - stego).abs() * amplify).clamp(0, 1),
            compressed,
            recovered,
            ((secret - recovered).abs() * amplify).clamp(0, 1),
        ]
        for column, (panel, title) in enumerate(zip(panels, titles)):
            axis = axes[row_index][column]
            image = panel[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title, fontsize=9)

    engine_name = "simulator" if engine == "sim" else "real file"
    fig.suptitle(f"StegTransX-CR qualitative results ({experiment} regime, {engine_name})")
    return _save(fig, os.path.join(figures_dir, f"fig_qualitative_{experiment}_{engine}"))


@torch.no_grad()
def save_sample_images(
    system: StegTransXCR,
    dataset,
    device: torch.device,
    samples_dir: str,
    experiment: str,
    chain: Sequence[Tuple[str, int]] = (("JPEG", 80),),
    engine: str = "sim",
    indices: Sequence[int] = (0, 1, 2),
) -> List[str]:
    """Write the individual PNGs behind the qualitative figure."""
    if engine == "real" and not _chain_supported(list(chain), engine):
        return []
    from PIL import Image

    os.makedirs(samples_dir, exist_ok=True)
    system.eval()
    condition = codec_index(chain[0][0] if chain else "JPEG")
    written: List[str] = []

    for sample_index in indices:
        cover, secret = dataset[sample_index]
        cover = cover.unsqueeze(0).to(device)
        secret = secret.unsqueeze(0).to(device)
        codec_tensor = torch.full((1,), condition, dtype=torch.long, device=device)

        stego = system.hiding(cover, secret, codec_tensor).clamp(0.0, 1.0)
        compressed = _compress(system, stego, list(chain), engine).clamp(0.0, 1.0)
        recovered = system.reveal(compressed)

        for name, tensor in (
            ("cover", cover),
            ("secret", secret),
            ("stego", stego),
            ("compressed", compressed),
            ("recovered", recovered),
        ):
            array = (tensor[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            path = os.path.join(samples_dir, f"{experiment}_{engine}_{sample_index:03d}_{name}.png")
            Image.fromarray(array.transpose(1, 2, 0)).save(path)
            written.append(path)
    return written


__all__ = [
    "CASCADE_SCENARIOS",
    "SINGLE_SCENARIOS",
    "evaluate_chain",
    "export_tables",
    "plot_convergence",
    "plot_experiment_comparison",
    "plot_quality_sweep",
    "qualitative_grid",
    "read_history",
    "run_benchmark",
    "save_sample_images",
    "write_csv",
    "write_latex_table",
]
