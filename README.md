# StegTransX-CR: code and trained models

**Version 1.0.0** · MIT licence · DOI [10.5281/zenodo.22894708](https://doi.org/10.5281/zenodo.22894708)

StegTransX-CR hides a full-resolution secret image inside a cover image so that
the secret can still be recovered after the stego image is recompressed with
JPEG, WebP, HEIF or AVIF, including chains of recompressions such as
JPEG → WebP → JPEG. The hiding network is a hybrid CNN–Transformer conditioned
on the target codec through frequency-adaptive attention. The reveal network
sees only the received image.

This release accompanies the article *StegTransX-CR: Codec-Conditioned
Transformer Image Hiding for JPEG, WebP, HEIF and AVIF Recompression*
(in preparation for the *International Journal of Intelligent Engineering and
Systems*). It contains:

- the training and evaluation code, byte-for-byte as used for the reported runs;
- the three trained models;
- the raw results files behind the article's tables and figures.

## Contents

| Path | What it is |
|---|---|
| `stegtransx_cr/` | Python package. It contains the hiding and reveal networks, frequency-adaptive attention, the differentiable JPEG/WebP/HEIF/AVIF simulators, real-encoder round trips, the losses, the training loop and the evaluation code. |
| `StegTransX_CR.ipynb` | The Colab notebook used for the runs, with its saved outputs |
| `smoke_test.py` | End-to-end run on tiny synthetic images (a few minutes on CPU) |
| `scripts/hide_reveal.py` | Command-line demo, with separate hide (sender) and reveal (receiver) commands |
| `scripts/verify_release.py` | Checks the checksums, safe loading, and that the full and weights-only models give identical outputs |
| `scripts/export_weights.py` | Rebuilds `checkpoints/weights/` from `checkpoints/full/` |
| `scripts/release_utils.py` | Loading and image helpers used by the scripts |
| `checkpoints/full/best_{single,multi,cascade}.pth` | Training checkpoints saved at the best validation epoch, about 40 MB each. Besides the weights, they hold the optimiser, scheduler, AMP-scaler and early-stopping state and the per-epoch history. |
| `checkpoints/weights/stegtransx_cr_{single,multi,cascade}.pth` | Weights-only files for inference, about 15 MB each: the hiding network, the reveal network and the frozen simulator |
| `results/tables/` | Benchmark results. `results_all.csv` holds all 186 rows (3 models × scenarios × simulated and real encoders), and `results_<model>.csv` splits them per model. The LaTeX tables are the ones written by `eval.export_tables`. |
| `results/history/history_{single,multi,cascade}.csv` | Per-epoch training and validation logs |
| `results/figures/` | PNG figures written by the notebook: convergence, losses, validation PSNR, quality sweeps and qualitative grids |
| `MODEL_CARD.md` | Model details, training set-up, results and limitations |
| `CITATION.cff`, `LICENSE`, `SHA256SUMS` | Citation metadata, licence and checksums of every file |

## The three models

| Model | Training compression (simulated) | Epochs run | Best epoch | Best validation secret PSNR |
|---|---|---|---|---|
| `single` | JPEG at quality 80 | 176 | 167 | 22.93 dB |
| `multi` | Codec and quality (50–95) drawn per sample | 159 | 150 | 23.48 dB |
| `cascade` | 1–3 recompressions per batch; the first stage uses the conditioning codec | 166 | 157 | 23.31 dB |

Secret-image PSNR on the 200 test pairs, with every codec at quality 80 run
through the real encoders:

| Model | No compression | JPEG | WebP | HEIF | AVIF | JPEG → WebP → JPEG | JPEG → WebP → HEIF → AVIF |
|---|---|---|---|---|---|---|---|
| `single` | 23.23 | 22.42 | 22.74 | 23.30 | 23.26 | 21.52 | 21.78 |
| `multi` | 23.93 | 22.35 | 22.56 | 23.64 | 22.98 | 21.35 | 21.38 |
| `cascade` | 23.88 | 22.60 | 22.89 | 23.65 | 23.21 | 21.61 | 21.67 |

The cover-to-stego PSNR is 20.77–20.96 dB, so the stego image is visibly
different from the cover. See `MODEL_CARD.md` for the full results and
limitations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`pillow-heif` and `pillow-avif-plugin` provide the real HEIF and AVIF
encoders. Without them, those codecs are still simulated, but real-encoder
round trips through them are reported as unavailable.

The models were trained with PyTorch 2.11.0 (CUDA 12.8) on an NVIDIA A100
80 GB in Google Colab. This release was checked on CPU with Python 3.11 and
PyTorch 2.14.0: `smoke_test.py` passed, and so did
`scripts/verify_release.py`.

## Quick start

Run all commands from this folder.

```bash
# 1. check the download
python scripts/verify_release.py

# 2. hide secret.png in cover.jpg, pass the stego image through real JPEG and
#    then WebP, and recover the secret (writes PNGs and the codec files)
python scripts/hide_reveal.py demo --cover cover.jpg --secret secret.png \
    --chain JPEG:80,WebP:80 --out demo_out

# 3. sender and receiver as separate steps
python scripts/hide_reveal.py hide --cover cover.jpg --secret secret.png \
    --codec WebP --quality 80 --out sent
python scripts/hide_reveal.py reveal --image sent/stego_WebP_q80.webp --out recovered.png
```

`--weights` chooses the model. The default is the multi-codec model; for
example, `--weights checkpoints/weights/stegtransx_cr_cascade.pth` uses the
cascade model. Images are centre-cropped and resized to 256 × 256, the
resolution the models were trained and tested at.

### From Python

```python
import torch
from stegtransx_cr.train import StegTransXCR

ckpt = torch.load("checkpoints/weights/stegtransx_cr_multi.pth",
                  map_location="cpu", weights_only=True)
system = StegTransXCR()
for part in ("hiding", "reveal", "simulator"):
    getattr(system, part).load_state_dict(ckpt[part])
system.eval()

# cover, secret: float tensors (B, 3, 256, 256) in [0, 1]
codec = torch.zeros(cover.shape[0], dtype=torch.long)   # 0 JPEG, 1 WebP, 2 HEIF, 3 AVIF
with torch.no_grad():
    stego = system.hiding(cover, secret, codec).clamp(0, 1)
    # ... save/compress/share stego, decode the received image ...
    recovered = system.reveal(received)
```

The full checkpoints contain the same three keys, so the same code loads them.
Both formats contain only tensors and plain Python containers, which was
checked with `pickletools`, so they load with `weights_only=True`. The
package's own `utils.load_checkpoint`, used for training resumption and by
`train.load_best`, passes `weights_only=False`. Use it only with checkpoints
from a trusted source.

## Reproducing the runs

1. Upload this folder to Google Drive. Open `StegTransX_CR.ipynb` in Colab on
   an A100, set `PROJECT_DIR` to the folder that contains `stegtransx_cr/`,
   and run the cells in order. The notebook downloads DIV2K (training) and
   COCO val2017 (validation and test).
2. Before a long run, check the pipeline:
   `python smoke_test.py --root smoke_run --epochs 1 --image-size 32`.
3. To evaluate the released models with the notebook, copy
   `checkpoints/full/best_<regime>.pth` into the run's `checkpoints/` folder
   (`runs/checkpoints/` in the notebook). The evaluation cells load them with
   `train.load_best` and rebuild the benchmark tables.

## Implementation notes

These are deliberate differences from the original specification, as
implemented in this code:

1. The reveal network receives only the image, never the codec index.
2. The hiding and reveal losses each combine a Laplacian pyramid term, a
   Charbonnier term and a range-restriction term.
3. The hiding-loss weight is 0 for epochs 1–30 and then rises linearly to 1 by
   epoch 150. Best-checkpoint selection and early stopping (patience 9, on
   validation secret PSNR) start at epoch 150.
4. The cosine learning-rate horizon is 500 epochs for `single` and 250 for
   `multi` and `cascade`. The single run started under a 500-epoch setting
   and was resumed at epoch 114 with a 250-epoch cap. The scheduler kept its
   500-epoch horizon.
5. The simulator is frozen and was not calibrated. `calibrate.py` is included,
   but its notebook cell was not executed, so the learned simulator modules
   remain at their identity initialisation.
6. Every test scenario is measured twice: through the differentiable simulator
   and through the real encoders.
7. The three regimes are trained independently, each from freshly initialised
   hiding and reveal networks.

## Checksums

```bash
shasum -a 256 -c SHA256SUMS      # macOS
sha256sum -c SHA256SUMS          # Linux
python scripts/verify_release.py # any platform
```

## Citation and licence

The DOI of this release is [10.5281/zenodo.22894708](https://doi.org/10.5281/zenodo.22894708). Citation metadata is in
`CITATION.cff`. The code and model weights are
released under the MIT licence (`LICENSE`). The training and test images are
not redistributed here. DIV2K and COCO have their own terms of use.
