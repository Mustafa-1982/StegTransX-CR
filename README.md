# StegTransX-CR: code and trained models

**Version 1.0.0** · MIT licence

> ### ⚠ The evaluation in this release has been superseded
>
> A later matched-budget study re-trained this architecture and its
> unconditioned predecessor under one identical protocol and reached the
> opposite conclusion about the method's value. **The v1.0.0 measurements below
> are not withdrawn — they reproduce — but the framing they supported does
> not.** See [What the matched-budget study found](#what-the-matched-budget-study-found).
>
> The code and weights in this repository are unchanged and remain the exact
> artefacts the later study evaluated.

StegTransX-CR hides a full-resolution secret image inside a cover image so that
the secret can still be recovered after the stego image is recompressed with
JPEG, WebP, HEIF or AVIF, including chains of recompressions such as
JPEG → WebP → JPEG. The hiding network is a hybrid CNN–Transformer conditioned
on the target codec through frequency-adaptive attention. The reveal network
sees only the received image.

This release contains:

- the training and evaluation code, byte-for-byte as used for the reported runs;
- the three trained models;
- the raw results files behind the v1 article's tables and figures.

## What the matched-budget study found

The v1.0.0 results were produced with early stopping on validation secret PSNR,
on 200 COCO test pairs, and were compared against the published numbers of the
predecessor architecture rather than against a run of it. A later study removed
both of those confounds. Thirteen models were trained under one fixed budget —
30,000 optimisation steps at batch 32 and 256 × 256, identical data, schedule,
degradation regime and seed — and evaluated on 1,000 paired COCO images and 100
DIV2K images through the real encoders.

**The codec-conditioned architecture did not improve on its unconditioned
predecessor.** Retrained under the same budget, StegTransX-V1 reached 25.53 dB
cover fidelity against 21.24 dB, and 27.41 dB secret recovery after real JPEG at
quality 80 against 21.99 dB, and led at every channel condition tested on both
datasets (paired Wilcoxon, n = 1,000; differences 5.22–6.43 dB, all
|d_z| > 4.9).

Three further findings bear directly on how this release should be read.

1. **Codec conditioning contributes nothing net.** Ablating the conditioning
   pathway under an otherwise identical run changed secret recovery by at most
   0.36 dB in either direction (mean −0.12 dB over ten channel conditions) and
   left cover fidelity unchanged. Yet declaring a *wrong* codec at inference
   costs 0.45 dB on average, and 0.91 dB when the channel is JPEG — so the label
   is genuinely used, but its information is available to the network by other
   routes.
2. **The objective is unstable over a long horizon.** In eight of the twelve
   codec-conditioned runs, validation cover fidelity peaks between steps 6,000
   and 9,000 and then declines by 2.1 to 3.3 dB, while the unconditioned
   baseline loses 0.28 dB from its peak. Early stopping, as used for v1.0.0,
   halts near that peak. A shorter budget therefore reports a better model than
   longer training produces.
3. **These models are not covert.** A supervised detector separates cover from
   stego images produced by the released `multi` weights with **0.9885 accuracy**,
   and reaches above 0.98 against every model tested from as few as fifty
   training pairs. The v1.0.0 model card listed steganalysis as "not evaluated";
   it has now been evaluated, and the answer is that no model in this family
   offers detection resistance.

### The released weights, re-measured

The three checkpoints in this repository were re-evaluated under the larger
protocol. The v1.0.0 numbers reproduce; the test set is simply bigger
(1,000 pairs rather than 200), which is why they shift slightly:

| Released model | cover PSNR | secret, JPEG q80 | secret, four-codec chain |
|---|---|---|---|
| `single` | 20.78 | 22.10 | 21.46 |
| `multi` | 20.66 | 22.11 | 21.13 |
| `cascade` | 20.52 | 22.35 | 21.40 |

For comparison, the predecessor architecture retrained under the matched budget
reaches 25.53 dB cover and 27.41 dB secret at JPEG q80. Retraining the
predecessor also lifts it far above its own released weights, by 4.87 dB cover
and 5.30 dB recovery — a larger gap than any architectural difference reported
in this literature, which is the central methodological point of the later
study.

### Where the later work lives

The experiment harness, the thirteen trained runs, the per-image evaluation
arrays and the aggregated digest are held separately. A Zenodo record covering
them is prepared but **not yet published**; this README will carry its DOI once
it is. The harness pins the exact commits it used — see `UPSTREAM.md`.

## Contents

| Path | What it is |
|---|---|
| `stegtransx_cr/` | Python package. It contains the hiding and reveal networks, frequency-adaptive attention, the differentiable JPEG/WebP/HEIF/AVIF simulators, real-encoder round trips, the losses, the training loop and the evaluation code. |
| `StegTransX_CR.ipynb` | The Colab notebook used for the v1 runs, with its saved outputs |
| `smoke_test.py` | End-to-end run on tiny synthetic images (a few minutes on CPU) |
| `scripts/hide_reveal.py` | Command-line demo, with separate hide (sender) and reveal (receiver) commands |
| `scripts/verify_release.py` | Checks the checksums, safe loading, and that the full and weights-only models give identical outputs |
| `scripts/export_weights.py` | Rebuilds `checkpoints/weights/` from `checkpoints/full/` |
| `scripts/release_utils.py` | Loading and image helpers used by the scripts |
| `checkpoints/full/best_{single,multi,cascade}.pth` | Training checkpoints saved at the best validation epoch, about 40 MB each. Besides the weights, they hold the optimiser, scheduler, AMP-scaler and early-stopping state and the per-epoch history. |
| `checkpoints/weights/stegtransx_cr_{single,multi,cascade}.pth` | Weights-only files for inference, about 15 MB each: the hiding network, the reveal network and the frozen simulator |
| `results/tables/` | **v1 benchmark results**, on 200 test pairs. `results_all.csv` holds all 186 rows (3 models × scenarios × simulated and real encoders). Superseded as evidence for the method's value; retained as the record of the v1 runs. |
| `results/history/history_{single,multi,cascade}.csv` | Per-epoch training and validation logs for the v1 runs |
| `results/figures/` | PNG figures written by the notebook: convergence, losses, validation PSNR, quality sweeps and qualitative grids |
| `UPSTREAM.md` | The exact commits of this package and of the baseline used by the matched-budget study |
| `MODEL_CARD.md` | Model details, training set-up, results and limitations |
| `CITATION.cff`, `LICENSE`, `SHA256SUMS` | Citation metadata, licence and checksums of every file |

## The three models

| Model | Training compression (simulated) | Epochs run | Best epoch | Best validation secret PSNR |
|---|---|---|---|---|
| `single` | JPEG at quality 80 | 176 | 167 | 22.93 dB |
| `multi` | Codec and quality (50–95) drawn per sample | 159 | 150 | 23.48 dB |
| `cascade` | 1–3 recompressions per batch; the first stage uses the conditioning codec | 166 | 157 | 23.31 dB |

Secret-image PSNR on the 200 v1 test pairs, with every codec at quality 80 run
through the real encoders:

| Model | No compression | JPEG | WebP | HEIF | AVIF | JPEG → WebP → JPEG | JPEG → WebP → HEIF → AVIF |
|---|---|---|---|---|---|---|---|
| `single` | 23.23 | 22.42 | 22.74 | 23.30 | 23.26 | 21.52 | 21.78 |
| `multi` | 23.93 | 22.35 | 22.56 | 23.64 | 22.98 | 21.35 | 21.38 |
| `cascade` | 23.88 | 22.60 | 22.89 | 23.65 | 23.21 | 21.61 | 21.67 |

The cover-to-stego PSNR is 20.77–20.96 dB, so the stego image is visibly
different from the cover. These are the v1 measurements; read them alongside
the section above. See `MODEL_CARD.md` for the full results and limitations.

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

This reproduces the **v1** protocol, with early stopping and 200 test pairs.
The matched-budget protocol is a different harness; see `UPSTREAM.md`.

## Implementation notes

These are deliberate differences from the original specification, as
implemented in this code:

1. The reveal network receives only the image, never the codec index.
2. The hiding and reveal losses each combine a Laplacian pyramid term, a
   Charbonnier term and a range-restriction term.
3. The hiding-loss weight is 0 for epochs 1–30 and then rises linearly to 1 by
   epoch 150. Best-checkpoint selection and early stopping (patience 9, on
   validation secret PSNR) start at epoch 150. **The later study identifies this
   stopping rule as load-bearing**: validation cover fidelity peaks near where
   early stopping halts and declines thereafter, so the rule selects a model
   that longer training does not produce.
4. The cosine learning-rate horizon is 500 epochs for `single` and 250 for
   `multi` and `cascade`. The single run started under a 500-epoch setting
   and was resumed at epoch 114 with a 250-epoch cap. The scheduler kept its
   500-epoch horizon.
5. The simulator is frozen and was not calibrated in the v1 runs. `calibrate.py`
   is included, but its notebook cell was not executed, so the learned simulator
   modules remain at their identity initialisation. The later study did run
   calibration, and found that substituting the calibrated simulator for the
   real encoders at evaluation time shifts secret PSNR by at most 0.014 dB.
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

Citation metadata is in `CITATION.cff`. The code and model weights are
released under the MIT licence (`LICENSE`). The training and test images are
not redistributed here. DIV2K and COCO have their own terms of use.
