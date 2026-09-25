# Model card: StegTransX-CR (v1.0.0)

> **Status.** The evaluation in this card has been superseded by a
> matched-budget study. The v1.0.0 measurements reproduce and are retained, but
> the conclusions drawn from them do not hold. The corrections are marked
> inline and summarised under [Superseded conclusions](#superseded-conclusions).

## Model details

- **Task.** Image-in-image hiding that must survive lossy recompression. The
  hiding network takes a cover image, a secret image and a codec index, and
  outputs a stego image. The reveal network takes the received, possibly
  recompressed, stego image and outputs the recovered secret.
- **Architecture.**
  - *Hiding network*: a U-shaped hybrid CNN–Transformer with 8 × 8 window
    attention and 4 heads. Frequency-adaptive attention at 1/4 resolution is
    conditioned on a 32-dimensional codec embedding.
  - *Reveal network*: four window-Transformer blocks at full resolution,
    with a sigmoid output.
  - *Codec simulator*: frozen differentiable JPEG, WebP, HEIF and AVIF
    simulators, used for training and for the simulated test scenarios.
- **Size.** 3.08 M trainable parameters (hiding 2.81 M, reveal 0.28 M).
- **Inputs and outputs.** RGB tensors of shape (B, 3, 256, 256) with values in
  [0, 1]. The codec index is a long tensor of shape (B,): 0 = JPEG, 1 = WebP,
  2 = HEIF, 3 = AVIF. The hiding network returns the unclamped sum of the cover
  and a residual. Clamp it to [0, 1] before saving or compressing.
- **Checkpoints.** Each regime has two files with identical weights:
  `checkpoints/full/best_<regime>.pth`, the training state at the best
  validation epoch, and `checkpoints/weights/stegtransx_cr_<regime>.pth`,
  which holds only the hiding, reveal and simulator state dictionaries.
  `scripts/verify_release.py` confirms that the two give bit-identical
  outputs.
- **Licence.** MIT.

## Superseded conclusions

A later study trained thirteen models under one fixed budget — 30,000
optimisation steps at batch 32 and 256 × 256, identical data, schedule,
degradation regime and seed — and evaluated them on 1,000 paired COCO images and
100 DIV2K images through the real encoders. Four conclusions in the original
card do not survive it.

| Original position | What the matched-budget study measured |
|---|---|
| The codec-conditioned design is the contribution | Ablating the conditioning pathway changes secret recovery by at most 0.36 dB either way (mean −0.12 dB over ten conditions) and leaves cover fidelity unchanged. The label *is* used — a wrong codec at inference costs 0.45 dB, 0.91 dB under JPEG — but a model denied it reaches the same accuracy by other routes. |
| Results compared against the predecessor's published numbers | Retrained under the identical budget, the unconditioned predecessor reaches 25.53 dB cover and 27.41 dB secret at JPEG q80, against 21.24 and 21.99 dB, and leads at every condition on both datasets (paired Wilcoxon, n = 1,000, all \|d_z\| > 4.9). |
| "Resistance to steganalysis: this was not evaluated" | It has been. A supervised detector reaches 0.9885 accuracy against the released `multi` weights, and above 0.98 against every model tested from fifty training pairs. **No model in this family is covert.** |
| "One training run per regime. No variance across seeds was measured." | Measured. Across three seeds the conditioned multi-codec model spans 21.16 to 24.01 dB of cover fidelity — a 2.85 dB seed-to-seed range, wider than most differences this card reports. |

A fifth point concerns the stopping rule rather than a claim. Validation cover
fidelity under this objective peaks between steps 6,000 and 9,000 and then
declines by 2.1 to 3.3 dB in eight of twelve conditioned runs, against 0.28 dB
for the unconditioned baseline. Early stopping on validation secret PSNR, as
used here, halts near that peak, so it selects a model that longer training does
not produce.

## Intended use

- Research on image hiding that survives JPEG, WebP, HEIF and AVIF
  recompression.
- Reproducing the v1 experiments, and serving as the artefact the
  matched-budget study evaluated.
- A baseline for comparison — with the caveat that the matched-budget study
  found the unconditioned predecessor stronger under equal training.

## Out of scope

- **Imperceptible embedding.** The cover-to-stego PSNR is about 21 dB, so the
  changes are visible.
- **Confidentiality.** The weights are public, so anyone can run the reveal
  network on a stego image made with them. No encryption or key is involved.
- **Resistance to steganalysis.** **Evaluated, and absent.** A detector trained
  on fifty cover/stego pairs exceeds 0.98 accuracy against every model in this
  family. Do not treat these models as covert.
- **Untested conditions.** The models were not tested on real social-media
  pipelines, resizing, cropping, other resolutions or non-photographic
  images.
- **Misuse.** Do not use the models to hide content from moderation or lawful
  inspection, or in any way that breaks applicable laws or platform terms.

## Training data

- **Training:** the 800 DIV2K training HR images, centre-cropped and resized
  (bicubic) to 256 × 256. The secret partner of each cover is re-drawn every
  epoch, and no augmentation is applied.
- **Validation and test:** the first 300 files of COCO val2017.
  - Files 0–79 form 50 validation pairs (seed 42).
  - Files 80–299 form 200 test pairs (seed 43).
  - Cover and secret are always different images, and the validation and test
    images do not overlap.
- The images are not redistributed with this release.

## Training procedure

- **Optimiser.** AdamW with learning rate 2 × 10⁻⁴, betas (0.5, 0.999) and
  weight decay 0.01. Batch size 32, 25 iterations per epoch, mixed precision,
  gradient clipping at 1.0.
- **Learning-rate schedule.** Cosine annealing to 2 × 10⁻⁶, stepped once per
  epoch. The horizon is 500 epochs for `single` (see note 4 in `README.md`)
  and 250 epochs for `multi` and `cascade`.
- **Loss.** The total loss is

  λ₁(t) · content(L_H) + restriction(L_H) + L_R + 0.1 · L_F

  - L_H and L_R each combine a 5-level Laplacian pyramid term, Charbonnier
    (weight 1) and range restriction (weight 10).
  - L_F compares the FFT magnitudes of the clamped stego image and the
    compressed image in 4 radial bands.
  - λ₁ is 0 for epochs 1–30 and rises linearly to 1 by epoch 150.
- **Validation and early stopping.** Validation runs every epoch through the
  regime's own simulated compression, with a fixed seed of 2024 and λ₁ = 1.
  From epoch 150, the best checkpoint is selected and early stopping (patience
  9 epochs, minimum improvement 10⁻³ dB) is applied, both on validation secret
  PSNR. **See the fifth point under Superseded conclusions: this rule is not
  neutral.**
- **Compression during training** (simulated, frozen, uncalibrated):
  - `single`: JPEG at quality 80.
  - `multi`: the codec is drawn uniformly per sample from the four codecs, and
    the quality is drawn uniformly from {50, …, 95}.
  - `cascade`: 1–3 stages per batch. The first stage uses the conditioning
    codec, and each later stage draws a new codec and quality per sample.
- **Seeds.** Global seed 42; training-sampler seed 1234.
- **Hardware and software.** NVIDIA A100-SXM4-80GB (Google Colab), PyTorch
  2.11.0+cu128.

| Regime | Epochs run | Best epoch | Best validation secret PSNR | Training time (sum of logged epoch times) |
|---|---|---|---|---|
| `single` | 176 | 167 | 22.93 dB | 1.06 h |
| `multi` | 159 | 150 | 23.48 dB | 1.02 h |
| `cascade` | 166 | 157 | 23.31 dB | 1.14 h |

## Evaluation

- **Test data and metrics.** All 200 test pairs at 256 × 256. The metrics are
  PSNR (data range 1), SSIM (Gaussian window 11, σ = 1.5), MAE and RMSE,
  computed per image and averaged.
- **Conditioning.** The hiding network is conditioned on the first codec of
  each scenario, or on JPEG when there is no compression.
- **Real encoders.** Pillow's JPEG (4:2:0 chroma subsampling) and WebP encoders,
  and the `pillow-heif` and `pillow-avif-plugin` plugins for HEIF and AVIF.
- **Simulator.** Every scenario is also run through the simulator.
- **Raw results.** All rows are in `results/tables/results_all.csv`.

### Real encoders, quality 80

Cover-to-stego fidelity when the hiding network is conditioned on JPEG:

| Model | PSNR (dB) | SSIM |
|---|---|---|
| `single` | 20.96 | 0.8282 |
| `multi` | 20.87 | 0.8325 |
| `cascade` | 20.77 | 0.8430 |

Across all real-encoder scenarios, the cover-to-stego PSNR stays within
20.77–20.96 dB.

Secret recovery, shown as PSNR in dB / SSIM:

| Scenario | `single` | `multi` | `cascade` |
|---|---|---|---|
| No compression | 23.23 / 0.7394 | 23.93 / 0.7544 | 23.88 / 0.7359 |
| JPEG | 22.42 / 0.6535 | 22.35 / 0.6317 | 22.60 / 0.6368 |
| WebP | 22.74 / 0.6810 | 22.56 / 0.6643 | 22.89 / 0.6622 |
| HEIF | 23.30 / 0.7336 | 23.64 / 0.7354 | 23.65 / 0.7195 |
| AVIF | 23.26 / 0.7085 | 22.98 / 0.6966 | 23.21 / 0.6897 |
| JPEG → JPEG | 22.32 / 0.6478 | 22.27 / 0.6284 | 22.48 / 0.6321 |
| JPEG → WebP | 22.02 / 0.6252 | 21.78 / 0.5978 | 22.06 / 0.6054 |
| JPEG → WebP → JPEG | 21.52 / 0.5926 | 21.35 / 0.5710 | 21.61 / 0.5782 |
| JPEG → HEIF → WebP | 21.99 / 0.6240 | 21.76 / 0.5972 | 22.04 / 0.6045 |
| JPEG → WebP → HEIF → AVIF | 21.78 / 0.6075 | 21.38 / 0.5789 | 21.67 / 0.5880 |

Every stage of every chain uses quality 80. The largest drop in secret PSNR
from the uncompressed case to any real chain is 2.58 dB (`multi`, JPEG → WebP
→ JPEG).

### The same weights on the larger protocol

These checkpoints were re-evaluated on 1,000 COCO pairs under the matched-budget
protocol. The v1 numbers reproduce; the differences are the larger test set, not
a discrepancy:

| Released model | cover PSNR | secret, JPEG q80 | secret, four-codec chain |
|---|---|---|---|
| `single` | 20.78 | 22.10 | 21.46 |
| `multi` | 20.66 | 22.11 | 21.13 |
| `cascade` | 20.52 | 22.35 | 21.40 |

The unconditioned predecessor, retrained under the same fixed budget, reaches
25.53 dB cover and 27.41 dB secret at JPEG q80 on that protocol.

### Simulator versus real encoders

The difference is simulated minus real secret PSNR, over qualities
{95, 90, 80, 65, 50}:

- **JPEG:** at most 0.05 dB in magnitude.
- **HEIF:** the simulator is pessimistic at every quality ≤ 80, by up to
  1.05 dB (`single`, quality 50).
- **WebP and AVIF:** the simulator is optimistic for some models and
  qualities, by up to +1.28 dB (WebP, `multi`, quality 65).

The per-quality values are in `results/tables/results_all.csv` (column `engine`).
The later study calibrated the non-JPEG branches, after which substituting the
simulator for the real encoders at evaluation time shifts secret PSNR by at most
0.014 dB.

## Limitations

- **Visible changes.** Embedding strength is high, and the stego image
  visibly differs from the cover (about 21 dB PSNR on the test set).
- **Not covert.** A detector trained on fifty cover/stego pairs exceeds 0.98
  accuracy against every model in this family.
- **Fidelity varies by image.** When `scripts/hide_reveal.py` was tested on
  scikit-image sample photographs, the cover-to-stego PSNR was 13.6–18.1 dB,
  and the outline of the secret was visible in the stego image.
- **Moderate recovery quality.** Secret recovery is about 21–24 dB PSNR.
- **Seed sensitivity.** Across three seeds under the matched budget, cover
  fidelity for this architecture spans 21.16–24.01 dB. The v1 runs are one seed
  each.
- **Evaluation scope of the v1 numbers.** 200 pairs from COCO val2017, at a
  single resolution (256 × 256).
- **Uncalibrated simulator in the v1 runs.** Its learned modules stayed at
  their identity initialisation. The WebP, HEVC and AV1 step sizes in
  `codecs/quant.py` are fitted analytic approximations.
- **Chain naming.** The chain labelled "Platform simulate" in the results is
  a local JPEG → HEIF → WebP chain, not a measurement on a real platform.
- **Unlogged figure in code comments.** A comment in `stegtransx_cr/config.py`
  states a gradient ratio of about 120× at initialisation. That figure was not
  logged in these runs and is not a reported result.

## Citation

Citation metadata is in `CITATION.cff`.
