# `v2/` — the matched-budget experiment harness

This directory holds the harness that produced the matched-budget study
described in the accompanying manuscript. It is the code that was run, not a
cleaned-up restatement of it: `exp/` and `bootstrap.sh` are copied verbatim
from the private run repository, whose only removed content is the result
branches themselves.

The conclusions reported in the top-level `README.md` and `MODEL_CARD.md`
come from this harness, not from the v1 notebook (`StegTransX_CR.ipynb`) or
from the v1 artefacts under `results/`. Where the two disagree, this one is
the later and better-controlled measurement.

## Protocol

Every run in the study was given the same budget, so that no configuration
could win by being trained longer:

| Setting | Value |
|---|---|
| Optimisation budget | 30,000 steps (fixed; no early stopping) |
| Batch size | 32 |
| Crop | 256 x 256 |
| Learning-rate schedule | cosine decay over the full budget |
| Reported checkpoint | the final one, not the best-validation one |

Reporting the final checkpoint rather than the best one is deliberate. Under
a fixed budget, "best validation checkpoint" rewards a run that happened to
peak early and then decayed, which is what the v1 evaluation did.

## Variants

| Name | What it is |
|---|---|
| `cond` | the full model, conditioned on the codec label |
| `nocond` | the same model with conditioning removed |
| `nofaa` | frequency-aware attention replaced by the identity |
| `nofreq` | the frequency-domain loss term removed |
| `stx` | StegTransX-V1, trained under the identical budget as a baseline |

Runs are named `<regime>-<variant>-s<seed>`, over the regimes `single`,
`multi` and `cascade`. The `cond` configuration was run at three seeds
(`s0`, `s1`, `s2`) in every regime; the ablations were run at `s0`.

## Evaluation

Evaluation goes through real encoders rather than the differentiable
simulator: libjpeg, libwebp, libheif and libavif, reached through Pillow and
its HEIF/AVIF plugins. The scenario set is

* 1 clean (no compression),
* 20 single-codec scenarios (4 codecs x 5 quality levels),
* 5 cascade scenarios,

and, separately, 24 codec-label mismatch scenarios in which the label handed
to the conditioning path names a different codec from the one actually
applied. Steganalysis uses SRNet (Boroumand, Chen and Fridrich, IEEE TIFS
2019); `exp/steganalysis.py` carries the implementation used.

## Layout

| Path | Role |
|---|---|
| `exp/train_v2.py` | the fixed-budget training loop |
| `exp/systems.py` | variant construction shared by training and evaluation |
| `exp/evaluate.py` | real-encoder evaluation |
| `exp/steganalysis.py` | SRNet steganalysis |
| `exp/stx_baseline.py` | the StegTransX-V1 baseline wrapper |
| `exp/stats.py` | paired statistics over per-image metric arrays |
| `exp/bench.py` | step-time benchmark of implementation options |
| `exp/jobs.py` | job runner (stage plans) |
| `exp/collect.py` | gathers result branches into one digest |
| `exp/common.py` | paths, package setup, speed patches, data and git helpers |
| `exp/hotfix.py` | real-encoder evaluation fix (see its docstring) |
| `exp/shims/` | minimal stand-ins for the three `mmcv` symbols the baseline imports |
| `bootstrap.sh` | pod entry point: dependencies, code checkout, job plan |

## Running it

`bootstrap.sh` expects a CUDA machine and pins the two external checkouts it
needs. Result transport is parameterised: set `GH_REPO` (or `PUSH_URL`) to a
repository you can push to, or `NO_PUSH=1` to keep results local. No
credential is embedded anywhere in this directory; git reads a token from the
environment through an askpass helper.

```bash
export WORK=/workspace NO_PUSH=1 JOBS='data,train:multi-cond-s0'
bash v2/bootstrap.sh
```
