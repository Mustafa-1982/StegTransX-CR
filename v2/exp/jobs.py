"""Job runner for Runpod pods.

JOBS syntax: comma-separated stages run in order; inside a stage, jobs joined
with '+' run in parallel as separate processes on the same GPU.

  data                      download DIV2K train/valid HR and COCO val2017
  calibrate                 fit the v2 simulator (documented settings), gap tables
  bench                     step-time benchmark of implementation options
  e0                        reproduce the released numbers with the v1 code path
  e1                        v2 evaluation of the three released models
  run:<regime>-<variant>-s<seed>[@steps=N]   train, evaluate, push
  eval:<run name>           evaluate an existing run again (pulls its branch)
  steg                      SRNet steganalysis over finished runs
  collect                   gather every r/* branch into the r/collect digest
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from typing import Dict, List

import numpy as np
import torch

from .common import (
    DATA_DIR, OUT_DIR, PKG_DIR, SMOKE, WORK, env_info, eval_sets, log, patch_attention,
    patch_real_parallel, patch_simulator_skips, push_results, remote_url, save_json,
    setup_package, torch_save,
)

SIM_V2 = os.path.join(OUT_DIR, "calibrate", "simulator_v2.pth")
V1_WEIGHTS = {r: os.path.join(PKG_DIR, "checkpoints", "weights", f"stegtransx_cr_{r}.pth")
              for r in ("single", "multi", "cascade")}
DATASETS = {
    "DIV2K_train_HR": ["https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
                       "https://huggingface.co/datasets/eugenesiow/Div2k/resolve/main/data/DIV2K_train_HR.zip"],
    "DIV2K_valid_HR": ["https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
                       "https://huggingface.co/datasets/eugenesiow/Div2k/resolve/main/data/DIV2K_valid_HR.zip"],
    "val2017": ["http://images.cocodataset.org/zips/val2017.zip"],
}
EXPECTED = {"DIV2K_train_HR": 800, "DIV2K_valid_HR": 100, "val2017": 5000}


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
def job_data() -> Dict:
    from concurrent.futures import ThreadPoolExecutor

    from .common import list_images

    os.makedirs(DATA_DIR, exist_ok=True)

    def fetch(name: str) -> str:
        target = os.path.join(DATA_DIR, name)
        if os.path.isdir(target) and len(list_images(target)) >= EXPECTED[name]:
            return f"{name}: present"
        for url in DATASETS[name]:
            archive = os.path.join(DATA_DIR, os.path.basename(url))
            try:
                t0 = time.time()
                if shutil.which("curl"):
                    subprocess.run(["curl", "-L", "--fail", "--silent", "--show-error", "--retry", "5",
                                    "--retry-delay", "5", "-o", archive, url], check=True)
                else:
                    urllib.request.urlretrieve(url, archive)
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(DATA_DIR)
                os.remove(archive)
                found = list_images(target)
                if len(found) >= EXPECTED[name]:
                    return f"{name}: {len(found)} images from {url.split('/')[2]} in {time.time() - t0:.0f}s"
            except Exception as err:
                log(f"download failed for {name} from {url.split('/')[2]}: {type(err).__name__}")
        raise RuntimeError(f"could not obtain {name}")

    with ThreadPoolExecutor(3) as ex:
        msgs = list(ex.map(fetch, list(DATASETS)))
    for m in msgs:
        log(m)
    eval_sets()  # build the COCO / DIV2K-valid caches once
    return {"datasets": msgs}


# --------------------------------------------------------------------------- #
def _gap_table(sim, cache: np.ndarray, dev, qualities=(50, 65, 80, 95)) -> List[Dict]:
    from stegtransx_cr.utils import psnr

    from .evaluate import run_chain_parallel

    rows = []
    x_u8 = cache
    x = torch.from_numpy(x_u8.transpose(0, 3, 1, 2).copy()).to(dev).float() / 255.0
    for codec in ("JPEG", "WebP", "HEIF", "AVIF"):
        t_codec = time.time()
        for q in qualities:
            with torch.no_grad():
                simulated = torch.cat([sim(x[i : i + 50], codec, q) for i in range(0, len(x), 50)])
            real, sizes = run_chain_parallel(x_u8, [(codec, q)])
            real_t = torch.from_numpy(real.transpose(0, 3, 1, 2).copy()).to(dev).float() / 255.0
            rows.append({"codec": codec, "quality": q,
                         "sim_psnr": float(psnr(simulated, x).mean()),
                         "real_psnr": float(psnr(real_t, x).mean()),
                         "sim_vs_real_psnr": float(psnr(simulated, real_t).mean())})
        log(f"  gap table {codec}: {len(qualities)} qualities x {len(x_u8)} images "
            f"in {time.time() - t_codec:.0f}s")
    return rows


def job_calibrate() -> Dict:
    """Fresh simulator (seed 42), calibrated like the v1 notebook's cell 12:
    calibrate_simulator(..., steps=600, batch_size=8), lr 1e-3, seed 7, on 800
    DIV2K training images centre-cropped and resized to 256 x 256."""
    setup_package()
    from stegtransx_cr.calibrate import calibrate_simulator
    from stegtransx_cr.codecs.simulator import MultiCodecSimulator

    from .common import build_eval_cache, div2k_train_files, synthetic_images

    dev = device()
    patch_real_parallel()
    out = os.path.join(OUT_DIR, "calibrate")
    os.makedirs(out, exist_ok=True)
    if SMOKE:
        train_cache = synthetic_images(16, 64, 3)
        gap_cache = synthetic_images(8, 64, 4)
        steps = 6
    else:
        train_cache = build_eval_cache(div2k_train_files(), os.path.join(DATA_DIR, "div2k_train_256.npy"), 256)
        gap_cache = eval_sets()["coco_val"]["cache"][200:400]
        steps = int(os.environ.get("CALIB_STEPS", 600))
    torch.manual_seed(42)
    sim = MultiCodecSimulator().to(dev)
    sim.freeze()
    log("gap table before calibration")
    before = _gap_table(sim, gap_cache, dev)
    log(f"calibrating {steps} steps")
    hist = calibrate_simulator(sim, train_cache, device=dev, steps=steps, batch_size=8, lr=1e-3, seed=7,
                               log_every=100)
    log("gap table after calibration")
    after = _gap_table(sim, gap_cache, dev)
    v1 = MultiCodecSimulator().to(dev)
    v1.load_state_dict(torch.load(V1_WEIGHTS["multi"], map_location="cpu", weights_only=False)["simulator"])
    v1.freeze()
    log("gap table for the v1 release simulator")
    v1_rows = _gap_table(v1, gap_cache, dev)
    torch_save(SIM_V2, {"simulator": sim.state_dict(), "steps": steps, "seed_init": 42, "seed": 7,
                        "batch_size": 8, "lr": 1e-3})
    res = {"before": before, "after": after, "v1_release": v1_rows, "env": env_info(),
           "loss_history": {k: [float(v) for v in vs] for k, vs in hist.items()}}
    save_json(os.path.join(out, "calibration.json"), res)
    for b, a, v in zip(before, after, v1_rows):
        log(f"gap {b['codec']:<5}q{b['quality']:<3} sim-vs-real before {b['sim_vs_real_psnr']:.2f} "
            f"after {a['sim_vs_real_psnr']:.2f} v1 {v['sim_vs_real_psnr']:.2f} dB")
    push_results("calibrate", out)
    return {"ok": True}


def ensure_sim_v2() -> Dict:
    """Load the canonical v2 simulator: local file or the r/calibrate branch."""
    if not os.path.exists(SIM_V2):
        if os.environ.get("GH_REPO") or os.environ.get("PUSH_URL"):
            tmp = os.path.join(WORK, "fetch_calibrate")
            shutil.rmtree(tmp, ignore_errors=True)
            subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", "r/calibrate", remote_url(), tmp],
                           check=False, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
            src = os.path.join(tmp, "results", "calibrate", "simulator_v2.pth")
            if os.path.exists(src):
                os.makedirs(os.path.dirname(SIM_V2), exist_ok=True)
                shutil.copy(src, SIM_V2)
    if not os.path.exists(SIM_V2):
        raise FileNotFoundError("simulator_v2.pth not available (run the calibrate job first)")
    return torch.load(SIM_V2, map_location="cpu", weights_only=False)["simulator"]


# --------------------------------------------------------------------------- #
def job_e0() -> Dict:
    """Released models through the unmodified v1 benchmark on the v1 test pairs."""
    setup_package()
    import csv

    from stegtransx_cr.data import FixedPairDataset
    from stegtransx_cr.eval import run_benchmark
    from stegtransx_cr.train import StegTransXCR, load_best
    from torch.utils.data import DataLoader

    dev = device()
    patch_real_parallel()  # same encoder, same bytes, off the main process
    out = os.path.join(OUT_DIR, "e0")
    os.makedirs(out, exist_ok=True)
    spec = eval_sets()["v1_test"]
    loader = DataLoader(FixedPairDataset(spec["cache"], spec["pairs"]), batch_size=16, shuffle=False)
    rows = []
    for regime in ("single", "multi", "cascade"):
        system = StegTransXCR()
        system = load_best(system, V1_WEIGHTS[regime], dev)
        rows += run_benchmark(system, loader, dev, experiment=regime, engines=("sim", "real"), verbose=False)
        log(f"e0 {regime} done")
    with open(os.path.join(out, "results_e0.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ref_path = os.path.join(PKG_DIR, "results", "results_all.csv")
    compare = {}
    if os.path.exists(ref_path):
        ref = {(r["experiment"], r["group"], r["scenario"], r["engine"]): r for r in csv.DictReader(open(ref_path))}
        diffs = {"cs_psnr": [], "sr_psnr": [], "sr_ssim": []}
        by_engine = {"sim": [], "real": []}
        for r in rows:
            k = (r["experiment"], r["group"], r["scenario"], r["engine"])
            if k in ref:
                for m in diffs:
                    diffs[m].append(abs(float(r[m]) - float(ref[k][m])))
                by_engine[r["engine"]].append(abs(float(r["sr_psnr"]) - float(ref[k]["sr_psnr"])))
        compare = {m: {"max_abs": max(v) if v else None, "mean_abs": float(np.mean(v)) if v else None}
                   for m, v in diffs.items()}
        compare["sr_psnr_by_engine"] = {e: {"max_abs": max(v) if v else None, "n": len(v)} for e, v in by_engine.items()}
    save_json(os.path.join(out, "e0_compare.json"), {"compare": compare, "env": env_info()})
    log(f"e0 compare: {json.dumps(compare)[:400]}")
    push_results("e0", out)
    return compare


def job_e1() -> Dict:
    from .evaluate import evaluate_system
    from .systems import load_system

    patch_attention("sdpa")
    patch_simulator_skips()
    dev = device()
    sets = eval_sets()
    out = os.path.join(OUT_DIR, "e1")
    from .evaluate import brief

    digest = {}
    for regime, path in V1_WEIGHTS.items():
        system = load_system(path, dev, variant="cond", regime=regime)
        res = evaluate_system(system, sets, dev, os.path.join(out, f"v1-{regime}"), label_test=True,
                              samples=(regime == "multi"), set_names=("coco_test", "div2k_val", "v1_test"))
        digest[regime] = brief(res["coco_test"])
        push_results("e1", out)
    save_json(os.path.join(out, "env.json"), env_info())
    push_results("e1", out)
    return digest


# --------------------------------------------------------------------------- #
def parse_run(spec: str):
    name, _, opts = spec.partition("@")
    overrides = {}
    for item in filter(None, opts.split(";")):
        k, v = item.split("=", 1)
        overrides[{"steps": "total_steps"}.get(k, k)] = v
    return name, overrides


def job_run(spec: str) -> Dict:
    from .evaluate import evaluate_system
    from .systems import load_system
    from .train_v2 import RunConfig, Runner

    name, overrides = parse_run(spec)
    env_over = {
        "total_steps": os.environ.get("TOTAL_STEPS"), "batch": os.environ.get("BATCH"),
        "amp": os.environ.get("AMP"), "val_every": os.environ.get("VAL_EVERY"),
        "compile": (os.environ.get("COMPILE") == "1") if os.environ.get("COMPILE") else None,
        "image_size": os.environ.get("IMAGE_SIZE"), "max_hours": os.environ.get("RUN_MAX_HOURS"),
    }
    env_over.update(overrides)
    if "compile" in overrides:
        env_over["compile"] = overrides["compile"] in ("1", "true", "True")
    base = name.split(".")[-1]  # "pilot.multi-cond-s0" style prefixes are allowed
    cfg = RunConfig.from_name(base.split(":")[-1], **env_over)
    cfg.name = name
    patch_attention(os.environ.get("ATTN", "sdpa"))
    patch_simulator_skips()
    dev = device()
    out = os.path.join(OUT_DIR, name)
    done = os.path.join(out, "train_done.json")
    if not os.path.exists(done):
        runner = Runner(cfg, dev, ensure_sim_v2(), out)
        runner.run()
        del runner
        torch.cuda.empty_cache()
    push_results(name, out, exclude_ext=(".tmp", "last.pth"))
    digest = {}
    hist = os.path.join(out, "history.csv")
    if os.path.exists(hist):
        import csv

        rows = list(csv.DictReader(open(hist)))
        if rows:
            last = rows[-1]
            digest["last_val"] = {k: round(float(last[k]), 3) for k in
                                  ("step", "val_cover_psnr", "val_secret_psnr", "val_secret_psnr_realjpeg80", "sec_per_step")}
    if os.environ.get("SKIP_EVAL", "0") != "1":
        from .evaluate import brief

        system = load_system(os.path.join(out, "weights_final.pth"), dev)
        res = evaluate_system(system, eval_sets(cfg.image_size), dev, os.path.join(out, "eval"),
                              label_test=True, samples=cfg.seed == 0)
        digest["coco_test"] = brief(res["coco_test"])
        push_results(name, out, exclude_ext=(".tmp", "last.pth"))
    return digest


def job_eval(name: str) -> Dict:
    from .evaluate import evaluate_system
    from .systems import load_system

    patch_attention("sdpa")
    patch_simulator_skips()
    dev = device()
    out = os.path.join(OUT_DIR, name)
    weights = fetch_run(name, full=True)   # works on a pod that never trained it
    system = load_system(weights, dev)
    evaluate_system(system, eval_sets(), dev, os.path.join(out, "eval"), label_test=True, samples=True)
    push_results(name, out, exclude_ext=(".tmp", "last.pth"))
    return {"ok": True}


def fetch_run(name: str, full: bool = False) -> str:
    """Make sure OUT_DIR/<name> holds the run's artefacts (pull its branch if needed).

    ``full`` also restores config.json, history.csv and any samples, so that a
    pod which only re-evaluates a run still pushes a complete branch back.
    """
    dest = os.path.join(OUT_DIR, name)
    path = os.path.join(dest, "weights_final.pth")
    if os.path.exists(path) and not full:
        return path
    if os.path.exists(path) and full and os.path.exists(os.path.join(dest, "history.csv")):
        return path
    tmp = os.path.join(WORK, "fetch", name)
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", f"r/{name}", remote_url(), tmp], check=True,
                   env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    src = os.path.join(tmp, "results", name)
    os.makedirs(dest, exist_ok=True)
    for root, _, names in os.walk(src):
        for fname in names:
            source = os.path.join(root, fname)
            target = os.path.join(dest, os.path.relpath(source, src))
            if os.path.exists(target):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy(source, target)
    log(f"fetched r/{name} into {dest}")
    return path


def job_steg() -> Dict:
    from .steganalysis import run_steganalysis
    from .systems import load_system

    patch_attention("sdpa")
    patch_simulator_skips()
    dev = device()
    names = [n for n in os.environ.get("STEG_RUNS", "").split(",") if n]
    models = {}
    for n in names:
        if n.startswith("v1-"):
            models[n] = load_system(V1_WEIGHTS[n[3:]], dev, variant="cond", regime=n[3:])
        else:
            models[n] = load_system(fetch_run(n), dev)
    sweep_name = os.environ.get("STEG_SWEEP_RUN", "")
    n_sweep = {sweep_name: [50, 100, 200, 400, 800]} if sweep_name else {}
    out = os.path.join(OUT_DIR, "steg")
    sets = eval_sets()
    res = run_steganalysis(models, sets["coco_test"]["cache"], dev, out, n_sweep=n_sweep,
                           steps=int(os.environ.get("STEG_STEPS", 1500)))
    save_json(os.path.join(out, "env.json"), env_info())
    push_results("steg", out)
    return res


def job_bench() -> Dict:
    from .bench import run_bench

    out = os.path.join(OUT_DIR, "bench")
    res = run_bench(out)
    push_results("bench", out)
    return res


# --------------------------------------------------------------------------- #
def run_one(job: str) -> None:
    out_status = os.path.join(OUT_DIR, "_status")
    os.makedirs(out_status, exist_ok=True)
    t0 = time.time()
    status = {"job": job, "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pod": os.environ.get("RUNPOD_POD_ID")}
    try:
        if job == "data":
            res = job_data()
        elif job == "calibrate":
            res = job_calibrate()
        elif job == "bench":
            res = job_bench()
        elif job == "e0":
            res = job_e0()
        elif job == "e1":
            res = job_e1()
        elif job.startswith("run:"):
            res = job_run(job[4:])
        elif job.startswith("eval:"):
            res = job_eval(job[5:])
        elif job == "steg":
            res = job_steg()
        elif job == "collect":
            from .collect import job_collect
            res = job_collect()
        else:
            raise ValueError(f"unknown job {job}")
        status.update({"ok": True, "result": res})
    except Exception as err:
        status.update({"ok": False, "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-4000:]})
        log(f"JOB FAILED {job}: {type(err).__name__}: {err}")
        traceback.print_exc()
    status["hours"] = (time.time() - t0) / 3600.0
    try:  # one compact line in the pod log as a fallback channel
        print("SUMMARY " + json.dumps({"job": job, "ok": status.get("ok"), "hours": round(status["hours"], 3),
                                       "result": status.get("result")}, default=str)[:6000], flush=True)
    except Exception:
        pass
    safe = job.replace(":", "_").replace("/", "_").replace("@", "_").replace(";", "_")
    save_json(os.path.join(out_status, f"{safe}.json"), status)
    push_results(f"status-{safe}", out_status)
    if not status.get("ok"):
        sys.exit(3)


def run_plan(plan: str) -> int:
    failures = 0
    for stage in [s for s in plan.split(",") if s.strip()]:
        jobs = [j.strip() for j in stage.split("+") if j.strip()]
        log(f"stage: {jobs}")
        if len(jobs) == 1:
            code = subprocess.run([sys.executable, "-m", "exp.jobs", "one", jobs[0]]).returncode
            failures += int(code != 0)
            if code != 0 and jobs[0] in ("data", "calibrate"):
                log("prerequisite failed; stopping the plan")
                return failures
            continue
        procs = [subprocess.Popen([sys.executable, "-m", "exp.jobs", "one", j]) for j in jobs]
        for p in procs:
            failures += int(p.wait() != 0)
    return failures


if __name__ == "__main__":
    mode, arg = sys.argv[1], sys.argv[2]
    if mode == "one":
        run_one(arg)
    elif mode == "plan":
        sys.exit(1 if run_plan(arg) else 0)
