"""Step-time benchmark of implementation options (each in a fresh process)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import torch

from .common import OUT_DIR, PKG_DIR, log, patch_attention, patch_simulator_skips, save_json

CONFIGS = [
    {"id": "v1_explicit_fp16_multi", "attn": "explicit", "amp": "fp16", "name": "multi-cond-s0"},
    {"id": "sdpa_bf16_multi", "attn": "sdpa", "amp": "bf16", "name": "multi-cond-s0"},
    {"id": "sdpa_bf16_compile_multi", "attn": "sdpa", "amp": "bf16", "compile": True, "name": "multi-cond-s0"},
    {"id": "sdpa_bf16_single", "attn": "sdpa", "amp": "bf16", "name": "single-cond-s0"},
    {"id": "sdpa_bf16_cascade", "attn": "sdpa", "amp": "bf16", "name": "cascade-cond-s0"},
    {"id": "stx_bf16_multi", "attn": "sdpa", "amp": "bf16", "name": "multi-stx-s0"},
]


def bench_one(cfg: Dict, warm: int = 15, timed: int = 40) -> Dict:
    from .common import GpuCropSampler, load_train_images
    from .train_v2 import RunConfig, Runner

    patch_attention(cfg.get("attn", "sdpa"))
    patch_simulator_skips()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rc = RunConfig.from_name(cfg["name"], amp=cfg.get("amp", "bf16"), total_steps=100000, val_every=10 ** 9,
                             compile=bool(cfg.get("compile", False)))
    if os.environ.get("SMOKE") == "1":
        rc.batch, rc.image_size = 4, (128 if "stx" in cfg["name"] else 64)
        warm, timed = 2, 3
    sim = torch.load(os.path.join(PKG_DIR, "checkpoints", "weights", "stegtransx_cr_multi.pth"),
                     map_location="cpu", weights_only=False)["simulator"]
    runner = Runner(rc, dev, sim, out_dir=os.path.join(OUT_DIR, "bench", "tmp_" + cfg["id"]))
    runner.crops = GpuCropSampler(load_train_images(dev, limit=64), rc.image_size, seed=0)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times: List[float] = []
    for i in range(warm + timed):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        cover, secret = runner.crops.batch(rc.batch)
        stego, _, _, losses = runner.forward(cover, secret, runner.sampler)
        runner.optimizer.zero_grad(set_to_none=True)
        runner.scaler.scale(losses["loss"]).backward()
        if rc.grad_clip:
            runner.scaler.unscale_(runner.optimizer)
            torch.nn.utils.clip_grad_norm_(runner.params, rc.grad_clip)
        runner.scaler.step(runner.optimizer)
        runner.scaler.update()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if i >= warm:
            times.append(time.time() - t0)
    times.sort()
    res = {"id": cfg["id"], "median_s_per_step": times[len(times) // 2], "mean_s_per_step": sum(times) / len(times),
           "batch": rc.batch, "loss": float(losses["loss"])}
    if dev.type == "cuda":
        res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return res


def run_bench(out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for cfg in CONFIGS:
        r = subprocess.run([sys.executable, "-m", "exp.bench", json.dumps(cfg)], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("BENCH ")]
        if line:
            results.append(json.loads(line[-1][6:]))
        else:
            results.append({"id": cfg["id"], "error": (r.stderr or "")[-1500:]})
        log(f"bench {results[-1]}")
        save_json(os.path.join(out_dir, "bench.json"), {"results": results})
    # two concurrent processes of the default configuration
    cfg = dict(CONFIGS[1], id="sdpa_bf16_multi_x2")
    procs = [subprocess.Popen([sys.executable, "-m", "exp.bench", json.dumps(cfg)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for _ in range(2)]
    conc = []
    for p in procs:
        out, err = p.communicate()
        line = [l for l in out.splitlines() if l.startswith("BENCH ")]
        conc.append(json.loads(line[-1][6:]) if line else {"error": err[-800:]})
    results.append({"id": "concurrent_x2", "runs": conc})
    log(f"bench concurrent {conc}")
    save_json(os.path.join(out_dir, "bench.json"), {"results": results})
    return {"n": len(results)}


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    print("BENCH " + json.dumps(bench_one(cfg)), flush=True)
