"""Evaluation through real encoders (PIL: libjpeg, libwebp, libheif, libavif)
and through the differentiable simulator, with per-image metrics.

Protocol (v2): the stego image is quantised to 8 bits (what a saved PNG holds)
before any metric or encoder sees it; cover metrics compare that 8-bit stego
with the cover. Encoder settings match the v1 release: PIL ``quality=q`` and
4:2:0 chroma subsampling for JPEG, defaults otherwise.
"""
from __future__ import annotations

import atexit
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .common import CODECS, log, pairs_to_tensors, quantize, save_json, setup_package
from .systems import System, codec_tensor

SINGLE_Q = (50, 65, 80, 90, 95)
CASCADES = {
    "double_same": [("JPEG", 80), ("JPEG", 80)],
    "double_cross": [("JPEG", 80), ("WebP", 80)],
    "triple_mixed": [("JPEG", 80), ("WebP", 80), ("JPEG", 80)],
    "platform": [("JPEG", 80), ("HEIF", 80), ("WebP", 80)],
    "full_chain": [("JPEG", 80), ("WebP", 80), ("HEIF", 80), ("AVIF", 80)],
}
LABEL_Q = (50, 80)
METRICS = ("psnr", "ssim", "mae", "rmse")


def chain_key(chain: Sequence[Tuple[str, int]]) -> str:
    return "none" if not chain else ">".join(f"{c}@{q}" for c, q in chain)


# --------------------------------------------------------------------------- #
# Real-encoder worker processes
# --------------------------------------------------------------------------- #
_FMT: Dict[str, str] = {}


def _init_formats() -> None:
    setup_package()
    from stegtransx_cr.codecs import real

    avail = real.available_codecs(refresh=True)
    for name in CODECS:
        if not avail.get(name):
            raise RuntimeError(f"real {name} encoder unavailable")
        _FMT[name] = real._WORKING_FORMAT.get(name) or real.PIL_FORMAT[name]


def _worker_init() -> None:
    torch.set_num_threads(1)
    _init_formats()


def run_chain(arrays: np.ndarray, chain: Sequence[Tuple[str, int]]):
    from PIL import Image

    out = np.empty_like(arrays)
    sizes = np.zeros((len(arrays), len(chain)), dtype=np.int64)
    for n, arr in enumerate(arrays):
        img = Image.fromarray(np.ascontiguousarray(arr))
        for s, (codec, q) in enumerate(chain):
            buf = io.BytesIO()
            kwargs = {"quality": int(q)}
            if codec == "JPEG":
                kwargs["subsampling"] = "4:2:0"
            img.save(buf, format=_FMT[codec], **kwargs)
            sizes[n, s] = buf.tell()
            buf.seek(0)
            with Image.open(buf) as dec:
                dec.load()
                img = dec.convert("RGB")
        out[n] = np.asarray(img, dtype=np.uint8)
    return out, sizes


def _task(args):
    key, start, arrays, chain = args
    out, sizes = run_chain(arrays, chain)
    return key, start, out, sizes


def _pool(workers: int) -> ProcessPoolExecutor:
    # max_tasks_per_child recycles a worker regularly: the HEIF/AVIF encoders
    # occasionally leave a worker in a state that ends with the OS killing it,
    # which breaks the whole pool.
    return ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                               mp_context=get_context("spawn"), max_tasks_per_child=8)


def default_workers() -> int:
    """Worker processes for the real encoders.

    ``os.cpu_count()`` reports the host's cores, not the pod's share, so the
    count is capped: every worker imports torch and each encode is
    single-threaded anyway.
    """
    if "EVAL_WORKERS" in os.environ:
        return max(1, int(os.environ["EVAL_WORKERS"]))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 2
    return max(1, min(16, cpus - 2))


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def _metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, np.ndarray]:
    setup_package()
    from stegtransx_cr.utils import image_metrics

    m = image_metrics(pred.float(), target.float())
    return {k: m[k].detach().float().cpu().numpy() for k in METRICS}


def _to_u8(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(0, 1) * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def _from_u8(a: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2))).to(device).float().div_(255.0)


# --------------------------------------------------------------------------- #
# Scenario list
# --------------------------------------------------------------------------- #
def scenarios(system: System, label_test: bool = True) -> List[Tuple[str, List[Tuple[str, int]], str]]:
    policy = system.label_policy()

    def label_for(chain):
        if policy == "const":
            return "const"
        if policy.startswith("fixed:"):
            return policy.split(":", 1)[1]
        return chain[0][0] if chain else "JPEG"

    out = [("none", [], label_for([]))]
    for codec in CODECS:
        for q in SINGLE_Q:
            out.append(("single", [(codec, q)], label_for([(codec, q)])))
    for name, chain in CASCADES.items():
        out.append((f"cascade:{name}", list(chain), label_for(chain)))
    if label_test and system.uses_label:
        for codec in CODECS:
            for q in LABEL_Q:
                right = label_for([(codec, q)])
                for lab in CODECS:
                    if lab != right:
                        out.append(("label", [(codec, q)], lab))
    return out


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_set(
    system: System,
    cache: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    device,
    engines: Sequence[str] = ("real", "sim"),
    label_test: bool = True,
    batch: int = 50,
    chunk: int = 25,
    workers: int = 0,
    sample_dir: str = "",
    n_samples: int = 4,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    system.eval()
    if getattr(system, "variant", "") == "stx":
        batch = min(batch, 8)  # the baseline's global attention is O(B N^2)
    t0 = time.time()
    n = len(pairs)
    scen = scenarios(system, label_test)
    labels = sorted({lab for _, _, lab in scen})
    per_image: Dict[str, np.ndarray] = {}
    summary: Dict[str, Dict] = {}

    # Secrets and covers as uint8 on the device (small), stego per label.
    cov_u8 = np.stack([cache[i] for i, _ in pairs])
    sec_u8 = np.stack([cache[j] for _, j in pairs])
    stego_u8: Dict[str, np.ndarray] = {}
    for lab in labels:
        chunks = []
        acc: Dict[str, List[np.ndarray]] = {k: [] for k in METRICS}
        for b0 in range(0, n, batch):
            cov, sec = pairs_to_tensors(cache, pairs[b0 : b0 + batch], device)
            lt = codec_tensor("JPEG" if lab == "const" else lab, cov.shape[0], device)
            stego = quantize(system.hide(cov, sec, lt).float())
            m = _metrics(stego, cov)
            for k in METRICS:
                acc[k].append(m[k])
            chunks.append(_to_u8(stego))
        stego_u8[lab] = np.concatenate(chunks)
        for k in METRICS:
            per_image[f"cover|{lab}|cs_{k}"] = np.concatenate(acc[k]).astype(np.float32)

    def score(engine: str, group: str, chain, lab: str, get_batch, extra: Dict = None) -> None:
        acc: Dict[str, List[np.ndarray]] = {k: [] for k in METRICS}
        for b0 in range(0, n, batch):
            rec = system.recover(get_batch(b0, min(n, b0 + batch))).float()
            sec = _from_u8(sec_u8[b0 : b0 + batch], device)
            m = _metrics(rec, sec)
            for k in METRICS:
                acc[k].append(m[k])
        key = f"{engine}|{chain_key(chain)}|{lab}"
        row = {"engine": engine, "group": group, "chain": chain_key(chain), "label": lab}
        for k in METRICS:
            arr = np.concatenate(acc[k]).astype(np.float32)
            per_image[f"{key}|sr_{k}"] = arr
            row[f"sr_{k}"] = float(arr.mean())
            row[f"cs_{k}"] = float(per_image[f"cover|{lab}|cs_{k}"].mean())
        if extra:
            row.update(extra)
        summary[key] = row

    # ---- no compression -------------------------------------------------
    def from_array(arr):
        return lambda a, b: _from_u8(arr[a:b], device)

    for group, chain, lab in scen:
        if not chain:
            for engine in engines:
                score(engine, group, chain, lab, from_array(stego_u8[lab]))

    # ---- simulator ----------------------------------------------------------
    if "sim" in engines:
        for group, chain, lab in scen:
            if not chain or group == "label":
                continue
            def sim_batch(a, b, lab=lab, chain=chain):
                return system.simulator.cascade(_from_u8(stego_u8[lab][a:b], device), list(chain))

            score("sim", group, chain, lab, sim_batch)

    # ---- real encoders ----------------------------------------------------
    # One scenario at a time: a whole-batch fan-out would hold one output buffer
    # and one pickled payload per chunk for every scenario at once (tens of GB).
    size_rows: Dict[str, Dict] = {}
    if "real" in engines:
        workers = workers or default_workers()
        for group, chain, lab in scen:
            if not chain:
                continue
            key = f"{chain_key(chain)}|{lab}"
            buf, sz = run_chain_parallel(stego_u8[lab], chain, workers=workers, chunk=chunk, key=key)
            score("real", group, chain, lab, from_array(buf),
                  {"bytes_first_stage": float(sz[:, 0].astype(np.float64).mean())})
            per_image[f"real|{chain_key(chain)}|{lab}|bytes"] = sz[:, 0].astype(np.int64)
            del buf
        # covers through each codec at q = 80, for the file-size comparison
        for codec in CODECS:
            ck = f"COVER:{codec}@80"
            buf, sz = run_chain_parallel(cov_u8, [(codec, 80)], workers=workers, chunk=chunk, key=ck)
            del buf
            size_rows[ck] = {"cover_bytes": float(sz[:, 0].mean())}
            per_image[f"coverbytes|{ck}"] = sz[:, 0].astype(np.int64)
        for codec in CODECS:
            ck = f"COVER:{codec}@80"
            sk = [k for k in summary if k.startswith(f"real|{codec}@80|") and summary[k]["group"] == "single"]
            if ck in size_rows and sk:
                size_rows[ck]["stego_bytes"] = summary[sk[0]]["bytes_first_stage"]

    # ---- qualitative samples ---------------------------------------------
    if sample_dir:
        from PIL import Image

        os.makedirs(sample_dir, exist_ok=True)
        lab = [l for g, c, l in scen if g == "single" and c == [("JPEG", 80)]][0]
        k = min(n_samples, n)
        received, _ = run_chain_local(stego_u8[lab][:k], [("JPEG", 80)])
        rec = _to_u8(system.recover(_from_u8(received, device)).float())
        for i in range(k):
            resid = np.clip(np.abs(cov_u8[i].astype(np.int16) - stego_u8[lab][i].astype(np.int16)) * 10, 0, 255).astype(np.uint8)
            err = np.clip(np.abs(sec_u8[i].astype(np.int16) - rec[i].astype(np.int16)) * 10, 0, 255).astype(np.uint8)
            for name, arr in (("cover", cov_u8[i]), ("secret", sec_u8[i]), ("stego", stego_u8[lab][i]),
                              ("received_jpeg80", received[i]), ("recovered_jpeg80", rec[i]),
                              ("residual_x10", resid), ("error_x10", err)):
                Image.fromarray(arr).save(os.path.join(sample_dir, f"{i:02d}_{name}.png"))
                # display copies for figures (JPEG q = 95); metrics never use these
                Image.fromarray(arr).save(os.path.join(sample_dir, f"{i:02d}_{name}_view.jpg"), quality=95)

    info = {"pairs": n, "labels": labels, "seconds": time.time() - t0, "filesize_q80": size_rows,
            "label_policy": system.label_policy()}
    return {"info": info, "rows": summary}, per_image


def brief(res: Dict) -> Dict:
    """Compact digest of one evaluation (real encoders, correct label)."""
    rows = res["rows"]
    out = {}
    for key, row in rows.items():
        eng, chain, lab = key.split("|")
        if eng != "real" or row["group"] == "label":
            continue
        if chain in ("none", "JPEG@80", "WebP@80", "HEIF@80", "AVIF@80", "JPEG@50", "AVIF@50",
                     "JPEG@80>WebP@80>HEIF@80>AVIF@80"):
            out[chain] = [round(row["cs_psnr"], 2), round(row["sr_psnr"], 2), round(row["sr_ssim"], 4)]
    return out


def run_chain_local(arrays: np.ndarray, chain):
    if not _FMT:
        _init_formats()
    return run_chain(arrays, chain)


_SHARED_POOL: Optional[ProcessPoolExecutor] = None


def shared_pool(workers: int = 0) -> ProcessPoolExecutor:
    """One worker pool per process; spawning it per call would dominate."""
    global _SHARED_POOL
    if _SHARED_POOL is None:
        _SHARED_POOL = _pool(workers or default_workers())
    return _SHARED_POOL


def reset_pool() -> None:
    """Drop the shared pool and make sure its workers are really gone.

    ``shutdown(wait=False)`` only signals; a worker stuck in a codec call
    survives it and, once this process exits, is reparented and keeps its
    memory for the rest of the pod's life.  That stranded memory is what
    hung the job that followed a failed evaluation, so the workers are
    killed explicitly here.
    """
    global _SHARED_POOL
    pool, _SHARED_POOL = _SHARED_POOL, None
    if pool is None:
        return
    procs = list(getattr(pool, "_processes", {}).values())
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for proc in procs:
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass
    for proc in procs:
        try:
            proc.join(timeout=5)
        except Exception:
            pass


atexit.register(reset_pool)


def run_chain_parallel(arrays: np.ndarray, chain, workers: int = 0, chunk: int = 0,
                       key: str = "chain", attempts: int = 3):
    """Same round-trip as ``run_chain``, spread over worker processes.

    Byte-for-byte identical to the serial path: the workers run the very same
    ``run_chain`` on disjoint slices. ``chunk = 0`` splits the batch so that
    every worker gets a slice. A worker that the OS kills takes the pool with
    it, so a broken pool is rebuilt and the whole slice retried; after
    ``attempts`` failures the round-trip falls back to this process.
    """
    chain = list(chain)
    workers = workers or default_workers()
    chunk = chunk or max(1, min(25, -(-len(arrays) // workers)))
    if len(arrays) <= chunk or workers <= 1:
        return run_chain_local(arrays, chain)
    for attempt in range(attempts):
        out = np.empty_like(arrays)
        sizes = np.zeros((len(arrays), len(chain)), dtype=np.int64)
        try:
            ex = shared_pool(workers)
            futures = [ex.submit(_task, (key, b0, arrays[b0 : b0 + chunk], chain))
                       for b0 in range(0, len(arrays), chunk)]
            for fut in as_completed(futures):
                _k, start, piece, sz = fut.result()
                out[start : start + len(piece)] = piece
                sizes[start : start + len(piece)] = sz
            return out, sizes
        except BrokenProcessPool:
            log(f"worker pool broke on {key}; rebuilding (attempt {attempt + 1}/{attempts})")
            reset_pool()
            time.sleep(2.0)
    log(f"worker pool kept breaking on {key}; encoding in this process instead")
    return run_chain_local(arrays, chain)


def evaluate_system(system: System, sets: Dict[str, Dict], device, out_dir: str,
                    engines=("real", "sim"), label_test: bool = True, samples: bool = False,
                    set_names: Sequence[str] = ("coco_test", "div2k_val")) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    everything = {}
    for name in set_names:
        spec = sets[name]
        log(f"evaluating {name} ({len(spec['pairs'])} pairs)")
        res, per_image = evaluate_set(
            system, spec["cache"], spec["pairs"], device, engines=engines,
            label_test=label_test and name == "coco_test",
            sample_dir=os.path.join(out_dir, "samples", name) if samples and name == "coco_test" else "",
        )
        save_json(os.path.join(out_dir, f"eval_{name}.json"), res)
        np.savez_compressed(os.path.join(out_dir, f"per_image_{name}.npz"), **per_image)
        everything[name] = res
        r = res["rows"]
        lab = res["info"]["labels"]
        pick = [k for k in r if k.startswith("real|JPEG@80|") and r[k]["group"] == "single"]
        if pick:
            row = r[pick[0]]
            log(f"  {name}: cover {row['cs_psnr']:.2f} dB | secret JPEG80 {row['sr_psnr']:.2f} dB "
                f"| {res['info']['seconds']:.0f}s | labels {lab}")
    return everything
