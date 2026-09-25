"""Hot fix for the real-encoder path of the evaluation.

The encoders run in worker processes. On most hosts a worker is killed by the
OS part-way through the HEIF round trip at quality 50 -- the container reports
a memory limit -- which breaks the pool. The original fallback then repeated
the work in the job process itself, where the same kill ends the job and loses
every scenario after it; nine of ten evaluations died that way.

The fallback here never encodes in this process. The batch is done slice by
slice, each slice in its own disposable single-worker pool, so a kill costs one
slice. A slice whose worker dies is halved and retried until a single image is
identified; that image is reported and left unencoded rather than taking the
job down. Its recovered secret is then measured without recompression, which
flatters that one image, so the count is logged and belongs in the write-up.

Applied from ``exp/__init__``, so every process that imports the harness has it.
"""
from __future__ import annotations

import os
import time
from itertools import count
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from . import evaluate as _ev
from .common import log, mem_free_gb, push_results, reap_orphans

HEAVY_WORKERS = 2


def _kill_pool(ex) -> None:
    procs = list(getattr(ex, "_processes", {}).values())
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for proc in procs:
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass


def run_chain_isolated(arrays, chain, key="chain", slice_size=50):
    """Same round trip as ``run_chain``, in disposable one-worker pools."""
    out = np.empty_like(arrays)
    sizes = np.zeros((len(arrays), len(chain)), dtype=np.int64)
    lost = []
    # Losing a handful of images to a codec is a result worth recording; losing
    # many means the encoder is broken here (a failing worker initialiser looks
    # exactly like a bad image), and silently returning mostly-unencoded images
    # would be far worse than failing the job.
    budget = max(10, len(arrays) // 50)
    todo = [(a, min(a + slice_size, len(arrays))) for a in range(0, len(arrays), slice_size)]
    while todo:
        a, b = todo.pop(0)
        ex = _ev._pool(1)
        try:
            _k, start, piece, sz = ex.submit(_ev._task, (key, a, arrays[a:b], chain)).result()
            out[start : start + len(piece)] = piece
            sizes[start : start + len(piece)] = sz
        except BrokenProcessPool:
            if b - a == 1:
                lost.append(a)
                out[a] = arrays[a]
                if len(lost) > budget:
                    raise RuntimeError(
                        f"{key}: the encoder failed on {len(lost)} images "
                        f"(first {lost[:5]}); treating this as a broken encoder, not a result")
            else:
                mid = (a + b) // 2
                todo[:0] = [(a, mid), (mid, b)]
        finally:
            _kill_pool(ex)
    if lost:
        log(f"{key}: {len(lost)} of {len(arrays)} images could not be encoded: {lost[:8]}")
    return out, sizes


def run_chain_parallel(arrays, chain, workers=0, chunk=0, key="chain", attempts=3):
    """Drop-in replacement whose fallback isolates instead of encoding here."""
    chain = list(chain)
    workers = workers or _ev.default_workers()
    # HEIF is the one codec that cannot be run at full width. libheif drives
    # x265, which sizes its frame buffers from the host core count, so a dozen
    # concurrent HEIF encoders on a 128-core host take the container from
    # comfortable to its 234 GiB ceiling in about 25 seconds and the OS kills a
    # worker. Measured on one pod: JPEG and WebP scenarios finish in 1-6 s each
    # with free memory flat, then HEIF@50 hits 97 % and breaks the pool. Every
    # other codec is unaffected, so the cap is applied to HEIF chains only and
    # the remaining scenarios keep full throughput.
    if any(str(stage[0]).upper().startswith("HEIF") for stage in chain):
        capped = min(workers, HEAVY_WORKERS)
        if capped != workers:
            log(f"{key}: HEIF in the chain, capping {workers} workers to {capped}")
        workers = capped
    chunk = chunk or max(1, min(25, -(-len(arrays) // workers)))
    # A running evaluation used to be silent from its first scenario to its
    # last, so the only way to ask whether a pod was alive was a host-level
    # utilisation gauge that cannot resolve worker processes on a 128-core
    # machine. One line per scenario answers it directly. This has to wrap
    # every path through the function, serial included: an earlier revision
    # returned from the serial branch above this point and the first run at
    # EVAL_WORKERS=1 was silent for its whole first set, which is exactly the
    # blindness the log was added to remove.
    started = time.time()
    n = next(_SCENARIO)
    log(f"scenario {n}: {key} x{len(arrays)} w{workers} (free {mem_free_gb():.0f} GB)")
    try:
        return _dispatch_chain(arrays, chain, workers, chunk, key, attempts)
    finally:
        log(f"scenario {n}: {key} done in {time.time() - started:.0f}s")


def _dispatch_chain(arrays, chain, workers, chunk, key, attempts):
    if len(arrays) <= chunk or workers <= 1:
        # Serial, in this process. Measured on an A100 host, 256x256, one
        # process: JPEG 1 ms/image, WebP 7 ms, HEIF@50 80 ms, HEIF@80 119 ms,
        # AVIF@50 107 ms, peak RSS 0.08 GB. A thousand HEIF images therefore
        # cost under two minutes and sixty megabytes. Twelve workers bought no
        # speedup at all on the real pod (WebP: 6-9 s per scenario against 7 s
        # measured serially) while producing every failure of this programme:
        # OOM kills, broken pools, bisection cascades, hour-long stalls. The
        # pool is not worth its risk, so EVAL_WORKERS=1 takes this path and the
        # isolation route survives only as a fallback for a genuinely bad image.
        try:
            return _ev.run_chain_local(arrays, chain)
        except Exception as exc:
            log(f"{key}: serial encode failed ({type(exc).__name__}: {exc}); isolating")
            return run_chain_isolated(arrays, chain, key=key)
    # ProcessPoolExecutor recycles a worker every max_tasks_per_child tasks, and
    # a worker stuck inside a codec call does not die when it is retired: it is
    # reparented to init and keeps its address space for the rest of the pod's
    # life. Over fifty scenarios that is what fills the container's cgroup and
    # makes the OS kill a live worker, which is what breaks the pool. Reap the
    # strays before each scenario rather than after the damage.
    stray = reap_orphans()
    if stray:
        log(f"reaped {stray} orphaned encoder workers before {key} "
            f"(free {mem_free_gb():.0f} GB)")
    return _run_chain_parallel(arrays, chain, workers, chunk, key, attempts)


_SCENARIO = count(1)


def _run_chain_parallel(arrays, chain, workers, chunk, key, attempts):
    for attempt in range(attempts):
        out = np.empty_like(arrays)
        sizes = np.zeros((len(arrays), len(chain)), dtype=np.int64)
        try:
            ex = _ev.shared_pool(workers)
            futures = [ex.submit(_ev._task, (key, b0, arrays[b0 : b0 + chunk], chain))
                       for b0 in range(0, len(arrays), chunk)]
            for fut in _ev.as_completed(futures):
                _k, start, piece, sz = fut.result()
                out[start : start + len(piece)] = piece
                sizes[start : start + len(piece)] = sz
            return out, sizes
        except BrokenProcessPool:
            log(f"worker pool broke on {key} (free {mem_free_gb():.0f} GB); "
                f"rebuilding (attempt {attempt + 1}/{attempts})")
            _ev.reset_pool()
            time.sleep(2.0)
    log(f"worker pool kept breaking on {key} (free {mem_free_gb():.0f} GB); isolating slice by slice")
    return run_chain_isolated(arrays, chain, key=key)




# --------------------------------------------------------------------------- #
# Scenario budget
# --------------------------------------------------------------------------- #
# The codec-label mismatch grid is 24 of the ~50 real-encoder scenarios. It
# demonstrates what the codec label does, which is a property of the method and
# is shown on the primary model; running it for every seed and ablation doubles
# both the wall clock and the peak memory of an evaluation that is already
# fighting the container's limit. LABEL_TEST=0 drops it for a given run.
# --------------------------------------------------------------------------- #
# Codec exclusion
# --------------------------------------------------------------------------- #
# eval:single-cond-s0 does not terminate on its HEIF scenarios. Three pods were
# lost to it: the first held 184 GiB of a 232.8 GiB container for seventy
# minutes at HEIF@50; the third, with CPU_CAP=16, showed no memory pressure at
# all and still did not finish that one scenario in twenty minutes. More
# threads were not faster and fewer threads were not slower, which is the
# signature of a hang rather than of cost - libheif/x265 buffering frames it
# never drains on this run's stego images. Every other evaluation encodes HEIF
# without trouble, so this is a property of that one model's output.
#
# SKIP_CODECS=HEIF drops every scenario whose chain touches HEIF, for that run
# only, and the restriction is stated in the manuscript rather than worked
# around. The alternative - a timeout that silently discards the images the
# encoder cannot handle - would put an unquantified hole inside a reported mean.
SKIP_CODECS = tuple(
    c.strip().upper() for c in os.environ.get("SKIP_CODECS", "").split(",") if c.strip()
)
_orig_scenarios = _ev.scenarios


def scenarios(system, label_test=True):
    out = _orig_scenarios(system, label_test)
    if not SKIP_CODECS:
        return out
    kept = [
        (group, chain, lab) for group, chain, lab in out
        if not any(str(stage[0]).upper() in SKIP_CODECS for stage in chain)
    ]
    log(f"SKIP_CODECS={','.join(SKIP_CODECS)}: {len(out) - len(kept)} of "
        f"{len(out)} scenarios dropped, {len(kept)} kept")
    return kept


_orig_evaluate_system = _ev.evaluate_system


def evaluate_system(system, sets, device, out_dir, engines=("real", "sim"),
                    label_test=True, samples=False, set_names=("coco_test", "div2k_val")):
    """One test set at a time, pushing after each.

    Evaluations on these hosts die without warning often enough that an
    all-or-nothing unit is the wrong shape: a death during the second set
    used to discard the first, which is the expensive one. Each set is
    evaluated and pushed on its own, so a later failure costs only the set
    that was still running.
    """
    if os.environ.get("LABEL_TEST", "1") != "1":
        label_test = False
        log("label-mismatch scenarios disabled for this run (LABEL_TEST=0)")
    run_dir = os.path.dirname(os.path.abspath(out_dir))
    run = os.path.basename(run_dir)
    everything = {}
    for i, name in enumerate(set_names):
        everything.update(_orig_evaluate_system(
            system, sets, device, out_dir, engines=engines, label_test=label_test,
            samples=samples, set_names=(name,)))
        if i + 1 < len(set_names):      # the job's own push covers the last one
            try:
                push_results(run, run_dir, exclude_ext=(".tmp", "last.pth"))
                log(f"pushed {run} after {name} ({i + 1}/{len(set_names)} sets)")
            except Exception as exc:    # a push must never lose the results
                log(f"interim push after {name} failed: {exc}")
    return everything


def apply() -> None:
    _ev.run_chain_parallel = run_chain_parallel
    _ev.run_chain_isolated = run_chain_isolated
    _ev.evaluate_system = evaluate_system
    if SKIP_CODECS:
        _ev.scenarios = scenarios
        # the cover-size table iterates CODECS directly inside evaluate_set
        _ev.CODECS = tuple(c for c in _ev.CODECS if c.upper() not in SKIP_CODECS)


apply()
