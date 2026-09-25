"""Shared paths, package setup, exact speed patches, data and git helpers."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

WORK = os.environ.get("WORK", "/workspace")
PKG_DIR = os.environ.get("PKG_DIR", os.path.join(WORK, "StegTransX-CR"))
STX_DIR = os.environ.get("STX_DIR", os.path.join(WORK, "third_party", "StegTransX"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(WORK, "data"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(WORK, "out"))
REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMOKE = os.environ.get("SMOKE", "0") == "1"

CODECS = ("JPEG", "WebP", "HEIF", "AVIF")


def reap_orphans() -> int:
    """Kill multiprocessing workers orphaned by a failed job.

    Every job runs in its own subprocess, so a worker that outlives a crash
    is reparented to the container's init and keeps whatever memory it held.
    Enough of them and the next job cannot allocate, which shows up as a
    silent hang rather than a failure.  Called between stages, when nothing
    of ours is meant to be running.

    A process qualifies only if it is one of CPython's own multiprocessing
    helpers *and* has been reparented to init, so a shell that merely
    mentions the string is never a candidate.
    """
    marks = ("from multiprocessing.spawn import spawn_main",
             "from multiprocessing.resource_tracker import main")
    killed = 0
    me = os.getpid()
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return 0
    for pid in pids:
        if int(pid) in (me, 1):
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().decode("utf-8", "replace").split("\x00")
            with open(f"/proc/{pid}/stat", "rb") as fh:
                ppid = int(fh.read().decode("utf-8", "replace").rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid != 1 or not argv or "python" not in os.path.basename(argv[0]):
            continue
        if not any(m in a for a in argv for m in marks):
            continue
        try:
            os.kill(int(pid), signal.SIGKILL)
            killed += 1
        except OSError:
            pass
    return killed


def mem_free_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return -1.0


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{stamp}] {msg}", flush=True)


def setup_package():
    """Make the released package importable (never modified in place)."""
    if PKG_DIR not in sys.path:
        sys.path.insert(0, PKG_DIR)
    import stegtransx_cr  # noqa: F401

    return stegtransx_cr


# --------------------------------------------------------------------------- #
# Exact speed patches
# --------------------------------------------------------------------------- #
_ATTN_MODE = {"mode": "explicit"}


def patch_attention(mode: str = "sdpa") -> None:
    """Swap WindowAttention.forward for a fused SDPA version (same maths).

    softmax(q k^T / sqrt(d) + B) v with the learned positional bias B as an
    additive mask; the default SDPA scale is 1/sqrt(head_dim), identical to the
    explicit implementation in the package.
    """
    setup_package()
    from stegtransx_cr.models import blocks

    if not hasattr(blocks.WindowAttention, "_explicit_forward"):
        blocks.WindowAttention._explicit_forward = blocks.WindowAttention.forward
    if mode == "explicit":
        blocks.WindowAttention.forward = blocks.WindowAttention._explicit_forward
        _ATTN_MODE["mode"] = mode
        return

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        q_src = self.norm_q(x)
        kv_src = q_src if context is None else self.norm_kv(context)
        q_src, pad_h, pad_w = blocks._pad_to_window(q_src, self.window)
        kv_src, _, _ = blocks._pad_to_window(kv_src, self.window)
        _, _, hp, wp = q_src.shape
        q_tokens = blocks.window_partition(q_src, self.window)
        kv_tokens = blocks.window_partition(kv_src, self.window)
        n_win, tokens, _ = q_tokens.shape
        hd = c // self.num_heads
        q = self.to_q(q_tokens)
        k, v = self.to_kv(kv_tokens).chunk(2, dim=-1)
        q = q.view(n_win, tokens, self.num_heads, hd).transpose(1, 2)
        k = k.view(n_win, tokens, self.num_heads, hd).transpose(1, 2)
        v = v.view(n_win, tokens, self.num_heads, hd).transpose(1, 2)
        bias = self.position_bias.to(dtype=q.dtype).unsqueeze(0).expand(n_win, -1, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(n_win, tokens, c)
        out = self.proj(out)
        out = blocks.window_reverse(out, self.window, b, c, hp, wp)
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out

    blocks.WindowAttention.forward = forward
    _ATTN_MODE["mode"] = mode


def patch_simulator_skips() -> None:
    """Skip learned simulator modules whose parameters make them an exact identity.

    A LearnedDistortion whose conv_out is all zeros returns x + 0, a deblocking
    filter or CDEF with zero strength returns x, and a delta loop-restoration
    kernel returns x. Skipping them changes neither values nor gradients. The
    flags are computed by :func:`refresh_identity_flags` after weights load.
    """
    setup_package()
    from stegtransx_cr.codecs import avif, modules

    pairs = [
        (modules.LearnedDistortion, "forward"),
        (modules.LearnedDeblockingFilter, "forward"),
        (avif.CDEFFilter, "forward"),
        (avif.LoopRestoration, "forward"),
    ]
    for cls, name in pairs:
        if hasattr(cls, "_orig_forward"):
            continue
        cls._orig_forward = getattr(cls, name)

        def make(orig):
            def fwd(self, x, *args, **kwargs):
                if getattr(self, "_is_identity", False):
                    return x
                return orig(self, x, *args, **kwargs)

            return fwd

        setattr(cls, name, make(cls._orig_forward))


@torch.no_grad()
def patch_real_parallel(workers: int = 0) -> None:
    """Run ``real.encode_decode`` on worker processes instead of serially.

    Same encoder, same settings, same uint8 conversion as the released helper —
    only the loop over the batch moves off the main process. Used by the
    calibration loop, whose cost is dominated by real HEIF/AVIF encoding.
    """
    setup_package()
    from stegtransx_cr.codecs import real as real_codecs

    if getattr(real_codecs, "_parallel_patched", False):
        return

    def encode_decode(batch: torch.Tensor, codec: str, quality: int) -> torch.Tensor:
        from .evaluate import run_chain_parallel

        if not real_codecs.is_available(codec):
            raise RuntimeError(f"real {codec} codec is unavailable in this environment")
        arr = (batch.detach().clamp(0, 1).cpu().float().numpy() * 255.0).round().astype(np.uint8)
        arr = np.ascontiguousarray(arr.transpose(0, 2, 3, 1))
        out, _sizes = run_chain_parallel(arr, [(codec, int(quality))], workers=workers)
        tensor = torch.from_numpy(np.ascontiguousarray(out.transpose(0, 3, 1, 2)))
        return tensor.to(device=batch.device, dtype=batch.dtype) / 255.0

    real_codecs.encode_decode = encode_decode
    real_codecs._parallel_patched = True
    log("patched real.encode_decode to run on worker processes")


def refresh_identity_flags(simulator) -> Dict[str, bool]:
    setup_package()
    from stegtransx_cr.codecs import avif, modules

    flags = {}
    for name, mod in simulator.named_modules():
        ident = None
        if isinstance(mod, modules.LearnedDistortion):
            w, b = mod.conv_out.weight, mod.conv_out.bias
            ident = bool((w == 0).all()) and (b is None or bool((b == 0).all()))
        elif isinstance(mod, modules.LearnedDeblockingFilter):
            ident = bool((mod.scale == 0).all())
        elif isinstance(mod, avif.CDEFFilter):
            ident = bool((mod.strength == 0).all())
        elif isinstance(mod, avif.LoopRestoration):
            hk = mod.half_kernel
            ident = bool(hk[0] > 0) and bool((hk[1:] == 0).all())
        if ident is not None:
            mod._is_identity = ident
            flags[name] = ident
    return flags


class IdentityFAA(torch.nn.Module):
    """Ablation: bottleneck without frequency-adaptive attention."""

    def forward(self, x, codec):  # noqa: D401
        return x


# --------------------------------------------------------------------------- #
# Images and datasets
# --------------------------------------------------------------------------- #
def list_images(directory: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    out = []
    for root, _, names in os.walk(directory):
        for name in names:
            if name.lower().endswith(exts):
                out.append(os.path.join(root, name))
    return sorted(out)


def _load_resize(args):
    path, size = args
    setup_package()
    from stegtransx_cr.data import load_and_resize

    return load_and_resize(path, size)


def build_eval_cache(files: Sequence[str], cache_path: str, size: int = 256, workers: int = 0) -> np.ndarray:
    """Centre-crop + bicubic resize exactly like the v1 package, cached as .npy."""
    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(files):
            return arr
    workers = workers or max(1, min(32, (os.cpu_count() or 2)))
    log(f"caching {len(files)} images -> {os.path.basename(cache_path)} ({workers} threads)")
    if workers > 1:
        # threads, not processes: Pillow releases the GIL while decoding, and a
        # fork after CUDA is initialised can deadlock the parent.
        from multiprocessing.pool import ThreadPool

        with ThreadPool(workers) as pool:
            arrays = pool.map(_load_resize, [(f, size) for f in files], chunksize=16)
    else:
        arrays = [_load_resize((f, size)) for f in files]
    arr = np.stack(arrays).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, arr)
    return arr


def synthetic_images(count: int, size: int, seed: int) -> np.ndarray:
    setup_package()
    from stegtransx_cr.data import make_synthetic_cache

    return make_synthetic_cache(count, size, seed)


def coco_files() -> List[str]:
    return list_images(os.path.join(DATA_DIR, "val2017"))


def div2k_train_files() -> List[str]:
    return list_images(os.path.join(DATA_DIR, "DIV2K_train_HR"))


def div2k_valid_files() -> List[str]:
    return list_images(os.path.join(DATA_DIR, "DIV2K_valid_HR"))


def eval_sets(size: int = 0) -> Dict[str, Dict]:
    """Validation and test sets used by every v2 run (disjoint COCO images).

    coco_val : covers COCO[0:100],     secrets COCO[100:200]     (100 pairs)
    coco_test: covers COCO[1000:2000], secrets COCO[2000:3000]   (1000 pairs)
    steganalysis training uses covers COCO[200:1000] with secrets COCO[3000:3800]
    div2k_val: DIV2K valid 0801-0900, secret = image (i + 50) mod 100 (100 pairs)
    v1_test  : the 200 v1 test pairs (COCO[80:300], build_pairs seed 43)
    """
    setup_package()
    from stegtransx_cr.data import build_pairs

    size = size or (64 if SMOKE else 256)
    if SMOKE:
        coco = synthetic_images(64, size, 11)
        div = synthetic_images(12, size, 12)
        return {
            "coco_val": {"cache": coco, "pairs": [(i, i + 8) for i in range(8)]},
            "coco_test": {"cache": coco, "pairs": [(16 + i, 32 + i) for i in range(12)]},
            "div2k_val": {"cache": div, "pairs": [(i, (i + 6) % 12) for i in range(6)]},
            "v1_test": {"cache": coco, "pairs": [(40 + i, 50 + i) for i in range(6)]},
        }
    files = coco_files()
    if len(files) < 5000:
        raise RuntimeError(f"COCO val2017 incomplete: {len(files)} files")
    coco = build_eval_cache(files[:5000], os.path.join(DATA_DIR, f"coco5000_{size}.npy"), size)
    dv_files = div2k_valid_files()
    div = build_eval_cache(dv_files, os.path.join(DATA_DIR, f"div2k_valid_{size}.npy"), size)
    v1_local = build_pairs(220, 200, 43)
    return {
        "coco_val": {"cache": coco, "pairs": [(i, 100 + i) for i in range(100)]},
        "coco_test": {"cache": coco, "pairs": [(1000 + i, 2000 + i) for i in range(1000)]},
        "div2k_val": {"cache": div, "pairs": [(i, (i + 50) % len(dv_files)) for i in range(len(dv_files))]},
        "v1_test": {"cache": coco, "pairs": [(80 + i, 80 + j) for i, j in v1_local]},
    }


def pairs_to_tensors(cache: np.ndarray, pairs: Sequence[Tuple[int, int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    covers = np.stack([cache[i] for i, _ in pairs]).transpose(0, 3, 1, 2)
    secrets = np.stack([cache[j] for _, j in pairs]).transpose(0, 3, 1, 2)
    c = torch.from_numpy(covers.copy()).to(device).float().div_(255.0)
    s = torch.from_numpy(secrets.copy()).to(device).float().div_(255.0)
    return c, s


def quantize(x: torch.Tensor) -> torch.Tensor:
    """What a saved 8-bit PNG holds: clamp to [0, 1] and round to 1/255."""
    return (x.clamp(0.0, 1.0) * 255.0).round() / 255.0


class GpuCropSampler:
    """Random-resized square crops of full-resolution DIV2K images, on the GPU.

    Crop side s is log-uniform in [size, min(H, W)], resized to size x size with
    antialiased bicubic interpolation, then one of the 8 dihedral transforms.
    Covers and secrets come from 2B distinct images per batch.
    """

    def __init__(self, images: List[torch.Tensor], size: int, seed: int) -> None:
        self.images = images
        self.size = size
        self.rng = np.random.default_rng(seed)

    def _crop(self, idx: int) -> torch.Tensor:
        img = self.images[idx]
        _, h, w = img.shape
        side_max = min(h, w)
        if side_max <= self.size:
            s = side_max
        else:
            s = int(round(math.exp(self.rng.uniform(math.log(self.size), math.log(side_max)))))
            s = max(self.size, min(side_max, s))
        y = int(self.rng.integers(0, h - s + 1))
        x = int(self.rng.integers(0, w - s + 1))
        patch = img[:, y : y + s, x : x + s].unsqueeze(0).float().div_(255.0)
        if s != self.size:
            patch = F.interpolate(patch, size=(self.size, self.size), mode="bicubic", antialias=True, align_corners=False)
        k = int(self.rng.integers(0, 8))
        if k & 1:
            patch = patch.flip(-1)
        if k & 2:
            patch = patch.flip(-2)
        if k & 4:
            patch = patch.transpose(-1, -2)
        return patch

    def batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        replace = len(self.images) < 2 * n
        idx = self.rng.choice(len(self.images), size=2 * n, replace=replace)
        crops = torch.cat([self._crop(int(i)) for i in idx], dim=0)
        crops = quantize(crops)
        return crops[:n].contiguous(), crops[n:].contiguous()

    def state(self) -> Dict:
        return {"bit_generator": self.rng.bit_generator.state}

    def load_state(self, state: Dict) -> None:
        self.rng.bit_generator.state = state["bit_generator"]


def _decode_png(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_train_images(device, limit: Optional[int] = None) -> List[torch.Tensor]:
    """DIV2K train HR decoded once and kept on the GPU as uint8 (about 6.6 GB)."""
    if SMOKE:
        arr = synthetic_images(16, 96, 5)
        return [torch.from_numpy(a.transpose(2, 0, 1).copy()).to(device) for a in arr]
    files = div2k_train_files()
    if limit:
        files = files[:limit]
    if len(files) < 800 and not limit:
        raise RuntimeError(f"DIV2K train incomplete: {len(files)} files")
    from multiprocessing.pool import ThreadPool

    workers = max(1, min(16, os.cpu_count() or 2))
    log(f"decoding {len(files)} DIV2K training images ({workers} threads)")
    out: List[torch.Tensor] = []
    # threads, not processes: this runs after the model is on the GPU, and
    # forking a CUDA-initialised process can deadlock. Pillow releases the GIL
    # while decoding, so threads parallelise well. Each image moves to the GPU
    # as it arrives instead of piling up 6.6 GB of host memory.
    with ThreadPool(workers) as pool:
        for arr in pool.imap(_decode_png, files, chunksize=4):
            out.append(torch.from_numpy(arr.transpose(2, 0, 1).copy()).to(device))
    return out


# --------------------------------------------------------------------------- #
# Checkpoints and JSON
# --------------------------------------------------------------------------- #
def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


def torch_save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def env_info() -> Dict:
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0]}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import PIL

        info["pillow"] = PIL.__version__
    except Exception:
        pass
    for mod in ("pillow_heif", "pillow_avif"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        from PIL import Image

        Image.init()
        info["avif_saver"] = getattr(Image.SAVE.get("AVIF"), "__module__", None)
        info["heif_saver"] = getattr(Image.SAVE.get("HEIF"), "__module__", None)
    except Exception:
        pass
    info["pod"] = os.environ.get("RUNPOD_POD_ID")
    return info


# --------------------------------------------------------------------------- #
# Git: push a job's result directory to its own orphan branch
# --------------------------------------------------------------------------- #
def _git(args: List[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return subprocess.run(["git"] + args, cwd=cwd, env=env, check=check, capture_output=True, text=True)


def remote_url() -> str:
    """Remote for pushes and fetches; the token is supplied by GIT_ASKPASS."""
    override = os.environ.get("PUSH_URL")
    if override:
        return override
    return f"https://x-access-token@github.com/{os.environ.get('GH_REPO', '')}.git"


def push_results(job: str, src_dir: str, exclude_ext: Sequence[str] = (".tmp",), max_mb: float = 95.0) -> bool:
    """Force-push ``src_dir`` as the single commit of branch ``r/<job>``."""
    if os.environ.get("NO_PUSH", "0") == "1":
        log(f"push skipped (NO_PUSH) for {job}")
        return False
    repo = os.environ.get("GH_REPO")
    if not repo and not os.environ.get("PUSH_URL"):
        log("push skipped: GH_REPO not set")
        return False
    stage = os.path.join(WORK, "push_stage", job.replace("/", "_"))
    subprocess.run(["rm", "-rf", stage], check=False)
    os.makedirs(stage, exist_ok=True)
    try:
        _git(["init", "-q"], stage)
        _git(["config", "user.name", "stegtransx-cr-runner"], stage)
        _git(["config", "user.email", "runner@users.noreply.github.com"], stage)
        _git(["checkout", "-q", "--orphan", f"r/{job}"], stage)
        dest = os.path.join(stage, "results", job)
        os.makedirs(dest, exist_ok=True)
        for root, _, names in os.walk(src_dir):
            for name in names:
                if name.endswith(tuple(exclude_ext)):
                    continue
                full = os.path.join(root, name)
                if os.path.getsize(full) > max_mb * 1e6:
                    continue
                rel = os.path.relpath(full, src_dir)
                target = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                subprocess.run(["cp", full, target], check=True)
        _git(["add", "-A"], stage)
        _git(["commit", "-q", "-m", f"{job} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"], stage)
        url = remote_url()
        for attempt in range(4):
            res = _git(["push", "-q", "-f", url, f"r/{job}"], stage, check=False)
            if res.returncode == 0:
                log(f"pushed r/{job}")
                return True
            msg = (res.stderr or "").strip().replace("\n", " ")[:300]
            log(f"push attempt {attempt + 1} failed (code {res.returncode}): {msg}")
            if any(s in msg.lower() for s in ("not found", "denied", "403", "401", "authentication")):
                break
            time.sleep(10 * (attempt + 1))
    except Exception as err:  # never let a push failure kill a run
        log(f"push error for {job}: {type(err).__name__}")
    return False
