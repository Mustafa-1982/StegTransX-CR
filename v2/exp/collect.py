"""Gather every r/<job> results branch into one compact digest and push r/collect.

Runs on a pod (it needs the repository token). The digest is small enough to be
read back through the GitHub API: CSV tables, a JSON of paired statistics, and
the per-run configuration and validation curves.  The heavy artefacts
(per-image .npz, weights, sample PNGs) stay on their own branches.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional

import numpy as np

from .common import OUT_DIR, WORK, log, push_results, remote_url, save_json
from .stats import paired, seed_summary

SRC = os.path.join(WORK, "collect_src")
DIGEST = os.path.join(OUT_DIR, "collect")
METRICS = ("psnr", "ssim", "mae", "rmse")


# --------------------------------------------------------------------------- #
def remote_branches() -> List[str]:
    res = subprocess.run(["git", "ls-remote", "--heads", remote_url(), "refs/heads/r/*"],
                         capture_output=True, text=True, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    out = []
    for line in res.stdout.splitlines():
        parts = line.split("refs/heads/")
        if len(parts) == 2 and parts[1].startswith("r/"):
            out.append(parts[1][2:])
    return sorted(out)


def fetch(job: str) -> str:
    """Clone branch r/<job> and return the directory holding its results."""
    dest = os.path.join(SRC, job.replace("/", "_"))
    if os.path.isdir(dest):
        return os.path.join(dest, "results", job)
    shutil.rmtree(dest, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", f"r/{job}", remote_url(), dest],
                   check=False, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    return os.path.join(dest, "results", job)


def read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def eval_dir(root: str) -> str:
    """Where a run keeps its evaluation artefacts.

    A trained run writes them to <root>/eval; the released-model runs of job e1
    write them straight into their own directory. Accept either layout.
    """
    nested = os.path.join(root, "eval")
    return nested if os.path.isdir(nested) else root


# --------------------------------------------------------------------------- #
def run_rows(run: str, root: str) -> List[Dict]:
    """Flatten every eval_<set>.json of one run into records."""
    rows = []
    for path in sorted(glob.glob(os.path.join(eval_dir(root), "eval_*.json"))):
        setname = os.path.basename(path)[5:-5]
        res = read_json(path)
        if not res or "rows" not in res:
            continue
        for key, row in res["rows"].items():
            rec = {"run": run, "set": setname, "key": key, "engine": row.get("engine", ""),
                   "group": row.get("group", ""), "chain": row.get("chain", ""),
                   "label": row.get("label", "")}
            for m in METRICS:
                rec[f"sr_{m}"] = row.get(f"sr_{m}", "")
                rec[f"cs_{m}"] = row.get(f"cs_{m}", "")
            for extra in ("bpp", "bytes", "size_ratio"):
                if extra in row:
                    rec[extra] = row[extra]
            rows.append(rec)
    return rows


def curve_rows(run: str, root: str, keep: int = 400) -> List[Dict]:
    path = os.path.join(root, "history.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        hist = list(csv.DictReader(fh))
    if len(hist) > keep:
        idx = np.unique(np.linspace(0, len(hist) - 1, keep).astype(int))
        hist = [hist[i] for i in idx]
    out = []
    for row in hist:
        rec = {"run": run}
        rec.update({k: row.get(k, "") for k in row})
        out.append(rec)
    return out


def per_image(root: str, setname: str = "coco_test") -> Dict[str, np.ndarray]:
    path = os.path.join(eval_dir(root), f"per_image_{setname}.npz")
    if not os.path.exists(path):
        return {}
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def main_keys(arrays: Dict[str, np.ndarray], metric: str = "psnr") -> Dict[str, str]:
    """Map 'engine|chain' -> full npz key, for the non-label scenarios."""
    out = {}
    for key in arrays:
        parts = key.split("|")
        if len(parts) != 4 or parts[0] == "cover" or parts[3] != f"sr_{metric}":
            continue
        engine, chain, _lab, _m = parts
        out.setdefault(f"{engine}|{chain}", key)
    return out


# --------------------------------------------------------------------------- #
def job_collect() -> Dict:
    os.makedirs(DIGEST, exist_ok=True)
    branches = [b for b in remote_branches() if not b.startswith("status-") and b != "collect"]
    log(f"branches: {branches}")

    runs: Dict[str, str] = {}          # run name -> results dir
    eval_records: List[Dict] = []
    curve_records: List[Dict] = []
    configs: Dict[str, Dict] = {}
    singles: Dict[str, Dict] = {}      # calibrate / bench / e0 / e1 / steg payloads

    for job in branches:
        root = fetch(job)
        if not os.path.isdir(root):
            log(f"  {job}: nothing fetched")
            continue
        if job in ("calibrate", "bench", "steg", "e0", "e1"):
            for path in sorted(glob.glob(os.path.join(root, "*.json"))):
                payload = read_json(path)
                if payload is not None:
                    singles[f"{job}/{os.path.basename(path)[:-5]}"] = payload
        if job.startswith("e1/") or job == "e1":
            for sub in sorted(glob.glob(os.path.join(root, "*", "eval_*.json"))
                              + glob.glob(os.path.join(root, "*", "eval", "eval_*.json"))):
                home = os.path.dirname(sub)
                if os.path.basename(home) == "eval":
                    home = os.path.dirname(home)
                regime = os.path.basename(home)
                if regime.startswith("v1-"):      # v1-multi -> multi, so that the
                    regime = regime[3:]           # v2-vs-v1release lookup matches
                runs.setdefault(f"e1:{regime}", home)
        cfg = read_json(os.path.join(root, "config.json"))
        if cfg is not None and os.path.exists(os.path.join(root, "history.csv")):
            runs[job] = root
            configs[job] = cfg

    for run, root in sorted(runs.items()):
        eval_records += run_rows(run, root)
        curve_records += curve_rows(run, root)
    log(f"runs: {sorted(runs)} ({len(eval_records)} eval rows)")

    # ---- paired statistics -------------------------------------------- #
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    comparisons: List[Dict] = []

    def arrays_for(run: str) -> Dict[str, np.ndarray]:
        if run not in cache:
            cache[run] = per_image(runs[run])
        return cache[run]

    def compare(a: str, b: str, tag: str) -> None:
        aa, bb = arrays_for(a), arrays_for(b)
        if not aa or not bb:
            return
        ka, kb = main_keys(aa), main_keys(bb)
        for scen in sorted(set(ka) & set(kb)):
            rec = paired(aa[ka[scen]], bb[kb[scen]])
            rec.update({"comparison": tag, "a": a, "b": b, "scenario": scen, "metric": "sr_psnr"})
            comparisons.append(rec)
        # Cover PSNR is one key per conditioning label. The earlier form only
        # emitted a row when each side had exactly one, which is true of a
        # fixed-label run and false of every multi-codec run (four labels), so
        # the contrast that matters most never appeared. Pair the labels the
        # two runs share instead, and keep a plain "cover" row on the JPEG
        # label so anything reading that name still finds it.
        ca = {k.split("|")[1]: k for k in aa if k.startswith("cover|") and k.endswith("cs_psnr")}
        cb = {k.split("|")[1]: k for k in bb if k.startswith("cover|") and k.endswith("cs_psnr")}
        shared = sorted(set(ca) & set(cb))
        for lab in shared:
            rec = paired(aa[ca[lab]], bb[cb[lab]])
            rec.update({"comparison": tag, "a": a, "b": b,
                        "scenario": f"cover|{lab}", "metric": "cs_psnr"})
            comparisons.append(rec)
        primary = "JPEG" if "JPEG" in shared else (shared[0] if shared else None)
        if primary:
            rec = paired(aa[ca[primary]], bb[cb[primary]])
            rec.update({"comparison": tag, "a": a, "b": b, "scenario": "cover",
                        "metric": "cs_psnr", "cover_label": primary})
            comparisons.append(rec)

    def parse(run: str):
        """'multi-cond-s0' -> (regime, variant, seed).

        config.json is written as {"cfg": {...}, "env": {...},
        "identity_flags": {...}}, so the training parameters sit one level down.
        Reading them off the top level returned (None, None, None) for every
        run, which silently emptied all three consumers of this function: the
        ablation contrasts, the v2-vs-v1 contrasts and the seed aggregation.
        The first collect run reported 1600 eval rows and 0 comparisons for
        exactly this reason.
        """
        cfg = configs.get(run, {}).get("cfg", {})
        return cfg.get("regime"), cfg.get("variant"), cfg.get("seed")

    trained = [r for r in runs if r in configs]
    for run in sorted(trained):
        regime, variant, seed = parse(run)
        if variant == "cond" or regime is None:
            continue
        base = next((r for r in trained if parse(r) == (regime, "cond", seed)), None)
        if base:
            compare(base, run, f"cond-vs-{variant}")
    for run in sorted(trained):
        regime, variant, seed = parse(run)
        if variant != "cond":
            continue
        v1 = f"e1:{regime}"
        if v1 in runs:
            compare(run, v1, "v2-vs-v1release")

    # ---- seed aggregation ---------------------------------------------- #
    seeds: Dict[str, Dict[str, List[float]]] = {}
    for rec in eval_records:
        if rec["set"] != "coco_test" or rec["group"] == "label" or rec["engine"] != "real":
            continue
        regime, variant, _ = parse(rec["run"])
        if regime is None:
            continue
        group = f"{regime}-{variant}"
        for metric in ("sr_psnr", "cs_psnr", "sr_ssim"):
            value = rec.get(metric, "")
            if value == "":
                continue
            seeds.setdefault(f"{group}|{rec['chain']}|{metric}", {}).setdefault("v", []).append(float(value))
    seed_rows = [dict(key=k, **seed_summary(v["v"])) for k, v in sorted(seeds.items())]

    # ---- write and push -------------------------------------------------- #
    def write_csv(name: str, rows: List[Dict]) -> None:
        if not rows:
            return
        fields: List[str] = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        with open(os.path.join(DIGEST, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow(row)

    write_csv("eval_rows.csv", eval_records)
    write_csv("curves.csv", curve_records)
    write_csv("seed_summary.csv", seed_rows)
    write_csv("comparisons.csv", comparisons)
    save_json(os.path.join(DIGEST, "configs.json"), configs)
    save_json(os.path.join(DIGEST, "singles.json"), singles)
    save_json(os.path.join(DIGEST, "index.json"),
              {"branches": branches, "runs": sorted(runs), "n_eval_rows": len(eval_records),
               "n_comparisons": len(comparisons)})
    push_results("collect", DIGEST)
    return {"ok": True, "runs": sorted(runs), "rows": len(eval_records), "comparisons": len(comparisons)}
