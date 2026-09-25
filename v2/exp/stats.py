"""Paired statistics over per-image metric arrays (no model code needed)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def bootstrap_ci(d: np.ndarray, resamples: int = 10000, seed: int = 0,
                 alpha: float = 0.05, chunk: int = 1000):
    """Percentile bootstrap CI of the mean of ``d`` (paired differences)."""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    done = 0
    while done < resamples:
        m = min(chunk, resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done:done + m] = d[idx].mean(axis=1)
        done += m
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_p(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=np.float64)
    if d.size < 2 or np.allclose(d, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def paired(a: Sequence[float], b: Sequence[float], seed: int = 0) -> Dict:
    """Paired comparison of two per-image metric vectors (a - b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    d = a - b
    lo, hi = bootstrap_ci(d, seed=seed)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    return {"n": int(n), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_wilcoxon": wilcoxon_p(d),
            "dz": float(d.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")}


def seed_summary(values: Sequence[float]) -> Dict:
    """Mean, sd and t-based 95% CI over a handful of seed-level means."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    out = {"n": int(v.size), "mean": float(v.mean()) if v.size else float("nan"),
           "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan")}
    if v.size > 1:
        try:
            from scipy.stats import t
            half = float(t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
        except Exception:
            half = float("nan")
        out["ci_lo"], out["ci_hi"] = out["mean"] - half, out["mean"] + half
    else:
        out["ci_lo"] = out["ci_hi"] = float("nan")
    return out
