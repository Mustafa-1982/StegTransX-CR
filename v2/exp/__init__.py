"""StegTransX-CR v2 experiment harness (Runpod).

Builds on the released ``stegtransx_cr`` package (v1.0.0) without modifying it:
speed patches (SDPA window attention, exact identity skips in the simulator)
are applied at import time by :mod:`exp.common` and are numerically exact.
"""

try:  # a hot fix must never be the reason the package fails to import
    from . import hotfix as _hotfix  # noqa: F401
except Exception:  # pragma: no cover
    pass
