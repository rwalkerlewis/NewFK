"""Smoke test for the JAX backend.

Skipped unless JAX is installed.  The current backend wraps the numpy
implementation and recasts to float32, so the test verifies that the
results are within float32 relative precision of the numpy backend.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fkpy.greens import compute_greens
from fkpy.model import LayeredModel

JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@pytest.mark.gpu
@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_jax_backend_matches_numpy_backend() -> None:
    arr = np.array(
        [
            [10.0, 5.5, 3.18, 2.65, 600.0, 300.0],
            [0.0, 6.5, 3.7, 2.85, 1000.0, 500.0],
        ]
    )
    model = LayeredModel.from_array(arr)
    kw = dict(
        src_depth_km=5.0,
        distances_km=np.array([50.0]),
        npts=128,
        dt=0.5,
        n_workers=1,
    )
    res_np = compute_greens(model=model, backend="numpy", **kw)  # type: ignore[arg-type]
    res_jx = compute_greens(model=model, backend="jax", **kw)  # type: ignore[arg-type]
    rel = np.max(np.abs(res_np.gf - res_jx.gf)) / np.max(np.abs(res_np.gf))
    assert rel < 1e-3
