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


@pytest.mark.gpu
@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_jax_batched_source_depths() -> None:
    """Batched ``compute_greens_for_depths`` returns a stack of GFs that
    matches the per-depth numpy result to float32 precision."""
    from fkpy.jax_backend.greens_jx import compute_greens_for_depths

    arr = np.array(
        [
            [10.0, 5.5, 3.18, 2.65, 600.0, 300.0],
            [10.0, 6.0, 3.4, 2.78, 800.0, 400.0],
            [0.0, 6.5, 3.7, 2.85, 1000.0, 500.0],
        ]
    )
    model = LayeredModel.from_array(arr)
    depths = np.array([3.0, 8.0, 15.0])
    distances = np.array([50.0, 100.0])
    batched = compute_greens_for_depths(
        model=model,
        src_depths_km=depths,
        distances_km=distances,
        npts=128,
        dt=0.5,
    )
    assert batched.shape == (3, 2, 10, 128)
    # Compare against a per-depth call.
    for i, d in enumerate(depths):
        single = compute_greens(
            model=model,
            src_depth_km=float(d),
            distances_km=distances,
            npts=128,
            dt=0.5,
            n_workers=1,
            backend="numpy",
        )
        rel = np.max(np.abs(batched[i] - single.gf.astype(np.float32))) / max(
            np.max(np.abs(single.gf)), 1e-30
        )
        assert rel < 1e-4, f"depth {d}: rel diff {rel:.3e}"
