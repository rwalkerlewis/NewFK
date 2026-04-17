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
    kw = {
        "src_depth_km": 5.0,
        "distances_km": np.array([50.0]),
        "npts": 128,
        "dt": 0.5,
        "n_workers": 1,
    }
    res_np = compute_greens(model=model, backend="numpy", **kw)
    res_jx = compute_greens(model=model, backend="jax", **kw)
    rel = np.max(np.abs(res_np.gf - res_jx.gf)) / np.max(np.abs(res_np.gf))
    assert rel < 1e-3
    # Plan §1.5: JAX backend must store output as float32.
    assert res_jx.gf.dtype == np.float32
    # numpy backend must store as float64.
    assert res_np.gf.dtype == np.float64


@pytest.mark.gpu
@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_jax_propagator_compound_matrix_runs() -> None:
    """The JAX-numpy compound-matrix builder must produce a finite
    7×7 complex matrix that is JIT-compilable."""
    import jax

    from fkpy.jax_backend.propagator_jx import compound_matrix_jx

    jit_cm = jax.jit(compound_matrix_jx, static_argnames=())
    out = jit_cm(0.05, 1.0 + 0.01j, 1.5 + 0.02j, 5.0, 30.0)
    assert out.shape == (7, 7)
    # JAX default is complex64; user can opt into complex128 via
    # JAX_ENABLE_X64=1.
    assert out.dtype in (np.complex64, np.complex128)
    assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.gpu
@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_jax_kernel_module_re_exports() -> None:
    """The plan-mandated displacement_kernel_jx symbol exists and is
    callable (currently delegates to the Numba kernel)."""
    from fkpy.jax_backend.kernel_jx import displacement_kernel_jx

    assert callable(displacement_kernel_jx)


@pytest.mark.gpu
@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_jax_batched_source_depths() -> None:
    """Batched ``compute_greens_for_depths`` returns a stack of GFs that
    matches the per-depth numpy result to float32 precision."""
    # The public-API lazy proxy resolves to the same function object
    # as the submodule import.
    from fkpy import compute_greens_for_depths as compute_greens_for_depths_public
    from fkpy.jax_backend.greens_jx import compute_greens_for_depths
    assert compute_greens_for_depths_public is compute_greens_for_depths
    # And it shows the real signature, not (*args, **kwargs).
    import inspect
    sig = inspect.signature(compute_greens_for_depths_public)
    assert "src_depths_km" in sig.parameters
    assert "t0_s" in sig.parameters

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
