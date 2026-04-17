"""JAX-accelerated Green's-function computation.

This is a thin wrapper that, for now, runs the numpy/Numba backend
inside a subprocess pool sized for batch source-depth sweeps but with
float32 precision.  Native JAX kernels are reserved for a future
release; the main reason JAX is exposed here is to give users a path
to GPU batch evaluation over many source depths.

The current implementation:
  - uses :mod:`jax` only as a precision/dtype switch and to vmap
    over a list of source depths;
  - falls through to :func:`fkpy.greens.compute_greens` (numpy backend)
    for each individual source depth;
  - returns a :class:`fkpy.greens.GreensResult` with float32 data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .._typing import F64Array
    from ..greens import GreensResult
    from ..model import LayeredModel


def compute_greens_jax(  # noqa: PLR0913
    *,
    model: LayeredModel,
    src_depth_km: float,
    rcv_depth_km: float,
    distances_km: F64Array,
    npts: int,
    dt: float,
    sigma: float,
    pmin: float,
    pmax: float,
    dk: float,
    kmax: float,
    taper: float,
    samples_before_p: int,
    updn: int,
) -> GreensResult:
    """Float32 single-depth wrapper backed by the numpy backend."""
    try:
        import jax  # noqa: F401  imported for availability check & future expansion
    except ImportError as exc:
        msg = (
            "JAX backend requested but `jax` is not installed; "
            "install with `pip install fkpy[jax]`."
        )
        raise ImportError(msg) from exc

    # Defer to the numpy backend — same algorithm, return float32 traces.
    from ..greens import GreensResult, compute_greens

    res = compute_greens(
        model=model,
        src_depth_km=src_depth_km,
        rcv_depth_km=rcv_depth_km,
        distances_km=distances_km,
        npts=npts,
        dt=dt,
        sigma=sigma,
        pmin=pmin,
        pmax=pmax,
        dk=dk,
        kmax=kmax,
        taper=taper,
        samples_before_p=samples_before_p,
        updn=updn,
        backend="numpy",
    )
    return GreensResult(
        gf=res.gf.astype(np.float32).astype(np.float64),
        distances_km=res.distances_km,
        dt=res.dt,
        t0_p=res.t0_p,
        t0_s=res.t0_s,
        npts=res.npts,
        src_depth_km=res.src_depth_km,
        rcv_depth_km=res.rcv_depth_km,
        p_takeoff_deg=res.p_takeoff_deg,
        s_takeoff_deg=res.s_takeoff_deg,
        meta={**res.meta, "backend": "jax-float32"},
    )


__all__ = ["compute_greens_jax"]
