"""Bessel functions used in the discrete wavenumber summation.

We need ``J_0, J_1, J_2`` evaluated at ``z = k · x`` for every ``(k, x)``
in the integration grid.  Numba JIT cannot call ``scipy.special`` from
inside an ``@njit`` function, so we precompute the Bessel table on the
host and pass it into the JIT loop as a plain numpy array
(:func:`fkpy.integrate.wavenumber_sum_loop_tabulated`).

Per-frequency reuse: when ``pmin == 0`` (the default) every ω-loop
iteration starts at the same ``k₀ = ½·dk``, so a single global Bessel
table built for the worst-case k-array can be sliced for every
frequency, saving ~50 % of the wall-clock time on the canonical
benchmark.  See :func:`fkpy.frequency.run_omega_loop`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0 as _j0
from scipy.special import j1 as _j1
from scipy.special import jv as _jv

from ._typing import F64Array


def precompute_bessel(k_array: F64Array, distances_km: F64Array) -> F64Array:
    """Return ``(n_k, n_dist, 3)`` table of ``J_0, J_1, J_2`` at ``k·x``.

    Parameters
    ----------
    k_array
        1-D array of wavenumbers (1/km).
    distances_km
        1-D array of source-receiver distances (km).
    """
    z = np.outer(k_array, distances_km)
    out = np.empty((k_array.size, distances_km.size, 3), dtype=np.float64)
    out[..., 0] = _j0(z)
    out[..., 1] = _j1(z)
    out[..., 2] = _jv(2.0, z)
    return out


__all__ = ["precompute_bessel"]
