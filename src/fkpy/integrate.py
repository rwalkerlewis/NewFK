"""Discrete wavenumber summation — Bouchon (1981, BSSA 71), eq. (9).

The displacement at receiver distance ``x`` and frequency ``ω`` is

.. math::

   u_m(x, \\omega) = \\frac{\\Delta k}{2\\pi}
                     \\sum_n k_n J_m(k_n x) U_m(k_n, \\omega),
                     \\quad m = 0, 1, 2

where ``U_m(k, ω)`` is the layer kernel returned by
:func:`fkpy.propagator.kernel` and ``J_m`` are Bessel functions of the
first kind.  The wavenumbers ``k_n = ω·p_min + (n + ½)·Δk`` cover
``[ω·p_min, sqrt(k_max² + (ω·p_max)²)]``.

The Bessel argument ``z = k·x`` requires evaluation of ``J_0, J_1, J_2``.
We *cannot* call :mod:`scipy.special` from inside a Numba ``@njit``
function, so we precompute ``J_m(k_n · x_j)`` on the host and pass the
table to :func:`wavenumber_sum_loop`.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from ._typing import C128Array, F64Array
from .bessel import precompute_bessel
from .propagator import kernel


@nb.njit(cache=True)
def wavenumber_sum_loop(  # noqa: PLR0913, PLR0915
    k_array: F64Array,
    bessel_table: F64Array,
    distances_km: F64Array,
    kp_sq: C128Array,
    ks_sq: C128Array,
    mu: F64Array,
    thickness_km: F64Array,
    s_input: C128Array,
    src_layer: int,
    rcv_layer: int,
    src_type: int,
    updn: int,
    flip: int,
) -> C128Array:
    """Inner loop over wavenumbers for one frequency.

    Returns ``out[i_dist, i_comp]`` for ``i_comp = 0..8`` corresponding
    to ``(Z, R, T)`` for ``n = 0, 1, 2``.  Multiplied by ``Δk / (2π)``
    *outside* this routine, where ``Δk`` is known.
    """
    n_k = k_array.size
    n_dist = distances_km.size
    out = np.zeros((n_dist, 9), dtype=np.complex128)
    for ik in range(n_k):
        k = k_array[ik]
        u = kernel(
            k,
            kp_sq,
            ks_sq,
            mu,
            thickness_km,
            s_input,
            src_layer,
            rcv_layer,
            src_type,
            updn,
        )
        for irec in range(n_dist):
            aj0 = bessel_table[ik, irec, 0]
            aj1 = bessel_table[ik, irec, 1]
            aj2 = bessel_table[ik, irec, 2]
            z = k * distances_km[irec]
            # n = 0  (Z, R, T)
            out[irec, 0] += u[0, 0] * aj0 * flip
            out[irec, 1] += -u[0, 1] * aj1
            out[irec, 2] += -u[0, 2] * aj1
            # n = 1
            nf1 = (u[1, 1] + u[1, 2]) * aj1 / z
            out[irec, 3] += u[1, 0] * aj1 * flip
            out[irec, 4] += u[1, 1] * aj0 - nf1
            out[irec, 5] += u[1, 2] * aj0 - nf1
            # n = 2
            nf2 = 2.0 * (u[2, 1] + u[2, 2]) * aj2 / z
            out[irec, 6] += u[2, 0] * aj2 * flip
            out[irec, 7] += u[2, 1] * aj1 - nf2
            out[irec, 8] += u[2, 2] * aj1 - nf2
    return out


@nb.njit(cache=True)
def wavenumber_sum_loop_tabulated(  # noqa: PLR0913
    k_array: F64Array,
    bessel_table: F64Array,
    distances_km: F64Array,
    kp_sq: C128Array,
    ks_sq: C128Array,
    mu: F64Array,
    thickness_km: F64Array,
    s_input: C128Array,
    src_layer: int,
    rcv_layer: int,
    src_type: int,
    updn: int,
    flip: int,
) -> C128Array:
    """Same as :func:`wavenumber_sum_loop` but with a pre-sliced
    Bessel table indexed by ``(ik, idist, m)`` for ``m = 0, 1, 2``.
    """
    n_k = k_array.size
    n_dist = distances_km.size
    out = np.zeros((n_dist, 9), dtype=np.complex128)
    for ik in range(n_k):
        k = k_array[ik]
        u = kernel(
            k, kp_sq, ks_sq, mu, thickness_km, s_input,
            src_layer, rcv_layer, src_type, updn,
        )
        for irec in range(n_dist):
            aj0 = bessel_table[ik, irec, 0]
            aj1 = bessel_table[ik, irec, 1]
            aj2 = bessel_table[ik, irec, 2]
            z = k * distances_km[irec]
            out[irec, 0] += u[0, 0] * aj0 * flip
            out[irec, 1] += -u[0, 1] * aj1
            out[irec, 2] += -u[0, 2] * aj1
            nf1 = (u[1, 1] + u[1, 2]) * aj1 / z
            out[irec, 3] += u[1, 0] * aj1 * flip
            out[irec, 4] += u[1, 1] * aj0 - nf1
            out[irec, 5] += u[1, 2] * aj0 - nf1
            nf2 = 2.0 * (u[2, 1] + u[2, 2]) * aj2 / z
            out[irec, 6] += u[2, 0] * aj2 * flip
            out[irec, 7] += u[2, 1] * aj1 - nf2
            out[irec, 8] += u[2, 2] * aj1 - nf2
    return out


__all__ = [
    "precompute_bessel",
    "wavenumber_sum_loop",
    "wavenumber_sum_loop_tabulated",
]
