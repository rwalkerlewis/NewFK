"""Causal Futterman attenuation.

Implements the complex velocity used inside the F-K kernel:

.. math::

    \\tilde v(\\omega) = v \\,\\bigl(1 + \\frac{\\ln(\\omega / 2\\pi Q_\\mathrm{ref})}{\\pi Q}
                          + \\frac{i}{2 Q}\\bigr)

This is Aki & Richards (2002), eq. (5.93), with the reference frequency
``Q_REF_HZ`` (default 1 Hz, see :data:`fkpy.constants.Q_REF_HZ`).  In
Lupei Zhu's `fk.f` this appears compactly as

.. code-block:: fortran

    att = clog(w / pi2) / pi + cmplx(0., 0.5)
    ka(i) = w / (a(i) * (1. + att / qa(i)))

i.e. the same expression, where ``w / 2π`` is in Hz.  ``log`` is the
principal branch, so the attenuation is causal (Kramers–Kronig).
"""

from __future__ import annotations

import numpy as np

from ._typing import C128Array, F64Array
from .constants import Q_REF_HZ, TWO_PI


def futterman_attenuation_factor(omega_rad_s: complex | C128Array) -> complex | C128Array:
    """Return ``ln(ω / 2π Q_ref) / π + i/2``.

    Independent of layer; the per-layer division by ``Q`` happens in
    :func:`complex_wavenumber_squared`.
    """
    return np.log(omega_rad_s / (TWO_PI * Q_REF_HZ)) / np.pi + 0.5j


def complex_wavenumber_squared(
    omega_rad_s: complex,
    velocity_kms: F64Array,
    q: F64Array,
) -> C128Array:
    """Return ``(ω / ṽ)²`` for each layer.

    Parameters
    ----------
    omega_rad_s
        Complex angular frequency ``ω̃ = ω − iσ`` (Bouchon 1981).
    velocity_kms
        Per-layer real velocity (vp or vs) in km/s.
    q
        Per-layer Q (Qp or Qs).

    Returns
    -------
    k_squared
        ``(ω / ṽ)²`` array, complex128.
    """
    att = futterman_attenuation_factor(omega_rad_s)
    v_complex = velocity_kms * (1.0 + att / q)
    k = omega_rad_s / v_complex
    return np.asarray(k * k, dtype=np.complex128)


__all__ = ["complex_wavenumber_squared", "futterman_attenuation_factor"]
