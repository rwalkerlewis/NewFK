"""Inverse rfft + Bouchon ``exp(σ·t)`` damping correction.

Per the plan:
- Use :func:`scipy.fft.irfft` with ``length = next_fast_len(npts)``
  for arbitrary trace lengths (not just powers of two).
- After IFFT, every sample is multiplied by ``exp(σ · t)`` *on the real
  output* — Lupei Zhu's `fk.f` revision history dated 2000-07-26 fixes
  a subtle bug where applying the correction to the complex spectrum
  before IFFT introduced long-period aliasing (Graves's correction).

Mathematical sketch
-------------------
With ``ω̃ = ω − iσ``, ``e^{iω̃t} = e^{iωt} · e^{σt}``, so the inverse
Fourier transform of ``U(ω̃)`` returns ``u(t) · e^{−σt}`` and we must
multiply by ``e^{+σt}`` to recover the physical trace.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len

from ._typing import C128Array, F64Array


def inverse_transform(
    sum_freq: C128Array,
    dt: float,
    sigma_rad_s: float,
    t0_s: F64Array,
    npts: int,
) -> F64Array:
    """Convert frequency-domain Green's functions to time domain.

    Parameters
    ----------
    sum_freq
        Shape ``(n_dist, n_comp, nfft2 + 1)`` — output of
        :func:`fkpy.frequency.run_omega_loop`.
    dt
        Time-domain sampling interval in seconds.
    sigma_rad_s
        ``σ`` in rad/s used for the complex frequency shift.
    t0_s
        Time origin per receiver in seconds (used to compensate the
        frequency-domain phase shift that already applied an ``exp(σt0)``
        factor inside :func:`fkpy.frequency.run_omega_loop`).
    npts
        Desired number of output samples per trace.

    Returns
    -------
    traces : (n_dist, n_comp, npts) float64
    """
    n_dist, n_comp, nfreq = sum_freq.shape
    nfft = max(2 * (nfreq - 1), npts)
    nfft = next_fast_len(nfft)
    nfft2 = nfft // 2 + 1
    padded = np.zeros((n_dist, n_comp, nfft2), dtype=np.complex128)
    padded[:, :, : sum_freq.shape[-1]] = sum_freq
    time = irfft(padded, n=nfft, axis=-1) * (nfft / (2.0 * dt * (nfreq - 1)))
    # The coefficient nfft / (2 dt (nfreq - 1)) reproduces the
    # normalisation of Lupei Zhu's `fftr(data, nfft/2, -dt)` which uses
    # the convention X(ω) = ∫ x(t) e^{-iωt} dt with sample step dt.

    t_axis = np.arange(npts, dtype=np.float64) * dt
    correction = np.exp(sigma_rad_s * (t0_s[:, None, None] + t_axis[None, None, :]))
    return np.asarray(time[..., :npts] * correction, dtype=np.float64)


__all__ = ["inverse_transform"]
