"""Numerical constants for fkpy.

Every constant carries a comment with paper + equation/section number,
per the plan code-quality rule.
"""

from __future__ import annotations

import math

# ----------------------------------------------------------------------
# Wavenumber-integration parameters (Bouchon 1981, BSSA 71)
# ----------------------------------------------------------------------
SIGMA_DEFAULT: float = 2.0
"""Imaginary frequency shift in cycles per trace length.

Used as ``σ = SIGMA_DEFAULT · dω / (2π)``.  Bouchon 1981 §3 introduces
the complex frequency ``ω̃ = ω − iσ`` to avoid wrap-around when the
wavenumber sum is truncated to a finite domain.  Lupei Zhu's `fk.f`
default (`sample_input`) is 2."""

DK_DEFAULT: float = 0.3
"""Wavenumber sampling density, in units of ``π / max(x_max, h_s)``.

Bouchon 1981 condition (2) requires
``dk ≤ 0.5 / (1 + sqrt((v_max·t_max)² − h_s²) / x_max)``.
Zhu's `fk.f` README recommends 0.2–0.5; Ampuero (2015) showed 0.3 is a
good default."""

KMAX_DEFAULT: float = 15.0
"""Maximum wavenumber at ω=0, in units of ``1 / h_s``.

Kernels decay as ``exp(−k·h_s)`` at zero frequency, so 15 e-foldings
guarantees < 1e-7 contribution from k > k_max.  Zhu's `fk.f` README
requires ``kmax ≥ 10``."""

PMIN_DEFAULT: float = 0.0
"""Minimum slowness in units of ``1 / v_s_at_source``.  Zhu's `fk` default."""

PMAX_DEFAULT: float = 1.0
"""Maximum slowness in units of ``1 / v_s_at_source``.

Setting this to 1 caps integration at body-wave slowness; reduce for
teleseismic, increase to capture surface waves."""

# ----------------------------------------------------------------------
# Frequency-domain taper (cosine roll-off below f_Nyq)
# ----------------------------------------------------------------------
TAPER_DEFAULT: float = 0.3
"""Fraction of the spectrum between (1 − TAPER)·f_Nyq and f_Nyq over which a
cosine taper rolls off.  Matches Zhu's `fk.f` ``lp`` input."""

# ----------------------------------------------------------------------
# Time-domain padding before the first arrival
# ----------------------------------------------------------------------
SAMPLES_BEFORE_P_DEFAULT: int = 50
"""Number of samples to reserve before the P arrival.  Matches Zhu's
``nb`` default."""

# ----------------------------------------------------------------------
# Attenuation
# ----------------------------------------------------------------------
Q_REF_HZ: float = 1.0
"""Futterman reference frequency.  Phase velocities are real at this
frequency.  Aki & Richards 2002, Eq. (5.93)."""

# ----------------------------------------------------------------------
# Numerical tolerances
# ----------------------------------------------------------------------
EPSILON_THICKNESS_KM: float = 1.0e-6
"""Layer thickness below which a layer is treated as the bottom
half-space.  Matches Zhu's `fk.f` ``epsilon``."""

EPSILON_VS_KMS: float = 1.0e-6
"""Floor for v_s.  A pure liquid layer (e.g. ocean) is set to this so the
SH Haskell entries do not divide by zero; standard `fk` workaround."""

# ----------------------------------------------------------------------
# Mathematical convenience
# ----------------------------------------------------------------------
TWO_PI: float = 2.0 * math.pi

__all__ = [
    "DK_DEFAULT",
    "EPSILON_THICKNESS_KM",
    "EPSILON_VS_KMS",
    "KMAX_DEFAULT",
    "PMAX_DEFAULT",
    "PMIN_DEFAULT",
    "Q_REF_HZ",
    "SAMPLES_BEFORE_P_DEFAULT",
    "SIGMA_DEFAULT",
    "TAPER_DEFAULT",
    "TWO_PI",
]
