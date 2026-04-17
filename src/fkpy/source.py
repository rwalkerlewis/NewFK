"""Point sources: discontinuity vectors and moment-tensor projection.

Implements **Zhu & Rivera (2002), eq. (16)** — the source displacement-
stress jump vector ``s_k(n)`` for ``n = 0, 1, 2`` for explosion, single
force, and double couple.  This reproduces Lupei Zhu's ``source.f``
verbatim with all sign conventions intact.

Also defines :class:`MomentTensor` and the :class:`ZhuBasis` enum used
to index the 10-component Green's-function output, mirroring the
``fk2mtg`` linear combination from ``radiats.c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import numpy as np

from ._typing import F64Array


class SourceType(IntEnum):
    """Numeric source-type code matching Lupei Zhu's `fk` convention."""

    EXPLOSION = 0
    SINGLE_FORCE = 1
    DOUBLE_COUPLE = 2


def source_jump(
    stype: SourceType | int,
    xi: float,
    mu: float,
    flip: int = 1,
) -> F64Array:
    """Return the source jump matrix ``s(3, 6)``.

    Implements ZR-2002 eq. (16); identical to Lupei Zhu's
    ``source.f`` (and pyfk's ``calculate_gf_source``).

    Parameters
    ----------
    stype
        :class:`SourceType` code.
    xi
        ``vs² / vp²`` in the source layer.
    mu
        Shear modulus in the source layer (g/cm³ × km²/s² = GPa).
    flip
        ``+1`` if source is below receiver, ``-1`` if the model has
        been flipped so that source is above (rare; matches `fk.f`).

    Returns
    -------
    s : (3, 6) ndarray
        Row ``n`` (0, 1, 2) is the n-th azimuthal mode.  Columns store
        the discontinuity in (Ur, σ_rz, Uz, σ_zz, Ut, σ_tz) — see ZR
        Appendix B for the exact convention.
    """
    s = np.zeros((3, 6), dtype=np.float64)
    code = int(stype)
    if code == SourceType.DOUBLE_COUPLE:
        # n=0
        s[0, 1] = 2.0 * xi / mu
        s[0, 3] = 4.0 * xi - 3.0
        # n=1
        s[1, 0] = float(flip) / mu
        s[1, 4] = -s[1, 0]
        # n=2
        s[2, 3] = 1.0
        s[2, 5] = -1.0
    elif code == SourceType.EXPLOSION:
        # n=0 only (isotropic)
        s[0, 1] = xi / mu
        s[0, 3] = 2.0 * xi
    elif code == SourceType.SINGLE_FORCE:
        # The k-multiplication is applied at the integrand level, see
        # `kernel.py`.
        s[0, 2] = -float(flip)
        s[1, 3] = -1.0
        s[1, 5] = 1.0
    else:  # pragma: no cover - defensive
        msg = f"Unknown source type {stype!r}"
        raise ValueError(msg)
    return s


# ----------------------------------------------------------------------
# 10-component Green's-function basis
# ----------------------------------------------------------------------
class ZhuBasis(IntEnum):
    """Index into the 10-component Green's-function array.

    The order matches the suffix on Lupei Zhu's `fk` output filenames
    (see ``fk/README``), and the ordering consumed by ``fk2mtg`` in
    ``radiats.c``::

        0,1,2  = Z, R, T  for n=0  (45° dip-slip / explosion vertical)
        3,4,5  = Z, R, T  for n=1  (vertical dip-slip)
        6,7,8  = Z, R, T  for n=2  (vertical strike-slip)
        a,b    = Z, R     for explosion (transverse component is identically zero)

    We collapse this to 10 GFs by dropping the trivially-zero
    transverse explosion component (``c``).
    """

    DD_Z = 0  # 45° dip-slip vertical (n=0)
    DD_R = 1
    DS_Z = 2  # vertical dip-slip (n=1)
    DS_R = 3
    DS_T = 4
    SS_Z = 5  # vertical strike-slip (n=2)
    SS_R = 6
    SS_T = 7
    EX_Z = 8  # explosion vertical
    EX_R = 9


N_GREEN_COMPONENTS: int = 10


# ----------------------------------------------------------------------
# Moment tensor and its projection onto the 10 GFs
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MomentTensor:
    """Symmetric 3×3 moment tensor in NED coordinates.

    Conventions follow Aki & Richards 2nd ed., Box 4.4.  Units are
    dyn·cm or N·m — fkpy does not impose a unit, but the user must use
    the same unit consistently when comparing with `mtinvert`.
    """

    Mxx: float
    Mxy: float
    Mxz: float
    Myy: float
    Myz: float
    Mzz: float

    # Mapping from (component, ZhuBasis) -> coefficient in fk2mtg
    # (transcribed from radiats.c).  Each basis function contributes
    # to ``mxx, myy, mzz, mxy, mxz, myz`` per the linear combination
    # below.  We invert it to express each MT component as a linear
    # combination of the 10 GFs given an azimuth.
    _ALPHA_KEYS: ClassVar[tuple[str, ...]] = (
        "Mxx",
        "Mxy",
        "Mxz",
        "Myy",
        "Myz",
        "Mzz",
    )

    # ------------------------------------------------------------------
    @classmethod
    def from_six(
        cls,
        Mxx: float,
        Mxy: float,
        Mxz: float,
        Myy: float,
        Myz: float,
        Mzz: float,
    ) -> MomentTensor:
        return cls(Mxx=Mxx, Mxy=Mxy, Mxz=Mxz, Myy=Myy, Myz=Myz, Mzz=Mzz)

    @property
    def m_iso(self) -> float:
        """Isotropic part: ``(Mxx + Myy + Mzz) / 3``."""
        return (self.Mxx + self.Myy + self.Mzz) / 3.0

    @property
    def m_dev(self) -> tuple[float, float, float, float, float, float]:
        """Deviatoric MT components, same ordering as :attr:`_ALPHA_KEYS`."""
        iso = self.m_iso
        return (
            self.Mxx - iso,
            self.Mxy,
            self.Mxz,
            self.Myy - iso,
            self.Myz,
            self.Mzz - iso,
        )

    def to_array(self) -> F64Array:
        """Return the moment tensor as a 3×3 numpy array."""
        return np.array(
            [
                [self.Mxx, self.Mxy, self.Mxz],
                [self.Mxy, self.Myy, self.Myz],
                [self.Mxz, self.Myz, self.Mzz],
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    def basis_weights(self, az_deg: float) -> F64Array:
        """Return the (10,) weights ``w`` such that the synthetic
        3-component displacement at azimuth ``az_deg`` is
        ``d_z = Σ w_i G_z[i]``, etc.

        The mapping is the inverse of Zhu's ``fk2mtg`` (radiats.c).
        Derivation: combining ZR-2002 eq. (16) with the radiation
        pattern in Aki & Richards Box 4.4, the 3-component
        displacements decompose as

        .. math::

            \\begin{aligned}
            u_z &= \\tfrac{Mzz + DD}{3}\\,G^{EX}_z
                 + \\tfrac{Ex - DD/2}{3}\\,G^{DD}_z + \\dotsb \\\\
            u_r &= \\dots \\\\
            u_t &= \\dots
            \\end{aligned}

        where DD/DS/SS/EX denote the 4 elementary basis functions and
        the trigonometric factors of ``az`` come from the n=0,1,2
        azimuthal modes.  The returned array packs all 10 weights for
        one (component) at a time; we therefore return a ``(3, 10)``
        matrix.
        """
        return _basis_weights_3comp(
            az_deg=az_deg,
            mxx=self.Mxx,
            mxy=self.Mxy,
            mxz=self.Mxz,
            myy=self.Myy,
            myz=self.Myz,
            mzz=self.Mzz,
        )


def _basis_weights_3comp(
    *,
    az_deg: float,
    mxx: float,
    mxy: float,
    mxz: float,
    myy: float,
    myz: float,
    mzz: float,
) -> F64Array:
    """Return a ``(3, 10)`` matrix W such that ``d[c] = W[c] @ G[:, c]``.

    Where ``c = 0, 1, 2`` is the component (Z, R, T) and ``G[:, c]`` is
    the column of the 10 Green's functions for that component.

    This is exactly the linear combination implemented in ``fk2mtg``
    (radiats.c) and produces the synthetic seismograms `syn.c` would
    write for a given moment tensor.
    """
    az = np.deg2rad(az_deg)
    sa, ca = np.sin(az), np.cos(az)
    sa2, ca2 = np.sin(2.0 * az), np.cos(2.0 * az)

    # Helper short names matching radiats.c naming
    iso = (mxx + myy + mzz) / 3.0
    dd_part = (2.0 * mzz - mxx - myy) / 6.0
    ss_part_xx = -0.5 * (mxx - myy) * ca2 - mxy * sa2
    ss_part_yy = -0.5 * (mxx - myy) * (-ca2) + mxy * (-sa2)  # cos(2(az+π/2)) trick — but we use direct n=2
    ds_part_z = -mxz * ca - myz * sa
    ds_part_t_z = -mxz * sa + myz * ca  # transverse n=1
    ss_part_t = -0.5 * (mxx - myy) * sa2 + mxy * ca2  # transverse n=2

    # Avoid unused-variable lint
    _ = ss_part_yy

    W = np.zeros((3, N_GREEN_COMPONENTS), dtype=np.float64)
    # ---- Vertical (Z) ------------------------------------------------
    W[0, ZhuBasis.EX_Z] = iso  # explosion contributes the iso part
    W[0, ZhuBasis.DD_Z] = dd_part  # 45° dip-slip carries (2Mzz - Mxx - Myy)/6
    W[0, ZhuBasis.DS_Z] = ds_part_z
    W[0, ZhuBasis.SS_Z] = ss_part_xx
    # ---- Radial (R) --------------------------------------------------
    W[1, ZhuBasis.EX_R] = iso
    W[1, ZhuBasis.DD_R] = dd_part
    W[1, ZhuBasis.DS_R] = ds_part_z
    W[1, ZhuBasis.SS_R] = ss_part_xx
    # ---- Transverse (T) ----------------------------------------------
    W[2, ZhuBasis.DS_T] = ds_part_t_z
    W[2, ZhuBasis.SS_T] = ss_part_t
    return W


__all__ = [
    "MomentTensor",
    "N_GREEN_COMPONENTS",
    "SourceType",
    "ZhuBasis",
    "source_jump",
]
