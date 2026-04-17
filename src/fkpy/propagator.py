"""Numerically-stable propagator for the layered F-K kernel.

We work in the reduced 5×5 compound-matrix formulation of Wang &
Herrmann (BSSA, 1980), which is mathematically equivalent to the
Kennett–Kerry reflection/transmission matrix recursion (Kennett 1983,
§6.3) — both schemes factor the growing ``exp(Re(ν)·d)`` exponentials
out of every layer matrix by construction.  The plain Haskell–Thomson
4×4 propagator does *not*: it contains ``cosh(ν·d)`` and ``sinh(ν·d)``
that overflow for thick layers at high frequency, which is why every
modern code adopts one of these stable formulations.

Justification (kept here in the source so it survives any refactor)::

    Why R/T (Kennett–Kerry) / compound matrix, not Haskell–Thomson:
    The Haskell propagator contains cosh(ν·d) and sinh(ν·d).  For
    thick layers at high frequency, ν·d acquires large positive real
    part and the matrix entries blow up by 10^several_hundred —
    overflow.  The Wang–Herrmann 6×6 compound matrix (used here) and
    Kennett–Kerry's R/T matrices both factor exp(-Re(ν)·d) out by
    construction; |R| ≤ 1 always and the recursion is unconditionally
    stable.  The two formulations are algebraically equivalent;
    we use the reduced 5×5 compound-matrix form because it ports
    cleanly from Lupei Zhu's `fk.f` reference, which has been
    validated for two decades.

The implementation here is a faithful Numba port of Lupei Zhu's
``kernel.f``, ``haskell.f``, and ``prop.f``, with cited cross-
references back to Zhu & Rivera (2002) (ZR) equations.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from ._typing import C128Array, F64Array
from .constants import EPSILON_THICKNESS_KM


# ----------------------------------------------------------------------
# Per-layer scratch quantities (helpers — pure functions for clarity)
# ----------------------------------------------------------------------
@nb.njit(cache=True)
def _sh_ch(  # noqa: PLR0913
    a: complex,
    kd: float,
) -> tuple[complex, complex, complex, float]:
    """Compute ``(Cosh(a kd), Sinh(a kd) / a, Sinh(a kd) * a, exp(-Re(a kd)))``.

    Identical to the ``sh_ch`` subroutine in Lupei Zhu's ``haskell.f``.
    The returned ``ex = exp(-Re(a kd))`` is used to scale every layer
    matrix entry by ``ex`` so that the resulting hyperbolic functions
    never overflow.
    """
    y = kd * a
    re = y.real
    im = y.imag
    ex = np.exp(-re)
    yhalf = 0.5 * complex(np.cos(im), np.sin(im))  # 0.5 e^{i Im(y)}
    xhalf = ex * ex * np.conj(yhalf)
    c = yhalf + xhalf  # cosh(y) * ex
    x_minus_y = yhalf - xhalf
    yy = x_minus_y / a  # = sinh(y)/a * ex
    xx = x_minus_y * a  # = sinh(y)*a * ex
    return c, yy, xx, ex


# ----------------------------------------------------------------------
# Compound matrix and Haskell matrix builders
# ----------------------------------------------------------------------
@nb.njit(cache=True)
def _layer_parameters(
    k: float,
    kp_sq: complex,
    ks_sq: complex,
    d_km: float,
    mu_layer: float,
) -> tuple[complex, complex, complex, complex, complex, float, float]:
    """Return ``(r, ra, rb, r1, mu2, kd, k2)`` for one layer.

    ``r = 2 / (k_s² / k²)``, ``ra = sqrt(1 - k_p²/k²)``,
    ``rb = sqrt(1 - k_s²/k²)``.  Matches ``layerParameter`` in
    Zhu's ``haskell.f``.
    """
    k2 = k * k
    kka = kp_sq / k2
    kkb = ks_sq / k2
    r = 2.0 / kkb
    kd = k * d_km
    mu2 = 2.0 * mu_layer
    ra = np.sqrt(1.0 - kka)
    rb = np.sqrt(1.0 - kkb)
    r1 = 1.0 - 1.0 / r
    return r, ra, rb, r1, mu2, kd, k2


@nb.njit(cache=True)
def _haskell_matrix(  # noqa: PLR0913, PLR0915
    Ca: complex,
    Ya: complex,
    Xa: complex,
    Cb: complex,
    Yb: complex,
    Xb: complex,
    exa: float,
    exb: float,
    r: complex,
    r1: complex,
    mu2: float,
) -> C128Array:
    """Return the 5×5 P-SV+SH layer matrix ``a``.

    Direct port of ``haskellMatrix`` in ``haskell.f``.  All entries are
    scaled by ``exa·exb`` (P-SV) or ``exb`` (SH) — see the docstring of
    :mod:`fkpy.propagator`.
    """
    a = np.zeros((5, 5), dtype=np.complex128)
    Ca = Ca * exb
    Xa = Xa * exb
    Ya = Ya * exb
    Cb = Cb * exa
    Yb = Yb * exa
    Xb = Xb * exa
    # P-SV
    a[0, 0] = r * (Ca - r1 * Cb)
    a[0, 1] = r * (r1 * Ya - Xb)
    a[0, 2] = (Cb - Ca) * r / mu2
    a[0, 3] = (Xb - Ya) * r / mu2

    a[1, 0] = r * (r1 * Yb - Xa)
    a[1, 1] = r * (Cb - r1 * Ca)
    a[1, 2] = (Xa - Yb) * r / mu2
    a[1, 3] = -a[0, 2]

    a[2, 0] = mu2 * r * r1 * (Ca - Cb)
    a[2, 1] = mu2 * r * (r1 * r1 * Ya - Xb)
    a[2, 2] = a[1, 1]
    a[2, 3] = -a[0, 1]

    a[3, 0] = mu2 * r * (r1 * r1 * Yb - Xa)
    a[3, 1] = -a[2, 0]
    a[3, 2] = -a[1, 0]
    a[3, 3] = a[0, 0]

    a[4, 4] = exb  # SH part: Haskell matrix replaced by exb (cf. haskell.f)
    return a


@nb.njit(cache=True)
def _compound_matrix(  # noqa: PLR0913, PLR0915
    Ca: complex,
    Ya: complex,
    Xa: complex,
    Cb: complex,
    Yb: complex,
    Xb: complex,
    exa: float,
    exb: float,
    r: complex,
    r1: complex,
    mu2: float,
) -> C128Array:
    """Return the 7×7 compound matrix ``c`` (5×5 P-SV ⊕ 2×2 SH).

    Direct port of ``compoundMatrix`` in ``haskell.f``.  Top-left 5×5
    is the W&H 1980 compound of the P-SV Haskell matrix, bottom-right
    2×2 is the SH Haskell matrix.  Scaled by ``exa·exb`` (P-SV) and
    ``exb`` (SH).
    """
    a = np.zeros((7, 7), dtype=np.complex128)
    CaCb = Ca * Cb
    CaYb = Ca * Yb
    CaXb = Ca * Xb
    XaCb = Xa * Cb
    XaXb = Xa * Xb
    YaCb = Ya * Cb
    YaYb = Ya * Yb
    ex = exa * exb
    r2 = r * r
    r3 = r1 * r1
    one = 1.0 + 0.0j
    two = 2.0 + 0.0j
    a[0, 0] = ((one + r3) * CaCb - XaXb - r3 * YaYb - two * r1 * ex) * r2
    a[0, 1] = (XaCb - CaYb) * r / mu2
    a[0, 2] = ((one + r1) * (CaCb - ex) - XaXb - r1 * YaYb) * r2 / mu2
    a[0, 3] = (YaCb - CaXb) * r / mu2
    a[0, 4] = (two * (CaCb - ex) - XaXb - YaYb) * r2 / (mu2 * mu2)

    a[1, 0] = (r3 * YaCb - CaXb) * r * mu2
    a[1, 1] = CaCb
    a[1, 2] = (r1 * YaCb - CaXb) * r
    a[1, 3] = -Ya * Xb
    a[1, 4] = a[0, 3]

    a[2, 0] = two * mu2 * r2 * (r1 * r3 * YaYb - (CaCb - ex) * (r3 + r1) + XaXb)
    a[2, 1] = two * r * (r1 * CaYb - XaCb)
    a[2, 2] = two * (CaCb - a[0, 0]) + ex
    a[2, 3] = -two * a[1, 2]
    a[2, 4] = -two * a[0, 2]

    a[3, 0] = mu2 * r * (XaCb - r3 * CaYb)
    a[3, 1] = -Xa * Yb
    a[3, 2] = -a[2, 1] / two
    a[3, 3] = a[1, 1]
    a[3, 4] = a[0, 1]

    a[4, 0] = mu2 * mu2 * r2 * (two * (CaCb - ex) * r3 - XaXb - r3 * r3 * YaYb)
    a[4, 1] = a[3, 0]
    a[4, 2] = -a[2, 0] / two
    a[4, 3] = a[1, 0]
    a[4, 4] = a[0, 0]

    # SH part, scaled by exb
    a[5, 5] = Cb
    a[5, 6] = -two * Yb / mu2
    a[6, 5] = -mu2 * Xb / two
    a[6, 6] = Cb
    return a


@nb.njit(cache=True)
def _e_vector(
    ra: complex,
    rb: complex,
    r1: complex,
    mu2: float,
) -> C128Array:
    """Top-halfspace boundary vector ``e``.

    First 5 entries are E|_{12}^{ij} for ij = 12, 13, 23, 24, 34;
    last two are the first column of the 2×2 SH E matrix.
    Direct port of ``eVector``.  Used only when the *top* layer is a
    halfspace (no free surface above).
    """
    e = np.zeros(7, dtype=np.complex128)
    e[0] = ra * rb - 1.0
    e[1] = mu2 * rb * (1.0 - r1)
    e[2] = mu2 * (r1 - ra * rb)
    e[3] = mu2 * ra * (r1 - 1.0)
    e[4] = mu2 * mu2 * (ra * rb - r1 * r1)
    e[5] = -1.0
    e[6] = mu2 * rb / 2.0
    return e


@nb.njit(cache=True)
def _initial_g(
    r: complex,
    ra: complex,
    rb: complex,
    r1: complex,
    mu2: float,
) -> C128Array:
    """Bottom-halfspace ``g`` vector — Zhu & Rivera 2002 eq. (33).

    First 5 entries are ``inverse(E)|_{ij}^{12}``; last two are the
    5th row of the SH ``E^{-1}``.  Equivalent to the radiation BC
    ``R_D^{halfspace} = 0`` in Kennett–Kerry form.
    """
    g = np.zeros(7, dtype=np.complex128)
    delta = r * (1.0 - ra * rb) - 1.0
    g[0] = mu2 * (delta - r1)
    g[1] = ra
    g[2] = delta
    g[3] = -rb
    g[4] = (1.0 + delta) / mu2
    g[5] = -1.0
    g[6] = 2.0 / (rb * mu2)
    return g


@nb.njit(cache=True)
def _propagate_g(g: C128Array, c: C128Array) -> C128Array:
    """Compute ``g_new = g · c`` (5×5 P-SV + 2×2 SH)."""
    out = np.zeros(7, dtype=np.complex128)
    for i in range(5):
        s = 0.0 + 0.0j
        for j in range(5):
            s += g[j] * c[j, i]
        out[i] = s
    out[5] = g[5] * c[5, 5] + g[6] * c[6, 5]
    out[6] = g[5] * c[5, 6] + g[6] * c[6, 6]
    return out


@nb.njit(cache=True)
def _initial_b() -> tuple[C128Array, C128Array, C128Array]:
    """Initialise (b, e, g) for the propagation:

    ``b`` = identity; ``e = (1,0,0,0,0,1,0)`` (top free surface);
    ``g = (0,0,0,0,1,0,1)`` (bottom free surface, will be overwritten
    by :func:`_initial_g` if the bottom layer is a half-space).
    """
    b = np.zeros((7, 7), dtype=np.complex128)
    e = np.zeros(7, dtype=np.complex128)
    g = np.zeros(7, dtype=np.complex128)
    for i in range(7):
        b[i, i] = 1.0
    e[0] = 1.0
    e[5] = 1.0
    g[4] = 1.0
    g[6] = 1.0
    return b, e, g


@nb.njit(cache=True)
def _propagate_b(b: C128Array, c: C128Array) -> C128Array:
    """Compute ``b_new = b · c`` (5×5 P-SV ⊕ 2×2 SH)."""
    out = np.zeros((7, 7), dtype=np.complex128)
    for i in range(5):
        for j in range(5):
            s = 0.0 + 0.0j
            for ll in range(5):
                s += b[i, ll] * c[ll, j]
            out[i, j] = s
    for i in range(5, 7):
        for j in range(5, 7):
            out[i, j] = b[i, 5] * c[5, j] + b[i, 6] * c[6, j]
    # Preserve identity on rows we don't touch (not used downstream but
    # keeps the matrix sane for debugging).
    return out


@nb.njit(cache=True)
def _initial_z(s_jump: C128Array, g: C128Array) -> C128Array:
    """Build the source row vector ``z(3, 5) = s · X|_{ij}^{12}``.

    Direct port of ``initialZ`` in ``prop.f``; combines the source
    discontinuity vector ``s`` with the bottom-side ``g`` vector.
    """
    z = np.zeros((3, 5), dtype=np.complex128)
    for i in range(3):
        z[i, 0] = -s_jump[i, 1] * g[0] - s_jump[i, 2] * g[1] + s_jump[i, 3] * g[2]
        z[i, 1] = s_jump[i, 0] * g[0] - s_jump[i, 2] * g[2] - s_jump[i, 3] * g[3]
        z[i, 2] = s_jump[i, 0] * g[1] + s_jump[i, 1] * g[2] - s_jump[i, 3] * g[4]
        z[i, 3] = -s_jump[i, 0] * g[2] + s_jump[i, 1] * g[3] + s_jump[i, 2] * g[4]
        z[i, 4] = s_jump[i, 4] * g[5] + s_jump[i, 5] * g[6]
    return z


@nb.njit(cache=True)
def _propagate_z(z: C128Array, a: C128Array) -> C128Array:
    """Apply the Haskell layer matrix to the z vector: ``z_new = z · a``."""
    out = np.zeros((3, 5), dtype=np.complex128)
    for i in range(3):
        for j in range(4):
            s = 0.0 + 0.0j
            for ll in range(4):
                s += z[i, ll] * a[ll, j]
            out[i, j] = s
        out[i, 4] = z[i, 4] * a[4, 4]
    return out


@nb.njit(cache=True)
def _separat_s(  # noqa: PLR0913
    s_input: C128Array,
    updn: int,
    src_type: int,
    ra: complex,
    rb: complex,
    r: complex,
    r1: complex,
    mu2: float,
) -> C128Array:
    """Apply the up- or down-going wave selection (``updn = ±1``).

    ``updn = 0`` returns the source as-is (full waveform).  ``updn =
    +1`` keeps only down-going waves at the source, ``-1`` keeps only
    up-going.  Direct port of ``separatS`` in ``kernel.f``.
    """
    out = s_input.copy()
    if updn == 0:
        return out

    # Number of azimuthal modes to update (3 for double couple,
    # 2 for single force, 1 for explosion).
    if src_type == 2:
        ii = 3
    elif src_type == 1:
        ii = 2
    else:
        ii = 1

    ra1 = 1.0 / ra
    rb1 = 1.0 / rb
    dum = float(updn) * r
    temp = np.zeros((4, 4), dtype=np.complex128)
    temp[0, 0] = 1.0
    temp[0, 1] = dum * (rb - r1 * ra1)
    temp[0, 2] = 0.0
    temp[0, 3] = dum * (ra1 - rb) / mu2
    temp[1, 0] = dum * (ra - r1 * rb1)
    temp[1, 1] = 1.0
    temp[1, 2] = dum * (rb1 - ra) / mu2
    temp[1, 3] = 0.0
    temp[2, 0] = 0.0
    temp[2, 1] = dum * (rb - r1 * r1 * ra1) * mu2
    temp[2, 2] = 1.0
    temp[2, 3] = dum * (r1 * ra1 - rb)
    temp[3, 0] = dum * (ra - r1 * r1 * rb1) * mu2
    temp[3, 1] = 0.0
    temp[3, 2] = dum * (r1 * rb1 - ra)
    temp[3, 3] = 1.0
    temp_sh = (float(updn) * 2.0 / mu2) * rb1

    for i in range(ii):
        for j in range(4):
            v = 0.0 + 0.0j
            for jj in range(4):
                v += temp[j, jj] * s_input[i, jj]
            out[i, j] = v / 2.0
        out[i, 4] = (s_input[i, 4] + temp_sh * s_input[i, 5]) / 2.0
        out[i, 5] = (s_input[i, 5] + s_input[i, 4] / temp_sh) / 2.0
    return out


# ----------------------------------------------------------------------
# Top-level kernel: U(k, ω) for one wavenumber
# ----------------------------------------------------------------------
@nb.njit(cache=True)
def kernel(  # noqa: PLR0912, PLR0913, PLR0915
    k: float,
    kp_sq: C128Array,
    ks_sq: C128Array,
    mu: F64Array,
    thickness_km: F64Array,
    s_input: C128Array,
    src_layer: int,
    rcv_layer: int,
    src_type: int,
    updn: int,
) -> C128Array:
    """Compute the 3×3 displacement kernel ``u[n, comp]`` at one ``k``.

    Parameters
    ----------
    k
        Real wavenumber, ``1/km``.
    kp_sq, ks_sq
        Per-layer ``(ω/ṽ)²`` for P and S, complex128.
    mu
        Per-layer shear modulus.
    thickness_km
        Per-layer thicknesses; the half-space row is 0.
    s_input
        ``(3, 6)`` source jump (output of :func:`fkpy.source.source_jump`).
    src_layer, rcv_layer
        Layer indices in the (possibly flipped) model with the source
        and receiver placed at the top of those layers.  Convention:
        ``src_layer >= rcv_layer`` (caller flips the model otherwise).
    src_type
        ``0=ex, 1=sf, 2=dc``.
    updn
        ``0`` (whole), ``+1`` (down-going only), ``-1`` (up-going only).

    Returns
    -------
    u : (3, 3) complex128
        ``u[n, 0]`` = vertical, ``u[n, 1]`` = radial,
        ``u[n, 2]`` = transverse, for azimuthal mode ``n = 0, 1, 2``.

    Notes
    -----
    This is the displacement kernel of Zhu & Rivera (2002), eq. (28),
    realised through the W&H 1980 compound-matrix recursion which is
    algebraically equivalent to the Kennett–Kerry R/T scheme.
    """
    n_layers = thickness_km.size
    s_complex = s_input.astype(np.complex128)
    b, e, g = _initial_b()
    z = np.zeros((3, 5), dtype=np.complex128)

    # Single-pass: for ilayer = n_layers-1 down to 0, build c, propagate g,
    # apply source jump, propagate z (or b) depending on (src_layer, rcv_layer).
    for ilayer in range(n_layers - 1, -1, -1):
        r, ra, rb, r1, mu2, kd, _k2 = _layer_parameters(
            k, kp_sq[ilayer], ks_sq[ilayer], thickness_km[ilayer], mu[ilayer]
        )
        is_bottom_halfspace = (
            ilayer == n_layers - 1 and thickness_km[ilayer] < EPSILON_THICKNESS_KM
        )
        is_top_halfspace = ilayer == 0 and thickness_km[0] < EPSILON_THICKNESS_KM

        if is_bottom_halfspace:
            g = _initial_g(r, ra, rb, r1, mu2)
            c = np.zeros((7, 7), dtype=np.complex128)  # unused below
        elif is_top_halfspace:
            e = _e_vector(ra, rb, r1, mu2)
            break
        else:
            Ca, Ya, Xa, exa = _sh_ch(ra, kd)
            Cb, Yb, Xb, exb = _sh_ch(rb, kd)
            c = _compound_matrix(Ca, Ya, Xa, Cb, Yb, Xb, exa, exb, r, r1, mu2)
            g = _propagate_g(g, c)

        if ilayer == src_layer:
            ss = _separat_s(s_complex, updn, src_type, ra, rb, r, r1, mu2)
            z = _initial_z(ss, g)

        if ilayer < src_layer:
            if ilayer >= rcv_layer:
                Ca, Ya, Xa, exa = _sh_ch(ra, kd)
                Cb, Yb, Xb, exb = _sh_ch(rb, kd)
                a = _haskell_matrix(Ca, Ya, Xa, Cb, Yb, Xb, exa, exb, r, r1, mu2)
                z = _propagate_z(z, a)
            else:
                # Below the receiver; accumulate compound product into b.
                b = _propagate_b(b, c)

    # Top free-surface BC contributes a factor of 2 to e[2]
    e[2] = 2.0 * e[2]

    rayl = (
        g[0] * e[0]
        + g[1] * e[1]
        + g[2] * e[2]
        + g[3] * e[3]
        + g[4] * e[4]
    )
    love = g[5] * e[5] + g[6] * e[6]

    g_top = np.zeros(7, dtype=np.complex128)
    for i in range(4):
        s = 0.0 + 0.0j
        for j in range(5):
            s += b[i, j] * e[j]
        g_top[i] = s
    g_top[2] = g_top[2] / 2.0
    g_top[5] = b[5, 5] * e[5] + b[5, 6] * e[6]

    u = np.zeros((3, 3), dtype=np.complex128)
    # Transcribed from kernel.f:
    #     dum    = z(i,2)*g(1)+z(i,3)*g(2)-z(i,4)*g(3)
    #     z(i,2) =-z(i,1)*g(1)+z(i,3)*g(3)+z(i,4)*g(4)
    #     z(i,1) = dum
    #     z(i,5) = z(i,5)*g(6)
    # then u(i,1)=dum*z(i,2)/rayl, u(i,2)=dum*z(i,1)/rayl,
    #       u(i,3)=dum*z(i,5)/love     where dum becomes 1 (sf) or k.
    for i in range(3):
        z1_new = z[i, 1] * g_top[0] + z[i, 2] * g_top[1] - z[i, 3] * g_top[2]
        z2_new = -z[i, 0] * g_top[0] + z[i, 2] * g_top[2] + z[i, 3] * g_top[3]
        z5_new = z[i, 4] * g_top[5]

        scale = 1.0 + 0.0j if src_type == 1 else complex(k, 0.0)
        u[i, 0] = scale * z2_new / rayl  # vertical (Z)
        u[i, 1] = scale * z1_new / rayl  # radial (R)
        u[i, 2] = scale * z5_new / love  # transverse (T)
    return u


__all__ = ["kernel"]
