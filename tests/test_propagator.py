"""Tests for the layer propagator (Numba kernel).

Cross-references:
- ZR-2002 eq. (17): explicit P-SV Haskell entries (this file).
- ZR-2002 eq. (33): bottom halfspace inverse-E vector (asserted indirectly
  via the regression test against Lupei Zhu's Fortran fk on the canonical
  5-layer model).
"""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.attenuation import complex_wavenumber_squared
from fkpy.constants import TWO_PI
from fkpy.propagator import _haskell_matrix, _layer_parameters, _sh_ch, kernel
from fkpy.source import SourceType, source_jump


@pytest.mark.fast
class TestKernelSmoke:
    """The kernel must run, return a (3, 3) complex array, and have the
    expected sparsity for an explosion source (n>0 should be zero)."""

    def test_explosion_only_n0(self, halfspace_model) -> None:
        omega = TWO_PI * 1.0
        sigma = 0.0
        w = complex(omega, -sigma)
        kp_sq = complex_wavenumber_squared(w, halfspace_model.vp_kms, halfspace_model.qp)
        ks_sq = complex_wavenumber_squared(w, halfspace_model.vs_kms, halfspace_model.qs)
        s = source_jump(SourceType.EXPLOSION, xi=float(halfspace_model.xi[0]),
                        mu=float(halfspace_model.mu_gpa[0]), flip=1).astype(np.complex128)
        u = kernel(
            k=0.05,
            kp_sq=kp_sq,
            ks_sq=ks_sq,
            mu=halfspace_model.mu_gpa,
            thickness_km=halfspace_model.thickness_km,
            s_input=s,
            src_layer=0,
            rcv_layer=0,
            src_type=int(SourceType.EXPLOSION),
            updn=0,
        )
        assert u.shape == (3, 3)
        # Explosion source has no n=1, n=2 contribution.
        assert np.allclose(u[1], 0.0, atol=1e-15)
        assert np.allclose(u[2], 0.0, atol=1e-15)

    def test_haskell_matrix_matches_zr_eq17(self) -> None:
        """K-K compound-matrix path agrees with the *direct* Haskell
        layer matrix (ZR-2002 eq. 17) to 1e-10 for moderate kd.

        The Numba-jitted _haskell_matrix returns entries scaled by
        ``exa·exb`` and ``exb`` (P-SV and SH respectively) so that
        thick-layer high-frequency overflows are avoided.  Multiplying
        out the scaling, we recover the literal eq. (17) entries
        ``H_ij = ν · (cosh(ν_α kd) − ξ cosh(ν_β kd))`` etc.
        """
        # Hand-picked layer with moderate kd so cosh/sinh are O(1).
        k = 0.3  # 1/km
        d = 5.0  # km
        vp = 6.0
        vs = 3.4
        rho = 2.7
        mu = rho * vs**2  # GPa
        omega = TWO_PI * 2.0
        kp_sq = complex_wavenumber_squared(complex(omega, 0.0),
                                           np.array([vp]), np.array([1e8]))[0]
        ks_sq = complex_wavenumber_squared(complex(omega, 0.0),
                                           np.array([vs]), np.array([1e8]))[0]
        r, ra, rb, r1, mu2, kd, _ = _layer_parameters(k, kp_sq, ks_sq, d, mu)
        Ca, Ya, Xa, exa = _sh_ch(ra, kd)
        Cb, Yb, Xb, exb = _sh_ch(rb, kd)

        a = _haskell_matrix(Ca, Ya, Xa, Cb, Yb, Xb, exa, exb, r, r1, mu2)

        # ----- analytic direct evaluation per ZR-2002 eq. (17) -----
        # (no rescaling).  ν·kd = ra·kd or rb·kd in our notation.
        nukda = ra * kd
        nukdb = rb * kd
        Ca_an = np.cosh(nukda)
        Sa_an = np.sinh(nukda)
        Cb_an = np.cosh(nukdb)
        Sb_an = np.sinh(nukdb)

        # Using sinh(x)/x = Y, sinh(x)*x = X with x = ν·kd, but Lupei's
        # convention uses Y = sinh(nukd)/ν and X = ν·sinh(nukd):
        Ya_an = Sa_an / ra
        Xa_an = Sa_an * ra
        Yb_an = Sb_an / rb
        Xb_an = Sb_an * rb

        # ZR-2002 eq. (17) — same algebraic form as Lupei's haskellMatrix
        # but with cosh/sinh in place of the rescaled (Ca·exb, Ya·exb,
        # Cb·exa, Yb·exa, ...) primitives.
        a_an = np.zeros((4, 4), dtype=np.complex128)
        a_an[0, 0] = r * (Ca_an - r1 * Cb_an)
        a_an[0, 1] = r * (r1 * Ya_an - Xb_an)
        a_an[0, 2] = (Cb_an - Ca_an) * r / mu2
        a_an[0, 3] = (Xb_an - Ya_an) * r / mu2

        a_an[1, 0] = r * (r1 * Yb_an - Xa_an)
        a_an[1, 1] = r * (Cb_an - r1 * Ca_an)
        a_an[1, 2] = (Xa_an - Yb_an) * r / mu2
        a_an[1, 3] = -a_an[0, 2]

        a_an[2, 0] = mu2 * r * r1 * (Ca_an - Cb_an)
        a_an[2, 1] = mu2 * r * (r1 * r1 * Ya_an - Xb_an)
        a_an[2, 2] = a_an[1, 1]
        a_an[2, 3] = -a_an[0, 1]

        a_an[3, 0] = mu2 * r * (r1 * r1 * Yb_an - Xa_an)
        a_an[3, 1] = -a_an[2, 0]
        a_an[3, 2] = -a_an[1, 0]
        a_an[3, 3] = a_an[0, 0]

        # The Numba result has every entry already multiplied by
        # exa·exb (since both Ca·exb and Cb·exa carry one factor each).
        # Equivalently the analytic answer is recovered by
        # _haskell_matrix(...) / (exa*exb).
        scale = exa * exb
        a_unscaled = a[:4, :4] / scale

        np.testing.assert_allclose(a_unscaled, a_an, rtol=1e-10, atol=1e-12)

    def test_double_couple_returns_finite_values(self, two_layer_model) -> None:
        omega = TWO_PI * 1.0
        w = complex(omega, -0.1)
        kp_sq = complex_wavenumber_squared(w, two_layer_model.vp_kms, two_layer_model.qp)
        ks_sq = complex_wavenumber_squared(w, two_layer_model.vs_kms, two_layer_model.qs)
        s = source_jump(SourceType.DOUBLE_COUPLE,
                        xi=float(two_layer_model.xi[1]),
                        mu=float(two_layer_model.mu_gpa[1]),
                        flip=1).astype(np.complex128)
        u = kernel(
            k=0.04,
            kp_sq=kp_sq,
            ks_sq=ks_sq,
            mu=two_layer_model.mu_gpa,
            thickness_km=two_layer_model.thickness_km,
            s_input=s,
            src_layer=1,
            rcv_layer=0,
            src_type=int(SourceType.DOUBLE_COUPLE),
            updn=0,
        )
        assert np.all(np.isfinite(u.real))
        assert np.all(np.isfinite(u.imag))
        # At least one element should be substantially non-zero.
        assert np.max(np.abs(u)) > 1e-8
