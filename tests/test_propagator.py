"""Tests for the layer propagator (Numba kernel).

Cross-references:
- ZR-2002 eq. (17): explicit P-SV Haskell entries.
- ZR-2002 eq. (33): bottom halfspace inverse-E vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.attenuation import complex_wavenumber_squared
from fkpy.constants import TWO_PI
from fkpy.propagator import kernel
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
