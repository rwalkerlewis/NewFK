"""Unit tests for fkpy.source — verifies port of Lupei Zhu's source.f."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.source import (
    N_GREEN_COMPONENTS,
    MomentTensor,
    SourceType,
    ZhuBasis,
    source_jump,
)


@pytest.mark.fast
class TestSourceJump:
    def test_explosion_jump(self) -> None:
        s = source_jump(SourceType.EXPLOSION, xi=0.3, mu=30.0, flip=1)
        assert s.shape == (3, 6)
        # Only n=0 (row 0) is nonzero
        assert np.allclose(s[1:], 0)
        # cols 1 and 3 only
        np.testing.assert_allclose(s[0, 1], 0.3 / 30.0)
        np.testing.assert_allclose(s[0, 3], 2.0 * 0.3)
        for c in (0, 2, 4, 5):
            assert s[0, c] == 0.0

    def test_double_couple_jump(self) -> None:
        s = source_jump(SourceType.DOUBLE_COUPLE, xi=0.25, mu=30.0, flip=1)
        # ZR-2002 eq. (16) DC jumps:
        np.testing.assert_allclose(s[0, 1], 2 * 0.25 / 30.0)
        np.testing.assert_allclose(s[0, 3], 4 * 0.25 - 3.0)
        np.testing.assert_allclose(s[1, 0], 1.0 / 30.0)
        np.testing.assert_allclose(s[1, 4], -1.0 / 30.0)
        np.testing.assert_allclose(s[2, 3], 1.0)
        np.testing.assert_allclose(s[2, 5], -1.0)

    def test_single_force_jump_flip(self) -> None:
        s_pos = source_jump(SourceType.SINGLE_FORCE, xi=0.3, mu=30.0, flip=1)
        s_neg = source_jump(SourceType.SINGLE_FORCE, xi=0.3, mu=30.0, flip=-1)
        # Only the n=0 term flips with the orientation
        np.testing.assert_allclose(s_pos[0, 2], -1.0)
        np.testing.assert_allclose(s_neg[0, 2], 1.0)
        np.testing.assert_allclose(s_pos[1], s_neg[1])


@pytest.mark.fast
class TestMomentTensor:
    def test_iso_dev_split(self) -> None:
        mt = MomentTensor(Mxx=1, Mxy=0, Mxz=0, Myy=2, Myz=0, Mzz=3)
        assert mt.m_iso == 2.0
        dev = mt.m_dev
        assert dev[0] == -1.0
        assert dev[3] == 0.0
        assert dev[5] == 1.0

    def test_basis_weights_iso_isotropic_only_in_ex(self) -> None:
        mt = MomentTensor(Mxx=1.0, Mxy=0.0, Mxz=0.0, Myy=1.0, Myz=0.0, Mzz=1.0)
        W = mt.basis_weights(az_deg=37.0)
        # An isotropic source contributes to EX_Z, EX_R only.
        assert W[0, ZhuBasis.EX_Z] == pytest.approx(1.0)
        assert W[1, ZhuBasis.EX_R] == pytest.approx(1.0)
        # No transverse component for isotropic source.
        assert np.allclose(W[2], 0.0)

    def test_basis_weights_shape(self) -> None:
        mt = MomentTensor(Mxx=1.0, Mxy=0.5, Mxz=-0.2, Myy=0.3, Myz=0.7, Mzz=-1.1)
        W = mt.basis_weights(az_deg=10.0)
        assert W.shape == (3, N_GREEN_COMPONENTS)

    def test_to_basis_weights_alias(self) -> None:
        mt = MomentTensor(Mxx=1.0, Mxy=0.5, Mxz=-0.2, Myy=0.3, Myz=0.7, Mzz=-1.1)
        np.testing.assert_array_equal(
            mt.basis_weights(az_deg=15.0),
            mt.to_basis_weights(az_deg=15.0),
        )
