"""Unit tests for the precomputed Bessel table."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j0, j1, jv

from fkpy.bessel import precompute_bessel


@pytest.mark.fast
class TestPrecomputeBessel:
    def test_shape(self) -> None:
        k = np.linspace(0.1, 1.0, 7)
        x = np.array([10.0, 50.0])
        table = precompute_bessel(k, x)
        assert table.shape == (7, 2, 3)

    def test_values_match_scipy(self) -> None:
        k = np.array([0.1, 0.5, 1.0])
        x = np.array([20.0, 100.0])
        table = precompute_bessel(k, x)
        for i, ki in enumerate(k):
            for j, xj in enumerate(x):
                z = ki * xj
                np.testing.assert_allclose(table[i, j, 0], j0(z), rtol=1e-12)
                np.testing.assert_allclose(table[i, j, 1], j1(z), rtol=1e-12)
                np.testing.assert_allclose(table[i, j, 2], jv(2.0, z), rtol=1e-12)

    def test_zero_argument(self) -> None:
        """At z = 0, J_0 = 1 and J_1 = J_2 = 0."""
        table = precompute_bessel(np.array([0.0]), np.array([5.0]))
        np.testing.assert_allclose(table[0, 0, 0], 1.0)
        np.testing.assert_allclose(table[0, 0, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(table[0, 0, 2], 0.0, atol=1e-12)
