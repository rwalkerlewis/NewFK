"""Unit tests for fkpy.attenuation."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.attenuation import (
    complex_wavenumber_squared,
    futterman_attenuation_factor,
)
from fkpy.constants import Q_REF_HZ, TWO_PI


@pytest.mark.fast
class TestFutterman:
    def test_factor_at_qref(self) -> None:
        """At ω = 2π·Q_REF_HZ, log(ω/2π Q_ref) = 0; so att = i/2."""
        omega = TWO_PI * Q_REF_HZ
        f = futterman_attenuation_factor(omega)
        assert np.isclose(f.real, 0.0, atol=1e-12)
        assert np.isclose(f.imag, 0.5, atol=1e-12)

    def test_q_inf_returns_input_velocity_phase(self) -> None:
        """Q -> infinity should return real wavenumber squared (= ω²/v²)."""
        omega = TWO_PI * 5.0
        v = np.array([5.0, 6.0])
        q = np.array([1e9, 1e9])
        ksq = complex_wavenumber_squared(omega, v, q)
        expected = (omega / v) ** 2
        assert np.allclose(ksq.real, expected, rtol=1e-6)
        assert np.allclose(ksq.imag, 0.0, atol=1e-3)

    def test_attenuation_increases_imaginary_part(self) -> None:
        """Lower Q means larger imaginary part of the wavenumber."""
        omega = TWO_PI * 5.0
        v = np.array([5.0])
        ksq_high = complex_wavenumber_squared(omega, v, np.array([10000.0]))
        ksq_low = complex_wavenumber_squared(omega, v, np.array([100.0]))
        assert abs(ksq_low.imag).item() > abs(ksq_high.imag).item()
