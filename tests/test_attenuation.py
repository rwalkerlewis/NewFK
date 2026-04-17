"""Unit tests for fkpy.attenuation."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.attenuation import (
    complex_velocity,
    complex_wavenumber_squared,
    futterman_attenuation_factor,
)
from fkpy.constants import Q_REF_HZ, TWO_PI  # noqa: F401  used in causality test


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

    def test_complex_velocity_matches_wavenumber_squared(self) -> None:
        """``(omega / complex_velocity)**2`` must equal complex_wavenumber_squared."""
        omega = TWO_PI * 3.0 - 0.05j
        v = np.array([5.0, 6.0])
        q = np.array([300.0, 800.0])
        v_c = complex_velocity(omega, v, q)
        k_from_v = (omega / v_c) ** 2
        k_direct = complex_wavenumber_squared(omega, v, q)
        np.testing.assert_allclose(k_from_v, k_direct, rtol=1e-12)

    def test_complex_velocity_real_at_qref(self) -> None:
        """At omega = 2 pi Q_REF_HZ the complex velocity has zero log-slope
        (real part equals v · (1 + 0/Q)) and imaginary part v/(2Q)."""
        omega = TWO_PI * Q_REF_HZ
        v = np.array([5.0])
        q = np.array([100.0])
        v_c = complex_velocity(omega, v, q)
        np.testing.assert_allclose(v_c.real, v, rtol=1e-12)
        np.testing.assert_allclose(v_c.imag, v / (2 * q), rtol=1e-12)

    def test_causality_kramers_kronig(self) -> None:
        """Futterman with Q_ref = 1 Hz is a *causal* attenuation
        operator: the impulse response in the time domain must be
        zero for ``t < 0`` (within numerical noise).

        We construct ``H(ω) = exp(i (ω/v_complex) x)`` for a fixed
        propagation distance, inverse-FFT it, and require that the
        signal energy be concentrated *after* ``t = x/v`` — i.e. the
        causal arrival — with negligible energy at strictly negative
        times (samples 0..N/4 in our IFFT, since the impulse arrives
        at sample N/2).
        """
        npts = 8192
        dt = 0.005
        f = np.fft.rfftfreq(npts, dt)
        omega = 2.0 * np.pi * f
        Q = 200.0
        v = 5.0
        # Pick x so the impulse lands near sample N/2 = 4096 (t = 20.48 s):
        x = v * 20.0  # ≈ 100 km; arrival at 20 s
        att = np.zeros_like(omega, dtype=np.complex128)
        att[1:] = np.log(omega[1:] / TWO_PI / Q_REF_HZ) / np.pi + 0.5j
        v_complex = v * (1.0 + att / Q)
        # apply a low-pass cosine taper to avoid the Nyquist ringing
        f_lp = 0.5 * f.max()
        taper = np.where(f < f_lp, 1.0, 0.5 * (1.0 + np.cos(np.pi * (f - f_lp) / f_lp)))
        taper = np.clip(taper, 0, 1)
        # Propagator
        k_complex = omega / v_complex
        spectrum = taper * np.exp(1j * k_complex * x)
        spectrum[0] = 0.0
        h = np.fft.irfft(spectrum, n=npts)
        # The arrival should be near sample npts*dt/x*v = ... actually
        # the IFFT places t = i*dt for i = 0..npts-1, so the arrival is
        # near sample x/v / dt = 4000.  Energy before sample 3500 should
        # be << energy from 3500..6000.
        early = float(np.sum(h[:3500] ** 2))
        late = float(np.sum(h[3500:6000] ** 2))
        assert late > 100.0 * early, (
            f"Futterman not causal: early energy {early:.3e}, "
            f"late energy {late:.3e}, ratio {late / max(early, 1e-30):.1f}"
        )
