"""Unit tests for the inverse transform + sigma damping correction."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.transform import inverse_transform


@pytest.mark.fast
class TestInverseTransform:
    def test_zero_spectrum_returns_zeros(self) -> None:
        npts = 64
        nfreq = npts // 2 + 1
        sf = np.zeros((1, 1, nfreq), dtype=np.complex128)
        out = inverse_transform(sf, dt=0.1, sigma_rad_s=0.0,
                                t0_s=np.zeros(1), npts=npts)
        assert out.shape == (1, 1, npts)
        np.testing.assert_allclose(out, 0.0, atol=1e-15)

    def test_dc_spectrum_returns_constant(self) -> None:
        """A spectrum that is 1 at DC and 0 elsewhere is the IFFT of a
        single delta at t=0; with the irfft + dt scaling we use it
        produces an approximately constant trace whose value equals
        ``1 / (2 (nfreq - 1) dt)`` times nfft / dt."""
        npts = 64
        nfreq = npts // 2 + 1
        sf = np.zeros((1, 1, nfreq), dtype=np.complex128)
        sf[0, 0, 0] = 1.0
        out = inverse_transform(sf, dt=0.1, sigma_rad_s=0.0,
                                t0_s=np.zeros(1), npts=npts)
        assert out.shape == (1, 1, npts)
        # All samples should be equal (constant signal) and finite.
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out[0, 0, 0], out[0, 0, -1])

    def test_sigma_correction_reverses_damping(self) -> None:
        """If we feed a spectrum that corresponds to a real impulse
        response damped by exp(-sigma t), the inverse_transform should
        cancel that damping out (multiply by exp(sigma t))."""
        npts = 64
        nfreq = npts // 2 + 1
        sigma = 0.05
        # Build a known delta impulse (impulse response = 1 at sample 1)
        time_signal = np.zeros(npts)
        time_signal[1] = 1.0
        sf = np.fft.rfft(time_signal).reshape(1, 1, nfreq)
        # If the spectrum was already damped (which it isn't, here),
        # the inverse_transform with non-zero sigma would amplify.
        # Verify the amplification factor at sample i is exp(sigma*i*dt).
        out_with_sigma = inverse_transform(
            sf, dt=0.1, sigma_rad_s=sigma, t0_s=np.zeros(1), npts=npts
        )
        out_without = inverse_transform(
            sf, dt=0.1, sigma_rad_s=0.0, t0_s=np.zeros(1), npts=npts
        )
        # ratio = exp(sigma * t) per sample.
        t = np.arange(npts) * 0.1
        ratio = out_with_sigma[0, 0] / np.where(np.abs(out_without[0, 0]) > 1e-12,
                                                out_without[0, 0], 1.0)
        # Only check where the unscaled trace is nonzero.
        mask = np.abs(out_without[0, 0]) > 1e-6
        np.testing.assert_allclose(ratio[mask], np.exp(sigma * t)[mask], rtol=1e-6)

    def test_t0_offset_shifts_damping_origin(self) -> None:
        """Non-zero t0_s shifts the time axis used in the exp(sigma·t)
        correction by t0_s."""
        npts = 32
        nfreq = npts // 2 + 1
        sigma = 0.1
        sf = np.zeros((1, 1, nfreq), dtype=np.complex128)
        sf[0, 0, 0] = 1.0
        out_t0_zero = inverse_transform(sf, dt=0.1, sigma_rad_s=sigma,
                                        t0_s=np.zeros(1), npts=npts)
        out_t0_one = inverse_transform(sf, dt=0.1, sigma_rad_s=sigma,
                                       t0_s=np.array([1.0]), npts=npts)
        ratio = out_t0_one[0, 0] / out_t0_zero[0, 0]
        np.testing.assert_allclose(ratio, np.exp(sigma * 1.0), rtol=1e-12)
