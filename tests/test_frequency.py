"""Tests for the omega-loop dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.attenuation import complex_wavenumber_squared
from fkpy.constants import TWO_PI
from fkpy.frequency import FrequencyJobConfig, run_omega_loop
from fkpy.source import SourceType, source_jump


def _build_cfg(
    model_arr: np.ndarray,
    distances: np.ndarray,
    npts: int,
    dt: float,
    src_layer: int,
    rcv_layer: int,
) -> FrequencyJobConfig:
    vp = model_arr[:, 1]
    vs = model_arr[:, 2]
    rho = model_arr[:, 3]
    qp = model_arr[:, 4]
    qs = model_arr[:, 5]
    thickness = model_arr[:, 0].copy()
    thickness[-1] = 0.0
    mu = rho * vs * vs

    s_input = source_jump(
        SourceType.EXPLOSION,
        xi=(vs[src_layer] / vp[src_layer]) ** 2,
        mu=float(mu[src_layer]),
        flip=1,
    ).astype(np.complex128)
    nfft2 = npts // 2
    dw = TWO_PI / (npts * dt)
    sigma = 2.0 * dw / TWO_PI
    hs = max(thickness[rcv_layer:src_layer].sum(), 1e-3)
    xmax = max(distances.max(), hs)
    dk_per_km = 0.3 * np.pi / xmax
    kc_per_km = 15.0 / hs
    pmin = 0.0
    pmax = 1.0 / float(vs[src_layer])
    return FrequencyJobConfig(
        distances_km=distances.astype(np.float64),
        vp_kms=vp.astype(np.float64),
        vs_kms=vs.astype(np.float64),
        qp=qp.astype(np.float64),
        qs=qs.astype(np.float64),
        mu=mu.astype(np.float64),
        thickness_km=thickness.astype(np.float64),
        s_input=s_input,
        src_layer=src_layer,
        rcv_layer=rcv_layer,
        src_type=int(SourceType.EXPLOSION),
        updn=0,
        flip=1,
        sigma_rad_s=sigma,
        pmin_per_kms=pmin,
        pmax_per_kms=pmax,
        dk_per_km=dk_per_km,
        kc_per_km=kc_per_km,
        nfft2=nfft2,
        dw_rad_s=dw,
        wc1=1,
        wc2=1,
        wc=nfft2,
        taper=np.pi / nfft2,
        filter_const=dk_per_km / TWO_PI,
        t0_s=np.zeros(distances.size, dtype=np.float64),
    )


@pytest.mark.fast
def test_run_omega_loop_returns_expected_shape() -> None:
    arr = np.array(
        [
            [10.0, 6.0, 3.4, 2.7, 1000.0, 500.0],
            [0.0, 6.5, 3.7, 2.85, 1500.0, 800.0],
        ]
    )
    distances = np.array([50.0, 100.0])
    cfg = _build_cfg(arr, distances, npts=64, dt=0.5, src_layer=1, rcv_layer=0)
    out = run_omega_loop(cfg, n_workers=1)
    # Shape: (n_dist, 9, nfft2 + 1)
    assert out.shape == (2, 9, 33)
    assert np.all(np.isfinite(out.real))
    assert np.all(np.isfinite(out.imag))


@pytest.mark.fast
def test_run_omega_loop_serial_matches_parallel() -> None:
    """Serial and parallel paths must produce identical outputs."""
    arr = np.array(
        [
            [10.0, 6.0, 3.4, 2.7, 1000.0, 500.0],
            [0.0, 6.5, 3.7, 2.85, 1500.0, 800.0],
        ]
    )
    distances = np.array([50.0])
    cfg = _build_cfg(arr, distances, npts=128, dt=0.5, src_layer=1, rcv_layer=0)
    serial = run_omega_loop(cfg, n_workers=1)
    parallel = run_omega_loop(cfg, n_workers=2)
    np.testing.assert_allclose(serial, parallel, rtol=1e-12, atol=1e-15)


@pytest.mark.fast
def test_complex_frequency_omega_zero_returns_zero() -> None:
    """The omega = 0 frequency bin is skipped (DC) and remains 0."""
    arr = np.array(
        [
            [10.0, 6.0, 3.4, 2.7, 1000.0, 500.0],
            [0.0, 6.5, 3.7, 2.85, 1500.0, 800.0],
        ]
    )
    distances = np.array([50.0])
    cfg = _build_cfg(arr, distances, npts=64, dt=0.5, src_layer=1, rcv_layer=0)
    out = run_omega_loop(cfg, n_workers=1)
    np.testing.assert_array_equal(out[:, :, 0], 0.0)
    # Check kp_sq computation directly at a finite frequency works
    omega = 5.0 * cfg.dw_rad_s
    w = complex(omega, -cfg.sigma_rad_s)
    kp_sq = complex_wavenumber_squared(w, cfg.vp_kms, cfg.qp)
    assert np.all(np.isfinite(kp_sq.real))
    assert np.all(np.isfinite(kp_sq.imag))
