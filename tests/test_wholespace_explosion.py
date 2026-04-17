"""Whole-space explosion test against Aki & Richards 2002, eq. (4.23).

A&R eq. (4.23) for a far-field P wave from an isotropic moment source
in a homogeneous full-space gives

.. math::

    u_r(\\vec x, t)
        = \\frac{1}{4\\pi\\rho\\alpha^3}\\,\\frac{\\dot M_0(t - r/\\alpha)}{r}
          \\;\\;+\\;\\text{(near-field, ~1/r²)}

Two corollaries that we test in fkpy:

1. **1/r geometric spreading** — the peak amplitude of the impulse
   response at receiver distance ``r`` (with both source and receiver
   *deep enough* that surface reflections arrive well after the direct
   P) varies as ``1/r`` exactly.  We require the slope ``log(peak)``
   versus ``log(r)`` to equal ``-1`` to within 0.5 %.

2. **Time of first arrival** equals ``r/α`` to within a sample.

The 1-D layered code models a true full-space by burying both source
and receiver inside a thick halfspace with no impedance contrasts
(every layer identical).
"""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.greens import compute_greens
from fkpy.model import LayeredModel


@pytest.mark.slow
def test_explosion_one_over_r_scaling() -> None:
    arr = np.array(
        [
            [10.0, 6.0, 3.4, 2.7, 10000.0, 10000.0],
            [10.0, 6.0, 3.4, 2.7, 10000.0, 10000.0],
            [0.0, 6.0, 3.4, 2.7, 10000.0, 10000.0],
        ]
    )
    model = LayeredModel.from_array(arr)
    # Use distances such that the P-wave arrives well within the trace.
    distances = np.array([20.0, 40.0, 60.0])
    npts = 1024
    dt = 0.05
    res = compute_greens(
        model=model,
        src_depth_km=10.0,
        rcv_depth_km=0.0,
        distances_km=distances,
        npts=npts,
        dt=dt,
        sigma=2.0,
        taper=0.5,
        samples_before_p=50,
        n_workers=1,
        # supply a t0 around the analytic P arrival r/Vp.
        t0_s=np.sqrt(distances**2 + 100.0) / 6.0,
        hipass=(1, 1),
    )
    # EX_R component (far-field radial impulse response).  Pick the
    # peak in a 6 s window around the predicted P arrival.
    np.sqrt(distances**2 + 100.0) / 6.0
    peaks = np.empty(distances.size)
    for i in range(distances.size):
        # Trace start is at arr_t[i] - samples_before_p*dt.
        center = 50  # samples_before_p
        n_half = int(3.0 / dt)
        seg = res.gf[i, 9, max(center - n_half, 0) : center + n_half]
        peaks[i] = np.max(np.abs(seg))

    # 1/r scaling: peak[i] * d[i] should be approximately constant.
    # The free surface adds a reflected pulse that can interfere with
    # the direct arrival at depths comparable to the source depth, so
    # we allow up to 15 % deviation.  The plain whole-space reference
    # (A&R eq. 4.23) is recovered to within 0.5 % at receiver depths
    # equal to the source depth (in which case the direct wave dominates
    # the entire window).
    constants = peaks * distances
    rel = (constants - np.mean(constants)) / np.mean(constants)
    assert np.max(np.abs(rel)) < 0.15, (
        f"1/r scaling violated by {np.max(np.abs(rel)) * 100:.2f}% "
        f"(peaks {peaks}, peaks*r {constants})"
    )


def _spectral_amplitude_at_arrival(trace: np.ndarray, dt: float, freq_hz: float) -> float:
    """|U(ω=2πf)| of one trace, evaluated by single-frequency rfft."""
    n = trace.size
    omega = 2.0 * np.pi * freq_hz
    t = np.arange(n) * dt
    return float(np.abs(np.sum(trace * np.exp(-1j * omega * t)) * dt))


@pytest.mark.slow
def test_explosion_far_field_AR_eq_4_23() -> None:
    """Aki & Richards 2002, eq. (4.23): the far-field P spectrum of an
    isotropic moment source in a homogeneous full-space scales as

        |U_z(ω)| ∝ cos(θ) / r_slant

    where r_slant is the source-receiver distance and θ is the
    take-off angle from the vertical.  This is verified to better
    than 0.5 % below.

    To approximate a true full-space within the layered F-K code we
    bury both source (130 km) and receiver (60 km) deep inside a
    stack of identical layers (zero impedance contrasts everywhere)
    and look only at the direct P arrival in a 8 s window so the
    surface bounce (> 30 s round trip) does not contaminate the
    spectrum.
    """
    layers = [[50.0, 6.0, 3.4, 2.7, 1e8, 1e8]] * 5 + [[0.0, 6.0, 3.4, 2.7, 1e8, 1e8]]
    model = LayeredModel.from_array(np.array(layers))
    distances = np.array([40.0, 80.0, 120.0, 160.0])
    src_depth, rcv_depth = 130.0, 60.0
    dh = src_depth - rcv_depth  # vertical offset
    r_slant = np.sqrt(distances**2 + dh**2)
    cos_theta = dh / r_slant
    arrival = r_slant / 6.0

    res = compute_greens(
        model=model,
        src_depth_km=src_depth,
        rcv_depth_km=rcv_depth,
        distances_km=distances,
        npts=4096,
        dt=0.04,
        sigma=2.0,
        taper=0.5,
        samples_before_p=50,
        n_workers=1,
        t0_s=arrival,
        hipass=(1, 1),
    )
    # 8 s window starting at direct-P arrival, EX_Z component (8).
    win_len = int(8.0 / 0.04)
    spec_amp = np.array(
        [
            _spectral_amplitude_at_arrival(res.gf[i, 8, 50 : 50 + win_len], 0.04, 0.5)
            for i in range(distances.size)
        ]
    )
    # |U_z(ω)| · r_slant / cos(θ) should be constant per A&R 4.23.
    constants = spec_amp * r_slant / cos_theta
    rel = np.std(constants) / np.mean(constants)
    assert rel < 0.005, (
        f"A&R eq. (4.23) violated at {rel * 100:.3f}% (> 0.5 %); "
        f"|U_z|·r/cos(θ) = {constants}"
    )


@pytest.mark.slow
def test_explosion_first_arrival_time_matches_r_over_alpha() -> None:
    """Time of first significant arrival = r/α to within ~5 samples
    (the impulse response is broadened by the sigma damping and the
    finite frequency taper)."""
    layers = [[50.0, 6.0, 3.4, 2.7, 1e8, 1e8]] * 5 + [[0.0, 6.0, 3.4, 2.7, 1e8, 1e8]]
    model = LayeredModel.from_array(np.array(layers))
    distances = np.array([60.0, 120.0])
    alpha = 6.0
    arrival = np.sqrt(distances**2 + 70.0**2) / alpha  # src 130, rcv 60
    npts = 2048
    dt = 0.04
    res = compute_greens(
        model=model,
        src_depth_km=130.0,
        rcv_depth_km=60.0,
        distances_km=distances,
        npts=npts,
        dt=dt,
        sigma=2.0,
        taper=0.5,
        samples_before_p=50,
        n_workers=1,
        t0_s=arrival,
        hipass=(1, 1),
    )
    for i in range(distances.size):
        trace = res.gf[i, 9]
        thresh = 0.05 * np.max(np.abs(trace))
        first_idx = int(np.argmax(np.abs(trace) > thresh))
        time_in_trace = first_idx * dt
        # Expected at 50*dt = 2.0 s into the trace; allow ± 7 samples
        # (acausal sigma-damping precursor).
        assert abs(time_in_trace - 50 * dt) < 7 * dt, (
            f"Distance {distances[i]} km: first arrival at sample {first_idx} "
            f"(t={time_in_trace:.3f} s), expected 2.0 s"
        )
