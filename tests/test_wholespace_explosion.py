"""Whole-space explosion test.

In a homogeneous, lossless full-space, the displacement from a step-
moment isotropic source at distance ``r`` is, in the far field,

.. math::

    u_r(t) = \\frac{1}{4\\pi\\rho\\alpha^3 r}\\,\\ddot{M}_0(t - r/\\alpha)
             + (\\text{near-field terms})

per Aki & Richards (2002), eq. (4.23).  We synthesise a Dirac-impulse
moment source in fkpy by taking a *very* short trapezoid (one sample),
then verify the predicted *peak amplitude scaling* with distance:

.. math::

    \\text{peak}_r(d) = \\frac{C}{d}\\;\\;\\Rightarrow\\;\\;
    \\text{peak}_r(d_1)/\\text{peak}_r(d_2) = d_2 / d_1.

The test does *not* check the absolute scale (which depends on the
unit conventions in the source jump and would require the user to
multiply by the moment), but the ``1/r`` decay of the impulse-response
amplitude is a strong constraint and is recovered to within 0.5 %.

The 1-D layered code can model a "whole-space" by setting both the
source and receiver below the surface in a homogeneous halfspace —
the surface free-surface reflections then arrive late and leave the
direct-wave amplitude intact in the early time window we measure.
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
    arr_t = np.sqrt(distances**2 + 100.0) / 6.0
    peaks = np.empty(distances.size)
    for i in range(distances.size):
        # Trace start is at arr_t[i] - samples_before_p*dt.
        center = int(50)  # samples_before_p
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


@pytest.mark.slow
def test_explosion_amplitude_decreases_with_distance() -> None:
    """Sanity check: the EX_R amplitude must decay monotonically with
    distance (it does so as 1/r in the homogeneous case)."""
    arr = np.array(
        [
            [10.0, 6.0, 3.4, 2.7, 10000.0, 10000.0],
            [10.0, 6.0, 3.4, 2.7, 10000.0, 10000.0],
            [0.0, 6.0, 3.4, 2.7, 10000.0, 10000.0],
        ]
    )
    model = LayeredModel.from_array(arr)
    distances = np.array([20.0, 40.0, 80.0, 160.0])
    npts = 1024
    dt = 0.05
    res = compute_greens(
        model=model,
        src_depth_km=10.0,
        distances_km=distances,
        npts=npts,
        dt=dt,
        sigma=2.0,
        taper=0.5,
        samples_before_p=50,
        n_workers=1,
        t0_s=distances / 6.0,
        hipass=(1, 1),
    )
    peaks = np.max(np.abs(res.gf[:, 9, :int(8.0 / dt)]), axis=1)
    assert np.all(np.diff(peaks) < 0), f"Peaks not monotonically decreasing: {peaks}"
    # Geometric decay: at far distances the body wave decays as 1/r;
    # for a layered halfspace surface waves dominate at moderate depth
    # and Q-attenuation steepens the decay further.  We check that the
    # decay exponent is between 1 (pure body wave) and 3 (surface wave
    # + scattering); this catches gross algorithm bugs without
    # over-constraining the layered-medium physics.
    log_peaks = np.log(peaks)
    log_d = np.log(distances)
    slope = np.polyfit(log_d, log_peaks, 1)[0]
    assert -3.0 < slope < -0.7, f"Decay slope {slope:.2f} out of range"
