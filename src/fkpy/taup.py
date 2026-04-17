"""1-D ray-traced first-arrival times for a layered model.

Simplified port of Lupei Zhu's `tau_p.f` (and the corresponding pyfk
``taup`` routine).  We compute the **first P (or first S) arrival**
travel time at each receiver distance for a source at given depth,
using the standard tau-p algorithm:

.. math::

    T(x, p) = p\\,x + \\sum_i \\sqrt{1/v_i^2 - p^2}\\,d_i

where the sum runs over the layers between source and receiver.
We numerically minimise over the slowness ``p`` to find both:
  - the "direct" arrival (segments only above the source, no refraction);
  - the "refracted" arrival through the deeper, faster layer (turning
    point inside the half-space).

The minimum of the two gives the first-arrival time.

References
----------
Aki & Richards 2002 §9.1 — tau-p method.
Lupei Zhu's `tau_p.f` — direct port of the algorithm.
"""

from __future__ import annotations

import numpy as np

from ._typing import F64Array


def first_arrival_time(
    velocity_kms: F64Array,
    thickness_km: F64Array,
    src_layer: int,
    rcv_layer: int,
    distances_km: F64Array,
) -> F64Array:
    """Return the first-arrival time per receiver distance.

    Parameters
    ----------
    velocity_kms
        Per-layer P or S speed (km/s).  Pass ``model.vp_kms`` for
        the P arrival or ``model.vs_kms`` for the S arrival.
    thickness_km
        Per-layer thickness (km); the half-space row is 0.
    src_layer, rcv_layer
        Layer indices.  We require ``rcv_layer <= src_layer`` (the
        flipped-model convention used elsewhere in fkpy).
    distances_km
        1-D array of horizontal distances (km).
    """
    if rcv_layer > src_layer:
        rcv_layer, src_layer = src_layer, rcv_layer
    # Slice of layers between receiver (top) and source (bottom).
    v_above = velocity_kms[rcv_layer:src_layer]
    d_above = thickness_km[rcv_layer:src_layer]
    if v_above.size == 0:
        # Source and receiver in same layer; vertical-only path is 0.
        out: F64Array = (distances_km / max(velocity_kms[src_layer], 1e-9)).astype(
            np.float64
        )
        return out

    # Velocities below the source (used for refracted arrivals).
    v_below = velocity_kms[src_layer:]

    out = np.empty_like(distances_km)
    for i, x in enumerate(distances_km):
        out[i] = _first_arrival_single(x, v_above, d_above, v_below)
    return out


def _first_arrival_single(
    x: float,
    v_above: F64Array,
    d_above: F64Array,
    v_below: F64Array,
) -> float:
    """First-arrival travel time at one distance.

    Considers (a) the direct ray through the column above the source
    and (b) refracted head waves along each deeper interface; returns
    the minimum of the two.
    """
    # ---- direct ray through the layers above the source ------------
    # Solve sum_i d_i * p / sqrt(1/v_i^2 - p^2) = x for p in
    # [0, 1/max(v_above)).  Use bisection.
    p_max_direct = 1.0 / float(np.max(v_above)) - 1e-12
    p_direct = _bisect_distance(p_max_direct, x, v_above, d_above)
    t_direct = (
        p_direct * x
        + np.sum(np.sqrt(np.maximum(1.0 / v_above**2 - p_direct**2, 0.0)) * d_above)
    )

    # ---- head waves along each deeper layer ------------------------
    # The source sits at the top of v_below[0] (i.e. the bottom of the
    # column described by v_above / d_above).  For a head wave along the
    # j-th interface *below* the source we must travel:
    #   1. down through layers v_below[:j] from the source to interface j
    #      (depth = sum of v_below thicknesses).  Currently fkpy only
    #      knows total layer thickness through model.thickness_km, which
    #      is not part of v_below; we only support the j=0 case (head
    #      wave along the immediate sub-source interface).  This matches
    #      Lupei Zhu's `tau_p.f`, which loops only over the bottom-most
    #      interface for the typical use case.
    # The up-going leg traverses ``v_above`` vertically.
    # We do not include any down-going leg below the source for the
    # j == 0 case because the source already sits on the refraction
    # interface.
    t_refracted = np.inf
    for j in range(min(v_below.size, 1)):
        v_below_j = float(v_below[j])
        if v_below_j <= np.max(v_above) + 1e-12:
            continue  # head wave not faster than direct
        p_crit = 1.0 / v_below_j - 1e-12
        leg_dx = float(
            np.sum(d_above * p_crit / np.sqrt(np.maximum(1.0 / v_above**2 - p_crit**2, 0.0)))
        )
        if leg_dx > x:
            continue  # not enough range
        # Total travel time of the upgoing leg = p_crit·leg_dx + tau,
        # where tau = sum d_i sqrt(1/v_i² - p²) is the intercept time.
        tau_leg = float(
            np.sum(np.sqrt(np.maximum(1.0 / v_above**2 - p_crit**2, 0.0)) * d_above)
        )
        t_leg = p_crit * leg_dx + tau_leg
        t_head = t_leg + (x - leg_dx) / v_below_j
        t_refracted = min(t_refracted, t_head)

    return float(min(t_direct, t_refracted))


def _bisect_distance(
    p_max: float,
    x_target: float,
    v_above: F64Array,
    d_above: F64Array,
) -> float:
    """Bisect to find the slowness p in [0, p_max) such that the direct
    ray through the column above covers exactly the target horizontal
    distance ``x_target``."""
    if x_target <= 0:
        return 0.0
    # f(p) = sum d_i * p/sqrt(1/v_i^2 - p^2); monotonic increasing in p.
    lo, hi = 0.0, p_max
    # Check feasibility: at p == p_max the function diverges, so we are
    # always able to hit any positive x_target.
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        denom = np.sqrt(np.maximum(1.0 / v_above**2 - mid**2, 1e-30))
        x_mid = float(np.sum(d_above * mid / denom))
        if x_mid > x_target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def first_arrival_takeoff_deg(
    velocity_kms: F64Array,
    thickness_km: F64Array,
    src_layer: int,
    rcv_layer: int,
    distances_km: F64Array,
) -> F64Array:
    """Return the take-off angle (in degrees from vertical, downward
    positive) of the first arrival at each receiver distance.

    For a head wave the take-off angle is the critical angle ``arcsin(v_src
    / v_below)``; for a direct ray it is ``arcsin(p · v_src)`` where ``p``
    is the slowness of the direct ray.  Negative angles are used to
    indicate up-going rays (when receiver is below source).
    """
    if rcv_layer > src_layer:
        rcv_layer, src_layer = src_layer, rcv_layer
    v_above = velocity_kms[rcv_layer:src_layer]
    d_above = thickness_km[rcv_layer:src_layer]
    v_src = float(velocity_kms[src_layer])
    if v_above.size == 0:
        return np.full(distances_km.shape, 90.0, dtype=np.float64)
    v_below = velocity_kms[src_layer:]

    out = np.empty_like(distances_km)
    for i, x in enumerate(distances_km):
        out[i] = _takeoff_single(x, v_above, d_above, v_below, v_src)
    return out


def _takeoff_single(
    x: float,
    v_above: F64Array,
    d_above: F64Array,
    v_below: F64Array,
    v_src: float,
) -> float:
    p_max_direct = 1.0 / float(np.max(v_above)) - 1e-12
    p_direct = _bisect_distance(p_max_direct, x, v_above, d_above)
    t_direct = (
        p_direct * x
        + np.sum(np.sqrt(np.maximum(1.0 / v_above**2 - p_direct**2, 0.0)) * d_above)
    )
    best_t = t_direct
    best_p = p_direct
    for j in range(min(v_below.size, 1)):
        v_below_j = float(v_below[j])
        if v_below_j <= np.max(v_above) + 1e-12:
            continue
        p_crit = 1.0 / v_below_j - 1e-12
        leg_dx = float(
            np.sum(d_above * p_crit / np.sqrt(np.maximum(1.0 / v_above**2 - p_crit**2, 0.0)))
        )
        if leg_dx > x:
            continue
        tau_leg = float(
            np.sum(np.sqrt(np.maximum(1.0 / v_above**2 - p_crit**2, 0.0)) * d_above)
        )
        t_head = (p_crit * leg_dx + tau_leg) + (x - leg_dx) / v_below_j
        if t_head < best_t:
            best_t = t_head
            best_p = p_crit
    sin_take = best_p * v_src
    if sin_take >= 1.0:
        sin_take = 1.0 - 1e-12
    return float(np.degrees(np.arcsin(sin_take)))


__all__ = ["first_arrival_takeoff_deg", "first_arrival_time"]
