"""Tests for the simplified tau-p first-arrival module."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.taup import first_arrival_time


@pytest.mark.fast
class TestTaup:
    def test_homogeneous_layer_direct_ray(self) -> None:
        """In a single-layer model the first arrival is just
        ``r/v`` where r = sqrt(x² + Δh²)."""
        v = np.array([6.0, 6.0])
        d = np.array([10.0, 0.0])
        # source at the bottom of layer 0 (depth 10 km), receiver at top
        x = np.array([20.0, 50.0, 100.0])
        t = first_arrival_time(v, d, src_layer=1, rcv_layer=0, distances_km=x)
        # Direct ray: travel through the 10 km layer at velocity 6 along
        # a slant path of length sqrt(x² + 10²).
        expected = np.sqrt(x**2 + 100.0) / 6.0
        np.testing.assert_allclose(t, expected, rtol=1e-3)

    def test_refraction_through_faster_halfspace(self) -> None:
        """A faster half-space below the source should yield a head-wave
        first arrival at far enough distances.  The source sits on the
        crust/mantle interface so the head wave has just one upgoing
        leg."""
        v = np.array([5.0, 8.0])
        d = np.array([10.0, 0.0])
        x = np.array([200.0])
        t = first_arrival_time(v, d, src_layer=1, rcv_layer=0, distances_km=x)
        cos_c = np.sqrt(39.0) / 8.0  # cos of critical angle
        t_leg = 10.0 / (5.0 * cos_c)
        leg_dx = 10.0 * (5.0 / 8.0) / cos_c
        expected = t_leg + (200.0 - leg_dx) / 8.0
        np.testing.assert_allclose(t[0], expected, rtol=1e-3)

    def test_far_field_first_arrival_uses_refraction(self) -> None:
        """At very large distance with a faster halfspace, the head-wave
        time grows like ``x / v_below`` (slope = 1/v_below = 1/8)."""
        v = np.array([5.0, 8.0])
        d = np.array([10.0, 0.0])
        x = np.array([300.0, 600.0, 1200.0])
        t = first_arrival_time(v, d, src_layer=1, rcv_layer=0, distances_km=x)
        slopes = np.diff(t) / np.diff(x)
        np.testing.assert_allclose(slopes, 1.0 / 8.0, rtol=1e-2)

    def test_same_layer_returns_horizontal_time(self) -> None:
        """If source and receiver are in the same layer, the first
        arrival is just ``x / v`` along the layer."""
        v = np.array([6.0, 6.0])
        d = np.array([10.0, 0.0])
        t = first_arrival_time(v, d, src_layer=0, rcv_layer=0,
                               distances_km=np.array([60.0]))
        np.testing.assert_allclose(t[0], 10.0, rtol=1e-12)
