"""Smoke test for the full compute_greens pipeline (very small)."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.greens import compute_greens
from fkpy.source import N_GREEN_COMPONENTS


@pytest.mark.fast
def test_compute_greens_smoke(two_layer_model) -> None:
    res = compute_greens(
        model=two_layer_model,
        src_depth_km=5.0,
        rcv_depth_km=0.0,
        distances_km=np.array([20.0, 40.0]),
        npts=128,
        dt=0.5,
        n_workers=1,
    )
    assert res.gf.shape == (2, N_GREEN_COMPONENTS, 128)
    assert np.all(np.isfinite(res.gf))
    # At least one Green's function trace should be non-trivial.
    assert np.max(np.abs(res.gf)) > 0.0
    # SAC stream conversion should succeed.
    stream = res.to_obspy_stream(azimuth_deg=37.0)
    assert len(stream) == 2 * N_GREEN_COMPONENTS
