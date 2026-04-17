"""Test that the Bouchon (1981) condition (2) is checked at runtime."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fkpy.greens import compute_greens
from fkpy.model import LayeredModel


@pytest.mark.fast
def test_bouchon_dk_warning_logged(caplog) -> None:
    """A `dk` larger than the Bouchon (1981) limit must trigger a
    WARNING-level log entry from the `fkpy` logger."""
    arr = np.array(
        [
            [10.0, 6.3, 3.5, 2.786, 1000.0, 500.0],
            [25.0, 6.3, 3.5, 2.786, 1000.0, 500.0],
            [0.0, 8.1, 4.7, 3.362, 1600.0, 800.0],
        ]
    )
    model = LayeredModel.from_array(arr)
    with caplog.at_level(logging.WARNING, logger="fkpy"):
        compute_greens(
            model=model,
            src_depth_km=10.0,
            distances_km=np.array([20.0]),  # small xmax → strict dk limit
            npts=2048,  # long trace → strict dk limit
            dt=0.1,
            dk=0.4,  # deliberately too large
            n_workers=1,
        )
    bouchon_warnings = [
        r for r in caplog.records
        if "Bouchon" in r.message and r.levelno == logging.WARNING
    ]
    assert bouchon_warnings, "Expected a Bouchon condition (2) warning."


@pytest.mark.fast
def test_bouchon_no_warning_when_dk_safe(caplog) -> None:
    arr = np.array(
        [
            [10.0, 6.3, 3.5, 2.786, 1000.0, 500.0],
            [25.0, 6.3, 3.5, 2.786, 1000.0, 500.0],
            [0.0, 8.1, 4.7, 3.362, 1600.0, 800.0],
        ]
    )
    model = LayeredModel.from_array(arr)
    with caplog.at_level(logging.WARNING, logger="fkpy"):
        compute_greens(
            model=model,
            src_depth_km=10.0,
            distances_km=np.array([200.0]),
            npts=128,
            dt=0.5,
            dk=0.05,
            n_workers=1,
        )
    bouchon_warnings = [
        r for r in caplog.records if "Bouchon" in r.message
    ]
    assert not bouchon_warnings
