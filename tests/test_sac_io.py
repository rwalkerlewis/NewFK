"""Tests for SAC I/O — verifies that the SAC headers carry the full
set of fields documented in the README and PLAN §2.11."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fkpy.greens import compute_greens
from fkpy.model import LayeredModel
from fkpy.source import N_GREEN_COMPONENTS


@pytest.mark.fast
def test_sac_headers_carry_all_required_fields() -> None:
    arr = np.array(
        [
            [10.0, 6.3, 3.5, 2.786, 1000.0, 500.0],
            [25.0, 6.3, 3.5, 2.786, 1000.0, 500.0],
            [0.0, 8.1, 4.7, 3.362, 1600.0, 800.0],
        ]
    )
    model = LayeredModel.from_array(arr)
    res = compute_greens(
        model=model,
        src_depth_km=10.0,
        distances_km=np.array([100.0, 200.0]),
        npts=128,
        dt=0.5,
        n_workers=1,
    )
    stream = res.to_obspy_stream(azimuth_deg=37.0, kstnm="ABC")
    assert len(stream) == 2 * N_GREEN_COMPONENTS
    for tr in stream:
        sac = tr.stats.sac
        # Required header fields per PLAN §2.11
        assert sac["kstnm"] == "ABC"
        assert "dist" in sac
        assert "az" in sac
        assert "baz" in sac
        assert "evdp" in sac
        assert "stel" in sac
        assert "b" in sac
        assert "e" in sac
        assert "o" in sac
        assert "delta" in sac
        assert "t1" in sac  # P first arrival
        assert "t2" in sac  # S first arrival
        assert "user1" in sac  # P take-off angle
        assert "user2" in sac  # S take-off angle
        assert sac["az"] == pytest.approx(37.0)
        assert sac["baz"] == pytest.approx(217.0)


@pytest.mark.fast
def test_write_sac_filename_convention(tmp_path: Path) -> None:
    arr = np.array(
        [
            [10.0, 6.3, 3.5, 2.786, 1000.0, 500.0],
            [0.0, 8.1, 4.7, 3.362, 1600.0, 800.0],
        ]
    )
    model = LayeredModel.from_array(arr)
    res = compute_greens(
        model=model,
        src_depth_km=5.0,
        distances_km=np.array([50.0]),
        npts=128,
        dt=0.5,
        n_workers=1,
    )
    prefix = str(tmp_path / "syn")
    files = res.write_sac(prefix=prefix, azimuth_deg=10.0)
    assert len(files) == N_GREEN_COMPONENTS
    # Per Lupei Zhu's fk filename convention, n=0 transverse (suffix .2)
    # is identically zero for DC and is dropped from fkpy's output.
    expected_suffixes = ("0", "1", "3", "4", "5", "6", "7", "8", "a", "b")
    for fname, suf in zip(files, expected_suffixes, strict=True):
        assert fname.endswith(f"{suf}.sac"), f"unexpected filename {fname}"
