"""End-to-end CLI tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from fkpy.cli import main


@pytest.mark.fast
def test_cli_compute_writes_hdf5(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nd"
    model_path.write_text(
        "10  6.3 3.5 2.786 1000 500\n"
        "0   8.1 4.7 3.362 1600 800\n"
    )
    out = tmp_path / "greens.h5"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compute",
            "--model", str(model_path),
            "--depth", "5",
            "--distances", "50,100",
            "--npts", "128",
            "--dt", "0.5",
            "--out", str(out),
            "--workers", "1",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert out.exists()
    with h5py.File(out, "r") as h5:
        gf = h5["gf"]
        assert gf.shape == (2, 10, 128)
        # All required attrs present
        for k in ("dt", "npts", "src_depth_km", "rcv_depth_km",
                  "sigma", "dk", "kmax", "fkpy_version",
                  "component_order"):
            assert k in gf.attrs, f"missing attribute {k}"
        assert "distances_km" in h5
        assert "t0_p" in h5
        assert "t0_s" in h5
        assert "p_takeoff_deg" in h5
        assert "s_takeoff_deg" in h5
        assert "model" in h5
        np.testing.assert_array_equal(h5["distances_km"][:], [50.0, 100.0])


@pytest.mark.fast
def test_cli_compute_with_pyfk_format(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nd"
    model_path.write_text(
        "10  3.5 6.3 2.786 500 1000\n"
        "0   4.7 8.1 3.362 800 1600\n"
    )
    out = tmp_path / "greens.h5"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compute",
            "--model", str(model_path),
            "--model-format", "pyfk",
            "--depth", "5",
            "--distances", "50",
            "--npts", "128",
            "--dt", "0.5",
            "--out", str(out),
            "--workers", "1",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert out.exists()
