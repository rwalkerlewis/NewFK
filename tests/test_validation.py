"""Input-validation tests for the public API."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy import LayeredModel, compute_greens
from fkpy.kernel import displacement_kernel
from fkpy.source import SourceType, source_jump


@pytest.mark.fast
class TestComputeGreensValidation:
    def _model(self) -> LayeredModel:
        return LayeredModel.from_array(
            np.array(
                [
                    [10.0, 6.3, 3.5, 2.786, 1000.0, 500.0],
                    [0.0, 8.1, 4.7, 3.362, 1600.0, 800.0],
                ]
            )
        )

    def test_zero_distance_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_greens(
                model=self._model(),
                src_depth_km=5.0,
                distances_km=np.array([0.0]),
                npts=128,
                dt=0.5,
                n_workers=1,
            )

    def test_negative_distance_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_greens(
                model=self._model(),
                src_depth_km=5.0,
                distances_km=np.array([-50.0]),
                npts=128,
                dt=0.5,
                n_workers=1,
            )

    def test_empty_distances_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            compute_greens(
                model=self._model(),
                src_depth_km=5.0,
                distances_km=np.array([]),
                npts=128,
                dt=0.5,
                n_workers=1,
            )

    def test_npts_too_small_rejected(self) -> None:
        with pytest.raises(ValueError, match="npts"):
            compute_greens(
                model=self._model(),
                src_depth_km=5.0,
                distances_km=np.array([50.0]),
                npts=1,
                dt=0.5,
                n_workers=1,
            )

    def test_zero_dt_rejected(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            compute_greens(
                model=self._model(),
                src_depth_km=5.0,
                distances_km=np.array([50.0]),
                npts=128,
                dt=0.0,
                n_workers=1,
            )

    def test_negative_dt_rejected(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            compute_greens(
                model=self._model(),
                src_depth_km=5.0,
                distances_km=np.array([50.0]),
                npts=128,
                dt=-0.5,
                n_workers=1,
            )


@pytest.mark.fast
class TestDisplacementKernelPublicAPI:
    def test_kernel_module_exports_displacement_kernel(self) -> None:
        """The plan §1.1 mandated `fkpy.kernel.displacement_kernel`
        symbol must be the canonical entry point and produce identical
        results to the underlying Numba implementation."""
        from fkpy import attenuation, propagator

        omega = 2.0 * np.pi * 1.0
        w = complex(omega, -0.05)
        model = LayeredModel.from_array(
            np.array(
                [
                    [10.0, 6.3, 3.5, 2.786, 1000.0, 500.0],
                    [0.0, 8.1, 4.7, 3.362, 1600.0, 800.0],
                ]
            )
        )
        kp_sq = attenuation.complex_wavenumber_squared(w, model.vp_kms, model.qp)
        ks_sq = attenuation.complex_wavenumber_squared(w, model.vs_kms, model.qs)
        s = source_jump(
            SourceType.DOUBLE_COUPLE,
            xi=float(model.xi[1]),
            mu=float(model.mu_gpa[1]),
            flip=1,
        ).astype(np.complex128)
        u_kernel = displacement_kernel(
            0.04, kp_sq, ks_sq, model.mu_gpa, model.thickness_km,
            s, 1, 0, int(SourceType.DOUBLE_COUPLE), 0,
        )
        u_propagator = propagator.kernel(
            0.04, kp_sq, ks_sq, model.mu_gpa, model.thickness_km,
            s, 1, 0, int(SourceType.DOUBLE_COUPLE), 0,
        )
        np.testing.assert_array_equal(u_kernel, u_propagator)
