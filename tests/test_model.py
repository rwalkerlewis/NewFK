"""Unit tests for fkpy.model."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.model import LayeredModel, ModelValidationError


@pytest.mark.fast
class TestLayeredModel:
    def test_from_array_basic(self) -> None:
        arr = np.array(
            [
                [10.0, 5.5, 3.2, 2.7, 1000.0, 500.0],
                [0.0, 6.5, 3.7, 2.9, 1500.0, 800.0],
            ]
        )
        m = LayeredModel.from_array(arr)
        assert m.n_layers == 2
        np.testing.assert_array_equal(m.thickness_km, [10.0, 0.0])
        np.testing.assert_array_equal(m.vp_kms, [5.5, 6.5])
        # mu = rho * vs^2 (g/cm^3 * km^2/s^2 == GPa)
        np.testing.assert_allclose(m.mu_gpa, [2.7 * 3.2**2, 2.9 * 3.7**2])
        # xi = (vs/vp)^2
        np.testing.assert_allclose(m.xi, [(3.2 / 5.5) ** 2, (3.7 / 6.5) ** 2])

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ModelValidationError):
            LayeredModel.from_array(np.zeros((3, 4)))
        with pytest.raises(ModelValidationError):
            LayeredModel.from_array(np.zeros((0, 6)))

    def test_invalid_physics(self) -> None:
        with pytest.raises(ModelValidationError):
            # Vp < sqrt(2)*Vs
            LayeredModel.from_array(np.array([[0.0, 3.0, 3.0, 2.7, 1000.0, 500.0]]))
        with pytest.raises(ModelValidationError):
            # negative density
            LayeredModel.from_array(np.array([[0.0, 5.0, 3.0, -1.0, 1000.0, 500.0]]))
        with pytest.raises(ModelValidationError):
            # Q < 1
            LayeredModel.from_array(np.array([[0.0, 5.0, 3.0, 2.7, 0.5, 500.0]]))

    def test_from_text(self, tmp_path) -> None:
        p = tmp_path / "model.nd"
        p.write_text(
            "# thickness vp vs rho Qp Qs\n"
            "10  5.5 3.2 2.7 1000 500\n"
            "0   6.5 3.7 2.9 1500 800\n"
        )
        m = LayeredModel.from_text(p)
        assert m.n_layers == 2

    def test_cumulative_depth(self) -> None:
        m = LayeredModel.from_array(
            np.array(
                [
                    [3.0, 5.0, 3.0, 2.7, 1000.0, 500.0],
                    [7.0, 5.5, 3.2, 2.8, 1000.0, 500.0],
                    [0.0, 6.5, 3.7, 2.9, 1500.0, 800.0],
                ]
            )
        )
        np.testing.assert_array_equal(m.cumulative_depth_km(), [3.0, 10.0, 10.0])

    def test_insert_interface_inside_layer(self, two_layer_model) -> None:
        new, idx = two_layer_model.insert_interface(4.0)
        assert new.n_layers == 3
        # The new layer is the one whose top is at depth 4 km.
        assert idx == 1
        np.testing.assert_allclose(new.thickness_km, [4.0, 6.0, 0.0])
        # Properties on either side of the inserted boundary match the
        # layer that was split.
        assert new.vp_kms[0] == new.vp_kms[1]

    def test_insert_interface_at_boundary_is_no_op(self, two_layer_model) -> None:
        new, idx = two_layer_model.insert_interface(10.0)
        assert new.n_layers == 2
        assert idx == 1

    def test_insert_interface_at_surface(self, two_layer_model) -> None:
        new, idx = two_layer_model.insert_interface(0.0)
        assert new.n_layers == 2
        assert idx == 0

    def test_round_trip_to_array(self, two_layer_model) -> None:
        arr = two_layer_model.to_array()
        m2 = LayeredModel.from_array(arr)
        np.testing.assert_array_equal(m2.to_array(), arr)

    def test_from_text_pyfk_format(self, tmp_path) -> None:
        # pyfk/fk convention: thickness vs vp rho Qs Qp
        p = tmp_path / "pyfk.nd"
        p.write_text(
            "10  3.5 6.3 2.786 500 1000\n"
            "0   4.7 8.1 3.362 800 1600\n"
        )
        m = LayeredModel.from_text(p, format="pyfk")
        # Expected fkpy: thickness vp vs rho Qp Qs
        np.testing.assert_allclose(m.vp_kms, [6.3, 8.1])
        np.testing.assert_allclose(m.vs_kms, [3.5, 4.7])
        np.testing.assert_allclose(m.qp, [1000, 1600])
        np.testing.assert_allclose(m.qs, [500, 800])

    def test_from_text_unknown_format(self, tmp_path) -> None:
        p = tmp_path / "x.nd"
        p.write_text("0 5 3 2.7 500 1000\n")
        with pytest.raises(ValueError):
            LayeredModel.from_text(p, format="bogus")
