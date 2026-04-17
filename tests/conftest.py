"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.model import LayeredModel


@pytest.fixture
def halfspace_model() -> LayeredModel:
    """Single-layer (= half-space) model: Vp=6, Vs=3.4, ρ=2.7, Q=10000."""
    arr = np.array([[0.0, 6.0, 3.4, 2.7, 10000.0, 10000.0]])
    return LayeredModel.from_array(arr)


@pytest.fixture
def two_layer_model() -> LayeredModel:
    arr = np.array(
        [
            [10.0, 5.5, 3.2, 2.7, 1000.0, 500.0],
            [0.0, 6.5, 3.7, 2.9, 1500.0, 800.0],
        ]
    )
    return LayeredModel.from_array(arr)


@pytest.fixture
def canonical_5layer() -> LayeredModel:
    from fkpy.benchmarks.canonical_zhu5 import CANONICAL_5LAYER

    return LayeredModel.from_array(CANONICAL_5LAYER)
