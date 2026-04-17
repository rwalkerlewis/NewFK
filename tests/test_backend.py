"""Tests for backend selection."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from fkpy._backend import current_backend


@pytest.mark.fast
class TestBackendSelection:
    def test_default_is_numpy(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert current_backend() == "numpy"

    def test_env_var_jax(self) -> None:
        with patch.dict(os.environ, {"FK_BACKEND": "jax"}):
            assert current_backend() == "jax"

    def test_env_var_numpy_explicit(self) -> None:
        with patch.dict(os.environ, {"FK_BACKEND": "numpy"}):
            assert current_backend() == "numpy"

    def test_override_takes_precedence(self) -> None:
        with patch.dict(os.environ, {"FK_BACKEND": "jax"}):
            assert current_backend("numpy") == "numpy"
            assert current_backend("jax") == "jax"

    def test_case_insensitive(self) -> None:
        with patch.dict(os.environ, {"FK_BACKEND": "JAX"}):
            assert current_backend() == "jax"
        assert current_backend("Numpy") == "numpy"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown FK_BACKEND"):
            current_backend("cuda")
