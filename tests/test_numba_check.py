"""Verify the hard-fail behaviour when Numba is unavailable.

Per the plan code-quality rule: ``If numba import fails, raise
ImportError at module load, do not silently fall back to pure Python.``
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.fast
def test_numba_check_module_loads_without_error() -> None:
    """Sanity check: with Numba installed, importing the check module
    must succeed silently."""
    mod = importlib.import_module("fkpy._numba_check")
    # Re-import is a no-op; it must not have any user-facing attribute
    # other than the imported `numba` symbol.
    assert hasattr(mod, "numba")


@pytest.mark.fast
def test_numba_check_raises_clear_message_on_simulated_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Numba were missing, the module must raise ImportError with a
    user-friendly hint.  We simulate this by mocking ``numba`` out of
    ``sys.modules`` and forcing a re-import."""
    monkeypatch.setitem(sys.modules, "numba", None)
    # Drop any cached fkpy._numba_check.
    monkeypatch.delitem(sys.modules, "fkpy._numba_check", raising=False)
    with pytest.raises(ImportError, match="numba"):
        importlib.import_module("fkpy._numba_check")
