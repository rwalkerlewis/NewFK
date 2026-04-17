"""Hard-fail at import time if Numba is unavailable.

Per the plan code-quality rule: ``If numba import fails, raise ImportError
at module load, do not silently fall back to pure Python.``
"""

from __future__ import annotations

try:
    import numba as numba  # noqa: F401  (used for its side-effect / availability)
except ImportError as exc:  # pragma: no cover - exercised by environment, not unit tests
    raise ImportError(
        "fkpy requires numba. Install with `pip install numba` or "
        "`uv pip install fkpy[dev]`."
    ) from exc
