"""Backend selection: numpy (Numba) vs JAX.

Reads ``FK_BACKEND`` from the environment (``"numpy"`` default, ``"jax"``
opts in to the JAX path).  Public consumers use :func:`current_backend`.
"""

from __future__ import annotations

import os
from typing import Literal

BackendName = Literal["numpy", "jax"]


def current_backend(override: str | None = None) -> BackendName:
    """Return the active backend, honouring an optional override.

    Parameters
    ----------
    override
        If given, used directly (``"numpy"`` or ``"jax"``).  Otherwise the
        ``FK_BACKEND`` environment variable is consulted.
    """
    name = (override or os.environ.get("FK_BACKEND", "numpy")).lower()
    if name not in ("numpy", "jax"):
        msg = f"Unknown FK_BACKEND={name!r}; expected 'numpy' or 'jax'."
        raise ValueError(msg)
    return name  # type: ignore[return-value]


__all__ = ["BackendName", "current_backend"]
