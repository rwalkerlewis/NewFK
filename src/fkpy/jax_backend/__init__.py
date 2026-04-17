"""JAX backend (optional).

Activated by ``FK_BACKEND=jax`` or ``backend="jax"`` in
:func:`fkpy.compute_greens`.  Imports are deferred so that JAX is
*not* a hard dependency.

Per plan §1.5: the JAX path uses ``float32`` / ``complex64`` by default
to make GPU memory budgets and inter-device transfers cheap.  The
numpy/Numba CPU path uses ``float64`` / ``complex128``.

Submodules:
    propagator_jx  — JAX-numpy compound matrix builder.
    kernel_jx      — JAX displacement-kernel placeholder
                     (delegates to the Numba implementation in v0.1).
    greens_jx      — :func:`compute_greens_jax` and
                     :func:`compute_greens_for_depths` (batched
                     evaluation across source depths).
"""

from __future__ import annotations

__all__ = [
    "compute_greens_for_depths",
    "compute_greens_jax",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy attribute access so ``fkpy.jax_backend.compute_greens_jax``
    works without forcing JAX to be importable at package-import time.

    The actual ImportError (with a helpful install hint) is raised
    only when the user attempts to call the JAX-dependent function.
    """
    if name in __all__:
        from . import greens_jx

        return getattr(greens_jx, name)
    raise AttributeError(f"module 'fkpy.jax_backend' has no attribute {name!r}")
