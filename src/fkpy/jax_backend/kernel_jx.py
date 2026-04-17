"""JAX-friendly displacement kernel.

This module is the JAX-numpy equivalent of :mod:`fkpy.kernel`.  It
exists for compositional reasons: a future v0.2 release will let
users compose ``compound_matrix_jx`` (from
:mod:`fkpy.jax_backend.propagator_jx`) with this kernel under
``jax.vmap`` / ``jax.jit`` to get a fully-XLA-compiled F-K kernel.

For v0.1 the displacement kernel itself is provided by the Numba
implementation; this module re-exports the *symbol* under the
``displacement_kernel_jx`` name so that downstream users who write
``from fkpy.jax_backend.kernel_jx import displacement_kernel_jx``
are not broken when v0.2 lands a true JAX kernel.
"""

from __future__ import annotations

from ..kernel import displacement_kernel as _displacement_kernel


def displacement_kernel_jx(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Placeholder for the future native-JAX displacement kernel.

    Currently delegates to the Numba implementation.
    """
    return _displacement_kernel(*args, **kwargs)


__all__ = ["displacement_kernel_jx"]
