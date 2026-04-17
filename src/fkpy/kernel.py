"""Displacement kernel U(k, ω) at the receiver.

The kernel combines:

1. The R/T (or compound-matrix) recursion of :mod:`fkpy.propagator` to
   propagate the source-side ``g`` vector from the bottom half-space up
   to the source layer and the receiver-side ``b`` vector from the free
   surface down to the receiver.
2. The source-jump vector :func:`fkpy.source.source_jump` (Zhu &
   Rivera 2002 eq. 16) that turns the (R, T, g, b) information into the
   actual ``z(3, 5)`` row vector.
3. The free-surface boundary condition that produces the final 3×3
   ``u[n, comp]`` displacement-kernel array.

Public function
---------------

The displacement kernel itself is implemented in :func:`fkpy.propagator.kernel`
because the inner loops there must be Numba-jitted and would generate a
circular import if split across modules.  This file re-exports the
function under the plan-mandated name :func:`displacement_kernel` so
callers can use the documented entry point.

References
----------
- Zhu & Rivera (2002), GJI 148, eq. (28) for the displacement kernel.
- Aki & Richards (2002), §6.5 for the underlying notation.
"""

from __future__ import annotations

from .propagator import kernel as displacement_kernel

__all__ = ["displacement_kernel"]
