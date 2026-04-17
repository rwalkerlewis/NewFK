"""JAX backend (optional).

Activated by ``FK_BACKEND=jax`` or ``backend="jax"`` in
:func:`fkpy.compute_greens`.  Imports are deferred so that JAX is
*not* a hard dependency.
"""

from __future__ import annotations
