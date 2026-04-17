"""fkpy — frequency-wavenumber synthetic seismograms for layered elastic half-spaces.

Public API:
    LayeredModel        — 1-D velocity / Q model.
    MomentTensor        — 3×3 moment tensor with Zhu-basis projection.
    ZhuBasis            — enum of the 10 Green's-function components.
    compute_greens(...) — main entry point.
    GreensResult        — frozen dataclass of computed Green's functions.
    write_sac(...)      — write SAC files matching Lupei Zhu's `fk` headers.
    to_obspy_stream(...) — convert to an ObsPy Stream.

References:
    Zhu, L. & Rivera, L. A. (2002), GJI 148, 619-627.
    Bouchon, M. (1981), BSSA 71, 959-971.
    Kennett, B. L. N. (1983), Seismic Wave Propagation in Stratified Media.
    Aki, K. & Richards, P. G. (2002), Quantitative Seismology, 2nd ed.
"""

from __future__ import annotations

# Hard-fail at import time if numba is missing — see plan §1.5.
from . import _numba_check as _numba_check  # noqa: F401  (side-effect import)
from ._logging import logger as logger
from .greens import GreensResult, compute_greens, compute_single_force_greens
from .model import LayeredModel, ModelValidationError
from .sac_io import to_obspy_stream, write_sac
from .source import (
    N_GREEN_COMPONENTS,
    MomentTensor,
    SourceType,
    ZhuBasis,
    source_jump,
)

__all__ = [
    "GreensResult",
    "LayeredModel",
    "ModelValidationError",
    "MomentTensor",
    "N_GREEN_COMPONENTS",
    "SourceType",
    "ZhuBasis",
    "compute_greens",
    "compute_greens_for_depths",
    "compute_single_force_greens",
    "logger",
    "source_jump",
    "to_obspy_stream",
    "write_sac",
]


def compute_greens_for_depths(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy proxy to :func:`fkpy.jax_backend.greens_jx.compute_greens_for_depths`.

    JAX is an *optional* dependency.  Import is deferred so that
    ``import fkpy`` succeeds even when JAX is not installed; the
    helpful ImportError is raised only when the user actually calls
    this function.
    """
    from .jax_backend.greens_jx import (
        compute_greens_for_depths as _compute_greens_for_depths,
    )

    return _compute_greens_for_depths(*args, **kwargs)

__version__ = "0.1.1"
