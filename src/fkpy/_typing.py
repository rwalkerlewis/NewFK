"""Numpy typing aliases used throughout fkpy.

Per the plan code-quality rule: ``Type hints everywhere; numpy.typing.NDArray
with dtype parameter.``
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64Array = NDArray[np.float64]
F32Array = NDArray[np.float32]
C128Array = NDArray[np.complex128]
C64Array = NDArray[np.complex64]
I32Array = NDArray[np.int32]

__all__ = ["C128Array", "C64Array", "F32Array", "F64Array", "I32Array"]
