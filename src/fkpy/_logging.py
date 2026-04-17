"""Package-scoped logger for fkpy.

Per the plan: no `print()` statements anywhere in the package; all
diagnostics flow through this logger. Library users can attach handlers
or set the level externally::

    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("fkpy").setLevel(logging.DEBUG)
"""

from __future__ import annotations

import logging

logger: logging.Logger = logging.getLogger("fkpy")
# Library convention: don't emit anything by default.
logger.addHandler(logging.NullHandler())

__all__ = ["logger"]
