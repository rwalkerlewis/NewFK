"""Canonical 5-layer benchmark.

Run with::

    python -m fkpy.benchmarks.canonical_zhu5

Reports wall-clock time for the configuration in the README/plan:
src 8 km, 10 distances 10–200 km, npts=2048, dt=0.1 s.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .._logging import logger
from ..greens import compute_greens
from ..model import LayeredModel

# 5-layer continental crust roughly à la Hadley-Kanamori, extended.
# thickness, vp, vs, rho, Qp, Qs
CANONICAL_5LAYER = np.array(
    [
        [3.0, 5.5, 3.18, 2.65, 600.0, 300.0],
        [12.0, 6.30, 3.64, 2.78, 800.0, 400.0],
        [15.0, 6.70, 3.87, 2.85, 1000.0, 500.0],
        [10.0, 7.20, 4.16, 2.95, 1500.0, 800.0],
        [0.0, 8.10, 4.68, 3.36, 2000.0, 1000.0],  # half-space
    ],
    dtype=np.float64,
)


def run(workers: int | None = 1) -> float:
    """Run the canonical benchmark and return wall-clock seconds."""
    model = LayeredModel.from_array(CANONICAL_5LAYER)
    distances = np.linspace(10.0, 200.0, 10)
    # Warm Numba caches
    _ = compute_greens(
        model=model,
        src_depth_km=8.0,
        distances_km=distances[:2],
        npts=128,
        dt=0.5,
        n_workers=1,
    )
    t0 = time.perf_counter()
    result = compute_greens(
        model=model,
        src_depth_km=8.0,
        distances_km=distances,
        npts=2048,
        dt=0.1,
        n_workers=workers,
    )
    elapsed = time.perf_counter() - t0
    logger.info("canonical_zhu5: %.2f s, gf shape %s", elapsed, result.gf.shape)
    return elapsed


def main() -> None:  # pragma: no cover - CLI entry
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    elapsed = run(workers=args.workers)
    print(f"{elapsed:.2f} s with {args.workers} worker(s)")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(f"{elapsed:.6f}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
