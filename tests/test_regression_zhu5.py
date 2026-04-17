"""Regression test against frozen Fortran-fk Green's functions.

Run parameters and pass criteria documented in `tests/data/README.md`:
xcorr > 0.999, peak-amplitude ratio within 1 %.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fkpy.greens import compute_greens
from fkpy.model import LayeredModel
from fkpy.source import ZhuBasis

DATA_DIR = Path(__file__).resolve().parent / "data"
NPZ_PATH = DATA_DIR / "canonical_zhu5.npz"
MODEL_PATH = DATA_DIR / "canonical_zhu5.model"


# Mapping from the 9-component fk output (axis order Z0 R0 T0 Z1 R1 T1 Z2 R2 T2)
# to the fkpy 10-component output (with the trivially-zero n=0 transverse
# dropped and Explosion components added).  Index pairs are (fk_idx, fkpy_idx).
_FK_TO_FKPY = (
    (0, ZhuBasis.DD_Z),
    (1, ZhuBasis.DD_R),
    # 2 -> n=0 transverse, identically zero in fk
    (3, ZhuBasis.DS_Z),
    (4, ZhuBasis.DS_R),
    (5, ZhuBasis.DS_T),
    (6, ZhuBasis.SS_Z),
    (7, ZhuBasis.SS_R),
    (8, ZhuBasis.SS_T),
)


@pytest.mark.reference
@pytest.mark.skipif(not NPZ_PATH.exists(), reason="frozen reference data missing")
def test_canonical_zhu5_matches_fortran_fk() -> None:
    ref = np.load(NPZ_PATH)
    model = LayeredModel.from_text(MODEL_PATH)
    distances = ref["distances_km"]
    t0 = ref["tfirst_s"].astype(np.float64)

    res = compute_greens(
        model=model,
        src_depth_km=float(ref["src_depth_km"]),
        distances_km=distances,
        npts=int(ref["npts"]),
        dt=float(ref["dt"]),
        sigma=float(ref["sigma"]),
        taper=float(ref["taper"]),
        samples_before_p=int(ref["samples_before_p"]),
        n_workers=1,
        t0_s=t0,
        hipass=(1, 1),
    )

    failures: list[str] = []
    for d_idx in range(distances.size):
        for fk_idx, fkpy_idx in _FK_TO_FKPY:
            ours = res.gf[d_idx, int(fkpy_idx)]
            theirs = ref["gf"][d_idx, fk_idx]
            n = min(ours.size, theirs.size)
            a = ours[:n]
            b = theirs[:n]
            if np.std(b) == 0.0:
                continue
            xc = float(np.corrcoef(a, b)[0, 1])
            amp_ratio = float(np.max(np.abs(a)) / max(np.max(np.abs(b)), 1e-30))
            if xc < 0.999:
                failures.append(
                    f"d={distances[d_idx]:g} comp={ZhuBasis(fkpy_idx).name} xc={xc:.4f}"
                )
            if not (0.99 <= amp_ratio <= 1.01):
                failures.append(
                    f"d={distances[d_idx]:g} comp={ZhuBasis(fkpy_idx).name} amp={amp_ratio:.3f}"
                )
    assert not failures, "\n".join(failures)
