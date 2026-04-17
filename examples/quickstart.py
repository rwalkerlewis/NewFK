"""README one-line example, expanded.

Run from the repo root::

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fkpy import LayeredModel, MomentTensor, ZhuBasis, compute_greens

HERE = Path(__file__).resolve().parent


def main() -> None:
    model = LayeredModel.from_text(HERE / "zhu5.model")
    print(model)

    distances = np.array([50.0, 100.0, 150.0])
    result = compute_greens(
        model=model,
        src_depth_km=8.0,
        distances_km=distances,
        npts=2048,
        dt=0.1,
        n_workers=1,
    )
    print(f"Computed gf{result.gf.shape} at distances {distances} km")
    print(f"P first arrivals (s): {result.t0_p}")
    print(f"S first arrivals (s): {result.t0_s}")
    for comp in (ZhuBasis.DD_Z, ZhuBasis.SS_R, ZhuBasis.EX_Z):
        peak = float(np.max(np.abs(result.gf[:, int(comp), :])))
        print(f"  peak |{comp.name}| = {peak:.3e}")

    # Project a known moment tensor onto the GFs and synthesise a
    # 3-component seismogram at the first receiver and 37° azimuth.
    mt = MomentTensor(Mxx=1.0, Mxy=0.3, Mxz=-0.2, Myy=0.5, Myz=0.1, Mzz=0.4)
    weights = mt.basis_weights(az_deg=37.0)
    print(f"Projected MT weights shape: {weights.shape}")

    # Synthesise the 3-component (Z, R, T) seismogram at receiver 0
    # by linearly combining the 10 Green's functions per the
    # `MomentTensor.basis_weights` matrix.
    synth_3comp = np.einsum("cg,gt->ct", weights, result.gf[0])
    print(f"3-component synthetic shape: {synth_3comp.shape}")
    for ic, name in enumerate(("Z", "R", "T")):
        print(f"  peak |{name}| = {np.max(np.abs(synth_3comp[ic])):.3e}")


if __name__ == "__main__":
    main()
