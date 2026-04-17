"""End-to-end test: forward a moment tensor through Green's functions,
then invert and check that we recover the input M_ij.

Procedure
---------
1. Compute the 10-component Green's-function set at one distance.
2. Pick a known moment tensor ``M_in`` (six independent components).
3. Synthesise the 3-component (Z, R, T) seismograms via
   ``MomentTensor.basis_weights``.
4. Set up the forward design matrix for inversion: each Mij contributes
   linearly to the 3 components through 8 of the 10 GFs.
5. Solve via :func:`numpy.linalg.lstsq` and verify
   ``‖M_out − M_in‖ / ‖M_in‖ < 0.01``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fkpy.greens import compute_greens
from fkpy.model import LayeredModel
from fkpy.source import MomentTensor


def _design_matrix_for_one_distance(
    gf_one_dist: np.ndarray, az_deg: float
) -> np.ndarray:
    """Build the (3*npts, 6) design matrix that maps M to (Z, R, T)·t.

    Mirrors :meth:`MomentTensor.basis_weights` — one row per
    (component, sample), one column per Mij.
    """
    npts = gf_one_dist.shape[-1]
    az = np.deg2rad(az_deg)
    sa, ca = np.sin(az), np.cos(az)
    sa2, ca2 = np.sin(2.0 * az), np.cos(2.0 * az)

    # gf indices: 0 DD_Z, 1 DD_R, 2 DS_Z, 3 DS_R, 4 DS_T,
    #             5 SS_Z, 6 SS_R, 7 SS_T, 8 EX_Z, 9 EX_R
    DD_Z = gf_one_dist[0]
    DD_R = gf_one_dist[1]
    DS_Z = gf_one_dist[2]
    DS_R = gf_one_dist[3]
    DS_T = gf_one_dist[4]
    SS_Z = gf_one_dist[5]
    SS_R = gf_one_dist[6]
    SS_T = gf_one_dist[7]
    EX_Z = gf_one_dist[8]
    EX_R = gf_one_dist[9]

    # For each component (Z, R, T) and each Mij, we know how the
    # 10 basis Green's functions combine — see fkpy.source._basis_weights_3comp.
    #
    # We invert that mapping symbolically:  define for each Mij the
    # contribution to (Z_t, R_t, T_t).  Coefficients drawn from
    # MomentTensor.basis_weights.
    cols = []
    # Mxx contribution
    iso_Mxx = 1.0 / 3.0
    dd_Mxx = -1.0 / 6.0
    ss_xx_Mxx = -0.5 * ca2
    ss_t_Mxx = -0.5 * sa2
    z = iso_Mxx * EX_Z + dd_Mxx * DD_Z + ss_xx_Mxx * SS_Z
    r = iso_Mxx * EX_R + dd_Mxx * DD_R + ss_xx_Mxx * SS_R
    t = ss_t_Mxx * SS_T
    cols.append(np.concatenate([z, r, t]))

    # Mxy
    ss_xx_Mxy = -sa2
    ss_t_Mxy = ca2
    z = ss_xx_Mxy * SS_Z
    r = ss_xx_Mxy * SS_R
    t = ss_t_Mxy * SS_T
    cols.append(np.concatenate([z, r, t]))

    # Mxz
    ds_z_Mxz = -ca
    ds_t_Mxz = -sa
    z = ds_z_Mxz * DS_Z
    r = ds_z_Mxz * DS_R
    t = ds_t_Mxz * DS_T
    cols.append(np.concatenate([z, r, t]))

    # Myy
    iso_Myy = 1.0 / 3.0
    dd_Myy = -1.0 / 6.0
    ss_xx_Myy = 0.5 * ca2
    ss_t_Myy = 0.5 * sa2
    z = iso_Myy * EX_Z + dd_Myy * DD_Z + ss_xx_Myy * SS_Z
    r = iso_Myy * EX_R + dd_Myy * DD_R + ss_xx_Myy * SS_R
    t = ss_t_Myy * SS_T
    cols.append(np.concatenate([z, r, t]))

    # Myz
    ds_z_Myz = -sa
    ds_t_Myz = ca
    z = ds_z_Myz * DS_Z
    r = ds_z_Myz * DS_R
    t = ds_t_Myz * DS_T
    cols.append(np.concatenate([z, r, t]))

    # Mzz
    iso_Mzz = 1.0 / 3.0
    dd_Mzz = 1.0 / 3.0
    z = iso_Mzz * EX_Z + dd_Mzz * DD_Z
    r = iso_Mzz * EX_R + dd_Mzz * DD_R
    cols.append(np.concatenate([z, r, np.zeros(npts)]))

    return np.column_stack(cols)


def _synthesise(gf_one_dist: np.ndarray, mt: MomentTensor, az_deg: float) -> np.ndarray:
    """Synthesise (Z, R, T) at one distance for moment tensor ``mt``."""
    G = _design_matrix_for_one_distance(gf_one_dist, az_deg)
    m = np.array([mt.Mxx, mt.Mxy, mt.Mxz, mt.Myy, mt.Myz, mt.Mzz])
    d = G @ m
    return d


@pytest.mark.slow
def test_mt_inversion_recovers_input_within_one_percent() -> None:
    arr = np.array(
        [
            [10.0, 5.5, 3.18, 2.65, 600.0, 300.0],
            [12.0, 6.3, 3.64, 2.78, 800.0, 400.0],
            [0.0, 6.7, 3.87, 2.85, 1000.0, 500.0],
        ]
    )
    model = LayeredModel.from_array(arr)
    res = compute_greens(
        model=model,
        src_depth_km=10.0,
        distances_km=np.array([100.0]),
        npts=512,
        dt=0.1,
        sigma=2.0,
        taper=0.5,
        samples_before_p=25,
        n_workers=1,
    )
    gf = res.gf[0]  # (10, npts)

    M_in = MomentTensor(Mxx=1.0, Mxy=0.3, Mxz=-0.2, Myy=0.5, Myz=0.1, Mzz=0.4)
    az = 37.0
    d = _synthesise(gf, M_in, az_deg=az)
    G = _design_matrix_for_one_distance(gf, az_deg=az)
    m_out, *_ = np.linalg.lstsq(G, d, rcond=None)

    M_in_arr = np.array(
        [M_in.Mxx, M_in.Mxy, M_in.Mxz, M_in.Myy, M_in.Myz, M_in.Mzz]
    )
    rel = np.linalg.norm(m_out - M_in_arr) / np.linalg.norm(M_in_arr)
    assert rel < 0.01, f"MT recovery error {rel:.4f}, M_out={m_out}, M_in={M_in_arr}"
