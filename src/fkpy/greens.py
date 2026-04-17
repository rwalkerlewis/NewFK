"""High-level Green's-function computation.

Public entry point: :func:`compute_greens`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._backend import current_backend
from ._logging import logger
from ._typing import F32Array, F64Array
from .constants import (
    DK_DEFAULT,
    H_SOURCE_RECEIVER_FLOOR_KM,
    KMAX_DEFAULT,
    PMAX_DEFAULT,
    PMIN_DEFAULT,
    SAMPLES_BEFORE_P_DEFAULT,
    SIGMA_DEFAULT,
    TAPER_DEFAULT,
    TWO_PI,
)
from .frequency import FrequencyJobConfig, run_omega_loop
from .model import LayeredModel
from .source import (
    N_GREEN_COMPONENTS,
    SourceType,
    ZhuBasis,
    source_jump,
)
from .transform import inverse_transform

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class GreensResult:
    """Computed Green's functions and metadata.

    Attributes
    ----------
    gf
        ``(n_dist, 10, npts)`` array of Green's functions for the 10
        Zhu-basis components defined by :class:`fkpy.source.ZhuBasis`.
    distances_km
        Receiver distances.
    dt
        Time-domain sampling interval (s).
    t0_p, t0_s
        Approximate first-arrival times for P and S (s).
    npts
        Trace length.
    meta
        Dict carrying every input parameter, for reproducibility.
    """

    # gf may be float32 when the JAX backend is used; otherwise float64.
    gf: F64Array | F32Array
    distances_km: F64Array
    dt: float
    t0_p: F64Array
    t0_s: F64Array
    npts: int
    src_depth_km: float
    rcv_depth_km: float
    p_takeoff_deg: F64Array = field(default_factory=lambda: np.zeros(0))
    s_takeoff_deg: F64Array = field(default_factory=lambda: np.zeros(0))
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_obspy_stream(self, *, azimuth_deg: float = 0.0, kstnm: str = "STA") -> Any:
        """Convert to an ObsPy ``Stream`` (one Trace per dist × component)."""
        from .sac_io import to_obspy_stream

        return to_obspy_stream(self, azimuth_deg=azimuth_deg, kstnm=kstnm)

    def write_sac(self, prefix: str, *, azimuth_deg: float = 0.0) -> list[str]:
        from .sac_io import write_sac

        return write_sac(self, prefix=prefix, azimuth_deg=azimuth_deg)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _approximate_first_arrivals(
    model: LayeredModel,
    src_layer: int,
    rcv_layer: int,
    distances_km: F64Array,
) -> tuple[F64Array, F64Array, F64Array, F64Array]:
    """Return ``(t0_p, t0_s, takeoff_p_deg, takeoff_s_deg)`` per distance
    via 1-D ray tracing.

    Uses :func:`fkpy.taup.first_arrival_time` and
    :func:`fkpy.taup.first_arrival_takeoff_deg` (simplified ports of
    Lupei Zhu's ``tau_p.f``) that pick the minimum of the direct ray
    and the head-wave refraction along the immediate sub-source
    interface.
    """
    from .taup import first_arrival_takeoff_deg, first_arrival_time

    t0p = first_arrival_time(
        model.vp_kms, model.thickness_km, src_layer, rcv_layer, distances_km
    )
    t0s = first_arrival_time(
        model.vs_kms, model.thickness_km, src_layer, rcv_layer, distances_km
    )
    take_p = first_arrival_takeoff_deg(
        model.vp_kms, model.thickness_km, src_layer, rcv_layer, distances_km
    )
    take_s = first_arrival_takeoff_deg(
        model.vs_kms, model.thickness_km, src_layer, rcv_layer, distances_km
    )
    return t0p, t0s, take_p, take_s


def _vertical_separation(
    model: LayeredModel, src_layer: int, rcv_layer: int
) -> float:
    if src_layer == rcv_layer:
        return 0.0
    lo, hi = sorted((src_layer, rcv_layer))
    return float(np.sum(model.thickness_km[lo:hi]))


# ----------------------------------------------------------------------
# Source jump for the 10 component output
# ----------------------------------------------------------------------
def _build_three_jumps(
    model: LayeredModel, src_layer: int, flip: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return source jumps for the double-couple basis (DC) and
    explosion (EX), already cast to complex128.

    The 10-component output is assembled from these two runs:
    DC contributes 8 components (n=0,1,2 × Z,R,T minus the n=2 trivially
    zero component), EX contributes Z and R for n=0.
    """
    xi = float(model.xi[src_layer])
    mu = float(model.mu_gpa[src_layer])
    s_dc = source_jump(SourceType.DOUBLE_COUPLE, xi=xi, mu=mu, flip=flip)
    s_ex = source_jump(SourceType.EXPLOSION, xi=xi, mu=mu, flip=flip)
    return s_dc.astype(np.complex128), s_ex.astype(np.complex128)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def compute_greens(  # noqa: PLR0913, PLR0915
    model: LayeredModel,
    *,
    src_depth_km: float,
    rcv_depth_km: float = 0.0,
    distances_km: F64Array | Sequence[float],
    npts: int,
    dt: float,
    sigma: float = SIGMA_DEFAULT,
    pmin: float = PMIN_DEFAULT,
    pmax: float = PMAX_DEFAULT,
    dk: float = DK_DEFAULT,
    kmax: float = KMAX_DEFAULT,
    taper: float = TAPER_DEFAULT,
    samples_before_p: int = SAMPLES_BEFORE_P_DEFAULT,
    n_workers: int | None = None,
    backend: str | None = None,
    updn: int = 0,
    t0_s: F64Array | float | None = None,
    hipass: tuple[int, int] | None = None,
) -> GreensResult:
    """Compute the 10-component Green's functions for a layered model.

    Parameters
    ----------
    model
        :class:`LayeredModel` describing the half-space.
    src_depth_km, rcv_depth_km
        Source and receiver depths from the free surface (km).
    distances_km
        1-D array of horizontal distances in km.
    npts, dt
        Time-domain length and sampling interval (s).
    sigma
        Imaginary frequency shift in cycles per trace.  The complex
        angular frequency is ``ω̃ = ω − iσ`` with
        ``σ = sigma · dω / (2π)`` (Bouchon 1981).  In the time domain
        this multiplies the response by ``exp(−σ·t)``; we compensate
        post-IFFT (Graves 2000).  Default :data:`fkpy.constants.SIGMA_DEFAULT`
        = 2 cycles per trace.
    pmin, pmax
        Slowness window in units of ``1 / v_s_at_source``.
        Defaults: ``pmin=0`` (full body-wave coverage),
        ``pmax=1`` (cap at the S-wave slowness).
    dk
        Wavenumber sampling, in units of ``π / max(x_max, h_s)``.
        Default 0.3 (Bouchon 1981 condition (2) is checked at
        runtime; a WARNING is emitted if violated).
    kmax
        Maximum wavenumber at ``ω=0`` in units of ``1 / h_s``.
        Default 15 — the kernel decays as ``exp(−k·h_s)`` so this
        gives < 1e-7 contribution from the truncation.
    taper
        Cosine roll-off fraction below f_Nyq (default 0.3).
    samples_before_p
        Number of trace samples reserved before the first arrival
        (default 50).
    n_workers
        Process-pool worker count.  ``None`` ⇒ ``os.cpu_count()``.
    backend
        ``"numpy"`` (default, Numba CPU) or ``"jax"`` (JAX backend).
    updn
        ``0`` whole field, ``+1`` down-going only, ``-1`` up-going only.
    t0_s
        Optional override for the per-distance first-arrival time used
        as the time origin.  Either a scalar or an array of length
        ``len(distances_km)``.  When ``None`` the value is computed by
        :mod:`fkpy.taup`.
    hipass
        Optional ``(wc1, wc2)`` integers defining a low-frequency
        cosine taper (zero below ``(wc1-1)·dω``, full above
        ``(wc2-1)·dω``).  ``None`` ⇒ no high-pass.

    Returns
    -------
    GreensResult
        ``gf`` shape ``(n_dist, 10, npts)``; component order matches
        :class:`fkpy.source.ZhuBasis`.

    See also
    --------
    fkpy.constants : default values cited above with paper + equation.
    """
    backend_name = current_backend(backend)
    if backend_name == "jax":
        from .jax_backend.greens_jx import compute_greens_jax

        return compute_greens_jax(
            model=model,
            src_depth_km=src_depth_km,
            rcv_depth_km=rcv_depth_km,
            distances_km=np.asarray(distances_km, dtype=np.float64),
            npts=npts,
            dt=dt,
            sigma=sigma,
            pmin=pmin,
            pmax=pmax,
            dk=dk,
            kmax=kmax,
            taper=taper,
            samples_before_p=samples_before_p,
            updn=updn,
            t0_s=t0_s,
            hipass=hipass,
        )

    distances_km = np.asarray(distances_km, dtype=np.float64).ravel()
    if distances_km.ndim != 1 or distances_km.size == 0:
        msg = "distances_km must be a 1-D array with at least one entry"
        raise ValueError(msg)
    if np.any(distances_km <= 0):
        msg = "distances_km must be strictly positive (use a small value to mimic 0)"
        raise ValueError(msg)
    if npts < 2:
        raise ValueError("npts must be ≥ 2")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    # --- Insert source / receiver interfaces and identify layer indices.
    # For now we require src_depth >= rcv_depth (the "source below receiver"
    # case).  The opposite case is handled by flipping the model.
    flip = 1
    if src_depth_km < rcv_depth_km:
        # Flip the model upside-down so the source is below the receiver.
        flipped = LayeredModel.from_array(model.to_array()[::-1].copy())
        # After the flip, surface depths swap; transform.
        model = flipped
        new_src = _vertical_separation(model, 0, model.n_layers - 1) - src_depth_km
        new_rcv = _vertical_separation(model, 0, model.n_layers - 1) - rcv_depth_km
        src_depth_km, rcv_depth_km = new_src, new_rcv
        flip = -1

    # Insert receiver first then source (so the indices we get for src
    # remain valid even if both fall in the same layer).
    model_rcv, rcv_layer = model.insert_interface(rcv_depth_km)
    model_full, src_layer = model_rcv.insert_interface(src_depth_km)
    if src_layer < rcv_layer:
        # Interface insertion order swapped indices; correct.
        src_layer, rcv_layer = rcv_layer, src_layer

    logger.debug(
        "src_layer=%d rcv_layer=%d n_layers=%d",
        src_layer,
        rcv_layer,
        model_full.n_layers,
    )

    # --- Build per-layer arrays.
    mu_arr = model_full.mu_gpa.astype(np.float64)
    thickness = model_full.thickness_km.astype(np.float64)
    thickness[-1] = 0.0  # halfspace marker
    vp = model_full.vp_kms.astype(np.float64)
    vs = model_full.vs_kms.astype(np.float64)
    qp = model_full.qp.astype(np.float64)
    qs = model_full.qs.astype(np.float64)

    # --- Wavenumber-integration grid.
    hs = max(_vertical_separation(model_full, src_layer, rcv_layer), H_SOURCE_RECEIVER_FLOOR_KM)
    xmax = max(float(np.max(distances_km)), hs)
    dk_per_km = dk * np.pi / xmax
    kc_per_km = kmax / hs
    vs_src = float(model_full.vs_kms[src_layer])
    pmin_per_kms = pmin / vs_src
    pmax_per_kms = pmax / vs_src

    # --- Bouchon (1981) condition (2): dk · x_max ≤ π · 0.5
    # gives 4 samples per Bessel period at distance x_max.  Lupei Zhu's
    # `fk.f` checks the related condition
    #     dk_norm ≤ 0.5 / (1 + sqrt((v_max·t_max)² − h_s²) / x_max)
    # which guarantees no wrap-around for a trace of duration t_max.
    vmax = float(np.max(model_full.vp_kms))
    t_max = npts * dt
    inner = (vmax * t_max) ** 2 - hs**2
    if inner > 0:
        dk_max = 0.5 / (1.0 + np.sqrt(inner) / xmax)
        if dk > dk_max:
            logger.warning(
                "dk=%.4f exceeds Bouchon (1981) condition (2) limit %.4f "
                "for v_max=%.2f km/s, t_max=%.1f s, x_max=%.1f km. "
                "Time-domain wrap-around may contaminate the trace; "
                "reduce dk or shorten npts.",
                dk, dk_max, vmax, t_max, xmax,
            )

    # --- Frequency grid (positive frequencies only; rfft layout).
    nfft2 = npts // 2
    dw = TWO_PI / (npts * dt)
    sigma_rad = sigma * dw / TWO_PI
    wc = max(int(nfft2 * (1.0 - taper)), 1)
    taper_rad = np.pi / (nfft2 - wc + 1)
    if hipass is None:
        wc1, wc2 = 1, 1
    else:
        wc1, wc2 = int(hipass[0]), int(hipass[1])
    if wc2 > wc:
        wc2 = wc
    if wc1 > wc2:
        wc1 = wc2

    # --- First-arrival approximations (used only as time alignment).
    t0_p_raw, t0_s_raw, take_p, take_s = _approximate_first_arrivals(
        model_full, src_layer, rcv_layer, distances_km
    )
    if t0_s is None:
        t0_first_arrival = t0_p_raw
    elif np.isscalar(t0_s):
        t0_first_arrival = np.full(
            distances_km.shape, float(t0_s),  # type: ignore[arg-type]
            dtype=np.float64,
        )
    else:
        t0_first_arrival = np.asarray(t0_s, dtype=np.float64)
    # Per Lupei Zhu's `fk.f`: t0_offset = t0 - tb*dt, the absolute time
    # represented by sample 0 of the output trace.  This is also the SAC
    # `b` header.
    t0_offset = np.maximum(t0_first_arrival - samples_before_p * dt, 0.0)

    filter_const = dk_per_km / TWO_PI

    # --- Build source jumps for the two runs (DC and EX).
    s_dc, s_ex = _build_three_jumps(model_full, src_layer, flip)

    # --- Run wavenumber integration twice (DC then EX).
    sum_dc = run_omega_loop(
        FrequencyJobConfig(
            distances_km=distances_km,
            vp_kms=vp,
            vs_kms=vs,
            qp=qp,
            qs=qs,
            mu=mu_arr,
            thickness_km=thickness,
            s_input=s_dc,
            src_layer=src_layer,
            rcv_layer=rcv_layer,
            src_type=int(SourceType.DOUBLE_COUPLE),
            updn=updn,
            flip=flip,
            sigma_rad_s=sigma_rad,
            pmin_per_kms=pmin_per_kms,
            pmax_per_kms=pmax_per_kms,
            dk_per_km=dk_per_km,
            kc_per_km=kc_per_km,
            nfft2=nfft2,
            dw_rad_s=dw,
            wc1=wc1,
            wc2=wc2,
            wc=wc,
            taper=taper_rad,
            filter_const=filter_const,
            t0_s=t0_offset,
        ),
        n_workers=n_workers,
    )
    sum_ex = run_omega_loop(
        FrequencyJobConfig(
            distances_km=distances_km,
            vp_kms=vp,
            vs_kms=vs,
            qp=qp,
            qs=qs,
            mu=mu_arr,
            thickness_km=thickness,
            s_input=s_ex,
            src_layer=src_layer,
            rcv_layer=rcv_layer,
            src_type=int(SourceType.EXPLOSION),
            updn=updn,
            flip=flip,
            sigma_rad_s=sigma_rad,
            pmin_per_kms=pmin_per_kms,
            pmax_per_kms=pmax_per_kms,
            dk_per_km=dk_per_km,
            kc_per_km=kc_per_km,
            nfft2=nfft2,
            dw_rad_s=dw,
            wc1=wc1,
            wc2=wc2,
            wc=wc,
            taper=taper_rad,
            filter_const=filter_const,
            t0_s=t0_offset,
        ),
        n_workers=n_workers,
    )

    # --- Inverse FFT both runs.
    traces_dc = inverse_transform(sum_dc, dt=dt, sigma_rad_s=sigma_rad, t0_s=t0_offset, npts=npts)
    traces_ex = inverse_transform(sum_ex, dt=dt, sigma_rad_s=sigma_rad, t0_s=t0_offset, npts=npts)

    # Pack into the 10-component output.
    n_dist = distances_km.size
    gf = np.zeros((n_dist, N_GREEN_COMPONENTS, npts), dtype=np.float64)
    # DC: components 0..8 = (Z, R, T) for n = 0, 1, 2
    gf[:, ZhuBasis.DD_Z, :] = traces_dc[:, 0, :]
    gf[:, ZhuBasis.DD_R, :] = traces_dc[:, 1, :]
    # n=0 transverse (T) is identically zero for DC
    gf[:, ZhuBasis.DS_Z, :] = traces_dc[:, 3, :]
    gf[:, ZhuBasis.DS_R, :] = traces_dc[:, 4, :]
    gf[:, ZhuBasis.DS_T, :] = traces_dc[:, 5, :]
    gf[:, ZhuBasis.SS_Z, :] = traces_dc[:, 6, :]
    gf[:, ZhuBasis.SS_R, :] = traces_dc[:, 7, :]
    gf[:, ZhuBasis.SS_T, :] = traces_dc[:, 8, :]
    # EX: only n=0 components are nonzero; transverse identically zero.
    gf[:, ZhuBasis.EX_Z, :] = traces_ex[:, 0, :]
    gf[:, ZhuBasis.EX_R, :] = traces_ex[:, 1, :]

    meta: dict[str, Any] = {
        "src_depth_km": src_depth_km,
        "rcv_depth_km": rcv_depth_km,
        "distances_km": distances_km,
        "npts": npts,
        "dt": dt,
        "sigma": sigma,
        "pmin": pmin,
        "pmax": pmax,
        "dk": dk,
        "kmax": kmax,
        "taper": taper,
        "samples_before_p": samples_before_p,
        "updn": updn,
        "flip": flip,
        "src_layer": src_layer,
        "rcv_layer": rcv_layer,
        "model_array": model_full.to_array(),
    }

    return GreensResult(
        gf=gf,
        distances_km=distances_km,
        dt=dt,
        t0_p=t0_p_raw,
        t0_s=t0_s_raw,
        npts=npts,
        src_depth_km=src_depth_km,
        rcv_depth_km=rcv_depth_km,
        p_takeoff_deg=take_p,
        s_takeoff_deg=take_s,
        meta=meta,
    )


def compute_single_force_greens(  # noqa: PLR0913
    model: LayeredModel,
    *,
    src_depth_km: float,
    rcv_depth_km: float = 0.0,
    distances_km: F64Array | Sequence[float],
    npts: int,
    dt: float,
    sigma: float = SIGMA_DEFAULT,
    pmin: float = PMIN_DEFAULT,
    pmax: float = PMAX_DEFAULT,
    dk: float = DK_DEFAULT,
    kmax: float = KMAX_DEFAULT,
    taper: float = TAPER_DEFAULT,
    samples_before_p: int = SAMPLES_BEFORE_P_DEFAULT,
    n_workers: int | None = None,
    updn: int = 0,
    t0_s: F64Array | float | None = None,
    hipass: tuple[int, int] | None = None,
) -> F64Array:
    """Compute the 6 single-force Green's functions per receiver.

    A single force has only ``n = 0`` (vertical force) and ``n = 1``
    (horizontal force) azimuthal modes — no ``n = 2``.

    Parameters
    ----------
    model
        :class:`LayeredModel` describing the half-space.
    src_depth_km, rcv_depth_km
        Source and receiver depths from the free surface (km).
    distances_km
        1-D array of horizontal distances in km.
    npts, dt
        Time-domain length and sampling interval (s).
    sigma, pmin, pmax, dk, kmax, taper, samples_before_p
        Numerical-integration parameters; see
        :func:`fkpy.greens.compute_greens` for full descriptions.
    n_workers, updn, t0_s, hipass
        See :func:`fkpy.greens.compute_greens`.

    Returns
    -------
    out : (n_dist, 6, npts) float64
        Axis-1 ordering follows Lupei Zhu's `fk` SF convention:
        ``[Z0, R0, T0, Z1, R1, T1]``.

    See also
    --------
    compute_greens : the canonical 10-component output for moment
        tensors.
    """
    distances_km = np.asarray(distances_km, dtype=np.float64).ravel()
    flip = 1
    model_rcv, rcv_layer = model.insert_interface(rcv_depth_km)
    model_full, src_layer = model_rcv.insert_interface(src_depth_km)
    if src_layer < rcv_layer:
        src_layer, rcv_layer = rcv_layer, src_layer

    mu_arr = model_full.mu_gpa.astype(np.float64)
    thickness = model_full.thickness_km.astype(np.float64)
    thickness[-1] = 0.0
    vp = model_full.vp_kms.astype(np.float64)
    vs = model_full.vs_kms.astype(np.float64)
    qp = model_full.qp.astype(np.float64)
    qs = model_full.qs.astype(np.float64)

    hs = max(_vertical_separation(model_full, src_layer, rcv_layer), H_SOURCE_RECEIVER_FLOOR_KM)
    xmax = max(float(np.max(distances_km)), hs)
    dk_per_km = dk * np.pi / xmax
    kc_per_km = kmax / hs
    vs_src = float(model_full.vs_kms[src_layer])
    pmin_per_kms = pmin / vs_src
    pmax_per_kms = pmax / vs_src

    nfft2 = npts // 2
    dw = TWO_PI / (npts * dt)
    sigma_rad = sigma * dw / TWO_PI
    wc = max(int(nfft2 * (1.0 - taper)), 1)
    taper_rad = np.pi / (nfft2 - wc + 1)
    if hipass is None:
        wc1, wc2 = 1, 1
    else:
        wc1, wc2 = int(hipass[0]), int(hipass[1])
    if wc2 > wc:
        wc2 = wc
    if wc1 > wc2:
        wc1 = wc2

    t0_p_raw, _, _, _ = _approximate_first_arrivals(
        model_full, src_layer, rcv_layer, distances_km
    )
    if t0_s is None:
        t0_first_arrival = t0_p_raw
    elif np.isscalar(t0_s):
        t0_first_arrival = np.full(distances_km.shape, float(t0_s))  # type: ignore[arg-type]
    else:
        t0_first_arrival = np.asarray(t0_s, dtype=np.float64)
    t0_offset = np.maximum(t0_first_arrival - samples_before_p * dt, 0.0)

    s_sf = source_jump(SourceType.SINGLE_FORCE, xi=float(model_full.xi[src_layer]),
                       mu=float(model_full.mu_gpa[src_layer]), flip=flip).astype(np.complex128)
    sum_sf = run_omega_loop(
        FrequencyJobConfig(
            distances_km=distances_km, vp_kms=vp, vs_kms=vs, qp=qp, qs=qs,
            mu=mu_arr, thickness_km=thickness, s_input=s_sf,
            src_layer=src_layer, rcv_layer=rcv_layer,
            src_type=int(SourceType.SINGLE_FORCE),
            updn=updn, flip=flip,
            sigma_rad_s=sigma_rad, pmin_per_kms=pmin_per_kms,
            pmax_per_kms=pmax_per_kms, dk_per_km=dk_per_km, kc_per_km=kc_per_km,
            nfft2=nfft2, dw_rad_s=dw, wc1=wc1, wc2=wc2, wc=wc,
            taper=taper_rad, filter_const=dk_per_km / TWO_PI, t0_s=t0_offset,
        ),
        n_workers=n_workers,
    )
    traces = inverse_transform(sum_sf, dt=dt, sigma_rad_s=sigma_rad,
                               t0_s=t0_offset, npts=npts)
    return np.asarray(traces[:, :6, :], dtype=np.float64)


__all__ = ["GreensResult", "compute_greens", "compute_single_force_greens"]
