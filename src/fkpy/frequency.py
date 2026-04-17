"""Frequency-loop dispatcher.

The discrete-wavenumber sum is independent across positive frequencies,
so we farm chunks of frequency indices to a
:class:`concurrent.futures.ProcessPoolExecutor` (per the plan).

The complex frequency shift used here is documented in detail in the
module docstring of :mod:`fkpy.transform`; a short reminder:

.. code-block::

    σ = sigma_input · dω / (2π)            [rad/s]
    ω̃ = ω − iσ                             (Bouchon 1981)
    ⇒ time-domain trace is multiplied by exp(−σ·t).
    Compensate later with exp(+σ·t) per sample (Graves 2000 fix).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from ._logging import logger
from ._typing import C128Array, F64Array
from .attenuation import complex_wavenumber_squared
from .integrate import precompute_bessel, wavenumber_sum_loop_tabulated


@dataclass(frozen=True)
class FrequencyJobConfig:
    """All ω-independent inputs to the per-frequency loop."""

    distances_km: F64Array
    vp_kms: F64Array
    vs_kms: F64Array
    qp: F64Array
    qs: F64Array
    mu: F64Array
    thickness_km: F64Array
    s_input: C128Array
    src_layer: int
    rcv_layer: int
    src_type: int
    updn: int
    flip: int
    sigma_rad_s: float
    pmin_per_kms: float
    pmax_per_kms: float
    dk_per_km: float
    kc_per_km: float
    nfft2: int
    dw_rad_s: float
    wc1: int
    wc2: int
    wc: int
    taper: float
    filter_const: float
    t0_s: F64Array


def _process_chunk(
    cfg: FrequencyJobConfig,
    j_indices: list[int],
    bessel_global: np.ndarray,
    k0_index_offset: int,
) -> tuple[list[int], C128Array]:
    """Compute the wavenumber sum for each frequency index in the chunk."""
    n_dist = cfg.distances_km.size
    out = np.zeros((len(j_indices), n_dist, 9), dtype=np.complex128)
    for ii, j in enumerate(j_indices):
        omega = j * cfg.dw_rad_s
        if omega == 0:
            continue
        w = complex(omega, -cfg.sigma_rad_s)
        kp_sq = complex_wavenumber_squared(w, cfg.vp_kms, cfg.qp)
        ks_sq = complex_wavenumber_squared(w, cfg.vs_kms, cfg.qs)
        # Wavenumber range for this ω (ZR fk.f convention)
        # When pmin=0, k0 is identical for every ω, so we can index
        # straight into the precomputed global Bessel table.  When
        # pmin>0, the k_array per ω starts at (omega*pmin + dk/2).
        k0_extra = omega * cfg.pmin_per_kms  # offset *above* the 0.5*dk start
        i_start = int(round(k0_extra / cfg.dk_per_km)) + k0_index_offset
        n_k_max = int(
            (np.sqrt(cfg.kc_per_km**2 + (cfg.pmax_per_kms * omega) ** 2)
             - (k0_extra + 0.5 * cfg.dk_per_km))
            / cfg.dk_per_km
        )
        n_k = max(0, min(n_k_max, bessel_global.shape[0] - i_start))
        if n_k <= 0:
            continue
        k_slice = bessel_global[i_start : i_start + n_k]
        # Compose actual k_array for the kernel (keep exact arithmetic)
        k_array = (
            (k0_extra + 0.5 * cfg.dk_per_km)
            + np.arange(n_k, dtype=np.float64) * cfg.dk_per_km
        )
        partial = wavenumber_sum_loop_tabulated(
            k_array,
            k_slice,
            cfg.distances_km,
            kp_sq,
            ks_sq,
            cfg.mu,
            cfg.thickness_km,
            cfg.s_input,
            cfg.src_layer,
            cfg.rcv_layer,
            cfg.src_type,
            cfg.updn,
            cfg.flip,
        )
        # Apply per-frequency filter & time-shift
        filt = cfg.filter_const
        if j > cfg.wc:
            filt *= 0.5 * (1.0 + np.cos((j - cfg.wc) * cfg.taper))
        if j < cfg.wc2:
            denom = max(cfg.wc2 - cfg.wc1, 1)
            filt *= 0.5 * (1.0 + np.cos((cfg.wc2 - j) * np.pi / denom))
        for irec in range(n_dist):
            phi = omega * cfg.t0_s[irec]
            shift = filt * complex(np.cos(phi), np.sin(phi))
            partial[irec, :] *= shift
        out[ii] = partial
    return j_indices, out


def run_omega_loop(
    cfg: FrequencyJobConfig,
    n_workers: int | None = None,
) -> C128Array:
    """Return ``sum_freq[i_dist, i_comp, j]`` for ``j = 0..nfft2``.

    Uses :class:`concurrent.futures.ProcessPoolExecutor` if
    ``n_workers > 1``; otherwise runs serially in the calling process
    (useful for tests and small problems where the fork overhead
    dominates).
    """
    n_dist = cfg.distances_km.size
    nfft2 = cfg.nfft2
    out = np.zeros((n_dist, 9, nfft2 + 1), dtype=np.complex128)

    j_indices = list(range(max(cfg.wc1, 1), nfft2 + 1))
    if not j_indices:
        return out

    # --- Precompute the global Bessel table once.
    omega_max = nfft2 * cfg.dw_rad_s
    k_max_global = (
        np.sqrt(cfg.kc_per_km**2 + (cfg.pmax_per_kms * omega_max) ** 2)
    )
    n_k_global = int(np.ceil((k_max_global - 0.5 * cfg.dk_per_km) / cfg.dk_per_km)) + 4
    k_global = (0.5 * cfg.dk_per_km) + np.arange(n_k_global, dtype=np.float64) * cfg.dk_per_km
    bessel_global = precompute_bessel(k_global, cfg.distances_km)

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    if n_workers <= 1 or len(j_indices) < 4:
        logger.debug("Serial ω-loop, %d frequencies", len(j_indices))
        idx, partial = _process_chunk(cfg, j_indices, bessel_global, 0)
        for ii, j in enumerate(idx):
            out[:, :, j] = partial[ii]
        return out

    chunk_size = max(1, len(j_indices) // (4 * n_workers))
    chunks = [j_indices[i : i + chunk_size] for i in range(0, len(j_indices), chunk_size)]
    logger.debug(
        "Parallel ω-loop with %d workers, %d frequencies, %d chunks",
        n_workers,
        len(j_indices),
        len(chunks),
    )
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_process_chunk, cfg, c, bessel_global, 0) for c in chunks]
        for f in as_completed(futures):
            idx, partial = f.result()
            for ii, j in enumerate(idx):
                out[:, :, j] = partial[ii]
    return out


__all__ = ["FrequencyJobConfig", "run_omega_loop"]
