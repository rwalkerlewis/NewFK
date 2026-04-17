"""SAC and ObsPy I/O.

Mirrors the SAC headers Lupei Zhu's `fk` writes for its Green's
functions (filename suffix conventions and key headers ``kstnm, dist,
az, baz, evdp, stel, b, e, o, delta, t1, t2``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .source import N_GREEN_COMPONENTS, ZhuBasis

if TYPE_CHECKING:  # pragma: no cover
    from obspy import Stream

    from .greens import GreensResult


# Ordering matches Lupei Zhu's filename suffixes (0..8, a, b).
_FK_SUFFIXES: tuple[str, ...] = (
    "0",  # DD_Z      n=0 vertical
    "1",  # DD_R      n=0 radial
    "2",  # DS_Z      n=1 vertical  (matches fk syn/  but our DS_T is index 4)
    "3",
    "4",
    "5",
    "6",
    "7",
    "a",  # EX_Z
    "b",  # EX_R
)


def to_obspy_stream(
    result: GreensResult,
    *,
    azimuth_deg: float = 0.0,
    kstnm: str = "STA",
    network: str = "XX",
) -> Stream:
    """Convert a :class:`GreensResult` to an ObsPy ``Stream``."""
    from obspy import Stream, Trace

    stream = Stream()
    n_dist = result.distances_km.size
    for irec in range(n_dist):
        for icomp in range(N_GREEN_COMPONENTS):
            data = result.gf[irec, icomp].astype("float32")
            stats = {
                "delta": result.dt,
                "starttime": 0.0,
                "network": network,
                "station": kstnm,
                "channel": _channel_for(icomp),
                "sac": {
                    "delta": result.dt,
                    "b": 0.0,
                    "e": (result.npts - 1) * result.dt,
                    "o": 0.0,
                    "kstnm": kstnm,
                    "kcmpnm": _channel_for(icomp),
                    "dist": float(result.distances_km[irec]),
                    "az": float(azimuth_deg),
                    "baz": float((azimuth_deg + 180.0) % 360.0),
                    "evdp": float(result.src_depth_km),
                    "stel": float(-result.rcv_depth_km),
                    "t1": float(result.t0_p[irec]),
                    "t2": float(result.t0_s[irec]),
                },
            }
            tr = Trace(data=data)
            tr.stats.delta = result.dt
            tr.stats.network = network
            tr.stats.station = kstnm
            tr.stats.channel = _channel_for(icomp)
            tr.stats.sac = stats["sac"]
            stream += tr
    return stream


def write_sac(
    result: GreensResult,
    *,
    prefix: str,
    azimuth_deg: float = 0.0,
    kstnm: str = "STA",
) -> list[str]:
    """Write SAC files following Lupei Zhu's `fk` filename convention.

    File names: ``{prefix}.{dist:.0f}.{suffix}.sac`` where ``suffix`` is
    one of ``0,1,...,8,a,b``.
    """
    from obspy import Trace

    stream = to_obspy_stream(result, azimuth_deg=azimuth_deg, kstnm=kstnm)
    written: list[str] = []
    n_dist = result.distances_km.size
    it = iter(stream)
    for irec in range(n_dist):
        for icomp in range(N_GREEN_COMPONENTS):
            tr: Trace = next(it)
            suffix = _FK_SUFFIXES[icomp]
            d = float(result.distances_km[irec])
            fname = f"{prefix}.{d:g}.{suffix}.sac"
            tr.write(fname, format="SAC")
            written.append(fname)
    return written


def _channel_for(icomp: int) -> str:
    """Return a 3-character SAC channel label for a Green's function index."""
    name = ZhuBasis(icomp).name  # e.g. "DD_Z", "EX_R"
    return name.replace("_", "")[:3]


__all__ = ["to_obspy_stream", "write_sac"]
