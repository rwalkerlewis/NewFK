"""Command-line entry point.

::

    fkpy compute --model model.nd --depth 8 \\
                 --distances 50,100,150 \\
                 --npts 2048 --dt 0.1 --out greens.h5
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import h5py
import numpy as np

from . import __version__
from ._logging import logger
from .greens import compute_greens
from .model import LayeredModel


@click.group(help="fkpy command-line interface.")
@click.option("-v", "--verbose", count=True, help="Increase log verbosity.")
def main(verbose: int) -> None:
    level = logging.WARNING - 10 * verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(name)s %(levelname)s %(message)s",
    )


@main.command("compute", help="Compute Green's functions and save to HDF5 (and optionally SAC).")
@click.option("--model", "model_path", required=True, type=click.Path(path_type=Path),
              help="Layered model text file.")
@click.option("--model-format", "model_format", default="fkpy",
              type=click.Choice(["fkpy", "pyfk"]),
              help="Column ordering for --model.  'fkpy' (default): "
                   "thickness vp vs rho Qp Qs; 'pyfk': thickness vs vp rho Qs Qp.")
@click.option("--depth", "src_depth_km", required=True, type=float, help="Source depth in km.")
@click.option("--rdep", "rcv_depth_km", default=0.0, type=float, help="Receiver depth in km.")
@click.option("--distances", "distances", required=True, type=str,
              help="Comma-separated list of receiver distances in km.")
@click.option("--npts", required=True, type=int, help="Number of time samples.")
@click.option("--dt", required=True, type=float, help="Sampling interval in seconds.")
@click.option("--sigma", default=2.0, type=float,
              help="Imaginary frequency shift (cycles per trace).")
@click.option("--dk", default=0.3, type=float, help="Wavenumber sampling, in pi/x_max units.")
@click.option("--kmax", default=15.0, type=float, help="kmax in 1/h_s units.")
@click.option("--workers", "n_workers", default=None, type=int,
              help="Worker processes for the omega-loop (defaults to CPU count).")
@click.option("--backend", default=None, type=click.Choice(["numpy", "jax"]),
              help="Override FK_BACKEND environment variable.")
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path),
              help="HDF5 output file.")
@click.option("--sac-prefix", "sac_prefix", default=None, type=str,
              help="Optional SAC output filename prefix.")
@click.option("--azimuth", default=0.0, type=float,
              help="Receiver azimuth in degrees (for SAC headers).")
def compute(  # noqa: PLR0913
    model_path: Path,
    model_format: str,
    src_depth_km: float,
    rcv_depth_km: float,
    distances: str,
    npts: int,
    dt: float,
    sigma: float,
    dk: float,
    kmax: float,
    n_workers: int | None,
    backend: str | None,
    out_path: Path,
    sac_prefix: str | None,
    azimuth: float,
) -> None:
    distances_km = np.array([float(x) for x in distances.split(",") if x.strip()], dtype=np.float64)
    model = LayeredModel.from_text(model_path, format=model_format)
    logger.info(
        "Computing GFs for %d distances, npts=%d, dt=%g, src_depth=%g km",
        distances_km.size,
        npts,
        dt,
        src_depth_km,
    )
    result = compute_greens(
        model=model,
        src_depth_km=src_depth_km,
        rcv_depth_km=rcv_depth_km,
        distances_km=distances_km,
        npts=npts,
        dt=dt,
        sigma=sigma,
        dk=dk,
        kmax=kmax,
        n_workers=n_workers,
        backend=backend,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        ds = h5.create_dataset(
            "gf", data=result.gf, compression="gzip", compression_opts=4
        )
        # Record every input parameter for reproducibility.
        ds.attrs["dt"] = result.dt
        ds.attrs["npts"] = result.npts
        ds.attrs["src_depth_km"] = result.src_depth_km
        ds.attrs["rcv_depth_km"] = result.rcv_depth_km
        ds.attrs["sigma"] = sigma
        ds.attrs["dk"] = dk
        ds.attrs["kmax"] = kmax
        ds.attrs["model_path"] = str(model_path)
        ds.attrs["model_format"] = model_format
        ds.attrs["fkpy_version"] = __version__
        ds.attrs["component_order"] = (
            "DD_Z DD_R DS_Z DS_R DS_T SS_Z SS_R SS_T EX_Z EX_R"
        )
        h5.create_dataset("distances_km", data=result.distances_km)
        h5.create_dataset("t0_p", data=result.t0_p)
        h5.create_dataset("t0_s", data=result.t0_s)
        h5.create_dataset("p_takeoff_deg", data=result.p_takeoff_deg)
        h5.create_dataset("s_takeoff_deg", data=result.s_takeoff_deg)
        h5.create_dataset("model", data=result.meta["model_array"])
    logger.info("Wrote %s", out_path)
    if sac_prefix:
        files = result.write_sac(prefix=sac_prefix, azimuth_deg=azimuth)
        logger.info("Wrote %d SAC files", len(files))


@main.command("bench", help="Run the canonical 5-layer benchmark and report wall time.")
@click.option("--workers", "n_workers", default=1, type=int,
              help="Worker processes for the omega-loop (default 1 = serial).")
def bench(n_workers: int) -> None:
    from .benchmarks.canonical_zhu5 import run

    elapsed = run(workers=n_workers)
    click.echo(f"{elapsed:.2f} s with {n_workers} worker(s)")


if __name__ == "__main__":  # pragma: no cover
    main()
