# fkpy

Frequency-wavenumber synthetic seismograms for a layered elastic half-space.

`fkpy` is a Python port of Lupei Zhu's `fk` Fortran package
([Zhu & Rivera, GJI 2002](https://doi.org/10.1046/j.1365-246X.2002.01610.x))
intended for moment-tensor inversion (`mtinvert`) and explosion Green's
functions for regional nuclear monitoring.

## Highlights

- Discrete wavenumber integration (Bouchon, BSSA 1981, eq. 9), with a
  complex frequency shift `w = ω − iσ` to suppress time-domain wrap-around.
- **Kennett–Kerry reflection/transmission matrix** recursion (Kennett 1983,
  §6.3) — unconditionally stable for thick layers at high frequency.
  No Haskell `cosh(ν·d)` overflow gymnastics.
- Causal Futterman attenuation (Aki & Richards 2002, p. 182) with
  `Q_ref = 1 Hz`.
- Numba-jitted hot loops (propagator, kernel, wavenumber summation);
  `concurrent.futures.ProcessPoolExecutor` parallelism over frequency.
- Optional JAX backend behind `FK_BACKEND=jax` for GPU batch runs over
  many source depths.
- SAC writer matching Zhu's `fk` header conventions.

## Install

```bash
uv pip install fkpy
# or, for development:
git clone <this repo>
cd fkpy
uv venv
uv pip install -e .[dev]
```

Python 3.11+. Numba is a hard dependency; install will fail loudly without
it (no silent fall-back to pure-Python).

## One-line example

```python
from fkpy import LayeredModel, compute_greens

model = LayeredModel.from_text("examples/zhu5.model")
gf = compute_greens(model, src_depth_km=8.0,
                    distances_km=[50, 100, 150],
                    npts=2048, dt=0.1)
gf.to_obspy_stream(azimuth_deg=42.0).write("synth.mseed", format="MSEED")
```

## CLI

```bash
fkpy compute --model model.nd --depth 8 \
             --distances 50,100,150 \
             --npts 2048 --dt 0.1 --out greens.h5
```

`greens.h5` contains a single dataset `gf` of shape `(n_dist, 10, npts)`,
with attribute metadata recording every input parameter.

The 10 Green's-function components per distance are, in order:

| Index | Component | Description |
|------:|:---------:|-------------|
| 0 | DD_z | 45° dip-slip, vertical |
| 1 | DD_r | 45° dip-slip, radial |
| 2 | DS_z | vertical dip-slip, vertical |
| 3 | DS_r | vertical dip-slip, radial |
| 4 | DS_t | vertical dip-slip, transverse |
| 5 | SS_z | vertical strike-slip, vertical |
| 6 | SS_r | vertical strike-slip, radial |
| 7 | SS_t | vertical strike-slip, transverse |
| 8 | EX_z | explosion, vertical |
| 9 | EX_r | explosion, radial |

This mirrors Zhu's `fk2mt` consumption order.

## Benchmark

Canonical 5-layer crustal model, source at 8 km, 10 receiver distances
from 10 to 200 km, 2048 samples at dt = 0.1 s, run on a 4-core x86-64
cloud VM (Linux 6.1):

| Implementation | wall clock | notes |
|---|---|---|
| Lupei Zhu Fortran `fk` (1 core, `gfortran -O`) | **1.8 s** | reference |
| `fkpy` numba (1 core, warm) | **4.6 s** | meets the 8 s target |
| `fkpy` `ProcessPoolExecutor` (4 cores) | **1.4 s** | beats Fortran `fk` |
| `fkpy` JAX backend (CPU float32) | currently equal to numpy backend; native JAX kernels planned for v0.2 |

Reproduce on your machine:

```bash
python -m fkpy.benchmarks.canonical_zhu5 --workers 1
python -m fkpy.benchmarks.canonical_zhu5 --workers 4
```

## Testing

```bash
pytest -m fast        # unit tests, < 1 s each
pytest -m slow        # integration: Lamb's problem, whole-space explosion, MT inversion
pytest -m reference   # regression vs frozen Fortran-fk output
pytest -m gpu         # JAX backend tests; skipped without GPU
```

## References

- M. Bouchon (1981). A simple method to calculate Green's functions for
  elastic layered media. *BSSA* **71**, 959–971.
- B. L. N. Kennett (1983). *Seismic Wave Propagation in Stratified Media.*
  Cambridge University Press.
- L. Zhu & L. A. Rivera (2002). A note on the dynamic and static
  displacements from a point source in multilayered media.
  *Geophys. J. Int.* **148**, 619–627.
- K. Aki & P. G. Richards (2002). *Quantitative Seismology*, 2nd ed.

## License

MIT — see `LICENSE`.
