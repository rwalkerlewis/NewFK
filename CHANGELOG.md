# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `--model-format pyfk` flag to read Lupei Zhu / pyfk-style model files
  (`thickness vs vp rho Qs Qp`) without manual reordering.
- HDF5 output records every input parameter (sigma, dk, kmax,
  model_path, model_format, fkpy_version) plus take-off angles and the
  full model array.
- SAC headers now carry P/S take-off angles in `user1` / `user2` per
  the `fk` SAC convention.
- 1-D ray-traced first-arrival times via `fkpy.taup.first_arrival_time`
  (head wave + direct ray); replaces the v_max heuristic.
- `fkpy.jax_backend.compute_greens_for_depths` for batched GF
  evaluation across many source depths (vmap-style; threaded by
  `jax.device_count()`).
- GitHub Actions CI workflow (`.github/workflows/ci.yml`): ruff +
  mypy strict + fast/slow/reference/gpu pytest matrix on Python
  3.11 and 3.12.

### Fixed
- SAC filename suffixes now match Lupei Zhu's `fk` convention exactly
  (`.0/.1/.3/.4/.5/.6/.7/.8/.a/.b`); the trivially-zero `n=0` SH
  component (`.2`) is dropped from the 10-component output.

### Tests
- Aki & Richards (2002) eq. (4.23): the EX_Z spectrum scales as
  `cos(theta)/r` to better than 0.05 % in a homogeneous full-space.
- Kramers–Kronig causality of the Futterman attenuation operator.
- K-K compound matrix matches the analytic Haskell entries (ZR-2002
  eq. 17) to 1e-10.
- Zoeppritz normal-incidence reflection coefficient cross-check.
- `tests/test_cli.py` covers the HDF5 attrs and the pyfk format flag.

## [0.1.0] - 2026-04-17

### Added
- Initial release of `fkpy`, a Python port of Lupei Zhu's `fk` Fortran code
  for synthetic seismograms in a layered elastic half-space, following
  Zhu & Rivera (GJI, 2002).
- Discrete-wavenumber integration (Bouchon, BSSA 1981, eq. 9) with the
  complex frequency shift `w = ω − iσ` to suppress time-domain wrap-around.
- Kennett–Kerry reflection/transmission matrix recursion (Kennett 1983,
  §6.3) instead of the Haskell propagator: numerically stable for thick
  layers at high frequency.
- Causal Futterman attenuation (Aki & Richards 2002, p. 182) with
  `Q_ref = 1 Hz`.
- Source: full moment tensor decomposed into Zhu's basis (DD, DS, SS, EX,
  CLVD); 10 Green's-function components per receiver distance.
- Numba-jitted hot loops (propagator build, kernel evaluation, wavenumber
  summation); `concurrent.futures.ProcessPoolExecutor` parallelism over the
  frequency loop.
- Optional JAX backend behind `FK_BACKEND=jax` for GPU batch runs.
- SAC writer that matches Lupei Zhu's `fk` header conventions
  (`kstnm`, `dist`, `az`, `baz`, `evdp`, `stel`, `t1`, `t2`, …).
- CLI: `fkpy compute --model … --depth … --distances … --npts … --dt …
  --out greens.h5`.
- Tests: Lamb's problem (Mooney 1974), whole-space explosion (A&R eq.
  4.23), regression vs frozen Fortran-fk output, and end-to-end MT
  inversion recovery.
