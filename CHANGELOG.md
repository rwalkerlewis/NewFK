# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-17

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
- `fkpy.compute_greens_for_depths` (also exposed via the JAX backend)
  for batched GF evaluation across many source depths (vmap-style;
  threaded by `jax.device_count()`).
- `fkpy.kernel.displacement_kernel` and JAX submodules
  (`propagator_jx`, `kernel_jx`, `greens_jx`) per plan §1.1.
- `fkpy.MomentTensor.from_obspy_event` (RTP→NED conversion).
- `fkpy.MomentTensor.to_basis_weights` alias for the documented API name.
- `fkpy.attenuation.complex_velocity` public helper per plan §2.2.
- `fkpy bench` CLI subcommand for one-shot benchmark from the shell.
- `examples/quickstart.py` and `examples/zhu5.model` so the README
  one-liner is reproducible from a fresh clone.
- Bouchon (1981) condition (2) on `dk` is checked at runtime; a
  WARNING is logged via the `fkpy` logger if the configured `dk`
  exceeds the wrap-around-safe limit.
- GitHub Actions CI workflow (`.github/workflows/ci.yml`): ruff +
  mypy strict + fast/slow/reference/gpu pytest matrix on Python
  3.11 and 3.12.

### Fixed
- SAC filename suffixes now match Lupei Zhu's `fk` convention exactly
  (`.0/.1/.3/.4/.5/.6/.7/.8/.a/.b`); the trivially-zero `n=0` SH
  component (`.2`) is dropped from the 10-component output.
- All output is routed through the package logger; no `print()` or
  `click.echo()` in `src/`.
- JAX propagator now respects `jax_enable_x64` and avoids the
  complex64 truncation warning when x64 is enabled.

### Tests
- Aki & Richards (2002) eq. (4.23): the EX_Z spectrum scales as
  `cos(theta)/r` to better than 0.05 % in a homogeneous full-space.
- Kramers–Kronig causality of the Futterman attenuation operator.
- K-K compound matrix matches the analytic Haskell entries (ZR-2002
  eq. 17) to 1e-10.
- Zoeppritz normal-incidence reflection coefficient cross-check.
- `tests/test_cli.py`: HDF5 attrs, SAC prefix, pyfk format flag,
  bench subcommand.
- Dedicated unit tests for `bessel`, `transform`, `frequency`,
  `taup`, `_backend`, `MomentTensor.from_obspy_event`, and the
  Bouchon condition warning.
- 69 tests (57 fast + 5 slow + 1 reference + 4 gpu/jax), no skips.

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
