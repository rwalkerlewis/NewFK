# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
