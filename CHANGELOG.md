# Changelog

All notable changes to pyoculus are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-19

**This release is a major rewrite and is not backwards-compatible with the 0.3.x series.**
Users relying on the previous API should pin to the last 0.3 release:

    pip install pyoculus==0.3.3

### Changed (breaking)
- The `Problem` abstraction has been split into two concepts:
  - `Map` — encapsulates the dynamical system itself, with methods `f`, `df`, etc.
  - `Field` — provides the interface to a measure-preserving field that will be integrated, 
  for example the magnetic field source (SPEC, simsopt, or analytic).
  - Code that previously subclassed `Problem` will not work without rewriting against
    the new `Map` / `Field` API.
- `Solver` classes have been refactored to operate on `Map` objects rather than `Problem`.
- Documentation has moved from Doxygen to Sphinx (Google/Napoleon-style docstrings).

### Added
- Calculation of stable and unstable manifolds around unstable fixed points.
- Calculation of homoclinic and heteroclinic orbits between fixed points.
- Turnstile area calculations using an action principle. 
- Fixed-point finding, Poincaré plotting, and rotational transform calculation
  in cylindrical coordinates.
- New fixed-point finding methods built on `scipy.optimize.root`.
- Analytic toy magnetic fields capable of producing tokamak-like configurations. 
- Tokamak-like near-axisymmetric fields sourced from an axisymmetric equlibrium 
  solver (LIUQUE) including non-axisymmetrically perturbed TCV-like equilibria.
- numpydoc-style docstrings across most of the codebase.
- SPEC and SPECTRE reintegration via the new `Field` interface; `SpectreBfield` for the
  successor SPECTRE code (imported lazily, so SPECTRE is not a hard dependency).
- New dependency: `dill` for saving long calculations. 

### Removed / not yet ported
- The jump Hamiltonian solver from 0.3.x has not been ported. Users needing it
  should pin to `pyoculus==0.3.3`.
- Multiprocessing: did not work great before, now completely omitted and awaiting
  a dedicated effort to implement. 

### Infrastructure
- Documentation now publishes via GitHub Actions to GitHub Pages.
- Repository moved to the `pyoculus` GitHub organization.

[1.0.0]: https://github.com/pyoculus/pyoculus/releases/tag/1.0.0
[0.3.3]: https://github.com/pyoculus/pyoculus/releases/tag/0.3.3
