# Physics regression suite

This suite adds **65 collected cases**, including parametrization, optional
backend cases, and strict expected failures. The ceiling is a scope constraint
for this change, not a request to add enough tests to reach an arbitrary
coverage percentage. Each case explains its physical oracle in its docstring.

Tests live beside the modules they exercise: `fields/`, `maps/`, `solvers/`,
`geo/`, and `utils/`. Integrator and runnable-example checks live in
`test_integrators.py` and `test_examples.py`. Independent numerical references
are in `helpers.py`; field and map fixtures are scoped to their directories.

Run from the repository root:

```sh
MPLBACKEND=Agg python -m pytest -q -ra tests
MPLBACKEND=Agg python -m pytest -q tests -m 'not slow'
MPLBACKEND=Agg COVERAGE_FILE=/tmp/pyoculus-physics.coverage python -m pytest -q -ra --cov=pyoculus --cov-branch --cov-report=term-missing --cov-report=xml:/tmp/pyoculus-physics.xml
```

The slow cases resolve separatrix branches, find homoclinic intersections, and
compare lobe action with shoelace area times the normal field evaluated once at
the contour's mean vertex position. No test downloads
data or depends on figures being displayed. External backends can initialize
MPI during imports, which requires an environment permitting MPI startup.

## Physical conventions and independent checks

- Cylindrical components are contravariant: physical toroidal field is
  `R * B[1]`. The section preserves `R * B[1] * dR * dZ`, not generally `dR * dZ`.
- Cylindrical derivative arrays have component rows; the toroidal interface
  has coordinate rows. Finite differences test those conventions explicitly.
- A vector potential must be contracted with the cylindrical metric in
  `A · dl`. Independent curl checks use its covariant components
  `(A_R, R**2 * A^phi, A_Z)`.
- Analytic integrable solutions, Biot–Savart quadrature, manufactured grid data,
  Fourier modes, finite differences, and invariant measures supply the oracles.
  Tests do not treat the implementation's own AD as its independent reference.
- Discrete angle seams are avoided for derivatives; periodic phase comparisons
  handle winding without mistaking a modulo jump for a physical displacement.
- Solver tolerances and initial displacements are fixed. Clinic searches use
  deterministic seeds and no random retries. The Lyapunov direction test saves
  and restores RNG state.

## Analytic manifold configuration

The field follows
[toymodel_q6.py](https://github.com/smiet/fusion_turnstile_figures/blob/main/toymodel_q6.py):
`R=6`, `Z=0`, `sf=1.16`, `shear=1.2`, a current loop of amplitude `-10` at
`(6,-5.5)`, and a Maxwell–Boltzmann perturbation `m=6`, `n=-1`, `d=2`,
amplitude `0.32`, centered on the displaced axis.

At this amplitude the tested period-three island pair is elliptic/hyperbolic.
The located period-six orbit is inverse-hyperbolic, with residue about 1.6065;
it is not assumed elliptic simply because the perturbation has poloidal mode 6.
Closure, exclusion of shorter periods, winding, and multipliers establish the
orbit classifications. Approximate seeds identify the orbits; saved coordinates
are not regression oracles.

One module-scoped field is shared read-only. Every test gets a fresh section
and solver. Tests changing perturbations create isolated field instances.
The default-list regression restores the shared production default in `finally`.

The lobe comparison uses the shoelace expression already present in
`ResonanceZone.flux_approx_SL`, multiplied by one evaluation of `R*B^phi` at the
mean vertex position. It does not implement a surface integrator or sample an
interior grid. This constant-field approximation has a deliberately looser
tolerance than the action-versus-line-integral comparison. Shoelace gives a
magnitude, whereas action and line integrals retain orientation.
`ResonanceZone.flux_approx_SL` itself still depends on a missing `contour()`
method, so its complete interface remains unvalidated.

## Defects and limits

Strict expected failures assert intended behavior and specify the expected
exception class. An unexpected pass requires removing or revising the marker.
They are defect regressions, **not passing physics validations**:

- Constructor mutation and aliasing of perturbation specifications, shared
  default lists, and stale individual-mode JIT caches.
- Missing toroidal derivative in `TwoWaves.dBdX`.
- Inconsistent physical/contravariant vector-potential conventions in the
  axisymmetric grid field.
- Broken standard-map argument plumbing and zero-time Jacobians in the
  cylindrical section and tokamap.
- Lost signs of inverse-hyperbolic eigenvalues in `manifold.eig`.
- Lyapunov's NumPy 2 incompatibility and uninitialized explicit-displacement
  normalization; QFM's missing boundary periodicity and abstract field adapter.
- Legacy M3DC1 construction and the removed pressure-jump-map base module.

Optional SPECTRE validation requires `SPECTRE_pyoculus_helper`; an installed
SPECTRE version without that API is reported as unavailable. Existing SPEC
tests still require `py_spec`. Fortran setup and the small Cartesian field
provider test adapter contracts; neither claims to validate a backend's field
solver. SDC contains no integrator implementation to exercise. Unsupported
coordinate-transform operations and blocked adapter physics remain uncovered.
The MGRID adapter is absent on `master`; its manufactured-field case skips
until that adapter is available. The existing SPECTRE tests also check for the
required helper API before collection.

The two example checks execute and render the working Greene and QFM examples.
No Lyapunov example is included until its existing implementation can run
without solver fixes.

## Validation results

On `master` at `7ff079e`, in the validation environment (NumPy 2.4.3, with JAX,
SIMSOPT, and netCDF4), the complete suite reports **69 passed, 5 skipped, and
18 expected failures**. The 65 new cases account for 45 passes, two
optional-backend skips, and all
18 expected failures. Production code is unchanged.

Coverage measured with `--cov=pyoculus --cov-branch`, comparing the original
master tests (24 passes and three backend-module skips) with the expanded suite:

| Scope | Before | After |
| --- | ---: | ---: |
| All lines | 28.97% | 59.91% |
| All branches | 13.55% | 40.26% |
| Manifold lines | 13.64% | 49.81% |
| Greene flux-surface lines | 16.90% | 66.20% |
| QFM solver lines | 5.84% | 82.47% |

Coverage includes code reached by expected failures; it is an execution measure,
not evidence that the documented defects are fixed. Optional-backend availability
and NumPy versions can change the skip and expected-failure counts.
