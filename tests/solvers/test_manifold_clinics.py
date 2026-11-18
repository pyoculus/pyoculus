"""Clinic finding and adaptive manifold tracing on the q=6 toy model.

The field is the configuration of smiet/fusion_turnstile_figures/toymodel_q6.py
(also used in the regression tests of PR #21): a circular current loop below a
displaced axis creates a period-1 X-point whose stable and unstable manifolds
form a homoclinic tangle around the magnetic axis. The m=1 X-point has two
homoclinic orbits per fundamental domain, whose turnstile lobes carry equal and
opposite flux.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("jax")
from pyoculus.fields import AnalyticCylindricalBfield as Field
from pyoculus.maps import CylindricalBfieldSection as Section
from pyoculus.solvers import FixedPoint, Manifold
from pyoculus.solvers.manifold import Clinic, find_zero_crossings

ROOT_ARGS = {"tol": 1e-8, "options": {"eps": 1e-5, "factor": 0.1}}


@pytest.fixture(scope="module")
def section():
    loop = dict(type="circular-current-loop", amplitude=-10, R=6, Z=-5.5)
    wave = dict(type="maxwell-boltzmann", m=6, n=-1, d=2, amplitude=0.32)
    field = Field.with_new_axis(6, 0, 1.16, 1.2, [loop, wave])
    axis = Section.without_axis(field, guess=[6.0, 0.0], finderargs={}, rtol=1e-11)
    return Section(field, R0=axis.R0, Z0=axis.Z0, rtol=1e-11, atol=1e-12)


@pytest.fixture(scope="module")
def xpoint(section):
    fp = FixedPoint(section)
    fp.find(1, [6.205, -4.50], tol=1e-11)
    fp.m = 1
    assert fp.successful and fp.GreenesResidue < 0
    return fp


@pytest.fixture(scope="module")
def tangle(section, xpoint):
    """Manifold with all clinics found by the exhaustive search (shared, read-mostly)."""
    m = Manifold(section, xpoint)
    m.find_clinics_exhaustive(root_args=ROOT_ARGS)
    return m


def _distance_to_curve(point, curve):
    """Euclidean distance from a point to a polyline."""
    a, ab = curve[:-1], np.diff(curve, axis=0)
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
    return np.min(np.linalg.norm(a + t[:, None] * ab - point, axis=1))


def _legs_meet(m, clinic):
    """Distance between the stable and unstable legs of a clinic orbit."""
    stable = m._map.f(-clinic.nint_s * m.fixedpoint_1.m, m.rfp_s + clinic.eps_s * m.vector_s)
    unstable = m._map.f(clinic.nint_u * m.fixedpoint_1.m, m.rfp_u + clinic.eps_u * m.vector_u)
    return np.linalg.norm(stable - unstable), stable


@pytest.mark.parametrize("bilinear", [False, True])
def test_zero_crossings_locate_isolated_root(bilinear):
    """The winding number certifies the one cell holding a root; the bilinear solve pins it.

    F is linear or bilinear in (x, y), so its interpolant on the grid is exact and
    the root is recovered to round-off. A NaN corner (e.g. a field line lost on a
    coil) must only disable the cells touching it.
    """
    x0, y0 = 0.3141, -0.2718
    X, Y = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 9), indexing="ij")
    if bilinear:
        Fx, Fy = (X - x0) * (Y + 2), (Y - y0) + 0.5 * (X - x0)
    else:
        Fx, Fy = 2 * (X - x0) - (Y - y0), (X - x0) + 3 * (Y - y0)
    Fx[0, 0] = np.nan
    xs, ys = find_zero_crossings(X, Y, Fx, Fy)
    assert len(xs) >= 1
    assert_allclose(xs, x0, atol=1e-12)
    assert_allclose(ys, y0, atol=1e-12)


def test_zero_crossings_report_root_on_grid_node():
    X, Y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), indexing="ij")
    xs, ys = find_zero_crossings(X, Y, X, Y)  # root exactly at the node (0, 0)
    assert len(xs) >= 1
    assert_allclose(xs, 0, atol=1e-12)
    assert_allclose(ys, 0, atol=1e-12)


def test_opoint_orients_manifolds_into_island(section, xpoint):
    """With the magnetic axis as O-point, both branches leave the X-point towards it."""
    axis = FixedPoint(section)
    axis.find(1, [section.R0, section.Z0])
    assert axis.successful and 0 < axis.GreenesResidue < 1
    m = Manifold(section, xpoint, opoint=axis)
    towards = axis.coords[0] - xpoint.coords[0]
    assert m.opoint is axis
    assert m.vector_s @ towards > 0
    assert m.vector_u @ towards > 0
    with pytest.raises(ValueError):
        Manifold(section, xpoint, opoint=FixedPoint(section))  # never found


@pytest.mark.parametrize("adaptive", [True, False])
@pytest.mark.parametrize(
    "eps_s, eps_u",
    [
        (1e-5, 1e-6),
        # eps_s >> eps_u: the unstable point outgrows the stable one while both
        # are still in the linear regime, which used to fake a crossing (n_s = 0)
        (9.263661e-06, 2.810464e-07),
    ],
)
def test_find_N_iterations_connect_the_legs(tangle, adaptive, eps_s, eps_u):
    """The iteration counts must be ones for which the legs actually meet at a clinic."""
    n_s, n_u = tangle.find_N(eps_s, eps_u, adaptive=adaptive)
    assert n_s >= 1 and n_u >= 1
    clinic = Clinic.from_guess(tangle, eps_s, eps_u, n_s, n_u, root_args=ROOT_ARGS)
    gap, _ = _legs_meet(tangle, clinic)
    assert gap < 1e-6


@pytest.fixture(scope="module")
def q3_chain(section):
    """Two neighbouring X-points of the q=3 chain and the O-point of the island between them."""
    x1 = FixedPoint(section)
    x1.find_with_iota(1, 3, np.array([7.8412, -0.5962]), tol=1e-11)
    x2 = FixedPoint(section)
    x2.find_with_iota(1, 3, x1.coords[1], tol=1e-11)
    opoint = FixedPoint(section)
    opoint.find_with_iota(1, 3, np.array([6.3644, 1.0906]), tol=1e-11)
    assert x1.GreenesResidue < 0 and 0 < opoint.GreenesResidue < 1
    return x1, x2, opoint


def test_ambiguous_default_direction_warns(section, q3_chain, caplog):
    """Both eigenvectors are nearly perpendicular to the X1->X2 line (|cos| ~ 0.06, 0.17).

    Defaulting towards the other X-point then picks the branches running to the
    third X-point of the chain; the user must be told, unless an O-point decides.
    """
    x1, x2, opoint = q3_chain
    with caplog.at_level("WARNING", logger="pyoculus.solvers.manifold"):
        Manifold(section, x1, x2)
    assert "ambiguous" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="pyoculus.solvers.manifold"):
        Manifold(section, x1, x2, opoint=opoint)
    assert "ambiguous" not in caplog.text


def test_heteroclinic_opoint_tangle_and_find_N(section, q3_chain):
    """Two X-points of the q=3 chain, oriented by the O-point of the island between them.

    The eigenvectors are nearly perpendicular to the X1->X2 line, so only the
    O-point reliably selects the branches that bound that island. With
    eps_s >> eps_u, adaptive find_N must still give iteration counts for which
    the stable leg of X1 and the unstable leg of X2 meet.
    """
    x1, x2, opoint = q3_chain
    m = Manifold(section, x1, x2, opoint=opoint)
    assert m.vector_s @ (opoint.coords[0] - m.rfp_s) > 0
    assert m.vector_u @ (opoint.coords[0] - m.rfp_u) > 0

    eps_s, eps_u = 1e-3, 1e-6
    n_s, n_u = m.find_N(eps_s, eps_u, adaptive=True)
    assert n_s >= 1 and n_u >= 1
    clinic = Clinic.from_guess(m, eps_s, eps_u, n_s, n_u, root_args=ROOT_ARGS)
    gap, point = _legs_meet(m, clinic)
    assert gap < 1e-6
    assert min(np.linalg.norm(point - m.rfp_s), np.linalg.norm(point - m.rfp_u)) > 0.1


def test_exhaustive_search_finds_both_homoclinic_orbits(tangle, section):
    """Two distinct orbits per fundamental domain whose lobes carry opposite flux.

    The hand-tuned clinic of the PR #21 tests is one of them: refining it again
    must be rejected as a duplicate (after shifting into the fundamental domain).
    """
    assert len(tangle.clinics) == 2
    for clinic in tangle.clinics:
        gap, point = _legs_meet(tangle, clinic)
        assert gap < 1e-6
        assert np.linalg.norm(point - tangle.rfp_s) > 1  # not collapsed onto the X-point
    known = Clinic.from_guess(tangle, 2.48677e-4, 7.54579e-6, 6, 5, root_args=ROOT_ARGS)
    assert not tangle.clinics.record_clinic(known)
    action = tangle.compute_turnstile_areas(n_joining=80)
    assert np.all(np.abs(action) > 0.1)
    assert_allclose(action.sum(), 0, atol=2e-6)


def test_single_clinic_from_jacobian_epsilon_estimate(section, xpoint, tangle):
    """Without guesses, the second-order linear-regime estimate still lands on a clinic."""
    m = Manifold(section, xpoint)
    m.find_clinic_single(root_args=ROOT_ARGS)
    assert len(m.clinics) == 1
    gap, _ = _legs_meet(m, m.clinics[0])
    assert gap < 1e-6
    assert not tangle.clinics.record_clinic(m.clinics[0])


def test_single_clinic_failure_returns_none_after_retries(section, xpoint):
    """One iteration per side cannot connect the legs; retries fail and nothing is recorded."""
    m = Manifold(section, xpoint)
    out = m.find_clinic_single(1e-5, 1e-5, n_s=1, n_u=1, nretry=2, root_args=ROOT_ARGS)
    assert out is None and m.clinics.is_empty

    class NoJacobian:  # map without df: epsilons cannot be estimated
        f = section.f

    m._map = NoJacobian()
    with pytest.raises(ValueError):
        m.find_clinic_single()


def test_find_clinics_keeps_positional_guesses(section, xpoint, tangle):
    """Old call style find_clinics(eps_s, eps_u, n_points) must still pass guesses, not alpha."""
    m = Manifold(section, xpoint)
    m.find_clinics(1e-5, 1e-6, 2, root_args=ROOT_ARGS)
    assert len(m.clinics) == 2
    assert not tangle.clinics.record_clinic(m.clinics[0])
    with pytest.raises(ValueError):  # first search fails -> clear error, not a TypeError
        m.find_clinics(1e-5, 1e-5, 2, n_s=1, n_u=1, root_args=ROOT_ARGS)


def test_expensive_solve_recovers_from_poor_guess(tangle):
    """A budget-limited root solve from a 20x overestimate fails; the grid restart rescues it."""
    root_args = {"tol": 1e-8, "options": {"eps": 1e-5, "factor": 0.1, "maxfev": 15}}
    clinic = tangle.clinics[1]
    guess = (20 * clinic.eps_s, 20 * clinic.eps_u, clinic.nint_s, clinic.nint_u)
    with pytest.raises(ValueError):
        Clinic.from_guess(tangle, *guess, root_args=root_args)
    rescued = Clinic.from_guess(tangle, *guess, expensive=True, root_args=root_args)
    assert_allclose([rescued.eps_s, rescued.eps_u], [clinic.eps_s, clinic.eps_u], rtol=1e-5)


def test_adaptive_manifolds_pass_through_clinics(section, xpoint, tangle):
    """Clinic points lie on both manifolds; refining the curvature tolerance converges onto them."""
    with pytest.raises(ValueError):
        Manifold(section, xpoint).compute_adaptive()

    def worst_miss(tol):
        stable, unstable = tangle.compute_adaptive(n_s_extra=0, n_u_extra=0, tol=tol)
        assert np.isfinite(stable).all() and np.isfinite(unstable).all()
        assert_allclose(stable[0], tangle.rfp_s + tangle.clinics.stable_epsilons[0] * tangle.vector_s)
        assert_allclose(unstable[0], tangle.rfp_u + tangle.clinics.unstable_epsilons[0] * tangle.vector_u)
        miss = max(
            _distance_to_curve(c.trajectory[c.nint_u], curve)
            for c in tangle.clinics
            for curve in (stable, unstable)
        )
        return miss, len(stable) + len(unstable)

    coarse, n_coarse = worst_miss(0.2)
    fine, n_fine = worst_miss(0.01)
    assert n_fine > n_coarse
    assert fine < 1e-4
    assert fine < coarse / 10
