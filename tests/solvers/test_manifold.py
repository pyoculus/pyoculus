"""Real field-line dynamics from smiet/fusion_turnstile_figures/toymodel_q6.py.

Keep its m=6, n=-1, amplitude=0.32 configuration exactly. At this strength a
q=3 elliptic/hyperbolic pair exists, while the located q=6 orbit is INVERSE
hyperbolic (residue > 1). A harmonic label alone does not determine the minimal
period or stability of every island orbit. Seeds below were established by a
deterministic search, but tests judge closure and winding, not saved coordinates.
"""

from types import SimpleNamespace as NS
import numpy as np
import pytest
from numpy.testing import assert_allclose
from tests.helpers import derivative

pytest.importorskip("jax")
from pyoculus.fields import AnalyticCylindricalBfield as Field
from pyoculus.maps import CylindricalBfieldSection as Section
from pyoculus.solvers import FixedPoint, Manifold, ResonanceZone
from pyoculus.solvers.manifold import Clinic, eig


@pytest.fixture(scope="module")
def model():
    """One shared field; no test may mutate it. Maps and solvers remain local."""
    loop = dict(type="circular-current-loop", amplitude=-10, R=6, Z=-5.5)
    wave = dict(type="maxwell-boltzmann", m=6, n=-1, d=2, amplitude=0.32)
    field = Field.with_new_axis(6, 0, 1.16, 1.2, [loop, wave])
    section = Section.without_axis(field, guess=[6.0, 0.0], finderargs={}, rtol=1e-11)
    return field, np.array([section.R0, section.Z0])


@pytest.fixture
def section(model):
    field, axis = model
    return Section(field, R0=axis[0], Z0=axis[1], rtol=1e-11, atol=1e-12)


def orbit(section, kind="hyperbolic"):
    seeds = {
        "hyperbolic": (3, [7.8412, -0.5962]),
        "elliptic": (3, [6.3644, 1.0906]),
        "inverse": (6, [8.7342, -0.2323]),
        "separatrix": (1, [6.205, -4.50]),
    }
    period, guess = seeds[kind]
    fp = FixedPoint(section)
    if kind == "separatrix":
        fp.find(1, guess, tol=1e-11)
        fp.m = 1
    else:
        fp.find_with_iota(1, period, np.array(guess), tol=1e-11)
    assert fp.successful, f"Deterministic {kind} seed failed"
    return fp


def test_displaced_axis_and_separatrix(model, section):
    """The loop displaces the magnetic axis; the added mode is centered there."""
    f, axis = model
    wave = f.perturbations_args[1]
    assert np.linalg.norm(axis - [6, 0]) > 0.1
    assert_allclose([wave["R"], wave["Z"]], axis, atol=1e-7)
    assert_allclose(section.f(1, axis), axis, atol=1e-8)
    fp = orbit(section, "separatrix")
    assert_allclose(section.f(1, fp.coords[0]), fp.coords[0], atol=1e-8)
    assert fp.GreenesResidue < 0


def test_period_three_island_stability_pair(section):
    """The q=3 O/X pair closes after three turns, winding once, not after six.

    Closure tolerance exceeds the ODE error; the shorter-period gap is physical
    and macroscopic. Residues distinguish elliptic and direct-hyperbolic points.
    """
    for kind in ["elliptic", "hyperbolic"]:
        fp = orbit(section, kind)
        x = fp.coords[0]
        assert_allclose(section.f(3, x), x, atol=1e-8)
        assert min(np.linalg.norm(section.f(t, x) - x) for t in [1, 2]) > 0.1
        assert_allclose(section.winding(3, x)[1], 2 * np.pi, atol=2e-8)
        values = np.linalg.eigvals(fp.Jacobian)
        assert_allclose(np.prod(values), 1, atol=2e-7)
        if kind == "elliptic":
            assert 0 < fp.GreenesResidue < 1
            assert_allclose(np.abs(values), 1, atol=2e-7)
        else:
            assert fp.GreenesResidue < 0
            assert np.min(values) < 1 < np.max(values)


def test_period_six_inverse_hyperbolic_orbit(section):
    """The q=6 orbit has negative reciprocal multipliers and reverses branches."""
    fp = orbit(section, "inverse")
    x = fp.coords[0]
    assert_allclose(section.f(6, x), x, atol=2e-8)
    assert min(np.linalg.norm(section.f(t, x) - x) for t in range(1, 6)) > 0.1
    assert_allclose(section.winding(6, x)[1], 2 * np.pi, atol=2e-8)
    values = np.linalg.eigvals(fp.Jacobian)
    assert np.all(values < 0)
    assert fp.GreenesResidue > 1
    assert_allclose(np.prod(values), 1, atol=2e-6)


def test_island_monodromy_chain_rule(section):
    """The return-map derivative follows all three orbit legs in their order."""
    fp = orbit(section)
    product = np.eye(2)
    for x in fp.coords[:-1]:
        product = section.df(1, x) @ product
    assert_allclose(fp.Jacobian, product, atol=2e-7)
    finite = derivative(lambda x: section.f(3, x), fp.coords[0], h=2e-5)
    assert_allclose(fp.Jacobian, finite, atol=3e-6)
    assert_allclose(fp.GreenesResidue, (2 - np.trace(finite)) / 4, atol=2e-6)


def test_manifold_eigendirections_and_growth(section):
    """An infinitesimal displacement grows by its Floquet multiplier per orbit."""
    fp = orbit(section)
    manifold = Manifold(section, fp)
    for value, vector in [
        (manifold.lambda_s, manifold.vector_s),
        (manifold.lambda_u, manifold.vector_u),
    ]:
        assert_allclose(fp.Jacobian @ vector, value * vector, atol=2e-8)
        eps = 1e-6
        displacement = section.f(3, fp.coords[0] + eps * vector) - section.f(
            3, fp.coords[0]
        )
        assert_allclose(displacement / eps, value * vector, atol=3e-5)
    assert manifold.lambda_s < 1 < manifold.lambda_u


def test_manifold_uses_island_return_period(section):
    """One manifold step is three toroidal turns, not one turn or the mode m=6."""
    fp = orbit(section)
    manifold = Manifold(section, fp)
    x = fp.coords[0] + 1e-5 * manifold.vector_u
    trajectory = manifold.integrate(x, nintersect=2)
    assert_allclose(trajectory[:, 0], [x, section.f(3, x), section.f(6, x)], atol=1e-10)
    assert np.linalg.norm(trajectory[1, 0] - section.f(1, x)) > 0.1


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="manifold.eig takes abs(eigenvalues), discarding inverse-hyperbolic signs",
)
def test_inverse_hyperbolic_eigenpairs_keep_sign(section):
    """A negative multiplier carries the displacement to the opposite branch."""
    fp = orbit(section, "inverse")
    ls, vs, lu, vu = eig(fp.Jacobian)
    assert_allclose(fp.Jacobian @ vs, ls * vs, atol=1e-8)
    assert_allclose(fp.Jacobian @ vu, lu * vu, atol=1e-8)


def test_manifold_seed_linearization_converges(section):
    """Taylor remainder is quadratic in seed distance on both unstable branches."""
    fp = orbit(section)
    m = Manifold(section, fp)
    x = fp.coords[0]
    for sign in [-1, 1]:
        v = sign * m.vector_u

        def error(eps):
            return np.linalg.norm(
                section.f(3, x + eps * v) - section.f(3, x) - fp.Jacobian @ (eps * v)
            )

        coarse, fine = error(2e-4), error(1e-4)
        assert 0.15 < fine / coarse < 0.4  # expected 1/4, above the integration floor


@pytest.mark.slow
def test_separatrix_manifold_geometry(section):
    """The script's manifold samples remain on trajectories of the same field.

    Verify propagated sample blocks rather than comparing a stored image.
    Reintegrating a late leg at tighter tolerance checks local tracing accuracy.
    """
    fp = orbit(section, "separatrix")
    m = Manifold(section, fp)
    m.compute(eps_s=9e-6, eps_u=8e-6, nint_s=8, nint_u=8, neps_s=80, neps_u=80)
    reference = Section(
        section._mf, R0=section.R0, Z0=section.Z0, rtol=1e-12, atol=1e-13
    )
    for points, direction in [(m.stable, -1), (m.unstable, 1)]:
        assert np.isfinite(points).all()
        blocks = points.reshape(9, 80, 2)
        assert_allclose(
            reference.f(direction, blocks[-2, 17]), blocks[-1, 17], atol=2e-8
        )
        assert np.max(np.linalg.norm(blocks[-1] - fp.coords[0], axis=1)) > 0.1


@pytest.mark.slow
def test_clinic_intersection_and_duplicate(section):
    """A clinic is one point reached independently along stable and unstable legs."""
    m = Manifold(section, orbit(section, "separatrix"))
    clinic = Clinic.from_guess(
        m,
        2.48677e-4,
        7.54579e-6,
        6,
        5,
        root_args={"tol": 1e-8, "options": {"eps": 1e-5, "factor": 0.1}},
    )
    stable = section.f(-clinic.nint_s, m.rfp_s + clinic.eps_s * m.vector_s)
    unstable = section.f(clinic.nint_u, m.rfp_u + clinic.eps_u * m.vector_u)
    assert_allclose(stable, unstable, atol=2e-7)
    assert np.linalg.norm(stable - m.rfp_s) > 1  # exclude collapse onto the saddle
    assert m.clinics.record_clinic(clinic)
    assert not m.clinics.record_clinic(clinic)
    assert len(m.clinics) == 1


@pytest.mark.slow
def test_lobe_flux_shoelace_and_gauge(section):
    """Action flux agrees approximately with area times the field at the center.

    The lobe is bounded by actual stable/unstable branches between two clinics.
    Use the same shoelace expression as ResonanceZone.flux_approx_SL, but evaluate
    R*B^phi once at the mean vertex position. This constant-field approximation
    is not a surface integrator. Its tolerance allows field variation across
    the lobe; refining the boundary cannot remove that approximation error.
    Compare magnitudes because shoelace discards orientation.
    """
    m = Manifold(section, orbit(section, "separatrix"))
    for es, eu, ns, nu in [
        (2.48677e-4, 7.54579e-6, 6, 5),
        (5.90031e-5, 4.89594e-5, 5, 6),
    ]:
        c = Clinic.from_guess(
            m,
            es,
            eu,
            ns,
            nu,
            root_args={"tol": 1e-8, "options": {"eps": 1e-5, "factor": 0.1}},
        )
        assert m.clinics.record_clinic(c)
    action = m.compute_turnstile_areas(n_joining=80)
    assert np.isfinite(action).all()
    assert abs(action[0]) > 0.1
    assert_allclose(action.sum(), 0, atol=2e-6)  # equal incoming/outgoing flux
    estimates = []
    for neps in [40, 160]:
        stable, unstable = m.get_lobe_boundary(5, neps=neps)
        boundary = np.vstack([stable, unstable])
        assert np.linalg.norm(stable[-1] - unstable[0]) < 2e-4
        assert np.linalg.norm(unstable[-1] - stable[0]) < 2e-4
        # Shoelace expression from ResonanceZone.flux_approx_SL. Do not call
        # that routine's million-point interior-field sampling for this estimate.
        R, Z = boundary.T
        area = 0.5 * abs(np.dot(R, np.roll(Z, 1)) - np.dot(Z, np.roll(R, 1)))
        center_R, center_Z = boundary.mean(axis=0)
        normal_field = (
            center_R * section._mf.B(np.array([center_R, section.phi0, center_Z]))[1]
        )
        estimates.append(area * normal_field)
    assert_allclose(estimates[-1], abs(action[0]), rtol=0.1)
    assert_allclose(estimates[-1], estimates[0], rtol=0.01)
    line = m._AdL_integral_points(boundary, is_closed=True)
    assert_allclose(abs(line), abs(action[0]), rtol=0.01)
    assert_allclose(
        m._AdL_integral_points(boundary[::-1], is_closed=True), -line, atol=1e-11
    )
    # Give this manifold a local potential wrapper; the shared field is untouched.
    original = section._mf

    class GaugeField:
        def A(self, x):
            R, _, Z = x
            return original.A(x) + [Z, 0, R]  # grad(R*Z), phi independent

    m._map = NS(phi0=0.0, _mf=GaugeField())
    assert_allclose(m._AdL_integral_points(boundary, is_closed=True), line, atol=1e-11)


def test_resonance_zone_action_and_chain_scaling():
    """Action sums remove stationary-orbit backgrounds and count all m islands.

    Controlled trajectory actions isolate this algebra from the clinic finder;
    the preceding tests validate actual field trajectories and line integrals.
    """
    fp1, fp2 = NS(m=3), NS(m=3)
    s = NS(lagrangian=lambda x, t: float(x[0]) * t)
    a = NS(
        _map=s,
        fixedpoint_1=fp1,
        fixedpoint_2=fp2,
        rfp_s=np.array([1.0, 0.0]),
        clinics=[NS(trajectory=np.zeros((3, 2)))],
        _compute_lagrangian_in_sections=lambda x: np.array([3.2, 3.3]),
    )
    b = NS(
        _map=s,
        fixedpoint_1=fp2,
        fixedpoint_2=fp1,
        rfp_s=np.array([2.0, 0.0]),
        clinics=[NS(trajectory=np.zeros((3, 2)))],
        _compute_lagrangian_in_sections=lambda x: np.array([6.1, 6.4]),
    )
    zone = ResonanceZone(a, b)
    assert_allclose(zone.area(), 1.0, atol=1e-14)
    assert_allclose(zone.area(whole_chain=True), 3.0, atol=1e-14)
