"""Return maps preserve magnetic flux and canonical symplectic structure."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from tests.helpers import derivative
from pyoculus.fields import TwoWaves
from pyoculus.maps import CylindricalBfieldSection, ToroidalBfieldSection


def test_cylindrical_exact_radial_flow(radial_field):
    """dR/dphi=aR has exponential flow, including negative and fractional turns."""
    section = CylindricalBfieldSection(radial_field, R0=2, Z0=0, rtol=1e-11)
    x = np.array([2.0, 0.4])
    assert_allclose(section.f(0, x), x, atol=0)
    for t in [1, -1, 0.25]:
        factor = np.exp(radial_field.a * section.dzeta * t)
        assert_allclose(section.f(t, x), [2 * factor, 0.4], atol=2e-9)
        assert_allclose(section.df(t, x), np.diag([factor, 1]), atol=2e-9)


def test_cylindrical_weighted_flux_and_action(radial_field):
    """The expanding R map preserves dR*dZ/R, not dR*dZ.

    Along this field A_i dx^i=-a*Z dphi, providing an independent action oracle.
    """
    section = CylindricalBfieldSection(radial_field, R0=2, Z0=0, rtol=1e-11)
    x = np.array([2.0, 0.4])
    end = section.f(1, x)
    det = np.linalg.det(section.df(1, x))
    assert abs(det - 1) > 0.1
    assert_allclose(det / end[0], 1 / x[0], atol=2e-9)
    assert_allclose(
        section.lagrangian(x, 1), -radial_field.a * x[1] * 2 * np.pi, atol=2e-9
    )
    assert_allclose(section.lagrangian(end, -1), -section.lagrangian(x, 1), atol=2e-9)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="Cylindrical zero-time tangent cache stores a list, then calls reshape",
)
def test_cylindrical_zero_time_tangent(radial_field):
    """An identity trajectory must have an identity tangent map."""
    section = CylindricalBfieldSection(radial_field, R0=2, Z0=0)
    assert_allclose(section.df(0, [2.0, 0.4]), np.eye(2), atol=0)


def test_toroidal_integrable_shear():
    """At k=0, rho is fixed and theta advances by rho*Delta_phi."""
    section = ToroidalBfieldSection(TwoWaves(k=0), rtol=1e-11)
    x = np.array([0.23, 0.7])
    dt = 2 * np.pi * 3
    assert_allclose(section.f(3, x), [0.23, (0.7 + 0.23 * dt) % (2 * np.pi)], atol=2e-9)
    assert_allclose(section.df(3, x), [[1, 0], [dt, 1]], atol=2e-9)
    assert_allclose(section.winding(3, x), [0.23, 0.23 * dt], atol=2e-9)


def test_toroidal_perturbed_tangent_and_inverse():
    """Hamiltonian flow is symplectic; derivative composition must follow the orbit."""
    section = ToroidalBfieldSection(TwoWaves(k=0.002), rtol=1e-11)
    x = np.array([0.31, 0.4])
    y = section.f(1, x)
    J = section.df(1, x)
    assert_allclose(J, derivative(lambda z: section.f(1, z), x), atol=2e-7)
    assert_allclose(np.linalg.det(J), 1, atol=2e-8)
    assert_allclose(section.df(2, x), section.df(1, y) @ J, atol=2e-7)
    assert_allclose(section.f(-1, y), x, atol=2e-8)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="StandardMap passes an extra w argument to its two-argument kernel",
)
def test_standard_map_integrable_limit():
    """The raised standard map at K=0 is a shear with theta in turns."""
    pytest.importorskip("jax")
    from pyoculus.maps.standard_map import StandardMap

    m = StandardMap(K=0)
    assert_allclose(m.f(1, [0.1, 0.2]), [0.8, 0.2], atol=1e-12)
    assert_allclose(m.df(1, [0.1, 0.2]), [[1, 1], [0, 1]], atol=1e-12)


def test_tokamap_inverse_and_symplecticity():
    """The tokamap preserves canonical dtheta*dpsi; compare away from angle seams."""
    pytest.importorskip("jax")
    from pyoculus.maps.tokamap import TokaMap

    m = TokaMap(K=0.2, w=0.6)
    x = np.array([0.2, 0.4])
    assert_allclose(m.f(-3, m.f(3, x)), x, atol=2e-12)
    J = np.asarray(m.df(1, x))
    assert_allclose(np.linalg.det(J), 1, atol=2e-12)
    assert_allclose(J, derivative(lambda y: m.f(1, y), x), atol=2e-8)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="TokaMap zero-iteration derivative calls identity with extra arguments",
)
def test_tokamap_zero_time_tangent():
    """Zero iterations must leave both a phase point and its tangent unchanged."""
    pytest.importorskip("jax")
    from pyoculus.maps.tokamap import TokaMap

    m = TokaMap(K=0.2)
    assert_allclose(m.f(0, [0.2, 0.4]), [0.2, 0.4], atol=0)
    assert_allclose(m.df(0, [0.2, 0.4]), np.eye(2), atol=0)
