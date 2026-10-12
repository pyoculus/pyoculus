"""Maxwell identities and manufactured fields, rather than shape-only checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from scipy.special import ellipk, ellipe
from tests.helpers import derivative, cylindrical_curl
from pyoculus.fields import TwoWaves, AxisymmetricCylindricalGridField


def test_two_waves_hamilton_equations():
    """Hamilton's equations give drho/dphi=-H_theta, dtheta/dphi=H_rho.

    Numerical derivatives of H are independent of the field's formulas.
    This driven Hamiltonian is not an energy-conservation test.
    """
    field = TwoWaves(k=0.03)
    x = np.array([0.43, 0.71, 0.29])

    def H(y):
        r, t, z = y
        return r * r / 2 - field.k * (np.cos(2 * t - z) / 2 + np.cos(3 * t - 2 * z) / 3)

    grad = derivative(H, x).ravel()
    assert_allclose(field.B(x), [-grad[1], grad[0], 1], atol=2e-10)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="TwoWaves.dBdX omits the derivative with respect to phi",
)
def test_two_waves_full_spatial_derivative():
    """The forcing phase varies in phi as well as theta; dB must include both."""
    field = TwoWaves(k=0.03)
    x = np.array([0.43, 0.71, 0.29])
    assert_allclose(field.dBdX(x)[1], derivative(field.B, x).T, atol=2e-9)


def test_two_waves_batch_and_periodicity():
    """Both waves have integer helicity, hence 2*pi-periodic angles."""
    field = TwoWaves(k=0.02)
    pts = np.array([[0.2, 0.3, 0.4], [0.6, 1.1, 0.8]])
    expected = np.array([field.B(p) for p in pts])
    assert_allclose(field.B_many(*pts.T), expected, atol=1e-14)
    assert_allclose(
        field.B_many(pts[:, 0], pts[:, 1] + 2 * np.pi, pts[:, 2] + 2 * np.pi),
        expected,
        atol=1e-14,
    )


def test_analytic_field_maxwell_identities():
    """Finite-difference curl(A) and div(B) check the cylindrical metric factors."""
    pytest.importorskip("jax")
    from pyoculus.fields import AnalyticCylindricalBfield

    field = AnalyticCylindricalBfield(
        6, 0, 1.16, 1.2, [dict(type="maxwell-boltzmann", m=6, n=-1, d=2, amplitude=0.1)]
    )
    x = np.array([6.7, 0.23, 0.6])
    B = field.B(x)
    d = derivative(field.B, x)
    assert_allclose(cylindrical_curl(field.A, x), B, atol=2e-8)
    assert_allclose(d[0, 0] + B[0] / x[0] + d[1, 1] + d[2, 2], 0, atol=2e-9)
    assert_allclose(field.dBdX(x)[1], d, atol=2e-8)


def test_gaussian_flux_generates_solenoidal_field():
    """A toroidal potential produces a poloidal field tangent to its flux levels."""
    pytest.importorskip("jax")
    from pyoculus.toybox.cylindrical_toybox import psi_gaussian, A_gaussian, psitob

    args = dict(R=6.0, Z=0.0, m=3, n=2, mu=1.0, sigma=0.5)
    x = np.array([6.7, 0.23, 0.6])
    psi = lambda y: psi_gaussian(y, **args)
    A = lambda y: A_gaussian(y, **args)
    B = np.asarray(psitob(psi)(x))
    grad = derivative(psi, x).ravel()
    assert_allclose(B, [-grad[2] / x[0], 0, grad[0] / x[0]], atol=2e-9)
    assert_allclose(cylindrical_curl(A, x), B, atol=2e-9)


def test_current_loop_biot_savart():
    """Unit mu0*I loop: independent line quadrature fixes normalization and sign.

    The point is off-axis and away from the filament; no singular regularizer
    is needed. 2e-6 accommodates the toybox's approximate elliptic integrals.
    """
    pytest.importorskip("jax")
    from pyoculus.toybox.cylindrical_toybox import psitob, psi_circularcurrentloop

    radius, height = 2.0, -0.3
    x = np.array([1.1, 0.0, 0.8])

    def integrand(t):
        source = np.array([radius * np.cos(t), radius * np.sin(t), height])
        dl = np.array([-radius * np.sin(t), radius * np.cos(t), 0])
        dr = x - source
        return np.cross(dl, dr) / (4 * np.pi * np.linalg.norm(dr) ** 3)

    expected = quad_vec(integrand, 0, 2 * np.pi, epsabs=1e-12)[0]
    B = np.asarray(psitob(psi_circularcurrentloop)(x, radius, height))
    assert_allclose(B, expected, rtol=2e-6, atol=1e-9)


def test_elliptic_integrals():
    """Loop-field elliptic functions agree with independent SciPy special functions."""
    pytest.importorskip("jax")
    from pyoculus.toybox.cylindrical_toybox import ellpk, ellpe

    m = np.array([0.01, 0.3, 0.8, 0.99])
    assert_allclose([ellpk(v) for v in m], ellipk(m), rtol=2e-7)
    assert_allclose([ellpe(v) for v in m], ellipe(m), rtol=2e-7)


def polynomial_grid():
    """Flux F=2*pi*R^2*Z gives B_R=-R and B_Z=2Z (exactly solenoidal)."""
    R, Z = np.linspace(1, 3, 6), np.linspace(-1, 1, 6)
    r, z = np.meshgrid(R, Z, indexing="ij")
    return R, Z, -r, 2 * z, np.ones_like(r), 2 * np.pi * r * r * z


def test_grid_flux_derivatives():
    """A polynomial flux detects swapped axes and missing 2*pi/R factors.

    2e-4 allows the iterative quintic spline construction's default accuracy.
    """
    field = AxisymmetricCylindricalGridField(*polynomial_grid())
    x = np.array([1.7, 0.3, 0.2])
    B, d = field.dBdX(x)
    assert_allclose(B, [-1.7, 1.0, 0.4], atol=2e-4)
    assert_allclose(d, np.diag([-1.0, 0.0, 2.0]), atol=2e-4)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Grid A and B use inconsistent physical/contravariant phi components",
)
def test_grid_vector_potential_curl():
    """Grid fields must satisfy curl(A)=B in the section's contravariant basis."""
    field = AxisymmetricCylindricalGridField(*polynomial_grid())
    x = np.array([1.7, 0.3, 0.2])
    assert_allclose(cylindrical_curl(field.A, x), field.B(x), atol=2e-4)


def test_manufactured_matlab_flux(tmp_path):
    """Loading an equilibrium must preserve flux orientation and field amplitudes."""
    from scipy.io import savemat

    R, Z, br, bz, bp, flux = polynomial_grid()
    rr, zz = np.meshgrid(R, Z)
    path = tmp_path / "equilibrium.mat"
    savemat(path, dict(rr=rr, zz=zz, Br=br.T, Bz=bz.T, Bphi=bp.T, Fx=flux.T))
    field = AxisymmetricCylindricalGridField.from_matlab_file(path)
    assert_allclose(field.B([1.7, 0.3, 0.2]), [-1.7, 1.0, 0.4], atol=2e-4)
