"""Coordinate changes cannot alter a vector or metric's physical meaning."""

from types import SimpleNamespace
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyoculus.geo import SurfacesToroidal
from tests.helpers import derivative


@pytest.mark.parametrize("method", ["cubic_spline", "pchip", "cubic_hermite"])
def test_radial_interpolation_definition(method):
    """Linear endpoints give identity except for the explicit zero-slope Hermite.

    Hermite uses s=3*rho^2-2*rho^3; testing that polynomial avoids incorrectly
    demanding an identity transform from its deliberately different slopes.
    """
    surfaces = SurfacesToroidal(nsurfaces=2, mpol=1, ntor=1)
    surfaces.construct_interpolant(method=method)
    r = np.array([0.2, 0.4, 0.8])
    out = surfaces.get_coords(r, r, r, derivative=2)
    if method == "cubic_hermite":
        s, ds, dds = 3 * r * r - 2 * r**3, 6 * r - 6 * r * r, 6 - 12 * r
    else:
        s, ds, dds = r, np.ones_like(r), np.zeros_like(r)
    assert_allclose(out.s, s, atol=1e-13)
    assert_allclose(out.ds[:, 0], ds, atol=1e-13)
    assert_allclose(out.dds[:, 0, 0], dds, atol=1e-13)
    assert_allclose(out.t, r, atol=1e-13)


def mode_surface(sym=True):
    """s=rho*(1+a*cos(theta-2*zeta)), with an optional sine component."""
    s = SurfacesToroidal(nsurfaces=2, mpol=1, ntor=1, Nfp=2, stellar_sym=sym)
    s.scn[1, 1, 1] = 0.07
    s.tsn[:, 1, 1] = 0.03
    if not sym:
        s.ssn[1, 1, 1] = 0.04
        s.tcn[:, 1, 1] = 0.02
    s.construct_interpolant()
    return s


@pytest.mark.parametrize("symmetric", [True, False])
def test_fourier_geometry_derivatives(symmetric):
    """A helical Fourier mode fixes signs, Nfp factors, and mixed derivatives."""
    s = mode_surface(symmetric)
    x = np.array([0.4, 0.7, 0.2])
    phase = x[1] - 2 * x[2]
    expected_s = x[0] * (
        1 + 0.07 * np.cos(phase) + (0 if symmetric else 0.04 * np.sin(phase))
    )
    c = s.get_coords(*x, derivative=2)

    def coords(y):
        p = s.get_coords(*y)
        return np.array([p.s[0], p.t[0], p.z[0]])

    assert_allclose(c.s, expected_s, atol=1e-13)
    assert_allclose(c.jacobi[0], derivative(coords, x), atol=2e-9)
    dJ = derivative(
        lambda y: s.get_coords(*y, derivative=1).jacobi[0].ravel(), x
    ).reshape(3, 3, 3)
    assert_allclose(c.djacobi[0], dJ, atol=2e-9)
    assert_allclose(c.jacobian, np.linalg.det(c.jacobi), atol=1e-13)
    assert_allclose(
        s.metric_transform(np.eye(3), c)[0], c.jacobi[0].T @ c.jacobi[0], atol=1e-13
    )


def test_contravariant_vector_and_density_transform():
    """J*v_new=v_old; a vector density additionally acquires det(J).

    Use a single point to test tensor algebra independently of batched solve
    conventions. Compare derivatives to finite differences, including dJ.
    """
    s = mode_surface(False)
    x = np.array([0.4, 0.7, 0.2])

    def point(y):
        c = s.get_coords(*y, derivative=2)
        return SimpleNamespace(**{k: np.asarray(v)[0] for k, v in vars(c).items()})

    c = point(x)
    v = np.array([0.2, 0.7, 1.0])
    transformed, dv = s.contra_vector_transform(
        v, c, derivative=True, dv=np.zeros((3, 3))
    )
    assert_allclose(c.jacobi @ transformed, v, atol=1e-13)
    expected_dv = derivative(lambda y: np.linalg.solve(point(y).jacobi, v), x).T
    assert_allclose(dv, expected_dv, atol=2e-9)
    density = s.contra_vector_transform(v, c, has_jacobian=True)
    assert_allclose(density, c.jacobian * transformed, atol=1e-13)


def test_surface_serialization_preserves_geometry(tmp_path):
    """Saving nonsymmetric surfaces must retain both sine and cosine information."""
    original = mode_surface(False)
    path = tmp_path / "surfaces.npz"
    original.write_surfaces_to_file(path)
    loaded = SurfacesToroidal()
    loaded.read_surfaces_from_file(path, Nfp=2)
    loaded.construct_interpolant()
    before = original.get_coords(0.4, 0.7, 0.2, derivative=2)
    after = loaded.get_coords(0.4, 0.7, 0.2, derivative=2)
    assert_allclose(after.jacobi, before.jacobi, atol=1e-13)
    assert_allclose(after.djacobi, before.djacobi, atol=1e-13)
