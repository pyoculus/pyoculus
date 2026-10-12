"""Manufactured fields and real equilibrium data check backend adapters."""

from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from tests.helpers import derivative


@pytest.mark.optional_backend
def test_mgrid_manufactured_solenoidal_field(tmp_path):
    """MAKEGRID stores physical B_phi, so conversion must divide by R.

    B=(R, 2R, -2Z) in the orthonormal basis is divergence-free. Linear grid
    data are reproduced by cubic splines; two groups check current weighting.
    """
    nc = pytest.importorskip("netCDF4")
    MgridBfield = pytest.importorskip(
        "pyoculus.fields.mgrid_bfield", reason="MGRID adapter is not present on master"
    ).MgridBfield

    path = tmp_path / "manufactured.nc"
    with nc.Dataset(path, "w") as ds:
        for name, size in [("radial", 6), ("toroidal", 8), ("vertical", 6)]:
            ds.createDimension(name, size)
        values = dict(
            nfp=2, kp=8, ir=6, jz=6, nextcur=2, rmin=1.0, rmax=3.0, zmin=-1.0, zmax=1.0
        )
        for name, value in values.items():
            ds.createVariable(name, "f8", ())[...] = value
        r = np.broadcast_to(np.linspace(1, 3, 6)[None, None, :], (8, 6, 6))
        z = np.broadcast_to(np.linspace(-1, 1, 6)[None, :, None], (8, 6, 6))
        for group, factor in [(1, 1.0), (2, 2.0)]:
            for name, field in [("br", r), ("bp", 2 * r), ("bz", -2 * z)]:
                ds.createVariable(
                    f"{name}_{group:03d}", "f8", ("toroidal", "vertical", "radial")
                )[:] = factor * field
    field = MgridBfield(path, extcur=[0.25, 0.75])
    x = np.array([1.7, 0.3, 0.2])
    B, d = field.dBdX(x)
    assert_allclose(B, 1.75 * np.array([1.7, 2.0, -0.4]), atol=2e-6)
    assert_allclose(d, 1.75 * np.diag([1, 0, -2]), atol=2e-6)
    assert_allclose(field.B(x + [0, np.pi, 0]), B, atol=2e-6)
    assert_allclose(d[0, 0] + B[0] / x[0] + d[1, 1] + d[2, 2], 0, atol=2e-6)


def test_simsopt_cartesian_derivative_bridge():
    """A varying Cartesian solenoidal field detects missing basis derivatives.

    A minimal provider exercises the adapter protocol without expensive coils;
    the existing device tests separately validate actual SIMSOPT fields.
    """
    pytest.importorskip("simsopt")
    from pyoculus.fields import SimsoptBfield

    class CartesianField:
        def set_points(self, points):
            self.points = np.asarray(points)

        def B(self):
            return self.points * np.array([1.0, 2.0, -3.0]) + [0.2, 0.3, 0.4]

        def dB_by_dX(self):
            return np.diag([1.0, 2.0, -3.0])[None, :, :]

    f = SimsoptBfield(1, CartesianField())
    p = np.array([2.0, 0.7, 0.3])
    B, d = f.dBdX(p)
    assert_allclose(B, f.B(p), atol=1e-13)
    assert_allclose(d, derivative(f.B, p), atol=2e-8)
    assert_allclose(d[0, 0] + B[0] / p[0] + d[1, 1] + d[2, 2], 0, atol=1e-12)


@pytest.mark.optional_backend
def test_spectre_field_derivatives():
    """The actual SPECTRE adapter differentiates the same field that it traces.

    Stay inside a volume, away from interfaces and coordinate singularities.
    This test intentionally has no py_spec dependency.
    """
    spectre = pytest.importorskip("spectre")
    if not hasattr(spectre, "SPECTRE_pyoculus_helper"):
        pytest.skip(
            "Installed SPECTRE lacks the SPECTRE_pyoculus_helper API required by this adapter"
        )
    from pyoculus.fields import SpectreBfield

    path = Path(__file__).resolve().parents[1] / "test_files" / "w7x_1vol_spectre.h5"
    f = SpectreBfield.from_h5_file(path, 0)
    x = np.array([0.2, 0.6, 0.3])
    B, d = f.dBdX(x)
    assert_allclose(B, f.B(x), atol=1e-12)
    assert_allclose(d, derivative(f.B, x).T, atol=2e-6, rtol=2e-5)
    assert_allclose(f.B(x + [0, 0, 2 * np.pi / f.Nfp]), B, atol=2e-8)
