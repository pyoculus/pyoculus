"""Quadratic-flux-minimizing pseudo-orbits and their Fourier representation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyoculus.solvers.qfm import rfft1D, irfft1D, rfft2D, irfft2D
from pyoculus.fields import TwoWaves
from pyoculus.solvers import (
    QFM,
)
from tests.helpers import derivative


def test_qfm_action_gradient_jacobian():
    """The Newton matrix must differentiate the pseudo-orbit residual, not B alone."""
    q = QFM(TwoWaves(k=0.002), pqMpol=4, pqNtor=4)
    q.action(1, 2, sguess=0.5)  # initializes the solver's Fourier collocation grid
    qN, Nfft = 8, 32
    r = np.zeros(qN + 1)
    r[0] = 0.51
    r[1] = 0.001
    zero = np.zeros(qN + 1)
    x = q._pack_dof(0.001, r, zero, zero, zero)
    args = (1, 2, 0.0, qN, Nfft)
    actual = q.action_gradient_jacobi(x, *args)
    expected = derivative(lambda y: q.action_gradient(y, *args), x, h=1e-6)
    assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="QFM boundary straightening refers to undefined self.Nfp",
)
def test_qfm_boundary_straightening():
    """An already straight integrable field should retain its constant-rho boundaries."""
    q = QFM(TwoWaves(k=0), pqMpol=4, pqNtor=4)
    surfaces = q.construct_qfms([1], [2], sguesslist=[0.5], verbose=False)
    assert_allclose(surfaces.scn[:, 0, 0], [0, 0.5, 1], atol=1e-10)


def test_fourier_helical_mode_roundtrip():
    """Fourier coefficients must retain DC normalization and helical sine signs."""
    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    one = 1.3 + 0.2 * np.cos(3 * t) - 0.4 * np.sin(2 * t)
    c, s = rfft1D(one)
    assert_allclose(irfft1D(c, s), one, atol=1e-13)
    two = (
        1.3
        + 0.2 * np.cos(2 * t[:, None] - t[None, :])
        - 0.4 * np.sin(t[:, None] + 2 * t[None, :])
    )
    c, s = rfft2D(two, mpol=3, ntor=3)
    assert_allclose(irfft2D(c, s, 32, 32), two, atol=1e-13)
