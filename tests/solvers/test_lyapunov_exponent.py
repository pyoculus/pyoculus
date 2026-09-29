"""Exact linear dynamics fix Lyapunov growth rates and normalization."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyoculus.solvers import (
    LyapunovExponent,
)
from tests.helpers import LinearMap


@pytest.mark.parametrize(
    "matrix,expected",
    [
        (np.array([[0.0, -1.0], [1.0, 0.0]]), 0.0),
        (np.diag([2.0, 0.5]), np.log(2) / 0.4),
    ],
)
@pytest.mark.xfail(
    not hasattr(np, "NaN"),
    strict=True,
    raises=AttributeError,
    reason="LyapunovExponent uses np.NaN, removed in NumPy 2",
)
def test_lyapunov_exact_exponents(matrix, expected):
    """Rigid rotation has zero growth; a saddle grows as log(lambda)/Delta_phi.

    Start at the exact fixed point to avoid roundoff from a diverging base orbit.
    A fixed RNG seed controls only the initial direction; the last averaging
    window suppresses its finite-time alignment transient to below 0.03.
    """
    state = np.random.get_state()
    try:
        np.random.seed(7)
        result = LyapunovExponent(
            LinearMap(matrix, dzeta=0.4), nsave=8, every=50
        ).compute(np.zeros(2))
    finally:
        np.random.set_state(state)
    assert_allclose(result[-1], expected, atol=0.03)


@pytest.mark.xfail(
    strict=True,
    raises=(AttributeError, UnboundLocalError),
    reason="Explicit eps leaves d0 undefined; NumPy 2 fails earlier at np.NaN",
)
def test_lyapunov_explicit_direction():
    """An eigenvector displacement has exactly log(2) growth per map step."""
    result = LyapunovExponent(LinearMap(np.diag([2.0, 0.5])), nsave=2, every=4).compute(
        np.zeros(2), eps=np.array([1e-7, 0.0])
    )
    assert_allclose(result, np.log(2), atol=1e-12)
