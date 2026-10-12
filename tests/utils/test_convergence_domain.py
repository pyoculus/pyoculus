"""An exact saddle fixes the expected basin of Newton convergence."""

import numpy as np
from numpy.testing import assert_allclose
from pyoculus.utils.convergence_domain import compute_convergence_domain
from tests.helpers import LinearMap


def test_convergence_domain_unique_saddle():
    """Every Newton seed of an invertible linear saddle should reach its origin."""

    class SaddleFlow(LinearMap):
        # FixedPoint's default root method evaluates +/- half a return period.
        # This is an exact continuous saddle flow, not a fractional discrete map.
        def __init__(self):
            super().__init__(np.diag([2.0, 0.5]))
            self.is_discrete = False

        def f(self, t, y):
            return np.array([2.0**t, 0.5**t]) * y

        def df(self, t, y):
            return np.diag([2.0**t, 0.5**t])

    m = SaddleFlow()
    grid, points = compute_convergence_domain(
        m, [-0.2, 0.3], [-0.1, 0.4], find_with_iota=False, t=1, tol=1e-10
    )
    assert len(points) == 1
    assert_allclose(points[0].coords[0], 0, atol=1e-10)
    assert_allclose(np.asarray(grid[2], dtype=float), 0, atol=0)
