"""Independent analytic fields, maps, and finite-difference references for tests."""

import numpy as np
from pyoculus.fields import CylindricalBfield
from pyoculus.maps import BaseMap


def derivative(fun, x, h=2e-5):
    """Centered differences, with output-component rows (no production AD)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack(
        [
            (np.asarray(fun(x + h * e)) - fun(x - h * e)) / (2 * h)
            for e in np.eye(len(x))
        ]
    )


def cylindrical_curl(A, x):
    """Curl of contravariant A, returned in the same coordinate basis."""
    R = x[0]

    def covariant(y):
        return np.asarray(A(y)) * [1, y[0] ** 2, 1]

    d = derivative(covariant, x)
    return np.array(
        [(d[2, 1] - d[1, 2]) / R, (d[0, 2] - d[2, 0]) / R, (d[1, 0] - d[0, 1]) / R]
    )


class RadialField(CylindricalBfield):
    """Solenoidal field with dR/dphi=a*R, dZ/dphi=0, flux weight 1/R."""

    def __init__(self, a=0.03, Nfp=1):
        super().__init__(Nfp)
        self.a = a

    def B(self, x):
        R = x[0]
        return np.array([self.a / R, 1 / R**2, 0.0])

    def dBdX(self, x):
        R = x[0]
        d = np.zeros((3, 3))
        d[:, 0] = [-self.a / R**2, -2 / R**3, 0]
        return self.B(x), d

    def A(self, x):
        R, _, Z = x
        return np.array([0.0, -self.a * Z / R**2, -np.log(R)])


class LinearMap(BaseMap):
    """Exact area-preserving linear dynamics; no angle wrapping or clipping."""

    def __init__(self, matrix, dzeta=1):
        super().__init__(dim=2, is_discrete=True, dzeta=dzeta)
        self.matrix = np.asarray(matrix)

    def f(self, t, y):
        return np.linalg.matrix_power(self.matrix, t) @ y

    def df(self, t, y):
        return np.linalg.matrix_power(self.matrix, t)
