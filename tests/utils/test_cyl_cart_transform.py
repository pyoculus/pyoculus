"""Cartesian and cylindrical coordinate Jacobians obey the chain rule."""

import numpy as np
from numpy.testing import assert_allclose
from pyoculus.utils import cyl_cart_transform as cct
from tests.helpers import derivative


def test_cartesian_cylindrical_chain_rule():
    """The coordinate Jacobian is the inverse of the cylindrical basis matrix."""
    p = np.array([2.3, 0.7, -0.4])
    xyz = cct.xyz(*p)
    assert_allclose(cct.rphiz(*xyz), p, atol=1e-13)
    assert_allclose(
        cct.xyz_inv_jac(*p), derivative(lambda y: cct.xyz(*y), p), atol=2e-9
    )
    assert_allclose(cct.xyz_jac(*p) @ cct.xyz_inv_jac(*p), np.eye(3), atol=1e-13)
    assert_allclose(cct.rphiz_jac(*xyz), cct.xyz_jac(*p), atol=1e-13)
