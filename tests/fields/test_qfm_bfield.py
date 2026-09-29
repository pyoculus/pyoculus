"""An identity QFM coordinate transform should preserve the magnetic field."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyoculus.fields import TwoWaves


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="QFMBfield does not implement abstract method A and cannot be constructed",
)
def test_qfm_field_identity_coordinates():
    """An identity surface transform must leave all field components unchanged."""
    from pyoculus.geo import SurfacesToroidal
    from pyoculus.fields.qfm_bfield import QFMBfield

    surfaces = SurfacesToroidal(mpol=1, ntor=1)
    surfaces.construct_interpolant()
    field = TwoWaves(k=0.002)
    transformed = QFMBfield(field, surfaces)
    x = np.array([0.4, 0.7, 0.2])
    assert_allclose(transformed.B(x), field.B(x), atol=1e-12)
