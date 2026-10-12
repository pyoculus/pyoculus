"""Record the legacy M3DC1 construction blocker."""

from types import SimpleNamespace as NS
import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="M3DC1Bfield lacks abstract A/dBdX implementations; constructor is obsolete",
)
def test_m3dc1_backend_entrypoint():
    """A provider's field should survive adapter construction unchanged.

    Construction currently blocks further physics checks; this records that
    limitation without claiming to validate the external field solver.
    """
    from pyoculus.fields.m3dc1_bfield import M3DC1Bfield

    provider = NS(evaluate=lambda x: np.array([0.0, 1.0, 0.0]))
    field = M3DC1Bfield(provider, R0=6.0, Z0=0.0)
    assert_allclose(field.B([6.0, 0.0, 0.0]), [0, 1, 0], atol=0)
