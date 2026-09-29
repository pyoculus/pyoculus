"""Greene's periodic approximants recover analytically known integrable surfaces."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyoculus.fields import TwoWaves
from pyoculus.maps import ToroidalBfieldSection
from pyoculus.solvers import (
    FixedPoint,
    FluxSurfaceGR,
)


def test_greene_integrable_convergents():
    """Golden-mean approximants lie at rho=n/m and have parabolic monodromy."""
    section = ToroidalBfieldSection(TwoWaves(k=0), rtol=1e-11)
    seeds = []
    for n, m in [(1, 2), (2, 3)]:
        fp = FixedPoint(section)
        fp.find_with_iota(n, m, np.array([n / m, 0.0]), tol=1e-9)
        assert fp.successful
        seeds.append(fp)
    result = FluxSurfaceGR(section).GreeneMethod(
        (np.sqrt(5) - 1) / 2, *seeds, nexpand=6, tol=1e-9
    )
    assert [(p.n, p.m) for p in result.fixedpoints] == [(1, 2), (2, 3), (3, 5), (5, 8)]
    for p in result.fixedpoints:
        assert_allclose(p.coords[:, 0], p.n / p.m, atol=1e-9)
        assert_allclose(p.GreenesResidue, 0, atol=1e-9)
        assert_allclose(
            section.winding(p.m, p.coords[0])[1], 2 * np.pi * p.n, atol=1e-8
        )


def test_greene_rejects_nonconsecutive_seeds():
    """Continuation requires consecutive convergents, not arbitrary rational tori."""
    section = ToroidalBfieldSection(TwoWaves(k=0))
    seeds = []
    for n, m in [(1, 2), (3, 5)]:
        fp = FixedPoint(section)
        fp.find_with_iota(n, m, np.array([n / m, 0.0]), tol=1e-9)
        assert fp.successful
        seeds.append(fp)
    with pytest.raises(ValueError, match="continued fraction"):
        FluxSurfaceGR(section).GreeneMethod((np.sqrt(5) - 1) / 2, *seeds, nexpand=6)
