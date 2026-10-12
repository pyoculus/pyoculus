"""Exact integrable trajectories check the Poincare sampling convention."""

import numpy as np
from numpy.testing import assert_allclose
from pyoculus.fields import TwoWaves
from pyoculus.maps import ToroidalBfieldSection
from pyoculus.solvers import (
    PoincarePlot,
)


def test_poincare_integrable_trajectories():
    """Poincare sampling must retain radius and advance the angle each full turn."""
    section = ToroidalBfieldSection(TwoWaves(k=0), rtol=1e-11)
    starts = np.array([[0.2, 0.3], [0.37, 0.4]])
    p = PoincarePlot(section, starts)
    p.compute(5)
    hits = p._hits
    assert np.isfinite(hits).all()
    assert_allclose(
        hits[:, :, 0],
        np.broadcast_to(starts[:, 0, None], hits[:, :, 0].shape),
        atol=2e-9,
    )
    # Successive angular increments avoid relying on whether the initial hit
    # is stored. Complex phases compare angles without branch-cut artifacts.
    advance = np.exp(1j * np.diff(hits[:, :, 1], axis=1))
    expected = np.broadcast_to(np.exp(2j * np.pi * starts[:, 0, None]), advance.shape)
    assert_allclose(advance, expected, atol=2e-8)
