"""Fresh perturbation specifications for field-state tests."""

import pytest


@pytest.fixture
def perturbation():
    return dict(type="maxwell-boltzmann", m=6, n=-1, d=2.0, amplitude=0.1)
