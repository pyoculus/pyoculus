"""Analytic field fixture for return-map tests."""

import pytest
from tests.helpers import RadialField


@pytest.fixture
def radial_field():
    return RadialField()
