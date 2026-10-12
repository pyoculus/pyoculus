"""Record the legacy pressure-jump Hamiltonian entrypoint blocker."""

import pytest


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="SPECPJH imports removed pyoculus.maps.spec_problem",
)
def test_spec_pjh_backend_entrypoint():
    """The pressure-jump Hamiltonian must be importable before testing its physics."""
    from pyoculus.maps.spec_pjh import SPECPJH

    assert callable(SPECPJH)
