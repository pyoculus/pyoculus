"""Fortran adapter metadata must preserve volume and Fourier conventions."""

from types import SimpleNamespace as NS
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_fortran_setup_preserves_volume_and_fourier_modes(monkeypatch):
    """One-based volume selection and transposition must preserve mode amplitudes.

    A stand-in for Fortran storage checks the boundary protocol only; it is not
    presented as an equilibrium or field-physics validation.
    """
    import pyoculus.utils.fortran as bridge

    variables = NS()
    monkeypatch.setattr(bridge, "fortran_module", NS(specvariables=variables))
    harmonics = np.arange(6.0).reshape(2, 3)
    physics = NS(
        Ntor=1, Mpol=1, Igeometry=3, Istellsym=1, Nfp=5, Lrad=[2, 3], rpol=1.0, rtor=2.0
    )
    output = NS(
        Mvol=2,
        mn=2,
        im=np.array([0, 1]),
        in_=np.array([0, 5]),
        Rbc=harmonics,
        Zbc=2 * harmonics,
        Rbs=3 * harmonics,
        Zbs=4 * harmonics,
    )
    potential = NS(
        **{key: [harmonics, 10 + harmonics] for key in ["Ate", "Ato", "Aze", "Azo"]}
    )
    data = NS(
        version=3.1,
        input=NS(physics=physics),
        output=output,
        vector_potential=potential,
    )
    bridge.setup_fortran_module(data, 2)
    assert variables.ivol == 2 and variables.nfp == 5
    assert not variables.lcoordinatesingularity
    assert_allclose(variables.ate, (10 + harmonics).T, atol=0)
    assert_allclose(variables.irbc, harmonics.T, atol=0)
    with pytest.raises(Exception, match="out of bound"):
        bridge.setup_fortran_module(data, 3)
