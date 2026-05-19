import numpy as np
import pytest

py_spec = pytest.importorskip("py_spec")
SPECout = py_spec.SPECout

spectre = pytest.importorskip("spectre")

from pyoculus.fields import SpecBfield, SpectreBfield
from pyoculus.maps import ToroidalBfieldSection
from pyoculus.solvers import PoincarePlot

def test_compare_pplot_trajectories_inner_volume():
    """
    Calculate the trajectories using the SPEC (fortran) methods and 
    the SPECTRE (numba) methods. 
    """
    pp_starts_st = np.linspace(np.array([-.999, 0]), np.array([0.999, 0]), 5)
    out = SPECout('tests/test_files/w7x_1vol_spectre.h5')
    out.input.physics.Istellsym = 1  #hack because the attribute name changed
    spec = SpecBfield(out, 1)   #spec 1-based
    sections = ToroidalBfieldSection(spec)
    pplots = PoincarePlot(sections, pp_starts_st, 5)
    pplots.compute(10)


    spectre = SpectreBfield.from_h5_file('tests/test_files/w7x_1vol_spectre.h5', 0)
    section = ToroidalBfieldSection(spectre)
    pplot = PoincarePlot(section, pp_starts_st, 5)
    pplot.compute(10)

    assert np.allclose(pplots._hits, pplot._hits, atol=1e-5)

#@pytest.mark.xfail(reason="Known failure: trajectory comparison between SPEC and SPECTRE")
def test_compare_pplot_trajectories_outer_volume():
    """
    Calculate the trajectories using the SPEC (fortran) methods and 
    the SPECTRE (numba) methods. 
    """
    num_pts = 5
    pp_starts_st = np.linspace(np.array([-.999, 0]), np.array([0.5, 0]), 5)
    out = SPECout('tests/test_files/w7x_1vol_spectre.h5')
    out.input.physics.Istellsym = 1  #hack because the attribute name changed
    spec = SpecBfield(out, 2)   #spec 1-based
    sections = ToroidalBfieldSection(spec)
    pplot_spec = PoincarePlot(sections, pp_starts_st, 5)
    pplot_spec.compute(num_pts)


    spectre = SpectreBfield.from_h5_file('tests/test_files/w7x_1vol_spectre.h5', 1)
    section = ToroidalBfieldSection(spectre)
    pplot = PoincarePlot(section, pp_starts_st, 3)
    pplot.compute(num_pts)

    assert np.allclose(pplot_spec._hits, pplot._hits, atol=1e-5, equal_nan=True)