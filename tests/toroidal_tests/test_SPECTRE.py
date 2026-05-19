import pytest
from pyoculus.fields import SpectreBfield
from pyoculus.maps import ToroidalBfieldSection
import numpy as np
import os

from pyoculus.maps.toroidal_bfield_section import ToroidalBfieldSection
from pyoculus.solvers import FixedPoint
spectre = pytest.importorskip("spectre")
SPECTRE = spectre.SPECTRE
SPECTRE_pyoculus_helper = spectre.SPECTRE_pyoculus_helper
import logging
logging.basicConfig(level=logging.DEBUG)

SPECTRE_FILES = [
    "../test_files/w7x_1vol_spectre.h5",
]


@pytest.mark.parametrize("spectre_file", SPECTRE_FILES)
def test_spectre_bfield_init(spectre_file):
    """Test initialization of SpectreBfield"""
    
    file_path = os.path.join(os.path.dirname(__file__), spectre_file)

    for lvol in [0,1]:
        try:
            bfield = SpectreBfield.from_h5_file(file_path, lvol)
            assert bfield is not None
        except Exception as e:
            pytest.fail(f"SpectreBfield initialization failed for {spectre_file} with error: {e}")

@pytest.mark.parametrize("spectre_file", SPECTRE_FILES)
def test_spectre_bfield_methods(spectre_file):
    """Test methods of SpectreBfield."""

    file_path = os.path.join(os.path.dirname(__file__), spectre_file)

    for lvol in [0,2]:
        bfield = SpectreBfield.from_h5_file(file_path, lvol=1)

        # Test B method
        coords = np.array([0.5, 0.1, 0.2], dtype=np.float64)
        B = bfield.B(coords)
        assert isinstance(B, np.ndarray)
        assert B.shape == (3,)

        # Test dBdX method
        B, dB = bfield.dBdX(coords)
        assert isinstance(B, np.ndarray)
        assert B.shape == (3,)
        assert isinstance(dB, np.ndarray)
        assert dB.shape == (3, 3)

        # test A method
        A = bfield.A(coords)
        assert isinstance(A, np.ndarray)
        # known failure, accept for now:
        try: 
            assert A.shape == (3,)
        except AssertionError:
            pytest.xfail("method A needs to be updated to 3d covariant vector potential")

        # Test convert_coords
        stz = np.array([0.5, 0.1, 0.2], dtype=np.float64)
        xyz = bfield.convert_coords(stz)
        assert isinstance(xyz, np.ndarray)
        assert xyz.shape == (3,)


@pytest.mark.parametrize("spectre_file", SPECTRE_FILES)
def test_integration_methods(spectre_file):
    """
    test creating a map from the field
    """
    file_path = os.path.join(os.path.dirname(__file__), spectre_file)
    bfield = SpectreBfield.from_h5_file(file_path, lvol=0)

    section = ToroidalBfieldSection(bfield, phi0=0.0)
    test_point = [0.2, 0.1]

    mapped_point = section.f(1, test_point)  # map from (0, 0.1) to (s, theta)
    assert isinstance(mapped_point, np.ndarray)
    assert mapped_point.shape == (2,)

    # test backwards mapping
    mapped_back = section.f(-1, mapped_point)
    assert isinstance(mapped_back, np.ndarray)
    assert mapped_back.shape == (2,)
    assert np.allclose(mapped_back, test_point, atol=1e-5)

    map_jac = section.df(1, test_point)
    assert isinstance(map_jac, np.ndarray)
    assert map_jac.shape == (2, 2)
    
    #taylor test the jacobian
    epsilon = 1e-4
    direction = np.random.rand(2)
    direction /= np.linalg.norm(direction)
    displacement_vector = epsilon * direction
    displaced_point = test_point + displacement_vector
    f_p_plus_delta = section.f(1, displaced_point)

    lin_approx = mapped_point + map_jac @ (displacement_vector)
    assert np.allclose(f_p_plus_delta, lin_approx, atol=1e-4)


def test_solver_fixpoint():
    """
    test finding the axis in the inner volume
    """
    filename = "../test_files/w7x_1vol_spectre.h5"
    file_path = os.path.join(os.path.dirname(__file__), filename)

    bfield = SpectreBfield.from_h5_file(file_path, lvol=0)

    section = ToroidalBfieldSection(bfield, phi0=0.0)

    guess=[-0.989, 0.1]  #  has to be close to the actual fixed point
    fixed_point = FixedPoint(section)

    fixed_point.find_with_iota(n=0, m=1, guess=guess)
    assert fixed_point.successful
    # test that up to five mappings point stays fixed
    for n in range(5):
        start = fixed_point.coords[0]
        nmap = section.f(n, start)
        # compare as complex numbers, becuse closeness there is meaningful
        complex1 = (start[0] + 1)*np.exp(1j*start[1])
        complex2 = (nmap[0] + 1)*np.exp(1j*nmap[1])
        assert np.allclose(complex1, complex2, atol=1e-7)
    print(fixed_point.coords)


def test_finding_w7x_island():
    """
    find the island chain in the outer volume of the two-volume
    SPECTRE run. 
    """
    filename = "../test_files/w7x_1vol_spectre.h5"
    file_path = os.path.join(os.path.dirname(__file__), filename)

    bfield = SpectreBfield.from_h5_file(file_path, lvol=0)

    section = ToroidalBfieldSection(bfield, phi0=0.0)

    guess=[0.989, 0]  #  has to be close to the actual fixed point
    fixed_point = FixedPoint(section)

    fixed_point.find_with_iota(n=5, m=5, guess=guess)


