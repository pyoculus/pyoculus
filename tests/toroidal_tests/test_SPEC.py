import pytest
import numpy as np

py_spec = pytest.importorskip("py_spec")
SPECout = py_spec.SPECout
from pyoculus.fields.spec_bfield import SpecBfield
from pyoculus.solvers import FixedPoint
import os

from pyoculus.maps.toroidal_bfield_section import ToroidalBfieldSection

SPEC_FILES = [
    "../test_files/cyl_SPEC.h5",
    "../test_files/slab_SPEC.h5",
    "../test_files/tor_SPEC.h5",
    "../test_files/oldh5_SPEC.h5"
]

@pytest.mark.parametrize("spec_file", SPEC_FILES)
def test_spec_bfield_init(spec_file):
    """Test initialization of SpecBfield for different SPEC files."""
    
    file_path = os.path.join(os.path.dirname(__file__), spec_file)
    spec_data = SPECout(file_path)
    lvol = 1
    
    # Increase lvol until a valid volume is found
    while spec_data.input.physics.Lrad[lvol - 1] == 0:
        lvol +=1

    if spec_file == "../test_files/oldh5_SPEC.h5":
        with pytest.raises(Exception, match="SPEC version >=3.0 is needed"):
            SpecBfield(spec_data, lvol)
    else:
        try:
            bfield = SpecBfield(spec_data, lvol)
            assert bfield is not None
        except Exception as e:
            pytest.fail(f"SpecBfield initialization failed for {spec_file} with error: {e}")

@pytest.mark.parametrize("spec_file", SPEC_FILES)
def test_spec_bfield_methods(spec_file):
    """Test methods of SpecBfield."""
    if spec_file == "../test_files/oldh5_SPEC.h5":
        pytest.skip("skip testing deprecated file")
    
    file_path = os.path.join(os.path.dirname(__file__), spec_file)
    spec_data = SPECout(file_path)
    lvol = 1
    
    # Increase lvol until a valid volume is found
    while spec_data.input.physics.Lrad[lvol - 1] == 0:
        lvol +=1

    bfield = SpecBfield(spec_data, lvol)

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

    # Test B_many method
    x1 = np.array([0.5, 0.6])
    x2 = np.array([0.1, 0.2])
    x3 = np.array([0.2, 0.3])
    B_many = bfield.B_many(x1, x2, x3)
    assert isinstance(B_many, np.ndarray)
    assert B_many.shape == (len(x1), 3)

    # Test dBdX_many method
    B_many, dB_many = bfield.dBdX_many(x1, x2, x3)
    assert isinstance(B_many, np.ndarray)
    assert B_many.shape == (len(x1), 3)
    assert isinstance(dB_many, np.ndarray)
    assert dB_many.shape == (len(x1), 3, 3)
    
    # Test convert_coords
    stz = np.array([0.5, 0.1, 0.2], dtype=np.float64)
    xyz = bfield.convert_coords(stz)
    assert isinstance(xyz, np.ndarray)
    assert xyz.shape == (3,)


@pytest.mark.parametrize("spec_file", SPEC_FILES)
def test_integration_methods(spec_file):
    """ 
    test creating a map from the field
    """
    if spec_file == "../test_files/oldh5_SPEC.h5":
        pytest.skip("skip testing deprecated file")

    file_path = os.path.join(os.path.dirname(__file__), spec_file)
    spec_data = SPECout(file_path)
    lvol = 1
    
    bfield = SpecBfield(spec_data, lvol)

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
    assert np.allclose(f_p_plus_delta, lin_approx, atol=1e-11)


def test_solver_fixpoint():
    """
    test finding the axis in the inner volume
    """
    spec_file = "../test_files/tor_SPEC.h5"

    file_path = os.path.join(os.path.dirname(__file__), spec_file)
    spec_data = SPECout(file_path)
    lvol = 1

    bfield = SpecBfield(spec_data, lvol)
    section = ToroidalBfieldSection(bfield, phi0=0.0)
    guess=[-0.989, 0.1]  #  has to be close to the actual fixed point
    fixed_point = FixedPoint(section)

    fixed_point.find_with_iota(n=0, m=1, guess=guess)
    assert fixed_point.successful
