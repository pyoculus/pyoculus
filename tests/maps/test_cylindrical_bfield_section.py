import pytest
import numpy as np
from pyoculus.maps.cylindrical_bfield_section import CylindricalBfieldSection
from pyoculus.fields import AnalyticCylindricalBfield


class TestCylindricalBfieldSection:

    @pytest.fixture(autouse=True)
    def setUp(self):
        """
        Set up the test case with a default AnalyticCylindricalBfield object and a CylindricalBfieldSection object.
        """
        self.R=5
        self.Z=0
        self.sf = 1.2
        self.shear = 1.
        self.tol = 1e-8
        self.mf = AnalyticCylindricalBfield(R=self.R, Z=self.Z, sf=self.sf, shear=self.shear)
        self.cylindrical_bfield_section = CylindricalBfieldSection(self.mf, R0=self.R, Z0=self.Z)

    def test_initialization(self):
        """
        sanity tests
        """
        assert self.cylindrical_bfield_section._mf.sf == self.sf
        assert self.cylindrical_bfield_section._mf.shear == self.shear
    
    def test_f(self):
        """
        axis of the toy field is at (5,0) and should come back to the same point after one turn
        """
        y0 = [5.0, 0.0]
        result = self.cylindrical_bfield_section.f(1, y0)
        assert len(result) == 2
        assert np.isclose(result[0], 5.0, atol = 1e-7)
        assert np.isclose(result[1], 0.0, atol = 1e-7)


    def test_df_map(self):
        """
        test the jacobian of the mapping; the rotational transform (1/sf) should be equal
        to the 2*pi*arccos(trace(J)/2). 
        Modulo because arccos is not unique."""
        y0 = [6.0, 0.5]
        jac = self.cylindrical_bfield_section.df(1, y0)
        f_y = self.cylindrical_bfield_section.f(1, y0)
        assert jac.shape == (2, 2)
        randangle = 2*np.pi*np.random.random()
        randvec = np.array([np.cos(randangle), np.sin(randangle) ])* 1e-4
        f_y_plus_delta = self.cylindrical_bfield_section.f(1, np.array(y0)+randvec)  # convert to numpy so you can add arrays sensibly. Also tests if f takes numpy arrays like it should

        # check that the jacobian is correct
        assert np.allclose(f_y_plus_delta-f_y, jac@randvec, atol=1e-6)
    
    def test_df_rottrans(self):
        """
        test the jacobian of the mapping; the rotational transform (1/sf) should be equal
        to the 2*pi*arccos(trace(J)/2). 
        Modulo because arccos is not unique."""
        y0 = [5.0, 0.0]   # axis of the toy field
        jac = self.cylindrical_bfield_section.df(1, y0)
        rottrans_df = np.arccos(np.trace(jac)/2)/(2*np.pi)  # rotational transform formula
        rottrans_map = 1- 1/self.sf
        assert np.isclose(rottrans_df, rottrans_map, atol = 1e-4)

    def test_lagrangian(self):
        # Does not test physics, only return shape
        y0 = [5.01, 0.0]
        result = self.cylindrical_bfield_section.lagrangian(y0, 1)
        assert result.size == 1

    def test_winding(self):
        y0 = [5.00001, 0.0]
        result = self.cylindrical_bfield_section.winding(1, y0)
        assert len(result) == 2
        assert np.isclose(result[1]/(2*np.pi), 1/self.sf, atol = 1e-5)   # remember result is in radians


    def test_dwinding(self):
        y0 = [5.1, 0.0]
        result = self.cylindrical_bfield_section.dwinding(1, y0)
        assert result.shape == (2, 2)

    def test_clear_cache(self):
        self.cylindrical_bfield_section.clear_cache()
        assert len(self.cylindrical_bfield_section.cache.cache) == 0
    
    def test_find_axis(self):
        self.cylindrical_bfield_section.find_axis(guess=[5.5,0])
        assert np.isclose(self.cylindrical_bfield_section.R0, 5.0, atol=1e-5)
        assert np.isclose(self.cylindrical_bfield_section.Z0, 0.0, atol=1e-5)

    def test_lagrangian_integration(self):
        """
        replace the vector potential with dl/||dl||_2 and confirm that the integral around
        the axis is 2*pi*R0
        """
        y0 = [5.0, 0.0]
        self.cylindrical_bfield_section._mf.A = lambda x: np.array([0, 1 / x[0], 0]) # Set A to dl/||dl||_2 
        result = self.cylindrical_bfield_section.lagrangian(y0, 1)  # reduces to \int dl over circle with radius R
        assert np.isclose(result, 2*np.pi*self.R, atol=1e-5)

