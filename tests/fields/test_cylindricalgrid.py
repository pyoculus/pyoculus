import unittest
import numpy as np
from pyoculus.maps.cylindrical_bfield_section import CylindricalBfieldSection
from pyoculus.fields import AnalyticCylindricalBfield
from pyoculus.fields import AxisymmetricCylindricalGridField

class TestCylindricalGridInterpolatedField(unittest.TestCase):

    def setUp(self):
        """
        Set up the test case with a default AxisymmetricCylindricalGridField object and a CylindricalBfieldSection object.
        """

        self.R=np.linspace(0.5,1.5,10)
        self.Z=np.linspace(-1,+1,15) 
        self.B_R = np.zeros((10,15))
        self.B_Z = np.zeros((10,15))
        self.B_phi =(1/self.R*np.ones((15,10))).T
        self.F_psi = np.zeros((10,15))
        self.mf = AxisymmetricCylindricalGridField(R=self.R, Z=self.Z, B_R=self.B_R, B_Z=self.B_Z, B_phi=self.B_phi,F_psi=self.F_psi)
        self.cylindrical_bfield_section = CylindricalBfieldSection(self.mf, R0=1, Z0=0)# tol=self.tol)

    
    def test_f(self):
        """
        axis of the toy field is at (5,0) and should come back to the same point after one turn
        """
        y0 = [1,0]
        result = self.cylindrical_bfield_section.f(1, y0)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 1.0, places = 7)
        self.assertAlmostEqual(result[1], 0.0, places = 7)


    def test_df_map(self):
        """
        test the jacobian of the mapping; the rotational transform (1/sf) should be equal
        to the 2*pi*arccos(trace(J)/2). 
        Modulo because arccos is not unique."""
        y0 = [1,0]
        jac = self.cylindrical_bfield_section.df(1, y0)
        f_y = self.cylindrical_bfield_section.f(1, y0)
        self.assertEqual(jac.shape, (2, 2))
        randangle = 2*np.pi*np.random.random()
        randvec = np.array([np.cos(randangle), np.sin(randangle) ])* 1e-4
        f_y_plus_delta = self.cylindrical_bfield_section.f(1, y0+randvec)  # convert to numpy so you can add arrays sensibly. Also tests if f takes numpy arrays like it should

        # check that the jacobian is correct
        self.assertTrue(np.allclose(f_y_plus_delta-f_y, jac@randvec, atol=1e-6))
    


    def test_lagrangian(self):
        # Does not test physics, only return shape
        y0 = [1,0]
        result = self.cylindrical_bfield_section.lagrangian(y0, 1)
        self.assertEqual(result.size, 1)



    def test_clear_cache(self):
        self.cylindrical_bfield_section.clear_cache()
        self.assertEqual(len(self.cylindrical_bfield_section.cache.cache), 0)
    

    def test_lagrangian_integration(self):
        """
        replace the vector potential with dl/||dl||_2 and confirm that the integral around
        the axis is 2*pi*R0
        """
        y0 = [1,0]
        self.cylindrical_bfield_section._mf.A = lambda x: np.array([0, 1 / x[0], 0]) # Set A to dl/||dl||_2 

        result = self.cylindrical_bfield_section.lagrangian(y0, 1)  # reduces to \int dl over circle with radius R
        self.assertAlmostEqual(result, 2*np.pi, places=5)

    def test_load_from_mat(self):
        """
        Test the loading from a .mat file
        """
        JFField = AxisymmetricCylindricalGridField.from_matlab_file('./examples/TCV_jellyfish/JF_75979_120.mat', with_perturbation=True)


if __name__ == '__main__':
    unittest.main()