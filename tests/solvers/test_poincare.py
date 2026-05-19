from pyoculus.solvers import PoincarePlot
from pyoculus.fields import AnalyticCylindricalBfield
from pyoculus.maps import CylindricalBfieldSection
import pytest
import numpy as np

class TestPoincarePlot:
    
        @pytest.fixture(autouse=True)
        def setUp(self):
            """
            Set up the test case with a default AnalyticCylindricalBfield object and a PoincarePlot object.
            """
            self.R = 5.
            self.Z = 0.
            self.sf = 1.2
            self.shear = 1.
            self.ntraj = 4
            self.mf = AnalyticCylindricalBfield(R=self.R, Z=self.Z, sf=self.sf, shear=self.shear)
            self.section = CylindricalBfieldSection(self.mf, R0=self.R, Z0=self.Z)
            self.rhos = np.linspace(1e-5, 1, self.ntraj)
            self.xs = np.stack([self.rhos+self.R, np.zeros(self.ntraj)], axis=1)
            self.poincare_plot = PoincarePlot(self.section, self.xs)
    
        def test_initialization(self):
            """
            sanity tests
            """
            assert self.mf.sf == self.sf
            assert self.mf.shear == self.shear

        def test_with_horizontal(self):
            """
            test the helper classmethods
            """
            horizontalplot = PoincarePlot.with_horizontal(self.section, 1, self.ntraj)
            assert len(horizontalplot.xs) == self.ntraj
            assert np.all(horizontalplot.xs[:, 1] == self.Z)

        def test_with_sections(self):
             """
             not implemented yet
             """
             pass
        
        def test_compute_iota(self):
            """
            test the helper classmethods
            """
            xs, iotas = self.poincare_plot.compute_iota(npts=60)
            toybox_expectation_q = self.sf + self.shear / 2 * self.rhos**2
            toybox_expectation_iota = 1/toybox_expectation_q
            np.testing.assert_allclose(iotas, toybox_expectation_iota, atol=1.1e-5)
        
        def test_plot(self):
            """
            test the plotting function, which should not error
            """
            self.poincare_plot.compute(npts=10)
            fig, ax = self.poincare_plot.plot()
            assert fig is not None
            assert ax is not None

        

            