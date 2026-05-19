## @file toroidal_bfield.py
#  @brief containing a problem class with magnetic fields in two cyclical coordinates for pyoculus ODE solver
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from .magnetic_field import MagneticField

class ToroidalBfield(MagneticField):
    """
    Class that sets up a magnetic field in toroidal system :math:`(s, \\theta, \\zeta)`.
    """
    def __init__(self, Nfp=1):
        """
        Initializes the ToroidalBfield class.
        
        Args:
            Nfp (int): The number of field periods. Default is 1. This parameter defines the periodicity of the magnetic field (:math:`T = 2*\\pi/n_\\text{fp}`).
        """
        self.Nfp = Nfp