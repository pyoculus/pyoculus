from .magnetic_field import MagneticField

class CylindricalBfield(MagneticField):
    """
    Cylindrical magnetic field class. The coordinate system should be :math:`(R, \\phi, Z)`.
    """

    def __init__(self, Nfp=1):
        """
        Initializes the CylindricalBfield object.
        
        Args:
            Nfp (int): The number of field periods. Default is 1. This parameter defines the periodicity of the magnetic field (:math:`T = 2*\\pi/n_\\text{fp}`).
        """
        self.Nfp = Nfp