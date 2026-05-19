## file spectre_bfield.py
#  Set up the SPECTRE magnetic field, which directly interfaces to ah SPECTRE object
#  in python memory
#  @copyright 2026 EPFL all rights reserved
#  
#  author Chris Smiet (christopher.smiet@epfl.ch)

from .toroidal_bfield import ToroidalBfield
import numpy as np
#from spectre import SPECTRE, SPECTRE_pyoculus_helper


class SpectreBfield(ToroidalBfield):
    """
    SPECTRE magnetic field class, which directly interfaces to a SPECTRE object in python memory

    This class only deals with a single SPECTRE volume at a time, in which the field is given in s, theta, zeta coordinates.
    The radial coordinate s goes from -1 to 1. 
    Theta and zeta coordinates from 0 to 2 pi.

    This class is a wrapper around the SPECTRE_helper, provided by the SPECTRE code, 
    which provides the functions to evaluate the magnetic field at given coordinates. 
    """
    ## the problem size, 2 for 1.5D/2D Hamiltonian system

    def __init__(self, SPECTRE_helper, lvol):
        """
        Initialize the SPECTRE magnetic field

        Parameters
        ----------
        SPECTRE_helper : SPECTRE_pyoculus_helper
            The SPECTRE helper object, which provides the functions to evaluate the magnetic field at given coordinates.
        lvol : int
            The index of the SPECTRE volume to use, starting from 0.
        """
        super().__init__(1)
        self.helper = SPECTRE_helper
        self.lvol = lvol
        self.B = self.helper.B
        self.dBdX = self.helper.dBdX
        self.Nfp = self.helper.Nfp
    
    @classmethod
    def from_h5_file(cls, h5_file, lvol):
        """
        Initialize the SPECTRE magnetic field from a h5 file

        Parameters
        ----------
        h5_file : str
            The path to the h5 file containing the SPECTRE data.
        lvol : int
            The index of the SPECTRE volume to use, starting from 0.

        Returns
        -------
        SpectreBfield
            The initialized SPECTRE magnetic field object.
        """
        from spectre.postprocess.main import SPECTREout
        from spectre import SPECTRE_pyoculus_helper
        helper = SPECTRE_pyoculus_helper(SPECTREout(h5_file), lvol)
        return cls(helper, lvol)

    def B(self, coords):
        return self.helper.B(coords)

    def dBdX(self, coords):
        return self.helper.dBdX(coords)

    def A(self, coords):
        return self.helper.A(coords) 
    
    def convert_coords(self, coords):
        return self.helper.convert_coords(coords)
    
    
