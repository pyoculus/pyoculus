"""
magnetic_field.py
=================

This module contains the abstract class for magnetic fields.

:authors:
    - Zhisong Qu (zhisong.qu@anu.edu.au)
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

from abc import ABC, abstractmethod
import numpy as np

class MagneticField(ABC):
    # def __init__(self):
    #     """
    #     Initializes the MagneticField class.
    #     """
    #     ## if the output magnetic field contains the jacobian factor or not
    #     # self.has_jacobian = False

    @abstractmethod
    def B(self, coords, *args):
        """
        Returns the contravariant magnetic fields at the given coordinates.

        Args:
            coords (array): The coordinates at which to calculate the magnetic fields.
            *args: Additional parameters.
        """
        raise NotImplementedError("A problem class should implement member function B.")

    @abstractmethod
    def dBdX(self, coords, *args):
        """
        Returns the contravariant components of the magnetic fields and their derivatives at the given coordinates.

        Args:
            coords (array): The coordinates at which to calculate the magnetic fields and their derivatives.
            *args: Additional parameters.

        Returns:
            array: The contravariant components of the magnetic field
            array: The contravariant components of the derivative of the magnetic fields
        """
        raise NotImplementedError(
            "A problem class should implement member function dBdX."
        )

    @abstractmethod
    def A(self, coords, *args):
        """
        Returns the contravariant components of the vector potential at given coordinates.

        Args:
            coords (array): The coordinates at which to calculate the vector potential.
            *args: Additional parameters.
        """
        raise NotImplementedError("Vector potential is not implemented.")

    # Many points implementation

    def B_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """
        Returns the contravariant magnetic fields at multiple coordinates.

        Args:
            x1arr, x2arr, x3arr (arrays): The coordinates at which to calculate the magnetic fields.
            input1D (bool, optional): If False, create a meshgrid with x1arr, x2arr and x3arr. If True, treat them as a list of points.
            *args: Additional parameters.
        """
        if input1D:
            xs = np.array([x1arr, x2arr, x3arr]).T
        else:
            xs = np.array(np.meshgrid(x1arr, x2arr, x3arr)).T.reshape(-1, 3)
        
        return np.array([self.B(x, *args) for x in xs])


    def dBdX_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """
        Returns the contravariant magnetic fields and their derivatives at multiple coordinates.

        Args:
            x1arr, x2arr, x3arr (arrays): The coordinates at which to calculate the magnetic fields and their derivatives.
            input1D (bool, optional): If False, create a meshgrid with x1arr, x2arr and x3arr. If True, treat them as a list of points.
            *args: Additional parameters.
        """
        raise NotImplementedError("dBdX_many is not implemented.")

    def A_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """
        Returns the contravariant vector potential at multiple coordinates.

        Args:
            x1arr, x2arr, x3arr (arrays): The coordinates at which to calculate the vector potential.
            input1D (bool, optional): If False, create a meshgrid with x1arr, x2arr and x3arr. If True, treat them as a list of points.
            *args: Additional parameters.
        """
        raise NotImplementedError("A_many is not implemented.")
