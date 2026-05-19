"""
base_solver.py
==================

Contains the abstract base class for solvers.

:authors:
    - Zhisong Qu (zhisong.qu@anu.edu.au)
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

from abc import ABC
import pyoculus.maps as maps


class BaseSolver(ABC):
    """
    Abstract base class for solvers.
    """

    class OutputData:
        """
        Class to hold the output data.
        """

        def __init__(self):
            pass

    def __init__(self, map: maps.BaseMap):
        """
        Initializes the BaseSolver object.

        Args:
            map (BaseMap): The map to use for the solver.
        """
        # Flag to note if the computation was performed successfuly
        self._successful = False

        # Check if the map is derived from BaseMap
        if not isinstance(map, maps.BaseMap):
            raise ValueError("The problem is not a derived type of BaseMap class.")

        self._map = map

    @property
    def successful(self):
        """
        Returns a boolean indicating if the solver was successful.
        """
        return self._successful
