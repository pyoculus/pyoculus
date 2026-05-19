"""
base_map.py
==================

Contains the abstract base class for maps.

:authors:
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseMap(ABC):
    """
    Defines an abstract base class for the map subclasses.

    A map object is a function that takes a point in phase space to another point. The transformation can be either discrete or continuous. A continuous transformation allows the map to be applied for a continuous time (:math:`f^t` where :math:`t` is a real number), while a discrete transformation allows the map to be applied only for discrete times (:math:`f^t` where :math:`t` is an integer).

    Attributes:
        dimension (int): The dimension of the phase space that the map operates on.
        is_discrete (bool): A flag indicating whether the map is discrete. If True, the map is discrete; if False, the map is continuous.
        domain (list of tuples, optional): The domain of the map. Each tuple should contain the lower and upper bounds for each dimension. If None, the domain is assumed to be :math:`(-\\infty, \\infty)` for each dimension.
    """

    def __init__(self, dim=2, is_discrete=False, domain=None, periodicity=None, dzeta=1):
        """
        Initializes BaseMap object.

        Args:
            dim (int): Dimension of the map.
            is_discrete (bool): Whether the map is discrete. If True, the map is discrete; if False, the map is continuous.
            domain (list of tuples, optional): The domain of the map. Each tuple should contain the lower and upper bounds for each dimension. If None, the domain is assumed to be :math:`(-\\infty, \\infty)` for each dimension.
        """
        if domain is None:
            domain = [(-np.inf, np.inf)]*dim
        if periodicity is None:
            periodicity = np.zeros(dim)

        self.dimension = dim
        self.is_discrete = is_discrete
        self.domain = domain
        self.periodicity = periodicity
        self.dzeta = dzeta
        
    @abstractmethod
    def f(self, t, y0):
        """
        This method represents the mapping function. It takes a point :math:`y_0` in the domain and returns its image under :math:`t` application of the map.

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The initial point in the phase space.
        """
        raise NotImplementedError("A BaseMap object should have a mapping f method.")

    @abstractmethod
    def df(self, t, y0):
        """
        Computes the Jacobian matrix of the map at :math:`y_0` after :math:`t` applications :math:`df^t = (\\frac{\\partial f^t}{\\partial x})_{i,j}`.

        The 

        .. math::
            J_f(x) = \\begin{bmatrix}
                \\partial_{x^1}f^1 & \\cdots & \\partial_{x^n}f^1 \\\\ 
                \\vdots & \\ddots & \\vdots \\\\
                \\partial_{x^1}f^n & \\cdots & \\partial_{x^n}f^n
            \\end{bmatrix}

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The point in phase space where the Jacobian is computed.
        """
        raise NotImplementedError(
            "A BaseMap object should have a jacobian mapping df method."
        )

    def lagrangian(self, y0, t=None):
        """
        Calculates the Lagrangian at a given point or the difference in Lagrangian between two points. The Lagrangian is as defined in the paper by Meiss (https://doi.org/10.1063/1.4915831).

        Args:
            y0 (array): The point at which to calculate the Lagrangian.
            t (int or float, optional): The number of times the map is applied from :math:`y_0`.

        Returns:
            float: Lagrangian at :math:`y_0`, or the difference in Lagrangian between :math:`y_1 = f^t(y_0)` and :math:`y_0`: :math:`(\\mathcal{L}(y_1)-\\mathcal{L}(y_0))` if :math:`t` is provided.
        """
        raise NotImplementedError("A BaseMap object should have a lagrangian method.")
    
    def winding(self, t, y0, y1=None):
        """
        Calculates how much the point :math:`y_0` winds around the point :math:`y_1` after applying the map :math:`t` times.

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The initial point in the phase space.
            y1 (array): The point around which :math:`y_0` winds. If None, the origin should be used.
        """
        raise NotImplementedError("A BaseMap object may have a winding mapping f_winding method.")
    
    def dwinding(self, t, y0, y1=None):
        """
        Calculates the Jacobian of the winding of :math:`y_0` around :math:`y_1` after applying the map :math:`t` times.

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The initial point in the phase space.
            y1 (array): The point around which :math:`y_0` winds. If None, the origin should be used.
        """
        raise NotImplementedError("A BaseMap object may have a jacobian winding mapping df_winding method.")
    
    # Domain related methods

    def into_domain(self, x):
        """Maps the point x into the domain of the map."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"Input must be a {self.dimension}-dimensional vector.")
        for i, (low, high) in enumerate(self.domain):
            if self.periodicity[i] == 1:
                x[i] = low + (x[i] - low) % (high - low)
            else:
                x[i] = np.clip(x[i], low, high)
        return x

    
    def in_domain(self, x):
        """
        Checks if the point x is within the domain of the map.
        """
        for i, (low, high) in enumerate(self.domain):
            if not low <= x[i] <= high and self.periodicity[i] == 0:
                return False
        return True

    def f_many(self, t, x_many): 
        """
        apply map f t times to all the rows of xx. 
        """
        res = []
        for row in x_many:
            try: 
                res.append(self.f(t, row))
            except: 
                res.append(np.full(self.dimension, np.nan))
        return np.array(res)
    
    # def distance(self, x, y):
    #     sum_xy_sq = 0
    #     for i, (low, high) in enumerate(self.domain):
    #         if self.periodicity[i] == 1:
    #             a = min
    #             db = 
    #         else:
    #             sum_xy_sq += (x[i] - y[i])**2
    #     return np.sqrt(sum_xy_sq)
