"""
standard_map.py 

Pyoculus implementation of the standard map with automatic differentiation

The map is 'raised' so that the identity mapping takes place at p=0.5

"""
from .base_map import BaseMap
import numpy as np

import logging
logger = logging.getLogger(__name__)

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    from jax import jit, jacfwd
    import jax.numpy as jnp
except ImportError as e:
    logger.warning("Could not import jax. Some functionalities will not be available.")
    raise e

@jit
def standardmap_f_pure(y, k):
    """
    Pyoculus implementation of the standard map with automatic differentiation

    The map is 'raised' so that the identity mapping takes place at p=0.5
    """
    theta_old = y[0]
    p_old = y[1]
    # applying standard map on old coordinates to get new coordinates
    p = p_old + (k/(2*np.pi))*np.sin(2*np.pi*theta_old)
    theta = theta_old + p + 0.5 % 1
    # return
    return np.array([theta, p])


@jit
def standardmap_frev_pure(y, k):
    """
    Inverse of the Tokamap as described in:
    Balescu, Radu, Mǎdǎlina Vlad, and F. Spineanu. "Tokamap: A Hamiltonian twist map for magnetic field lines in a toroidal geometry." Physical Review E 58.1 (1998): 951.

    The theta funtion is not directly invertible, so a simple Newtons method with analytical derivatives is used to solve for the inverse.
    
    with W(psi) the winding function monotonically increasing from w up. 
    """
    p = y[1]
    theta = y[0]
    theta_old = theta - p - 0.5 % 1
    p_old = p - (k/(2*np.pi))*np.sin(2*np.pi*theta_old)
    return jnp.array([theta_old, p_old])

def return_f_t_pure(t):
    """
    return a pure (jax-transformable) function for the t-times applied map
    Used to automatically differentiate the map and get the gradients

    Note that for reverse mappings, we have to differentiate through a 
    Newtons iteration, so the accuracy is probably bad. 
    """
    if t>0:
        @jit
        def standardmap_f_t_pure(y, K, w):
            for _ in range(t):
                y = standardmap_f_pure(y, K, w)
            return y
        return standardmap_f_t_pure
    elif t<0:
        @jit
        def standardmap_frev_t_pure(y, K, w):
            for _ in range(abs(t)):
                y = standardmap_frev_pure(y, K, w)
            return y
        return standardmap_frev_t_pure
    else: # in the idiot edge case of the zero-map, do the idiot thing
        return lambda x:x


class StandardMap(BaseMap):
    """
    two-dimensional iterated map described in:
    Balescu, Radu, Mǎdǎlina Vlad, and F. Spineanu. "Tokamap: A Hamiltonian twist map for magnetic field lines in a toroidal geometry." Physical Review E 58.1 (1998): 951.

    This is a map that is slightly more complex than the standard map or the nontwist map, but is a two-dimensional iterated map for which it is easier to define the (monotonic) twist profile or safety factor profile. 
    """
    def __init__(self, K=0, w=0.666):
        """
        Initializes the TokaMap. Arguments: 
        - K: float, a chaos parameter
        - w_0: float, the safety factor at the "axis" \psi or y = 0
        - w_1: float, the safety factor at the "boundary" \psi or y = 1
        """
        super().__init__(2, is_discrete=True, domain=[[0, np.pi*2], [0, 1]], periodicity=[0, 1])
        self.K = K
        self.w = w

        self._f = lambda y: standardmap_f_pure(y, self.K, self.w)
        self._f_rev = lambda y: standardmap_frev_pure(y, self.K, self.w)
        self.return_f_t_pure = return_f_t_pure
        self.df_dict = {}

    
    def f(self, t, y0):
        """
        This method represents the mapping function. It takes a point :math:`y_0` in the domain and returns its image under :math:`t` application of the map.

        Args:
            t (float or int): The number of times the map is applied.
            y0 (array): The initial point in the phase space.
        """
        y = np.copy(y0)
        if t>0: 
            for _ in range(t):
                y = self._f(y)
            return y
        if t<0: 
            for _ in range(abs(t)):
                y = self._f_rev(y)
            return y
    
    def df(self, t, y0):
        """
        Computes the Jacobian matrix of the map at :math:`y_0` after :math:`t` applications :math:`df^t = (\\frac{\\partial f^t}{\\partial x})_{i,j}`.

        Uses a cache to avoid re-creating and re-compiling functions. 
        """
        if t in self.df_dict:
            return self.df_dict[t](y0)
        else:
            self.df_dict[t] = lambda y: jacfwd(self.return_f_t_pure(t))(y, self.K, self.w)
            return self.df_dict[t](y0)
    
