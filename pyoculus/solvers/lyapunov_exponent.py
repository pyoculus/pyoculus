## @file lyapunov_exponent.py
# Containing a class for computing the Lyapunov Exponent
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)

from .base_solver import BaseSolver
from ..utils.plot import create_canvas
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LyapunovExponent(BaseSolver):
    """
    Class for computing the Lyapunov Exponent
    """

    def __init__(self, problem, nsave=20, every=100):
        self.nsave = nsave
        self.every = every

        super().__init__(problem)

    def compute(self, x, eps=None, base=np.exp(1)):
        """
        Compute the maximal Lyapunov exponents
        
        J. C. Sprott, Chaos and Time-Series Analysis (Oxford University Press, 2003), pp.116-117.
        
        Args:
            x: the initial point
            eps: the perturbation, default is a random direction with norm of the square root of the machine precision
            base: the base of the logarithm, default is exp(1)
        
        Returns:
            np.array: the maximal Lyapunov exponents
        """
        self._lyapunov_exp = np.NaN * np.zeros(self.nsave, dtype=np.float64)
        self._dis = np.NaN * np.zeros((self.nsave + 1)*self.every, dtype=np.float64)

        if eps is None:
            random_dir = np.random.rand(self._map.dimension)
            # Advice to get the square root of the machine precision
            d0 = np.sqrt(np.finfo(float).eps)
            eps = d0*random_dir/np.linalg.norm(random_dir)

        # if isinstance(self._map, pr.integrated_map):
        #     self.dt = self._map.dt
        # else:
        #     self.dt = 1
        dt = self._map.dzeta

        x_nearby = x + eps

        for i in range((self.nsave + 1) * self.every):
            # Map the points
            try:
                x_new = self._map.f(1, x)
                x_nearby_new = self._map.f(1, x_nearby)
            except Exception as e:
                logger.error(f"Error in mapping the points, stopping the computation after {i+1} iterations : {e}")
                break

            # Get their new distance
            self._dis[i] = np.linalg.norm(x_nearby_new - x_new)

            # Update the positions
            x_nearby = x_new + d0*(x_nearby_new - x_new)/self._dis[i]
            x = x_new

        for i in range(1, self.nsave+1):
            # Compute the Lyapunov exponent by averaging the logarithm of di/d0
            self._lyapunov_exp[i-1] = np.mean(np.log(self._dis[: i*self.every]/d0)/np.log(base))/dt

        self._successful = True

        return self._lyapunov_exp

    def plot(self, **kwargs):

        if not self._successful:
            raise Exception("A successful call of compute() is needed")

        fig, ax, kwargs = create_canvas(**kwargs)

        # plotting the points
        ax.plot(
            np.log10(self.every*np.arange(1, self.nsave+1)), np.log10(self._lyapunov_exp), **kwargs
        )

        ax.set_xlabel("Log10 Num of iters")
        ax.set_ylabel("Log10 Maximal LE")

        return fig, ax