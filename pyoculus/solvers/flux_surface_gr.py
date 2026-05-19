## @file flux_surface_gr.py
#  @brief A class for finding flux surfaces using Greene's residue method
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#

from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from ..utils.continued_fraction import expandcf, fromcf
from ..utils.plot import create_canvas
import numpy as np
import logging

logger = logging.getLogger(__name__)


## Class that used to set up the flux surface finder.
class FluxSurfaceGR(BaseSolver):

    def __init__(
        self, map
    ):
        """
        Set up the class of the flux surface point finder using Greene's method
        """

        if map.dimension != 2:
            raise ValueError("Finding the should be 2D")

        super().__init__(map)


    def GreeneMethod(
        self,
        iota,
        fp_1=None,
        fp_2=None,
        nexpand=10,
        **kwargs,
    ):
        """
        Look for the flux surface with a given rotation number using Greene's residue method.

        Args:
            iota: the irrational! rotation number of the flux surface
            fixed_point_left: a sucessfully found FixPoint to mark the left bound of the flux surface,
                                its rotation number needs to be in the convergent sequence of iota
            fixed_point_right a sucessfully found FixPoint to mark the right bound of the flux surface,
                                its rotation number needs to be in the convergent sequence of iota and next to fixed_point_left
            n_expand=10 the number of terms in the continued fraction expansion of iota, used to approximate the flux surface
            **kwargs: additional arguments for the fixed point finding method

        Returns:
            `fdata.MackayResidue` -- the Mackay Residue of the fixed points
            `fdata.fixed_points` -- all the fixed point located
            `fdata.rmnc`, fdata.rmns`, `fdata.zmnc`, `fdata.zmns` -- the Fourier harmonics
        """

        # iota will be divided by Nfp
        # if isinstance(self._map, pyoculus.map.ToroidalBfieldSection):
        # if isinstance(self._map, maps.ToroidalBfieldSection):
        #     iota = iota / self.Nfp

        # continued fraction expansion of the input irrational
        ais = expandcf(iota, nexpand)
        fp_left, fp_right = None, None

        for i in range(nexpand-2):
            n1, m1 = fromcf(ais[:i + 1])
            n2, m2 = fromcf(ais[:i + 2])
            
            logger.info(f"n1 = {n1}, m1 = {m1}, n2 = {n2}, m2 = {m2}")

            # Check if the fixed points match and assign accordingly
            if (n1, m1, n2, m2) == (fp_1.n, fp_1.m, fp_2.n, fp_2.m):
                fp_left, fp_right = fp_1, fp_2
                nstart = i
            elif (n2, m2, n1, m1) == (fp_1.n, fp_1.m, fp_2.n, fp_2.m):
                fp_left, fp_right = fp_2, fp_1
                nstart = i

        if fp_left is None or fp_right is None:
            raise ValueError("The fixed points are not found in the continued fraction expansion")

        fixedpoints = [fp_left, fp_right]
        
        self._nstart = nstart
        for i in range(nstart + 2, nexpand):
            n, m = fromcf(ais[: i + 1])
            iotatarget = n / m

            x_left = fixedpoints[-2].coords[0]
            x_right = fixedpoints[-1].coords[1]

            iota_left = fixedpoints[-2].n / fixedpoints[-2].m
            iota_right = fixedpoints[-1].n / fixedpoints[-1].m

            # Interpolate between x_left and x_right
            x_guess = x_left + (x_right - x_left) / (iota_right - iota_left) * (
                iotatarget - iota_left
            )
            x_guess = np.array([x_guess[0], 0])
            
            # Find the next fixed point
            nextfixedpoint = FixedPoint(self._map)
            logger.info(f"Searching for the fixedpoint with n = {n}, m = {m} at x_guess = {x_guess}")
            nextfixedpoint.find_with_iota(
                n, m, x_guess, **kwargs
            )

            if not nextfixedpoint.successful:
                raise Exception("Fixed point not found")

            fixedpoints.append(nextfixedpoint)

        # save the fixed points found
        self.fixedpoints = fixedpoints

        # assemble the output data
        rdata = FluxSurfaceGR.OutputData()
        rdata.fixedpoints = fixedpoints

        # put the flag as successful
        self._successful = True

        return rdata

    def plot(
        self, **kwargs
    ):
        if not self.successful:
            raise Exception("A successful call of compute() is needed")

        self.fixedpoints[-1].plot(**kwargs)

    def plot_residue(self, **kwargs):

        gamma = (np.sqrt(5) + 1) / 2

        xlist_greene = np.arange(
            self._nstart + 1, self._nstart + 1 + len(self.fixedpoints)
        )
        greenes_list = np.zeros(len(self.fixedpoints), dtype=np.float64)

        for ii, fp in enumerate(self.fixedpoints):
            greenes_list[ii] = fp.GreenesResidue

        xlist_Mackay = xlist_greene[1:]
        Mackay_list = np.zeros(len(self.fixedpoints) - 1, dtype=np.float64)

        for ii in range(len(self.fixedpoints) - 1):
            Mackay_list[ii] = (
                self.fixedpoints[ii].GreenesResidue
                + gamma * self.fixedpoints[ii + 1].GreenesResidue
            ) / (1.0 + gamma)

        fig, ax, kwargs = create_canvas(**kwargs)

        ax.plot(xlist_greene, greenes_list, '-x', label="Greene")
        ax.plot(xlist_Mackay, Mackay_list, '-o', label="Mackay")
        ax.plot(
            xlist_greene, 0.25 * np.ones_like(greenes_list), label="Stable bound"
        )

        ax.legend()
        ax.set_xlabel("Order of fixed point", fontsize=20)
        ax.set_ylabel("Residue", fontsize=20)

        return fig, ax