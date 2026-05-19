"""
two_waves.py
==================

Perturbed slab model described in S.R. Hudson, Phys. Plasmas 11, 677 (2004).

:authors:
    - Zhisong Qu (zhisong.qu@anu.edu.au)
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

from .toroidal_bfield import ToroidalBfield
import numpy as np


class TwoWaves(ToroidalBfield):
    """
        A very simple (but still hard) system described in `S.R. Hudson, Phys. Plasmas 11, 677 (2004) <https://doi.org/10.1063/1.1640379>`_.

        The model mimics the behavior of magnetic fields in a two-wave system. The magnetic field is defined as:
            
        .. math::
            \\mathbf{B} = \\nabla \\rho \\times \\nabla \\theta + \\nabla\\phi\\times \\nabla\\chi(\\rho, \\theta, \\phi)
        
        The Hamiltonian :math:`\\chi` associated to the magnetic field is:

        .. math::
            \\chi(\\rho, \\theta, \\phi) = \\frac{\\rho^2}{2} - k \\left[ \\frac{1}{2} \\cos (2\\theta - \\phi) + \\frac{1}{3} \\cos(3\\theta - 2\\phi) \\right]

        The components of the magnetic fields are thus given by:

        .. math::
            B^\\rho = - k \\left[ \\sin (2\\theta - \\phi) + \\sin(3\\theta - 2\\phi)\\right], \\quad B^\\theta = \\rho , \\quad B^\\phi = 1

        And using the :math:`\\phi` variable as a time variable of the Hamiltonian system gives for the derivatives of :math:`\\rho, \\theta`:

        .. math::
            \\dot{\\rho} = \\frac{B^\\rho}{B^\\phi} \\quad  \\dot{\\theta} = \\frac{B^\\theta}{B^\\phi}

        Attributes:
            k (float): Parameter giving the strength of the perturbation.
    """

    def __init__(self, k=0.002):
        """
        Set up the problem
        
        Arguments:
            k (float): Parameter giving the strength of the perturbation.
        """
        super().__init__()
        self._k = k
        self.Nfp = 1
        self.has_jacobian = True

    @property
    def k(self):
        """
        Parameter giving the strength of the perturbation.
        """
        return self._k

    def B(self, coords, *args):
        """! Returns magnetic fields
        @param coordinates
        @param *args extra parameters
        @returns the contravariant magnetic fields
        """

        # q = theta, p = rho, t = phi
        q = coords[1]
        p = coords[0]
        t = coords[2]

        dqdt = p
        dpdt = -self.k * (np.sin(2 * q - t) + np.sin(3 * q - 2 * t))
        dtdt = 1

        return np.array([dpdt, dqdt, dtdt], dtype=np.float64)

    def dBdX(self, coords, *args):
        """! Returns magnetic fields
        @param coordinates
        @param *args extra parameters
        @returns B, dBdX, the contravariant magnetic fields, the derivatives of them
        """
        q = coords[1]
        p = coords[0]
        t = coords[2]

        dqdt = p
        dpdt = -self.k * (np.sin(2 * q - t) + np.sin(3 * q - 2 * t))
        dtdt = 1

        dpdq = -self.k * (2.0 * np.cos(2 * q - t) + 3.0 * np.cos(3 * q - 2 * t))
        dqdp = 1.0

        dBu = np.zeros([3, 3], dtype=np.float64)

        dBu[0, 1] = dqdp
        dBu[1, 0] = dpdq

        Bu = np.array([dpdt, dqdt, dtdt], dtype=np.float64)

        return Bu, dBu

    def A(self, coords, *args):
        pass

    def convert_coords(self, incoords):
        """! Convert coordinates for Poincare plots
        @param incoords \f$(p,q,t)\f$
        @returns \f$(p,q \ \mbox{mod} \ 2\pi,t \ \mbox{mod} \ 2\pi)\f$
        """
        return np.array(
            [
                incoords[0],
                np.mod(incoords[1], 2.0 * np.pi),
                np.mod(incoords[2], 2.0 * np.pi),
            ],
            dtype=np.float64,
        )

    def B_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """! Returns magnetic fields, with multipy coordinate inputs
        @param x1arr the first coordinates. Should have the same length as the other two if input1D=True.
        @param x2arr the second coordinates. Should have the same length as the other two if input1D=True.
        @param x3arr the third coordinates. Should have the same length as the other two if input1D=True.
        @param input1D if False, create a meshgrid with sarr, tarr and zarr, if True, treat them as a list of points
        @param *args extra parameters
        @returns the contravariant magnetic fields
        """
        x1arr = np.atleast_1d(x1arr)
        x2arr = np.atleast_1d(x2arr)
        x3arr = np.atleast_1d(x3arr)

        if not input1D:
            size = (x1arr.size, x2arr.size, x3arr.size)
            p = np.broadcast_to(x1arr[:, np.newaxis, np.newaxis], size).flatten()
            q = np.broadcast_to(x2arr[np.newaxis, :, np.newaxis], size).flatten()
            t = np.broadcast_to(x3arr[np.newaxis, np.newaxis, :], size).flatten()
            n = p.size

        else:
            p = x1arr
            q = x2arr
            t = x3arr
            n = p.size

        dqdt = p
        dpdt = -self.k * (np.sin(2 * q - t) + np.sin(3 * q - 2 * t))
        dtdt = np.ones_like(p)

        Blist = np.stack([dpdt, dqdt, dtdt], -1)

        return Blist

    def dBdX_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """! Returns magnetic fields
        @param x1arr the first coordinates. Should have the same length as the other two if input1D=True.
        @param x2arr the second coordinates. Should have the same length as the other two if input1D=True.
        @param x3arr the third coordinates. Should have the same length as the other two if input1D=True.
        @param input1D if False, create a meshgrid with sarr, tarr and zarr, if True, treat them as a list of points
        @param *args extra parameters
        @returns B, dBdX, the contravariant magnetic fields, the derivatives of them
        """
        x1arr = np.atleast_1d(x1arr)
        x2arr = np.atleast_1d(x2arr)
        x3arr = np.atleast_1d(x3arr)

        if not input1D:
            size = (x1arr.size, x2arr.size, x3arr.size)
            p = np.broadcast_to(x1arr[:, np.newaxis, np.newaxis], size).flatten()
            q = np.broadcast_to(x2arr[np.newaxis, :, np.newaxis], size).flatten()
            t = np.broadcast_to(x3arr[np.newaxis, np.newaxis, :], size).flatten()
            n = p.size

        else:
            p = x1arr
            q = x2arr
            t = x3arr
            n = p.size

        dqdt = p
        dpdt = -self.k * (np.sin(2 * q - t) + np.sin(3 * q - 2 * t))
        dtdt = np.ones_like(p)

        Blist = np.stack([dpdt, dqdt, dtdt], -1)

        dpdq = -self.k * (2.0 * np.cos(2 * q - t) + 3.0 * np.cos(3 * q - 2 * t))
        dqdp = np.ones_like(dpdq)

        zeros = np.zeros_like(dpdq)

        dBplist = np.stack([zeros, dpdq, zeros], -1)
        dBqlist = np.stack([dqdp, zeros, zeros], -1)
        dBtlist = np.stack([zeros, zeros, zeros], -1)

        dBlist = np.stack([dBplist, dBqlist, dBtlist], -1)

        return Blist, dBlist
