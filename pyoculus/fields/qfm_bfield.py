## @file qfm_bfield.py
#  @brief Setup the magnetic field system - using the Quadratic minimising surfaces coordinates
#  @author Zhisong Qu (zhisong.qu@anu.edu.au)
#
from .toroidal_bfield import ToroidalBfield
from ..geo.interpolate_coordinates import SurfacesToroidal
import numpy as np


## Class that used to setup the bfield problem based on QFM
#
class QFMBfield(ToroidalBfield):
    def __init__(self, pb: ToroidalBfield, surfaces: SurfacesToroidal):
        """! Set up the problems based on a known magnetic field and the QFM surfaces
        @param pb  the TorodialBfield problem we will used as the original coordinates and fields
        @param surfaces  the QFM surfaces
        """
        super().__init__()
        self.pb = pb
        self.surfaces = surfaces
        self.poincare_plot_type = pb.poincare_plot_type
        self.poincare_plot_xlabel = pb.poincare_plot_xlabel
        self.poincare_plot_ylabel = pb.poincare_plot_ylabel
        self.Nfp = pb.Nfp
        self.has_jacobian = pb.has_jacobian

    def B(self, coords, *args):
        """! Returns magnetic fields
        @param coords \f$(\rho,\vartheta,\zeta)\f$
        @param *args extra parameters
        @returns the contravariant magnetic fields
        """

        r = np.atleast_1d(coords[0])
        t = np.atleast_1d(coords[1])
        z = np.atleast_1d(coords[2])

        coordsout = self.surfaces.get_coords(r, t, z, derivative=1, input1D=True)
        coordsin = np.array([coordsout.s[0], coordsout.t[0], coordsout.z[0]])
        B = np.atleast_2d(self.pb.B(coordsin))
        B = self.surfaces.contra_vector_transform(
            B, coordsout, has_jacobian=self.has_jacobian, derivative=False
        ).flatten()

        return B

    def dBdX(self, coords, *args):
        """! Returns magnetic fields
        @param coords \f$(s,\theta,\zeta)\f$
        @param *args extra parameters
        @returns B, dBdX, the contravariant magnetic fields, the derivatives of them
        """

        r = np.atleast_1d(coords[0])
        t = np.atleast_1d(coords[1])
        z = np.atleast_1d(coords[2])

        coordsout = self.surfaces.get_coords(r, t, z, derivative=2, input1D=True)
        coordsin = np.array([coordsout.s[0], coordsout.t[0], coordsout.z[0]])
        B, dBdX = self.pb.dBdX(coordsin)
        B, dBdX = self.surfaces.contra_vector_transform(
            np.atleast_2d(B),
            coordsout,
            has_jacobian=self.has_jacobian,
            derivative=True,
            dv=dBdX[np.newaxis, :],
        )

        return B[0], dBdX[0]

    def B_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """! Returns magnetic fields, with multipy coordinate inputs
        @param x1arr the first coordinates. Should have the same length as the other two if input1D=True.
        @param x2arr the second coordinates. Should have the same length as the other two if input1D=True.
        @param x3arr the third coordinates. Should have the same length as the other two if input1D=True.
        @param input1D if False, create a meshgrid with sarr, tarr and zarr, if True, treat them as a list of points
        @param *args parameter
        @returns the contravariant magnetic fields
        """

        r = x1arr
        t = x2arr
        z = x3arr

        if len(args) == 0:
            coordsout = self.surfaces.get_coords(r, t, z, derivative=1, input1D=input1D)
        else:
            coordsout = args[0]
            args.pop()

        newr = coordsout.s
        newt = coordsout.t
        newz = coordsout.z

        arr_shape = newr.shape

        B = self.pb.B_many(newr.flatten(), newt.flatten(), newz.flatten(), True, *args)
        B = np.reshape(B, np.concatenate([arr_shape, [3]]))
        B = self.surfaces.contra_vector_transform(
            B, coordsout, has_jacobian=self.has_jacobian
        )

        return B

    def dBdX_many(self, x1arr, x2arr, x3arr, input1D=True, *args):
        """! Returns magnetic fields
        @param x1arr the first coordinates. Should have the same length as the other two if input1D=True.
        @param x2arr the second coordinates. Should have the same length as the other two if input1D=True.
        @param x3arr the third coordinates. Should have the same length as the other two if input1D=True.
        @param input1D if False, create a meshgrid with sarr, tarr and zarr, if True, treat them as a list of points
        @param *args extra parameters
        @returns B, dBdX, the contravariant magnetic fields, the derivatives of them
        """
        r = x1arr
        t = x2arr
        z = x3arr

        if len(args) == 0:
            coordsout = self.surfaces.get_coords(r, t, z, derivative=2, input1D=input1D)
        else:
            coordsout = args[0]
            args.pop()

        B, dBdX = self.pb.dBdX_many(r, t, z, input1D, *args)
        B, dBdX = self.surfaces.contra_vector_transform(
            B, coordsout, has_jacobian=self.pb.has_jacobian, derivative=True, dv=dBdX
        )

        return B, dBdX

    def convert_coords(self, stz):
        """! Python wrapper for getting the xyz coordinates from stz
        @param stz  the stz coordinate
        @returns the xyz coordinates
        """

        r = np.atleast_1d(stz[0])
        t = np.atleast_1d(stz[1])
        z = np.atleast_1d(stz[2])

        coordsout = self.surfaces.get_coords(r, t, z, derivative=0, input1D=True)

        return self.pb.convert_coords(
            np.array([coordsout.s[0], coordsout.t[0], coordsout.z[0]])
        )
