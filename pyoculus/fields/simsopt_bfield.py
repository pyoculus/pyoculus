from .cylindrical_bfield import CylindricalBfield
import pyoculus.utils.cyl_cart_transform as cct
from typing import Union
import numpy as np
from typing import TYPE_CHECKING

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from simsopt.field import MagneticField, InterpolatedField, BiotSavart
    from simsopt.geo import SurfaceXYZFourier, SurfaceClassifier


class SimsoptBfield(CylindricalBfield):
    """
    
    """
    
    def __init__(self, Nfp : int, mf : "MagneticField", interpolate: Union[bool, "InterpolatedField"] = False, **kwargs):
        """
        Create a SimsoptBfield, Pyoculus's bridge to the simsopt MagneticField object. 
        
        Args: 
        Nfp (int): The number of field periods. Default is 1. This parameter defines the periodicity of the magnetic field (:math:`T = 2*\\pi/n_\\text{fp}`).
        mf (MagneticField): A simsopt MagneticField object.
        interpolate (bool or InterpolatedField): If True, the field will be interpolated. If an InterpolatedField object is passed, it will be used as the interpolating field.
        **kwargs: Additional parameters to be passed to the InterpolatedField constructor if interpolate is True. This should contain a kwarg 'surf' which is the boundary of the interpolated region. 
        """
        
        super().__init__(Nfp)

        self._mf = mf
        self._interpolating = False

        if interpolate:
            self._interpolating = True
            if type(interpolate)-__name__ == "InterpolatedField":
                self._mf_B = interpolate
            else:
                from simsopt.geo import SurfaceClassifier
                from simsopt.field import InterpolatedField
                surf = kwargs.get('surf', None)
                deltah = kwargs.get('deltah', 0.05)
                degree = kwargs.get('degree', 3)
                stellsym = kwargs.get('stellsym', True)
                if surf is None:
                    raise ValueError("If interpolate is True, a surf object (limit of interpolatedfield) must be passed as a kwarg.")

                p = kwargs.get('p', 2)
                h = kwargs.get('h', 0.03)
                self.surfclassifier = SurfaceClassifier(surf, h=h, p=p)

                def skip(rs, phis, zs):
                    rphiz = np.asarray([rs, phis, zs]).T.copy()
                    dists = self.surfclassifier.evaluate_rphiz(rphiz)
                    skip = list((dists < -deltah).flatten())
                    return skip
                skyping = kwargs.get('skyping', skip)

                n = kwargs.get('n', 20)
                rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
                zs = surf.gamma()[:, :, 2]
                rrange = (np.min(rs), np.max(rs), n)
                phirange = (0, 2 * np.pi / Nfp, n * 2)
                zrange = (0, np.max(zs), n // 2)

                self._mf_B = InterpolatedField(
                    mf,
                    degree,
                    rrange,
                    phirange,
                    zrange,
                    True,
                    nfp=Nfp,
                    stellsym=stellsym,
                    skip=skyping,
                )
        else:
            self._mf_B = mf

    @classmethod
    def from_coils(cls, coils, Nfp, **kwargs):
        from simsopt.field import BiotSavart
        mf = BiotSavart(coils)
        return cls(Nfp, mf, **kwargs)

    # Methods of the MagneticField class

    def B(self, rphiz):
        """
        
        """
        xyz = cct.xyz(*rphiz)
        xyz = np.reshape(xyz, (-1, 3))
        self._mf_B.set_points(xyz)

        B_cart = self._mf_B.B().flatten()

        return cct.vec_cart2cyl(B_cart, *rphiz)

    def dBdX(self, rphiz):
        """
        
        """
        xyz = cct.xyz(*rphiz)
        xyz = np.reshape(xyz, (-1, 3))
        self._mf.set_points(xyz)

        B_cart = self._mf.B()
        dBdX_cart = self._mf.dB_by_dX().reshape(3, 3)
        
        return cct.vec_cart2cyl(B_cart, *rphiz), cct.mat_cart2cyl(dBdX_cart, *rphiz) + cct.dinvJ_matrix(B_cart, *rphiz)

    def A(self, rphiz):
        """
        
        """
        xyz = cct.xyz(*rphiz)
        xyz = np.reshape(xyz, (-1, 3))
        self._mf.set_points(xyz)

        A_cart = self._mf.A().flatten()

        return cct.vec_cart2cyl(A_cart, *rphiz)


#def surf_from_coils(coils, **kwargs):
#    """
#    not very robust way of trying to fit a surface that intersects the coils. 
#    """
#    logger.info(f"Using surf_from_coils with parameters: {kwargs}")
#    logger.warning("Using surf_from_coild can result in weird surfaces. Use with caution.")
#    
#    mpol = kwargs.get('mpol', 3)
#    ntor = kwargs.get('ntor', 3)
#    stellsym = kwargs.get('stellsym', False)
#    nfp = kwargs.get('nfp', 1)
#
#    ncoils = kwargs.get('ncoils', None)
#    
#    nphi, ntheta = len(coils), len(coils[0].curve.gamma())
#    qpts_theta = np.linspace(0, 1, ntheta, endpoint=False)
#    qpts_phi = np.linspace(0, 1, nphi, endpoint=False)
#
#    surf = SurfaceXYZFourier(
#        mpol=mpol,
#        ntor=ntor,
#        stellsym=stellsym,
#        nfp=nfp,
#        quadpoints_phi=qpts_phi,
#        quadpoints_theta=qpts_theta
#    )
#    centroids = np.array([np.mean(coil.curve.gamma(), axis=0) for coil in coils])
#
#    phis = np.arctan2(centroids[:, 1], centroids[:, 0])
#    indices = np.argsort(phis)
#    gamma_curves = [coils[i].curve.gamma() for i in indices]
#    if ncoils is not None:
#        gamma_curves = np.stack([gamma if (i // ncoils) % 2 != 0 else gamma[::-1] for i, gamma in enumerate(gamma_curves)])
#    surf.least_squares_fit(gamma_curves)
#
#    return surf