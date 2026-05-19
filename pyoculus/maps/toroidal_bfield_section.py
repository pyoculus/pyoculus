from .integrated_map import IntegratedMap
from ..fields.toroidal_bfield import ToroidalBfield
import numpy as np


class ToroidalBfieldSection(IntegratedMap):
    """
    Class that sets up a Map given by following the a magnetic field in toroidal system :math:`(s, \\theta, \\zeta)`.
    """

    def __init__(self, toroidalbfield : ToroidalBfield, phi0=0., **kwargs):

        if not isinstance(toroidalbfield, ToroidalBfield):
            raise ValueError("The input should be a ToroidalBfield object.")
        else:
            self._mf = toroidalbfield

        domain = [(-1, 1), (0, 2*np.pi)]

        periodicity = [0, 1]

        super().__init__(dim=2, domain=domain, periodicity=periodicity, dzeta=2*np.pi/toroidalbfield.Nfp, ode = self._ode_rhs_tangent, **kwargs)

        self.phi0 = phi0

    ## BaseMap methods

    def f(self, t, y0):
        """
        
        Returns: 
            
        """
        # Set the integrator
        if self._integrator.rhs is not self._ode_rhs:
            self._integrator.set_rhs(self._ode_rhs)

        y_new = self._integrate(t, y0)
        return self.into_domain(y_new)

    def df(self, t, y0):
        ic = np.array([*y0, 1.0, 0.0, 0.0, 1.0])
        self._integrator.set_rhs(self._ode_rhs_tangent)
        return self._integrate(t, ic)[2:6].reshape([2, 2]).T

    def lagrangian(self, y0, t):
        self._integrator.set_rhs(self._ode_rhs_A)
        return self._integrate(t, y0)

    def winding(self, t, y0, y1=None):
        # Set the integrator
        if self._integrator.rhs is not self._ode_rhs:
            self._integrator.set_rhs(self._ode_rhs)
        
        y_new = self._integrate(t, y0)
        return np.array([y_new[0], y_new[1] - y0[1]])
    
    def dwinding(self, t, y0, y1=None):
        return self.df(t, y0)

    ## Integration methods

    def _integrate(self, t, y0):
        """
        Integrates the ODE for a number of periods.
        """
        if t==0:
            return y0
        else: 
            dphi = t * self.dzeta
            y = np.array(y0)
            self._integrator.set_initial_value(self.phi0, y)
            return self._integrator.integrate(self.phi0 + dphi)

    def _ode_rhs(self, phi, st, *args):
        """
        Calculates the right-hand side (RHS) of the ODE.

        Args:
            phi (float): The current cylindrical angle.
            st (array): The cylindrical coordinates :math:`(s, \\theta)` in the ODE.
            *args: Additional parameters for the ODE.

        Returns:
            array: The RHS of the ODE.
        """
        stz = np.array([st[0], st[1], phi])
        B = self._mf.B(stz, *args)
        return np.array([B[0] / B[2], B[1] / B[2]])

    def _ode_rhs_tangent(self, phi, y, *args):
        """
        Calculates the right-hand side (RHS) of the ODE with differential of the dependent variables.

        Args:
            phi (float): The current cylindrical angle.
            st (array): The cylindrical coordinates :math:`(s, \\theta, ds_1, d\\theta_1, ds_2, d\\theta_2))` in the ODE.
            *args: Additional parameters for the ODE.

        Returns:
            array: The RHS of the ODE.
        """
        stz = np.array([y[0], y[1], phi])
        Bu, dBu = self._mf.dBdX(stz, *args)

        deltax = np.reshape(y[2:], [2, 2])
        gBzeta = Bu[2]

        M = dBu[0:2, 0:2] * gBzeta - dBu[0:2, 2, np.newaxis] * Bu[0:2]

        deltax = deltax @ M / gBzeta**2

        df = np.zeros([6])
        df[0:2] = Bu[0:2] / Bu[2]
        df[2:6] = deltax.flatten()

        return df

    def _ode_rhs_A():
        raise NotImplementedError("A is not implemented.")