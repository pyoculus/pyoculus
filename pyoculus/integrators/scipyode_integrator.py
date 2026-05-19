"""
scipyode_integrator.py
==================

Contains wrapper for the scipy.integrate.ode ODE solver class.

:authors: 
    - Zhisong Qu (zhisong.qu@anu.edu.au)
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

from .base_integrator import BaseIntegrator
from scipy.integrate import ode


class ScipyODEIntegrator(BaseIntegrator):
    """
    Wrapper for the scipy.integrate.ode ODE solver.

    This class serves as a wrapper and is notably useful to use the explicit Runge-Kutta methods implemented in `scipy.integrate.ode`.

    Attributes:
        ode (callable): Function for the right-hand side of the ODE.
        args (tuple): Additional arguments to pass to the ODE function.
        type (str): Type of integrator. Notably: 'dopri5' for RK45, 'dop853' for RK853. Defaults to 'dopri5'.
        rtol (float): Relative tolerance for the solver.
        nsteps (int): Maximum number of integration steps.

    For more information on the scipy ODE solver, see the `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_.
    """

    def __init__(self, **params):
        """
        Sets up the ODE solver.

        Args:
            **params: Arbitrary keyword arguments.
                - ode (callable): Function for the right-hand side of the ODE. Must be of the form f(t, x, *args).
                - args (tuple, optional): Additional arguments to pass to the ODE function. Defaults to ().
                - type (str, optional): Type of integrator. Notably: 'dopri5' for RK45, 'dop853' for RK853. Defaults to 'dopri5'.
                - rtol (float, optional): Relative tolerance for the solver. Defaults to 1e-10.
                - nsteps (int, optional): Maximum number of integration steps. Defaults to 10000.
        """

        # if "ode" not in params.keys():
        #     raise ValueError("ODE function not provided")

        # check if the ode is provided. If not, raise an error
        if "type" not in params.keys():
            params["type"] = "dopri5"  # set default to RK45

        if "rtol" not in params.keys():
            params["rtol"] = 1e-10  # set to default value
        self.rtol = params["rtol"]

        if "nsteps" not in params.keys():
            params["nsteps"] = 20000

        if "args" not in params.keys():
            params["args"] = ()
        self.args = params["args"]

        super().__init__(params)

        # set up the integrator
        self.set_rhs(params["ode"])


    def set_initial_value(self, t, x):
        """
        Sets up the initial value for the ODE solver.

        Args:
            t (float): The start time.
            x (array_like): The initial state vector.
        """

        self.integrator.set_initial_value(x, t)
        self.integrator.set_f_params(*self._params["args"])
        # try:
        self.rhs(t, x, *self.args)
        # except:
        #   raise "ODE function not callable"

        super().set_initial_value(t, x)

    def integrate(self, tend):
        """
        Integrates the ODE until :math:`t_\\text{end}`.

        Args:
            tend (float): The target end time.

        Returns:
            array_like: The new state vector at time :math:`t_\\text{end}`.
        """
        x_new = self.integrator.integrate(tend)

        if not self.integrator.successful():
            raise Exception("Integration failed")

        self.x = x_new
        self.t = tend
        return x_new

    def set_rhs(self, rhs):
        """
        Changes the RHS function to solve.

        Args:
            rhs (callable): The new RHS function.
        """
        self.rhs = rhs
        ode_params = self._params.copy()
        ode_params.pop("ode")
        ode_params.pop("args")
        ode_params.pop("type")
        self.integrator = ode(self.rhs).set_integrator(
            self._params["type"], **ode_params
        )

    def __copy__(self):
        """
        Returns a copy of self, to use if wanting to compute in parallel.

        Returns:
            RKIntegrator: A copy of the current integrator instance.
        """
        # set up a new integrator
        return ScipyODEIntegrator(**self._params)
