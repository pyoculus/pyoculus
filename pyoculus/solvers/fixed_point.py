"""
fixed_point.py
==================

Contains the class for finding fixed points of a map.

:authors:
    - Zhisong Qu (zhisong.qu@anu.edu.au)
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

from .base_solver import BaseSolver
import pyoculus.maps as maps
from ..utils.plot import create_canvas
from scipy.optimize import root, root_scalar
import numpy as np
from numpy.typing import ArrayLike
from typing import Union
import logging

logger = logging.getLogger(__name__)


class FixedPoint(BaseSolver):
    """
    Class to find fixed points of a map, i.e. points that satisfy :math:`f^t(x) = x`.
    """

    def __init__(self, map):  #,  t : Union[float,  int]=1, guess : ArrayLike = None, nrestart : int = 0, method : str = "newton",):
        # if constraints is None:
        #     constraints = np.NaN * np.ones(map.dimension)
        # elif len(constraints) != map.dimension:
        #     raise ValueError("The constraints should have the same dimension as the map domain.")
        # elif all([c is not np.NaN for c in constraints]):
        #     raise ValueError("Their must be at least one unconstrained dimension.")
        # self._constraints = constraints
    
        self._found_by_iota = False
#        self.t = t
#        self._guess = guess
#        self._nrestart = nrestart
#        self._method = method
        self._n = None
        self._m = None
        super().__init__(map)
        

    ## Properties

    @property
    def iotaslash(self):
        if not self.successful:
            raise ValueError("Fixed point not found.")
        elif not self._found_by_iota:
            raise ValueError("Fixed point not found with winding number.")
        else:
            return self._n / self._m
        
    @property
    def n(self):
        if not self.successful:
            raise ValueError("Fixed point not found.")
        elif not self._found_by_iota:
            raise ValueError("Fixed point not found with winding number.")
        else:
            return self._n
    
    @property
    def m(self):
        if not self.successful:
            raise ValueError("Fixed point not found.")
        else:
            return self._m
    
    @m.setter
    def m(self, m):
        """
        set m to use maps that cannot be found with winding number
        """
        if not self.successful:
            raise ValueError("Fixed point not found.")
        self._found_by_iota = True
        self._m = m

    

    ## Findings fixed points methods

    def find(self, t : Union[float,  int], guess : ArrayLike = None, nrestart : int = 0, method : str = "scipy.root", **kwargs):
        """
        Tries to find a fixed point of the map applied :math:`t` times.

        Args:
            t: the number of iterations of the map
            guess: the initial guess of the fixed point
            nrestarts: the maximum number of restart with different random initial conditions
            method: the method to use to find the fixed point, default is 'newton'. 
            **kwargs: additional arguments for the method

        Returns:
            FixedPoint.OutputData: the output data of the fixed point search
                - coords: the coordinates of the fixed point
                - Jacobian: the Jacobian of the fixed point
                - GreenesResidue: the Greene's Residue of the fixed point
        
        Notes:
            There are several methods implemented, 
            *newton* is a simple Newton's method implemented in this class. it evaluates the full map.  
            *scipy.root* uses the scipy root method to find the fixed point without derivative evaluation, it also finds the point using only a half-map forwards and backwards, which should help with convergence to very unstable fixed points
            *scipy.derivs* is the same as root, but also uses the calculated derivatives.
            *scipy.1D* is a 1D solver that finds the fixed point on a symmetry plane using a 1D bisection method and looking a zero in the difference in Z in the half-maps (defalut solver is brentq, specified by the keyword scipy_method="..")
            
            For IntegratedMaps, often best results are obtained with 'scipy.root', which does not use derivatives. This is because 
            the derivative evaluation is often more expensive than several map evaluations and most scipy solvers internally construct a jacobian from the evaluations.
            For s 

        """

        # Check the iteration number is correct
        if self._map.is_discrete and not isinstance(t, int):
            raise ValueError("The iteration number should be an integer for a discrete map.")

        # Setup the search
        self.t = t
        self._m = t
        self.history = []
        x_fp = None
        self._successful = False

        # Check the guess is correct
        if guess is None:
            guess = self.random_initial_guess()
        elif len(guess) != self._map.dimension:
            raise ValueError("The guess should have the same dimension as the map domain.")
        elif not self._map.in_domain(guess):
            raise ValueError("The guess is not in the domain of the map.")

        # Setup and check the method
        if method == "newton":
            method_fun = self._newton_method
        elif method == "scipy.root":
            method_fun = self._scipy_root
        elif method == "scipy.derivs":
            method_fun = self._scipy_derivs
        elif method == "scipy.1D":
            method_fun = self._scipy_1d_symmetry
        elif method == "scipy.1D_2D":
            method_fun = self._scipy_1d_2d
#        elif method == "scipy.stellsym.2D":
#            method_fun = self.scipy_stellsym_2d
        else:
            raise ValueError(f"Method {method} is not implemented.")
        self._method = method

        # run the solver, if failed, try a different random initial condition
        guess0 = guess.copy()
        for i in range(nrestart + 1):
            try:
                x_fp = method_fun(guess, **kwargs)
            except Exception as e:
                logger.info(f"Search {i} - failed: {e}")

            if x_fp is not None:
                break
            elif i < nrestart:
                logger.info(f"Search {i+1} starting from a random initial guesss.")
                guess = self.random_initial_guess(prev_guess=guess0)

        # now we go and get all the fixed points by iterating the map
        if x_fp is not None:
            logger.info(f"Found fixed point at {x_fp}. Computing additional data...")
            rdata = self.record_data(x_fp)

            # Set the successful flag
            self._successful = True
        else:
            rdata = None
            logger.info(f"Fixed point search unsuccessful for t={self.t}.")

        return rdata

    def find_with_iota(self, n : int, m : int, guess : ArrayLike, x_axis : ArrayLike = None, nrestart : int = 0, method : str = "scipy.root", **kwargs):
        """
        Tries to find the fixed point of a map with winding number :math:`\\iota/2\\pi = q^{-1} = n/m` around x_axis.s

        Args:
            n (int): the numerator of the winding number
            m (int): the denominator of the winding number
            guess (array): the initial guess of the fixed point
            x_axis (array): the point around which the winding number is calculated
            nrestarts (int): the maximum number of restart with different random initial conditions
            method (str): the method to use to find the fixed point, default is 'newton'
            **kwargs: additional arguments for the method        

        Returns:
            FixedPoint.OutputData: the output data of the fixed point search
                - coords: the coordinates of the fixed point
                - Jacobian: the Jacobian of the fixed point
                - GreenesResidue: the Greene's Residue of the fixed point
                - MeanResidue: --
        """

        # Setup the x_axis if not provided
        if x_axis is None:
            if isinstance(self._map, maps.ToroidalBfieldSection):
                x_axis = np.array([-1., 0.])
            elif isinstance(self._map, maps.CylindricalBfieldSection):
                x_axis = np.array([self._map.R0, self._map.Z0])
            else:
                logger.warning("No x_axis provided, using the zero vector.")
                x_axis = np.zeros(self._map.dimension)
        elif len(x_axis) != self._map.dimension:
            raise ValueError("The x_axis should have the same dimension as the map domain.")

        # Setup of the poloidal m and toroidal mode numbers
        if not isinstance(n, int) or not isinstance(m, int):
            raise ValueError("n and m should be integers")

        n = np.sign(n*m)*np.abs(n)
        m = int(np.abs(m))
        self._n = n
        self._m = m

        # Setup the search
        self.t = m
        self.history = []
        x_fp = None
        self._successful = False

        # Check the guess is right
        if guess is None:
            guess = self.random_initial_guess()
        elif len(guess) != self._map.dimension:
            raise ValueError("The guess should have the same dimension as the map domain.")
        elif not self._map.in_domain(guess):
            raise ValueError("The guess is not in the domain of the map.")

        # Setup and check the method
        if method == "newton":
            method_fun = self._newton_method_winding
        elif method == "1D":
            method_fun = self._scipy_winding_1D
        elif method == "scipy.root":
            method_fun = self._scipy_winding
        else:
            raise ValueError(f"Method {method} is not implemented.")
        self._method = method

        # run the solver, if failed, try a different random initial condition
        guess0 = guess.copy()
        for i in range(nrestart + 1):
            try:
                x_fp = method_fun(guess, x_axis, **kwargs)
            except Exception as e:
                logger.info(f"Search {i} - failed: {e}")

            if x_fp is not None:
                break
            elif i < nrestart:
                logger.info(f"Search {i+1} starting from a random initial guesss.")
                guess = self.random_initial_guess(prev_guess=guess0)

        # now we go and get all the fixed points by iterating the map
        if x_fp is not None:
            logger.info(f"Found fixed point at {x_fp}. Computing additionnal data...")
            self._found_by_iota = True
            rdata = self.record_data(x_fp)

            # Set the successful flag
            self._successful = True
        else:
            logger.info(f"Fixed point search unsuccessful for iotaslash=n/m={n}/{m}.")
            rdata = None

        return rdata

    ## Utils methods

    def random_initial_guess(self, prev_guess=None, max_step=0.5):
        """
        Returns a random initial point uniformly sampled from the domain of the map.

        If prev_guess is provided and the map is a CylindricalBfieldSection,
        the sampling region is restricted to prev_guess ± max_step (intersected with the domain).

        Args:
            prev_guess (np.array): the previous guess to sample around
            max_step (float): the maximum step size from prev_guess in each dimension
        """
        domain = self._map.domain
        lows = np.array([low for (low, high) in domain])
        highs = np.array([high for (low, high) in domain])

        if prev_guess is not None and isinstance(self._map, maps.CylindricalBfieldSection):
            lows = np.maximum(lows, prev_guess - max_step)
            highs = np.minimum(highs, prev_guess + max_step)

        return np.random.uniform(lows, highs)

    def record_data(self, x_fp):
        """
        Record some additional data about the fixed point, such as the Jacobian, the Greene's Residue, and the Mean Residue for each iteration of the map.

        Args:
            x_fp (array): Fixed point coordinates
            is_winding (bool)
        """

        self.Jacobian = self._map.df(self.t, x_fp)
        self.GreenesResidue = 0.25 * (2.0 - np.trace(self.Jacobian))
        if self._found_by_iota:
            self.MeanResidue = np.power(
                np.abs(self.GreenesResidue) / 0.25, 1 / float(self._m)
            )
        
        self.coords = np.array([np.asarray(self._map.f(i, x_fp)) for i in range(self.t + 1)])  # uses cache (cast to np array if analytic map returns other)


        # Create an output
        rdata = FixedPoint.OutputData()  # dummy holder class (legacy)
        rdata.coords = self.coords.copy()
        rdata.Jacobian = self.Jacobian.copy()
        rdata.GreenesResidue = self.GreenesResidue.copy()
        if self._found_by_iota:
            rdata.MeanResidue = self.MeanResidue.copy()

        return rdata
    
    @property
    def eigenvalues(self):
        """
        Compute the eigenvalues of the Jacobian of the fixed point
        """
        if not self.successful:
            raise ValueError("Fixed point not found.")
        return np.linalg.eigvals(self.Jacobian)
    
    @property
    def RZcoords(self):
        """
        Return the RZ coordinates of the fixed point, applying a transform if the 
        field is a ToroidalBfield (rho theta)
        """
        if not self.successful:
            raise ValueError("Fixed point not found.")
        if isinstance(self._map, maps.ToroidalBfieldSection):
            converted_coords = [self._map._mf.convert_coords(np.append(coord, self._map.phi0)) for coord in self.coords]
            return np.array(converted_coords)[:, ::2]
        else: 
            return self.coords
        
    
    @property
    def polarcoords(self):
        """
        Return the polar coordinates of the fixed point, applying a transform if the 
        field is a ToroidalBfield (rho theta)
        """
        if not self.successful:
            raise ValueError("Fixed point not found.")
        if isinstance(self._map, maps.CylindricalBfieldSection):
            converted_coords = [self._map.to_rhotheta(np.append(coord, self._map.phi0)) for coord in self.coords]
            return np.array(converted_coords)[:, ::2]
        else: 
            return self.coords

    @property
    def topological_index(self):
        """
        return the topological index of the fixed point
        """
        if not self.successful:
            raise ValueError("Fixed point not found.")

        return np.sign(self.GreenesResidue)

    @property
    def rotational_transform(self):
        """
        return the rotational transform if the fixed point is Elliptic
        """
        if not self.successful:
            raise ValueError("Fixed point not found.")
        if np.abs(np.trace(self.Jacobian)) >= 2:
            raise ValueError("rotational transform only defined for elliptic fixed point, this point is hyperbolic")
        return np.arccos(np.trace(self.Jacobian)/2)/(2*np.pi)

    """
    Solver methods.

    They are private methods that are used to solve the fixed point problem. They should either return the coordinates of the fixed point if the search was successful or None if the search was not. 
    
    They can be of two types depending whether they need to be used with the winding number or not.
    """

    def _newton_method(self, guess, niter=100, tol=1e-10):
        x = np.array(guess, dtype=np.float64)
        self.history.append(x.copy())
        succeeded = False

        for i in range(niter):
            logger.info(f"Newton {i} - x : {x}")
            df = self._map.df(self.t, x)
            x_evolved = self._map.f(self.t, x)

            # Stop if the resolution is good enough
            logger.info(f"Newton {i} - delta_x : {x_evolved-x}")
            if np.linalg.norm(x_evolved - x) < tol:
                succeeded = True
                break

            # Newton's step
            delta_x = x_evolved - x
            step = np.linalg.solve(df - np.eye(self._map.dimension), -1 * delta_x)
            x_new = self._map.into_domain(x + step)

            # Update the variables
            logger.info(f"Newton {i} - step : {x_new-x}")
            x = x_new

            if not self._map.in_domain(x):
                logger.info(f"Newton {i} - out of domain")
                return None

            self.history.append(x.copy())

        if succeeded:
            return x
        else:
            return None

    def _scipy_root(self, guess, **kwargs):
        """
        Use the scipy root method to find the fixed point. 
        Finds the minimum of :math:f^{t/2}-f^{-t/2} as it should be better behaved. 
        """
        scipy_method = kwargs.pop("scipy_method", "hybr")

        def fun(x):
            x = self._map.into_domain(x) # map back to domain
            logger.info(f"Newton - xx : {x}")
            diff = self._map.f(self.t/2, x) - self._map.f(-self.t/2, x)
            logger.info(f"Newton - diff : {diff}")
            return diff
        
        self._scipy_root_res = root(fun, guess, method=scipy_method, **kwargs)
        if not self._scipy_root_res.success:
            logger.info(f"Scipy root failed: {self._scipy_root_res.message}")
            return None

        return np.copy(self._scipy_root_res.x)
    
    def _scipy_derivs(self, guess, **kwargs):
        """
        find the fixed point by mapping half the period forward, half the period backwards. 
        :math:`f^{t/2}(x) = f^{-t/2}(x)`
        """

        scipy_method = kwargs.pop("scipy_method", "hybr")
        def fun(x):
            logger.info(f"Newton - xx : {x}")
            forwardjac = self._map.df(self.t/2, x)
            forwardf = self._map.f(self.t/2, x)
            backwardjac = self._map.df(-self.t/2, x)
            backwardf = self._map.f(-self.t/2, x)

            return forwardf - backwardf, (forwardjac - backwardjac)

        self._scipy_root_res = root(fun, guess, jac=True, method=scipy_method, **kwargs)
        
        if not self._scipy_root_res.success:
            logger.info(f"Scipy root failed: {self._scipy_root_res.message}")
            return None
       
        return np.copy(self._scipy_root_res.x)
    
    def _scipy_1d_symmetry(self, guess, **kwargs):
        """
        use a 1d solver to find the fixed point in a symmetry plane. 
        In stellarator symmetric devices, it is guaranteed there is a fixed point on Z=0.
        R is varied and a zero of the Z component of the half map difference is found.

        With bracketing functions it is useful to add a bracket_infinity_sign keyword +- 1, to choose 
        if a failed evaluation should return positive or negative infinity. (sign of the return must be opposite
        on each limit of the bracket)
    
        """
        if guess[-1] != 0.:
            raise ValueError("1d finding only works on the Z=0 line")
        if not (self._map.phi0 == 0. or np.isclose(self._map.phi0 / self._map.dzeta, 0.5)):
            raise ValueError("1d finding only works on a symmetry plane")
        scipy_method = kwargs.pop("scipy_method", "brentq")
        
        bracket = kwargs.pop("bracket", None)
        bracket_infinity_sign = np.sign(kwargs.pop("bracket_infinity", -1))  # sign of the infinity to return when the map is not calculated
        if scipy_method in ["toms748", "ridder", "brentq", "brenth", "bisect"]:
            if bracket is None:
                bracket = guess[0] * np.array([0.99, 1.01])
                logger.info(f"Newton - bracket not given for method that requires it, using 1 percent margin, bracket = {bracket}")
        
        def fun(R):
            logger.info(f"Newton - xx : {R}")
            try:
                diff = (self._map.f(self.t/2, np.append(R, 0.)) - self._map.f(-self.t/2, np.append(R, 0.)))
                zdiff = diff[-1]
            except Exception as e:
                logger.info(f"Newton - failed at xx {R}: {e}")
                zdiff = np.inf*bracket_infinity_sign

            logger.info(f"Newton - zdiff : {zdiff}")
            return zdiff

        self._scipy_root_res = root_scalar(fun, x0=guess[0], method=scipy_method, bracket=bracket, **kwargs)
        
        if not self._scipy_root_res.success:
            logger.info(f"Scipy root failed: {self._scipy_root_res.message}")
            return None
        
        return np.array([self._scipy_root_res.root, 0])

    def _scipy_1d_2d(self, guess, **kwargs):
        """
        use a 1d solver to find the fixed point in a symmetry plane, then refine with a 
        2d search on the found point. 
        In stellarator symmetric devices, it is guaranteed there is a fixed point on Z=0.
        Nevertheless, due to numerical error the fixed point of the integration may be slightly offset. 
        In this method, first R is varied and a zero of the Z component of the half map difference is found, then that point is used as the starting point of the 2d search. 


        With bracketing functions it is useful to add a bracket_infinity_sign keyword +- 1, to choose 
        if a failed evaluation should return positive or negative infinity. (sign of the return must be opposite
        on each limit of the bracket)
    
        """
        if not np.isclose(guess[-1], 0., atol=1e-7):
            raise ValueError("1d finding only works on the Z=0 line")
        if not (self._map.phi0 == 0. or np.isclose(self._map.phi0 / self._map.dzeta, 0.5)):
            raise ValueError("1d finding only works on a symmetry plane")
        scipy_method = kwargs.pop("scipy_method", "brentq")
        
        bracket = kwargs.pop("bracket", None)
        bracket_infinity_sign = np.sign(kwargs.pop("bracket_infinity", -1))  # sign of the infinity to return when the map is not calculated
        if scipy_method in ["toms748", "ridder", "brentq", "brenth", "bisect"]:
            if bracket is None:
                bracket = guess[0] * np.array([0.99, 1.01])
                logger.info(f"Newton - bracket not given for method that requires it, using 1 percent margin, bracket = {bracket}")
        
        def fun_1d(R):
            logger.info(f"Newton - xx : {R}")
            try:
                diff = (self._map.f(self.t/2, np.append(R, 0.)) - self._map.f(-self.t/2, np.append(R, 0.)))
                zdiff = diff[-1]
            except Exception as e:
                logger.info(f"Newton - failed at xx {R}: {e}")
                zdiff = np.inf*bracket_infinity_sign

            logger.info(f"Newton 1d - zdiff : {zdiff}")
            return zdiff

        self._scipy_root_res_1d = root_scalar(fun_1d, x0=guess[0], method=scipy_method, bracket=bracket, **kwargs)
        guess2d = [self._scipy_root_res_1d.root, 0]

        def fun_2d(x):
            logger.info(f"Newton 2d - xx : {x}")
            return self._map.f(self.t/2, x) - self._map.f(-self.t/2, x)
        
        self._scipy_root_res = root(fun_2d, guess2d)
        
        if not self._scipy_root_res.success:
            logger.info(f"Scipy root failed: {self._scipy_root_res.message}")
            return None

        return np.copy(self._scipy_root_res.x)


    def _newton_method_winding(self, guess, xaxis=None, niter=100, tol=1e-10):
        """
        Newton's method to find the fixed point using the change in total
        winding, based on a toroidal coordinate system located on the
        axis. 
        Guess should be provided in the coordinates natural to the map, 
        and will be converted to toroidal internally.
        """
        x = np.array(guess)

        self.history.append(x.copy())
        succeeded = False
        target_dtheta = self._n * self._map.dzeta

        for i in range(niter):
            logger.info(f"Newton {i} - x : {x}")

            dW = self._map.dwinding(self.t, x, xaxis)
            x_winding = self._map.winding(self.t, x, xaxis)
            logger.info(
                f"Newton {i} - x_winding : {x_winding}"
            )

            if isinstance(self._map, maps.ToroidalBfieldSection):
                rhotheta = x
            elif isinstance(self._map, maps.CylindricalBfieldSection):
                rhotheta = self._map.to_rhotheta(x)

            # Stop if the resolution is good enough
            delta_x = np.array([x_winding[0] - rhotheta[0], x_winding[-1] - target_dtheta])
            logger.info(f"Newton {i} - delta_x_rhodtheta : {delta_x}")
            if abs(delta_x[-1]) < tol:
                succeeded = True
                break

            # Newton's step
            step = np.linalg.solve(dW - np.eye(self._map.dimension), -1 * delta_x)
            rhotheta_new = self._map.into_domain(rhotheta + step)

            # Update the variables
            logger.info(f"Newton {i} - step: {step}")
            
            if isinstance(self._map, maps.ToroidalBfieldSection):
                x = rhotheta_new
            elif isinstance(self._map, maps.CylindricalBfieldSection):
                x = self._map.to_RZ(rhotheta_new)
                logger.info(f"Newton {i} coordinate_transform - x_new: {x}")

            if not self._map.in_domain(x):
                logger.info(f"Newton {i} - out of domain")
                return None

            self.history.append(x.copy())

        if succeeded:
            return x
        else:
            return None
    
        
    def _scipy_winding(self, guess, xaxis=None, **kwargs):
        """
        Use the scipy root method to find the fixed point using the change in total
        winding, based on a toroidal coordinate system located on the
        axis. 
        Guess should be provided in the coordinates natural to the map, 
        and will be converted to toroidal internally.
        """
        scipy_method = kwargs.pop("scipy_method", "hybr")
        target_dtheta = self._n * self._map.dzeta

        def fun(x):
            logger.info(f"Newton - xx : {x}")
            if not self._map.in_domain(x):
                logger.info(f"Newton - out of domain")
                return None
            x_winding = self._map.winding(self.t, x, xaxis)
            logger.info(f"Newton - x_winding : {x_winding}")

            if isinstance(self._map, maps.ToroidalBfieldSection):
                rhotheta = x
            elif isinstance(self._map, maps.CylindricalBfieldSection):
                rhotheta = self._map.to_rhotheta(x)

            delta_x = np.array([x_winding[0] - rhotheta[0], x_winding[-1] - target_dtheta])
            logger.info(f"Newton - delta_x_rhodtheta : {delta_x}")
            return delta_x
        
        self._scipy_root_res = root(fun, guess, method=scipy_method, **kwargs)
        
        if not self._scipy_root_res.success:
            logger.info(f"Scipy root failed: {self._scipy_root_res.message}")
            return None

        return np.copy(self._scipy_root_res.x)

    def _scipy_winding_1D(self, guess, xaxis=None, **kwargs):
        """
        Use the scipy root method to find the fixed point using the change in total
        winding, based on a toroidal coordinate system located on the
        axis. 
        Guess should be provided in the coordinates natural to the map, 
        and will be converted to toroidal internally.
        """
        scipy_method = kwargs.pop("scipy_method", "hybr")
        target_dtheta = self._n * self._map.dzeta

        if guess[-1] != 0.:
            raise ValueError("1d finding only works on the Z=0 or theta=0 line")

        def fun(x_1d):
            xx = np.array([x_1d, 0.])
            logger.info(f"Newton - xx : {xx}")
            x_winding = self._map.winding(self.t, xx, xaxis)
            logger.info(f"Newton - x_winding : {x_winding}")

            if isinstance(self._map, maps.ToroidalBfieldSection):
                s_theta = xx
            elif isinstance(self._map, maps.CylindricalBfieldSection):
                s_theta = self._map.to_rhotheta(xx)

            delta_x = np.array([x_winding[0] - s_theta[0], x_winding[-1] - target_dtheta])
            logger.info(f"Newton - delta_x_rhodtheta : {delta_x}")
            return delta_x
        
        self._scipy_root_res = root(fun, guess, method=scipy_method, **kwargs)
        
        if not self._scipy_root_res.success:
            logger.info(f"Scipy root failed: {self._scipy_root_res.message}")
            return None

        return np.copy(np.array([self._scipy_root_res.x[0], 0.]))

    def plot(
        self, plottype='RZ', plot_all = True, **kwargs
    ):
        """
        Plot the fixed point. 
        Args: 
            plottype: the type of plot to make, either 'RZ' or 'polar' or None. if None, the natural coordinats of the map will be plotted. 
            plot_all: if True, all the iterates of the fixed point will be plotted, otherwise only the fixed point itself will be plotted.
            **kwargs: additional arguments for the scatter plot, such as color, size, etc.
        """

        if not self.successful:
            raise Exception("A successful call of compute() is needed")
        elif self._map.dimension != 2:
            raise ValueError("I can only plot 2D fixed points.")

        fig, ax, kwargs = create_canvas(**kwargs)

        if kwargs.get("marker", None) is None:
            if self.GreenesResidue > 1:
                # Alternating hyperbolic fixed point
                kwargs["marker"] = "s"
            elif self.GreenesResidue < 0:
                # Hyperbolic fixed point
                kwargs["marker"] = "X"
            elif self.GreenesResidue < 1:
                # Elliptic fixed point
                kwargs["marker"] = "o"
        
        if plottype == 'RZ':
            plotpoints = self.RZcoords
        elif plottype == 'polar':
            plotpoints = self.polarcoords
        elif plottype is None:
            plotpoints = self.coords
       
        if plot_all:
            ax.scatter(plotpoints[:, 0], plotpoints[:, 1], **kwargs)
        else:
            ax.scatter(plotpoints[0, 0], plotpoints[0, 1], **kwargs)

        return fig, ax
