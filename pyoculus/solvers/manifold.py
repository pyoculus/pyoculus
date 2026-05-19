import pyoculus.maps as maps
from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from ..utils.plot import create_canvas, clean_bigsteps
from scipy.optimize import root, minimize
from typing import Iterator, Literal, Union, Iterable
from numpy.typing import NDArray
import dill as pickle
# from functools import total_ordering
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

import logging

logger = logging.getLogger(__name__)


def eig(jacobian):
    """
    Compute stable and unstable eigenvalues/eigenvectors of a fixed point.

    This function calculates the eigenvalues and eigenvectors of a given Jacobian matrix
    and separates them into stable and unstable components based on their magnitudes.

    Args:
        jacobian (np.ndarray): A 2x2 Jacobian matrix at the fixed point.

    Returns:
        tuple:
            lambda_s (float): The stable eigenvalue (:math:`\\vert\\lambda\\vert < 1`)
            vector_s (np.ndarray): The corresponding stable eigenvector
            lambda_u (float): The unstable eigenvalue (:math:`\\vert\\lambda\\vert > 1`)
            vector_u (np.ndarray): The corresponding unstable eigenvector

    Examples:
        >>> J = np.array([[1.5, 0.5], [0.5, 2.0]])
        >>> lambda_s, v_s, lambda_u, v_u = eig(J)
    """

    eigRes = np.linalg.eig(jacobian)
    eigenvalues = np.abs(eigRes[0])

    # Eigenvectors are stored as columns of the matrix eigRes[1], transposing it to access them as np.array[i]
    eigenvectors = eigRes[1].T

    # Extract the index of the stable and unstable eigenvalues
    s_index, u_index = 0, 1
    if eigenvalues[0].real > eigenvalues[1].real:
        s_index, u_index = 1, 0

    return (
        eigenvalues[s_index],
        eigenvectors[s_index],
        eigenvalues[u_index],
        eigenvectors[u_index],
    )


class Clinic:
    """A class representing the trajectory of a homoclinic/heteroclinic point.

    This class handles the computation and storage of a heteroclinic/homoclinic trajectory, which represent intersections between stable and unstable manifolds of fixed points.

    Args:
        manifold: The :class:`Manifold` object associated to the fixed points and map analyzed.
        eps_s (float): Initial distance in the linear regime along stable manifold direction.
        eps_u (float): Initial distance in the linear regime along unstable manifold direction.
        n_s (int): Number of iterations to apply to the intersection closest to the stable fixed point.
        n_u (int): Number of iterations to apply to the intersection closest to the unstable fixed point.

    Attributes:
        eps_s (float): Distance parameter along stable manifold.
        eps_u (float): Distance parameter along unstable manifold.
        nint_s (int): Number of stable iterations.
        nint_u (int): Number of unstable iterations.
        _fundamental_segments (dict): Fundamental domain bounds.
        _trajectory (np.ndarray): Computed clinic orbit.
        _path_s (np.ndarray): Stable manifold path.
        _path_u (np.ndarray): Unstable manifold path.
        _xend_s (np.ndarray): End point on stable manifold.
        _xend_u (np.ndarray): End point on unstable manifold.
    """

    def __init__(
        self, manifold: "Manifold", eps_s: float, eps_u: float, n_s: int, n_u: int
    ) -> None:
        self._manifold = manifold
        self.eps_s = eps_s
        self.eps_u = eps_u
        self.nint_s = n_s
        self.nint_u = n_u
        self._fundamental_segments = None
        self._trajectory = None
        self._path_s = None
        self._path_u = None
        self._xend_s = None
        self._xend_u = None

    @classmethod
    def from_guess(cls, manifold: "Manifold", eps_s: float, eps_u: float, n_s: int, n_u: int, ERR=1e-3, **kwargs):
        """
        Search a homo/hetero-clinic point.

        This function attempts to find the intersection point of the stable and unstable manifolds by iteratively adjusting the provided epsilon guesses using scipy root-finding algorithm.

        Args:
            guess_eps_s (float): Initial guess for the stable manifold epsilon.
            guess_eps_u (float): Initial guess for the unstable manifold epsilon.
            **kwargs: Additional keyword arguments.
                - root_args (dict): Arguments to pass to the root-finding function.
                - ERR (float): Error tolerance for verifying the linear regime (default: 1e-3).
                - n_s (int): Number of times the map needs to be applied for the stable manifold.
                - n_u (int): Number of times the map needs to be applied for the unstable manifold.

        Returns:
            tuple: A tuple containing the found epsilon values for the stable and unstable manifolds (eps_s, eps_u).
        """

        logger.debug(f"Using n_s, n_u - {n_s}, {n_u}")

        # Updating the default root finding parameters
        root_kwargs = {"jac": False}
        root_kwargs.update(kwargs.get("root_args", {}))
        use_jac = root_kwargs.get("jac")

        logger.debug(f"Using root finding parameters: {root_kwargs}")

        rfp_s = manifold.rfp_s
        rfp_u = manifold.rfp_u
        vector_s = manifold.vector_s
        vector_u = manifold.vector_u
        map_multiple = manifold.fixedpoint_1.m

        # Verifying that epsilons lie in linear regime
        stable_error = manifold.error_linear_regime(eps_s, rfp_s, manifold.vector_s, direction=-1)

        if (stable_error > ERR):
            raise ValueError(f"Stable epsilon guess is not in the linear regime, error is {stable_error}.")
        unstable_error =  manifold.error_linear_regime(eps_u, rfp_u, manifold.vector_u, direction=+1)
        if (unstable_error > ERR):
            raise ValueError(f"Unstable epsilon guess is not in the linear regime, error is {unstable_error}.")
        

        # Evolution function for the root finding without and with jacobian
        def evolution_no_jac(eps, n_s, n_u):
            eps_s, eps_u = eps
            r_s = rfp_s + eps_s * vector_s
            r_u = rfp_u + eps_u * vector_u

            r_s_evolved = manifold._map.f(-1 * n_s * map_multiple, r_s)
            r_u_evolved = manifold._map.f(n_u * map_multiple, r_u)
            logger.debug(f"mapping to : {r_s_evolved}, {r_u_evolved}")

            return (r_s_evolved, r_u_evolved, r_s_evolved - r_u_evolved)

        def evolution_with_jac(eps, n_s, n_u):
            eps_s, eps_u = eps
            r_s = rfp_s + eps_s * vector_s
            r_u = rfp_u + eps_u * vector_u

            jac_s = manifold._map.df(-1 * n_s * map_multiple, r_s)
            r_s_evolved = manifold._map.f(-1 * n_s * map_multiple, r_s)

            jac_u = manifold._map.df(n_u * map_multiple, r_u)
            r_u_evolved = manifold._map.f(n_u * map_multiple, r_u)

            return (
                r_s_evolved,
                r_u_evolved,
                r_s_evolved - r_u_evolved,
                np.array([jac_s @ vector_s, -jac_u @ vector_u]),
            )

        # Root finding
        def residual(logeps, n_s, n_u):
            eps_s, eps_u = np.exp(logeps)
            logger.debug(f"Current epsilon pair (eps_s, eps_u) : {eps_s, eps_u}")

            # if not in the fundamental segments then it should get back there, to work on maybe...

            if use_jac:
                _, _, diff, r_jac = evolution_with_jac([eps_s, eps_u], n_s, n_u)
                diff_jac = r_jac * np.array([eps_s, eps_u])
            else:
                _, _, diff = evolution_no_jac([eps_s, eps_u], n_s, n_u)

            logger.debug(f"Current difference : {diff}")

            if use_jac:
                return diff, diff_jac
            else:
                return diff

        root_obj = root(
            residual,
            np.log([eps_s, eps_u]),
            args=(n_s, n_u),
            **root_kwargs,
        )

        # Checking status and logging the result
        logger.info(f"Root search status : {root_obj.message}")
        logger.debug(f"Root search object : {root_obj}")

        if not root_obj.success:
            raise ValueError("Homo/Heteroclinic search was not successful.")

        eps_s, eps_u = np.exp(root_obj.x)

        logger.info(
            f"Success! Found epsilon pair (eps_s, eps_u) : {eps_s:.3e}, {eps_u:.3e} gives a difference of {root_obj.fun}."
        )

        return cls(manifold, eps_s, eps_u, n_s, n_u)
        
    @classmethod
    def with_deflation(cls, manifold:"Manifold", eps_s:float, eps_u: float, n_s: int, n_u: int, **kwargs):
        """
        find a clinic point using a deflation method and bounds to remove previously found
        clinic points. 
        """
        logger.debug(f"Using n_s, n_u - {n_s}, {n_u}")
        # Updating the default root finding parameters
        root_kwargs = {"jac": False}
        root_kwargs.update(kwargs.get("root_args", {}))
        use_jac = root_kwargs.get("jac")
        logger.debug(f"Using root finding parameters: {root_kwargs}")

        if manifold.clinics.size <1: 
            raise ValueError("No clinics in the set, cannot use the deflation method.")

        rfp_s = manifold.rfp_s
        rfp_u = manifold.rfp_u
        vector_s = manifold.vector_s
        vector_u = manifold.vector_u
        map_multiple = manifold.fixedpoint_1.m
        fundamental_stable = manifold.clinics.fundamental_segments["stable"]
        fundamental_unstable = manifold.clinics.fundamental_segments["unstable"]

        # the first and last return point get mapped to +- infinity, we ignore them
        found_inner_epsilons = np.stack([manifold.clinics.stable_epsilons[1:-1], manifold.clinics.unstable_epsilons[1:-1]], axis=1)

        def map_optimization_variables_to_epsilons(xx_opt, fundamental_stable, fundamental_unstable):
            """
            we need to stick to the fundamental section, so we use the hyperbolic tangent
            to map our optimization variables to the epsilons in this section
            """
            eps_s = fundamental_stable[0] + (fundamental_stable[1] - fundamental_stable[0]) * (np.tanh(.1*xx_opt[0]) + 1) / 2
            eps_u = fundamental_unstable[0] + (fundamental_unstable[1] - fundamental_unstable[0]) * (np.tanh(.1* xx_opt[1]) + 1) / 2
            return eps_s, eps_u
       
        def map_epsilons_to_optimization_variables(eps_s, eps_u, fundamental_stable, fundamental_unstable):
            """
            we need to stick to the fundamental section, so we use the hyperbolic tangent
            to map our optimization variables to the epsilons in this section
            """
            eps_s = np.arctanh(2 * (eps_s - fundamental_stable[0]) / (fundamental_stable[1] - fundamental_stable[0]) - 1) *10
            eps_u = np.arctanh(2 * (eps_u - fundamental_unstable[0]) / (fundamental_unstable[1] - fundamental_unstable[0]) - 1) *10
            return eps_s, eps_u
        
        def deflated_residual(xx_opt, n_s, n_u, found_epsilons_in_optimization_variables):
            """
            deflation residual function for the optimization problem. 
            Deflation consists of multiplying the residual by $1 + \Sum 1/|xx-xx_found|^2)$, which blows up at the found points. 

            Since we expect the roots to be first order, we can use second order deflation.  
            """
            print(f'xx_opt: {xx_opt}')
            eps_s, eps_u = map_optimization_variables_to_epsilons(xx_opt, fundamental_stable, fundamental_unstable)
            print(f'eps_s, eps_u: {eps_s, eps_u}')
            r_s = rfp_s + eps_s * vector_s
            r_u = rfp_u + eps_u * vector_u

            r_s_evolved = manifold._map.f(-1 * n_s * map_multiple, r_s)
            r_u_evolved = manifold._map.f(n_u * map_multiple, r_u)

            diff = r_s_evolved - r_u_evolved

            # Deflation term: 1 plus sum of inverse of squared distances to previously found epsilon points
            # we do the deflation in epsilon space, the optimization space is only to limit to
            # the fundamental section.
            deflation_term = 1 + (1e-3* np.sum(xx_opt**2)) + np.sum([1/(np.linalg.norm(found_roots - xx_opt)**2) for found_roots in found_epsilons_in_optimization_variables])
            print(f'difference: {diff}, eps_s, eps_u : {eps_s, eps_u}, optimization variables : {xx_opt} deflation_term: {deflation_term}')
            return diff * deflation_term
        
        found_epsilons_in_optimization_variables = [map_epsilons_to_optimization_variables(*found_pair, fundamental_stable, fundamental_unstable) for found_pair in found_inner_epsilons]
        print(f'found_epsilons_in_optimization_variables: {found_epsilons_in_optimization_variables}')

        root_obj = root(
            deflated_residual,
            map_epsilons_to_optimization_variables(eps_s, eps_u, fundamental_stable, fundamental_unstable),
            args=(n_s, n_u, found_epsilons_in_optimization_variables),
            **root_kwargs,
        )
        
        # Checking status and logging the result
        logger.info(f"Root search status : {root_obj.message}")
        logger.debug(f"Root search object : {root_obj}")

        if not root_obj.success:
            raise ValueError("Homo/Heteroclinic search was not successful.")

        eps_s, eps_u = map_optimization_variables_to_epsilons(root_obj.x, fundamental_stable, fundamental_unstable)

        logger.info(
            f"Success! Found epsilon pair (eps_s, eps_u) : {eps_s:.3e}, {eps_u:.3e} gives a difference of {root_obj.fun}."
        )

        return cls(manifold, eps_s, eps_u, n_s, n_u)
        



    @property
    def trajectory(self):
        """Get the complete trajectory of the clinic point.

        Computes the trajectory by integrating along stable and unstable
        manifolds if not already calculated.

        Returns:
            np.ndarray: Array containing the orbit from unstable to stable fixed point.
        """

        if self._trajectory is not None:
            return self._trajectory

        path_u = self._manifold.integrate(
            self._manifold.rfp_u + self.eps_u * self._manifold.vector_u, self.nint_u, +1
        )[:, 0, :]
        path_s = self._manifold.integrate(
            self._manifold.rfp_s + self.eps_s * self._manifold.vector_s, self.nint_s - 1, -1
        )[:, 0, :]

        self._trajectory = np.concatenate((path_u, path_s[::-1, :]))

        return self._trajectory

    @property
    def x_end_s(self):
        """Get the endpoint on the stable manifold.

        Returns:
            np.ndarray: Coordinates of the end point on stable manifold.
        """
        if self._xend_s is not None:
            return self._xend_s
        elif self._path_s is not None:
            self._xend_s = self._path_s[-1, :]
        else:
            self.trajectory
            self._xend_s = self._path_s[-1, :]

        return self._xend_s

    @property
    def x_end_u(self):
        """Get the endpoint on the unstable manifold.

        Returns:
            np.ndarray: Coordinates of the end point on unstable manifold.
        """
        if self._xend_u is not None:
            return self._xend_u
        elif self._path_u is not None:
            self._xend_u = self._path_u[-1, :]
        else:
            self.trajectory
            self._xend_u = self._path_u[-1, :]

        return self._xend_u

    @property
    def fundamental_segments(self):
        """Get the fundamental domain boundaries.

        Returns:
            dict: Contains 'stable' and 'unstable' segment bounds.
        """
        if self._fundamental_segments is None:
            bnd_s, bnd_u = self._fundamental_segments_from_eps()
            self._fundamental_segments = {"stable": bnd_s, "unstable": bnd_u}
        return self._fundamental_segments

    # Private methods

    def _fundamental_segments_from_eps(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Calculate the fundamental segment on unstable and stable manifolds.

        Computes the bounds of fundamental domains by evolving points along
        the manifolds and measuring their distances from fixed points.

        Returns:
            tuple: A tuple containing two tuples:
                - (eps_s, upperbound_s): The initial ε_s and computed upper bound
                  for the stable manifold.
                - (eps_u, upperbound_u): The initial guess and computed upper bound
                  for the unstable manifold.
        """
        # Initial points along the manifolds
        r_s = self._manifold.rfp_s + self.eps_s * self._manifold.vector_s
        r_u = self._manifold.rfp_u + self.eps_u * self._manifold.vector_u

        # Evolve the points along the manifolds
        # r_s_unevolved = self._map.f(+1*self.fixedpoint_1.m, r_s)
        # r_s_evolved   = self._map.f(-1*self.fixedpoint_1.m, r_s)
        r_s_evolved = self._manifold.integrate(r_s, 1, -1)[1, 0]

        # r_u_unevolved = self._map.f(-1*self.fixedpoint_1.m, r_u)
        # r_u_evolved   = self._map.f(+1*self.fixedpoint_1.m, r_u)
        r_u_evolved = self._manifold.integrate(r_u, 1, +1)[1, 0]

        # Measure the distance from the fixed points with usual two norm
        upperbound_s = np.linalg.norm(r_s_evolved - self._manifold.rfp_s)
        upperbound_u = np.linalg.norm(r_u_evolved - self._manifold.rfp_u)

        return (self.eps_s, upperbound_s), (self.eps_u, upperbound_u)


class ClinicSet:
    """A collection of homoclinics/heteroclinics with unstable (:math:`>_u`) ordering.

    This class manages multiple :class:`Clinic` objects, maintaining their ordering and ensuring
    proper fundamental domain representation. It uses the first clinic added to manage the fundamental domain boundaries.

    Args:
        manifold: The :class:`Manifold` object associated to the fixed points and the map analyzed.

    Attributes:
        _clinics_list (list): List of Clinic objects.
        fundamental_segments (dict): Fundamental domain boundaries.
        nint_pair (tuple): Default iteration numbers (n_s, n_u).
        total_number_of_points: Return the total number of points in the fundamental clinic trajectory.
        stable_segment: Return the stable segment of the fundamental domain.
        unstable_segment: Return the unstable segment of the fundamental domain.

    Methods:
        record_clinic: Add a new clinic point to the collection.
        reset: Clear all stored clinics.
        is_empty: Check if the clinic set is empty.

    Examples:
        >>> a_manifold = Manifold(...)
        >>> clinic_set = ClinicSet(a_manifold)
        >>> clinic_set.record_clinic(myclinic)
    """

    DEFAULT_TOLERANCE = 1e-4
    MAX_ITERATIONS = 20

    def __init__(self, manifold: "Manifold") -> None:
        self._manifold = manifold
        self.reset()

    def __len__(self) -> int:
        return len(self._clinics_list)

    @property
    def size(self) -> int:
        """Number of clinic points in the set."""
        return len(self._clinics_list)

    @property
    def is_empty(self) -> bool:
        """Check if the clinic set is empty."""
        return len(self._clinics_list) == 0

    # Make the class indexable
    def __getitem__(self, index: int) -> Clinic:
        return self._clinics_list[index]

    # Make the class iterable
    def __iter__(self) -> Iterator[Clinic]:
        return iter(self._clinics_list)

    # Public methods
    @property
    def total_number_of_points(self):
        """
        the number of elements in the fundamental clinic trajectory
        """
        if self.is_empty:
            logger.warning('No clinics in the set, returning 0')
            return 0
        return sum(self.nint_pair)

    @property
    def stable_segment(self):
        """
        the stable segment of the fundamental domain
        """
        if not self.is_empty:
            return self.fundamental_segments["stable"]
        else: return None

    @property
    def unstable_segment(self):
        """
        the stable segment of the fundamental domain
        """
        if not self.is_empty:
            return self.fundamental_segments["unstable"]
        else: return None

    @property
    def stable_epsilons(self):
        """
        return the stable epsilons of the clinics, including the end of the fundamental section
        """
        if self.is_empty:
            return []
        return [clinic.eps_s for clinic in self._clinics_list] + [self.stable_segment[1]]

    @property
    def unstable_epsilons(self):
        """
        return the unstable epsilons of the clinics, including the end of the fundamental section
        """
        if self.is_empty:
            return []
        return [clinic.eps_u for clinic in self._clinics_list] + [self.unstable_segment[1]]

    @property
    def stable_shifts(self):
        """
        return the stable shifts of the clinics
        """
        if self.is_empty:
            return []
        epsilons = self.stable_epsilons
        epsilon_range = epsilons[-1] - epsilons[0]
        return [(thiseps - epsilons[0])/epsilon_range for thiseps in epsilons[1:-1]]

    @property
    def unstable_shifts(self):
        """
        return the unstable shifts of the clinics (excluding the clinic defining the fundamental section)
        """
        if self.is_empty:
            return []
        epsilons = self.unstable_epsilons
        epsilon_range = epsilons[-1] - epsilons[0]
        return [(thiseps - epsilons[0])/epsilon_range for thiseps in epsilons[1:-1]]

    @property
    def first_epsilons(self):
        """
        return the stable and unstable epsilons of the first clinic in the set
        """
        if self.is_empty:
            return None, None
        return self._clinics_list[0].eps_s, self._clinics_list[0].eps_u

    def record_clinic(
        self, clinic: Clinic, **kwargs
    ) -> bool:
        """Record a new clinic point in the fundamental domain.

        Creates and stores a new :class:`Clinic` object after converting the given parameters
        to their fundamental domain representation.

        Args:
            eps_s (float): Initial distance along stable manifold.
            eps_u (float): Initial distance along unstable manifold.
            n_s (int): Number of iterations for stable manifold.
            n_u (int): Number of iterations for unstable manifold.
            **kwargs: Additional keyword arguments:
                tol (float, optional): Tolerance for comparing epsilon values. Defaults to 1e-2.

        Returns:
            bool: True if the clinic was successfully added, False otherwise.

        Note:
            If this is the first clinic point, it establishes the fundamental domain boundaries.
            Otherwise, parameters are converted to their fundamental domain representation.
        """

        # test if self._clinic_list is empty
        if not self._clinics_list:
            self.fundamental_segments = clinic.fundamental_segments
            self.nint_pair = (clinic.nint_s, clinic.nint_u)
            self._clinics_list.append(clinic)
            return True
        else:
            tol = kwargs.get("tol", self.DEFAULT_TOLERANCE)
            fundamental_eps_s, n_s_shift = self._find_fundamental_eps(clinic.eps_s, "stable", tol=tol)
            fundamental_eps_u, n_u_shift = self._find_fundamental_eps(clinic.eps_u, "unstable", tol=tol)
            if n_s_shift != 0 or n_u_shift != 0:
                logger.warning(
                    f"Shifted clinic by ({n_s_shift}, {n_u_shift}) iterations to the fundamental domain."
                )
                # calculate a new clinic from the shifted values to ensure it is proper.
                clinic = Clinic.from_guess(self._manifold, fundamental_eps_s, fundamental_eps_u, clinic.nint_s + n_s_shift, clinic.nint_u + n_u_shift)

            if self._no_clinics_similar(clinic, tol):
                if not self._test_in_fundamental_segment(clinic.eps_s, "stable", tol) or not self._test_in_fundamental_segment(clinic.eps_u, "unstable", tol):
                    raise ValueError("in recalculating shifted clinic it escaped the fundamental domain.")
                self._clinics_list.append(clinic)
                self._orderize()
                logger.warning("Homo/heteroclinic recorded and ordered.")
                return True
            else:
                logger.warning(f"Homo/heteroclinic already recorded, total clinics = {self.size}. skipping...")
                return False

    def _no_clinics_similar(self, clinic, tol) -> bool:
        """
        Returns True if no clinics in the list are similar
        """
        stable_epsilon_test = not (np.any([np.isclose(clinic.eps_s, other.eps_s, rtol=tol, atol=0.) for other in self._clinics_list ]))  # not any isclose
        unstable_epsilon_test = not (np.any([np.isclose(clinic.eps_u, other.eps_u, rtol=tol, atol=0.) for other in self._clinics_list ]))  # not any isclose
        return stable_epsilon_test and unstable_epsilon_test  # none is close

    def reset(self) -> None:
        """Reset the clinic set to its initial empty state.

        Clears:
            - All stored clinic points
            - Fundamental segment boundaries
            - Default iteration numbers (nint_pair)
        """
        self._clinics_list = []
        self.fundamental_segments = None
        self.nint_pair = None

    # Private methods

    def _orderize(self) -> None:
        """Sorts the internal list of clinic points based on their eps_u values."""
        self._clinics_list = [
            self._clinics_list[i]
            for i in np.argsort([x.eps_u for x in self._clinics_list])
        ]

    def _test_in_fundamental_segment(self, eps: float, which: Literal["stable", "unstable"], tol : float
    ) -> bool:
        """
        test an epsilon is in the fundamental segment of the first clinic in the set (plusminus relative tolerance)
        """
        thissegment = self.fundamental_segments[which]
        return thissegment[0]* (1 - tol) < eps < thissegment[1] *(1 + tol)

    def _find_fundamental_eps(
        self, eps: float, which: Literal["stable", "unstable"], tol, **kwargs
    ) -> tuple[float, int]:
        """
        Find the epsilon parameter lying within the fundamental segment (stable or unstable) of the ClinicSet for a given epsilon value.

        Args:
            eps (float): The initial epsilon value to convert.
            which (str): Either "stable" or "unstable", which manifold eps lies on.
            **kwargs: Additional keyword arguments:
                max_iters (int, optional): Maximum number of iterations to find the fundamental eps.

        Returns:
            tuple: (fundamental_eps, n_shift)
                - fundamental_eps (float): The equivalent eps in fundamental segment.
                - n_shift (int): Number of iterations needed for the shift.

        Raises:
            ValueError: If which is neither "stable" nor "unstable".
            RuntimeError: If fundamental epsilon is not found within max_iters iterations.
        """
        if which == "stable":
            rfp, eigenvector, forward_dir = (
                self._manifold.rfp_s,
                self._manifold.vector_s,
                -1,
            )
        elif which == "unstable":
            rfp, eigenvector, forward_dir = (
                self._manifold.rfp_u,
                self._manifold.vector_u,
                +1,
            )
        else:
            raise ValueError(
                f"Invalid manifold selection: {which}. Must be 'stable' or 'unstable'"
            )

        fund = self.fundamental_segments[which]
        if fund[0] <= eps < fund[1]: # Already in the fundamental segment
            return eps, 0

        r_cur = rfp + eps * eigenvector
        eps_approx = eps
        n_shift = 0

        # If the eps is less then the lower bound, then the direction should be the correct direction for the type of manifold, otherwise the opposite direction
        if eps_approx < fund[0]:
            map_dir = forward_dir
            increase_n = -1 # if the eps is less than the lower bound, then we to map forward and use less steps
        elif eps > fund[1]:
            map_dir = -1 * forward_dir
            increase_n = +1  # if the eps is greater than the upper bound, then we to map backward and use more steps

        # Set a maximum number of iterations
        max_iterations = kwargs.get("max_iters", self.MAX_ITERATIONS)
        for _ in range(max_iterations):
            if fund[0]*(1 - tol) <= eps_approx < fund[1]*(1 + tol):  # within segment plusminus relative tolerance
                logger.debug(f"Found the fundamental segment for {which} manifold with {n_shift} steps.")
                return eps_approx, n_shift
            r_cur = self._manifold._map.f(map_dir * self._manifold.fixedpoint_1.m, r_cur)
            eps_approx = np.linalg.norm(r_cur - rfp)
            n_shift += increase_n
            logger.debug(f"Current epsilon (from norm calculation): {eps_approx}, fundamental segment is between {fund[0]} and {fund[1]}")

        raise RuntimeError(
            "Failed to find a solution within the maximum number of iterations"
        )

    def order(self):
        """
        Order the homo/hetero-clinic points with the induced linear ordering of the unstable manifold >_u.
        """
        self._clinics_list = [
            self._clinics_list[i]
            for i in np.argsort([x.eps_u for x in self._clinics_list])
        ]

    def reset(self):
        """
        remove all known clinics and start afresh.
        """
        self._clinics_list = []
        self.fundamental_segments = None
        self.nint_pair = None




class Manifold(BaseSolver):
    """Class for computing and analyzing a tangle composed of one stable and one unstable manifold of fixed points.

    This class handles the computation of stable and unstable manifolds for fixed points, including finding homoclinic/heteroclinic intersections and calculating the turnstile flux of the tangle.

    We need to resolve some duplicity to determine the direction in which to follow the manifolds of each x-point (M*v=e.v. * v => M*-v = -e.v. * v), this is done with the 'dir' keywords.

    Args:
        map (maps.base_map): The map defining the dynamical system.
        fixedpoint_1 (FixedPoint): First fixed point to consider.
        fixedpoint_2 (FixedPoint, optional): Second fixed point to consider if the manifolds go from one to the other.
        dir1 (str, optional): Direction
                       if type == str, can be ('+' or '-') to multiply eigenvector of the Jacobian.
                       if type == float, approximate angle (with respect of the x-axis) of the manifold to follow
                       if type == np.ndarray, the (approximate) vector of the manifold to follow
        dir2 (str, optional): Direction
                       if type == str, can be ('+' or '-') for the second manifold.
                       if type == float, approximate angle (with respect of the x-axis) of the manifold to follow
                       if type == np.ndarray, the (approximate) vector of the manifold to follow
        first_stable (bool, optional): Whether to follow the stable or unstable manifold departing from the first fixed point. Defaults to True.

    Attributes:
        fixedpoint_1 (FixedPoint): First fixed point.
        fixedpoint_2 (FixedPoint): Second fixed point.
        rfp_s (np.ndarray): Stable fixed point coordinates.
        rfp_u (np.ndarray): Unstable fixed point coordinates.
        vector_s (np.ndarray): Stable eigenvector.
        vector_u (np.ndarray): Unstable eigenvector.
        lambda_s (float): Stable eigenvalue.
        lambda_u (float): Unstable eigenvalue.
        stable (np.array): Stable manifold points.
        unstable (np.array): Unstable manifold points.
        clinics (ClinicSet): Set of homoclinic/heteroclinic intersections.
        turnstile_areas (np.array): Turnstile areas of the tangle.

    Methods:
        choose: Choose the stable and unstable directions for the manifold.
        show_directions: Plot the fixed points and their stable/unstable directions.
        show_current_directions: Plot the current stable and unstable directions.
        error_linear_regime: Metric to evaluate if a point is in the linear regime of a fixed point.
        start_config: Compute a starting configuration for the manifold drawing.
        find_epsilon: Find the epsilon that lies in the linear regime.
        compute_manifold: Compute the stable or unstable manifold.
        compute: Compute the stable and unstable manifolds.
        plot: Plot the stable and unstable manifolds.
        find_N: Find the number of times the map needs to be applied for the stable and unstable points to cross.
        find_clinic_single: Find a single homoclinic/heteroclinic intersection.
        find_clinic: Find all homoclinic/heteroclinic intersections.
        compute_turnstile_areas: Compute the turnstile areas of the tangle.

    Raises:
        TypeError: If fixed points are not FixedPoint instances.
        ValueError: If fixed points are not successfully computed.
    """

    def __init__(
        self,
        map: maps.base_map,
        fixedpoint_1: FixedPoint,
        fixedpoint_2: FixedPoint = None,
        dir1: Union[str, float, NDArray] = None,
        dir2: str = None,
        first_stable: bool = None,
    ) -> None:
        """
        Initialize the Manifold class by providing two fixed points and specifying if the first fixedpoint is stable. If only one fixed point is specified, a homoclinic connection is assumed.
        Directions are specified with the string '+' or '-' to indicate which eigenvectors of the fixed points to follow (remember that if $v$ is an eingenvector, $-v$ is as well).

        Args:
            map (maps.base_map): The map to use for the computation.
            fixedpoint_1 (FixedPoint): first fixed point
            fixedpoint_2 (FixedPoint, optional): second fix point if not homoclinic manifold. Defaults to None.
        """

        # Check that the fixed points are correct FixedPoint instances
        if not isinstance(fixedpoint_1, FixedPoint):
            raise TypeError("Fixed point must be an instance of FixedPoint class")
        if not fixedpoint_1.successful:
            raise ValueError("Need a successful fixed point to compute the manifold")

        if isinstance(fixedpoint_2, FixedPoint):
            if not fixedpoint_2.successful:
                raise ValueError(
                    "Need a successful fixed point to compute the manifold"
                )

        # Initialize the directions and the dictionnaries
        if fixedpoint_2 is not None:
            self.fixedpoint_1 = fixedpoint_1
            self.fixedpoint_2 = fixedpoint_2
        else:
            self.fixedpoint_1 = fixedpoint_1
            self.fixedpoint_2 = fixedpoint_1
            dir1 = '+'
            dir2 = '+'

        # Setting the fast and slow directions
        if dir1 is None:
            dir1 = self.fixedpoint_2.coords[0] - self.fixedpoint_1.coords[0]
        if dir2 is None:
            dir2 = self.fixedpoint_1.coords[0] - self.fixedpoint_2.coords[0]
        if first_stable is None:
            first_stable = True

        self.choose(dir1, dir2, first_stable)

        # Initialize the clinic set
        self.clinics = ClinicSet(self)
        self._areas = None

        # Initialize the BaseSolver
        super().__init__(map)

    def choose(
        self, dir_1: Union[Literal["+", "-"], float, Iterable], dir_2: Union[Literal["+", "-"], float, Iterable], first_stable: bool
    ) -> None:
        """Choose manifold stable and unstable directions to define your :class:`Manifold` problem.

        You must choose directions away from the fixed point in which the manifolds actually intersect. The good orientation is the one for which you could create the manifold by going away from the fixed point. Be carefull to this point, otherwise other manifold computations such as clinic finding will fail.

        Hint: Use :meth:`Manifold.show_directions` to help you choose here. This plot shows the fixed points and the stable eigenvector (and its negative) in green, the unstable eigenvector (and it's negative) in red.

        Args:
            dir_1 (str): '+' or '-' for the stable direction.
            dir_2 (str): '+' or '-' for the unstable direction.
            first_stable  (bool): Whether to follow the stable or unstable manifold from the first point.
        """
        if first_stable:
            self.fp_s = self.fixedpoint_1
            self.fp_u = self.fixedpoint_2
            stabledir = dir_1
            unstabledir = dir_2
        else:
            self.fp_u = self.fixedpoint_1
            self.fp_s = self.fixedpoint_2
            stabledir = dir_2
            unstabledir = dir_1

        # deal with the stable fixed point first:
        self.rfp_s = self.fp_s.coords[0]
        self.lambda_s, self.vector_s, _, _ = eig(self.fp_s.Jacobian)  # algorithm can give either positive or negative
        if type(stabledir) == str and stabledir in ["+", "-"]:
            self.vector_s *= (-1) ** int(stabledir == "-")  # negate if minus provided
        elif isinstance(stabledir, float):
            self.vector_s *= (-1)**int(np.dot(self.vector_s, np.array([np.cos(stabledir), np.sin(stabledir)])) < 0)  # negate if dot negative
        elif isinstance(stabledir, Iterable) and len(stabledir) == 2:
            self.vector_s *= (-1)**int(np.dot(self.vector_s, np.asarray(stabledir)) < 0)  # negate if dot negative
        else:
            raise ValueError("Invalid dir_1 given, either '+'/'-', float or len-2 array")

        # deal with the unstable fixed point
        self.rfp_u = self.fp_u.coords[0]
        _, _, self.lambda_u, self.vector_u = eig(self.fp_u.Jacobian)
        if type(unstabledir) == str and unstabledir in ["+", "-"]:
            self.vector_u *= (-1) ** int(unstabledir == "-") # negate if minus provided
        elif isinstance(unstabledir, float):
            self.vector_u *= (-1)**int(np.dot(self.vector_u, np.array([np.cos(unstabledir), np.sin(unstabledir)])) < 0) # negate if dot negative
        elif isinstance(unstabledir, Iterable) and len(unstabledir) == 2:
            self.vector_u *= (-1)**int(np.dot(self.vector_u, np.asarray(unstabledir)) < 0) # negate if dot negative
        else:
            raise ValueError("Invalid dir_2 given, either '+'/'-', float or len-2 array")

        self._stable_trajectory = None
        self._unstable_trajectory = None

    @classmethod
    def show_directions(
        cls, fp_1: FixedPoint, fp_2: FixedPoint, **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot fixed points and their stable/unstable directions.

        Helper function to plot the fixed points and their stable and unstable direction. Usefull to look at which direction need to be considered for the inner and outer manifolds before creating a class and analyzed them.

        Args:
            fp_1 (FixedPoint): First fixed point.
            fp_2 (FixedPoint): Second fixed point.
            **kwargs: Optional visualization parameters:
                pcolors (list): Colors for fixed points.
                vcolors (list): Colors for eigenvectors.
                vscale (int): Scale for eigenvectors.
                dvtext (float): Text distance as fraction.

        Returns:
            tuple: (figure, axis) matplotlib objects.
        """
        # Defaults
        pcolors = kwargs.get("pcolors", ["tab:blue", "tab:orange"])
        vcolors = kwargs.get("vcolors", ["tab:green", "tab:red"])
        vscale = kwargs.get("vscale", 18)
        dvtext = kwargs.get("dvtext", 0.005)

        # Set the figure and ax
        fig, ax, kwargs = create_canvas(**kwargs)

        # Choose the fixed points and their directions
        rfp_1 = fp_1.coords[0]
        _, p1_vector_s, _, p1_vector_u = eig(fp_1.Jacobian)
        rfp_2 = fp_2.coords[0]
        _, p2_vector_s, _, p2_vector_u = eig(fp_2.Jacobian)

        # Plot the fixed points
        ax.scatter(
            *rfp_1,
            marker="X",
            s=100,
            label="Fixed point 1",
            zorder=999,
            color=pcolors[0],
            edgecolor="black",
            linewidth=1,
        )
        ax.scatter(
            *rfp_2,
            marker="X",
            s=100,
            label="Fixed point 2",
            zorder=999,
            color=pcolors[1],
            edgecolor="black",
            linewidth=1,
        )
        # ax.text(*(rfp_1), '1', zorder=999, ha='center', va='center')
        # ax.text(*(rfp_2), '2', zorder=999, ha='center', va='center')

        # Plot the eigenvectors
        def plot_eigenvectors(ax, rfp, vectors):
            for vector, color in zip(vectors, vcolors):
                # Positive and negative arrows
                p_arrow = FancyArrowPatch(
                    rfp,
                    rfp + vector / vscale,
                    arrowstyle="-|>",
                    color=color,
                    mutation_scale=10,
                )
                n_arrow = FancyArrowPatch(
                    rfp,
                    rfp - vector / vscale,
                    arrowstyle="-|>",
                    color=color,
                    mutation_scale=10,
                )

                # Add the arrows to the plot
                ax.add_patch(p_arrow)
                ax.add_patch(n_arrow)

                # Add the text
                ax.text(
                    *(rfp + vector * (dvtext + 1 / vscale)),
                    "+",
                    zorder=999,
                    color=color,
                    fontsize="large",
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
                ax.text(
                    *(rfp - vector * (dvtext + 1 / vscale)),
                    "-",
                    zorder=999,
                    color=color,
                    fontsize="large",
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

        plot_eigenvectors(ax, rfp_1, [p1_vector_s, p1_vector_u])
        plot_eigenvectors(ax, rfp_2, [p2_vector_s, p2_vector_u])

        return fig, ax

    def show_current_directions(self, vscale=1/5, vcolors=None, **kwargs):
        """Plot the current stable and unstable directions.

        Args:
            **kwargs: Optional visualization parameters:
                vcolors (list): Colors for eigenvectors.
                vscale (int): Scale for the eigenvectors.
                dvtext (float): Text distance as fraction.

        Returns:
            tuple: (figure, axis) matplotlib objects.
        """
        # Defaults

        # Set the figure and ax
        fig, ax, kwargs = create_canvas(**kwargs)
        if vcolors is None:
            vcolors = ["xkcd:royal blue", "xkcd:magenta"]

        s_arrow = FancyArrowPatch(
            self.rfp_s,
            self.rfp_s + self.vector_s * vscale,
            arrowstyle="-|>",
            color=vcolors[0],
            mutation_scale=10,
            **kwargs
        )
        u_arrow = FancyArrowPatch(
            self.rfp_u,
            self.rfp_u + self.vector_u * vscale,
            arrowstyle="-|>",
            color=vcolors[1],
            mutation_scale=10,
            **kwargs
        )

        # Add the arrows to the plot
        ax.add_patch(u_arrow)
        ax.add_patch(s_arrow)


    @property
    def first_epsilons(self):
        """
        return the stable and unstable epsilons of the first clinic in the set
        """
        return self.clinics.first_epsilons

    def error_linear_regime(
        self,
        epsilon: float,
        rfp: np.ndarray,
        eigenvector: np.ndarray,
        direction: int = 1,
    ) -> float:
        """Calculate error in linear regime approximation.

        Metric to estimate if the point rfp + epsilon * eigenvector is in the linear regime of rfp point.

        Args:
            epsilon (float): Distance from fixed point.
            rfp (np.ndarray): Fixed point coordinates.
            eigenvector (np.ndarray): Eigenvector to check.
            direction (int, optional): Integration direction. Defaults to 1.

        Returns:
            float: Error metric for linear approximation.
        """
        # Initial point and evolution
        rEps = rfp + epsilon * eigenvector
        rz_path = self.integrate(rEps, 1, direction)

        # Direction of the evolution
        eps_dir = rz_path[1, 0, :] - rz_path[0, 0, :]
        norm_eps_dir = np.linalg.norm(eps_dir)
        eps_dir_norm = eps_dir / norm_eps_dir

        # Use the dot product to see if: cos(angle btw eps_dir_norm and eigenvector) is close to 1
        return np.abs(1 - np.dot(eps_dir_norm, eigenvector))

    ### Computation of the manifolds
    def start_config(self, epsilon, rfp, eigenvalue, eigenvector, neps, direction=1):
        """
        Compute a starting configuration for the manifold drawing. It takes a point in the linear regime
        and devide the interval from the point to its evolution after one nfp into neps points. The interval
        is computed geometrically.

        Args:
            epsilon (float): initial epsilon
            rfp (np.array): fixed point
            eigenvalue (float): eigenvalue of the fixed point
            eigenvector (np.array): eigenvector of the fixed point
            neps (int): number of points
            direction (int): direction of the integration (1 for forward, -1 for backward)

        Returns:
            np.array: array of starting points (shape (neps, 2))
        """
        # Initial point and evolution
        rEps = rfp + epsilon * eigenvector
        rz_path = self.integrate(rEps, 1, direction)

        # Direction of the evolution
        eps_dir = rz_path[1, 0, :] - rz_path[0, 0, :]
        norm_eps_dir = np.linalg.norm(eps_dir)
        eps_dir_norm = eps_dir / norm_eps_dir

        # Geometric progression from log_eigenvalue(epsilon) to log_eigenvalue(epsilon + norm_eps_dir)
        eps = np.logspace(
            np.log(epsilon) / np.log(eigenvalue),
            np.log(epsilon + norm_eps_dir) / np.log(eigenvalue),
            neps,
            base=eigenvalue,
            endpoint=False
        )

        Rs = rfp[0] + eps * eps_dir_norm[0]
        Zs = rfp[1] + eps * eps_dir_norm[1]
        return np.array([Rs, Zs]).T

    def find_epsilon(self, which: str, eps_guess=1e-3):
        """
        Find the epsilon that lies in the linear regime.
        """
        if which == "stable":
            rfp, eigenvector, direction = self.rfp_s, self.vector_s, -1
        elif which == "unstable":
            rfp, eigenvector, direction = self.rfp_u, self.vector_u, +1
        else:
            raise ValueError("Invalid manifold selection.")

        find_eps = lambda x: self.error_linear_regime(
            x, rfp, eigenvector, direction=direction
        )
        minobj = minimize(find_eps, eps_guess, bounds=[(0, 1)], tol=1e-12)

        if not minobj.success:
            raise ValueError("Search for minimum of the linear error failed.")

        esp_root = minobj.x[0]
        logger.info(
            f"Search for minimum of the linear error succeeded, epsilon = {esp_root:.5e}."
        )
        return esp_root

    def compute_manifold(self, which: str, eps=None, **kwargs):
        """
        Compute the stable or unstable manifold.

        Args:
            eps (float): epsilon in the stable or unstable direction
            compute_stable (bool): whether to compute the stable or unstable manifold

        Keyword Args:
            eps_guess (float): guess for epsilon (if eps is not given)
            neps (int): number of points in the starting configuration
            nint (int): number of intersections

        Returns:
            np.array: array of points on the manifold
        """
        # Check the manifold selection
        if which not in ["stable", "unstable"]:
            raise ValueError("Invalid manifold selection.")
        compute_stable = True if which == "stable" else False

        # Set the right parameters
        if compute_stable:
            rfp, vector, lambda_, goes = self.rfp_s, self.vector_s, self.lambda_s, -1
        else:
            rfp, vector, lambda_, goes = self.rfp_u, self.vector_u, self.lambda_u, 1

        # If the epsilon is not given, find the best one in the linear regime
        eps_guess = kwargs.get("eps_guess", 1e-3)
        if eps is None:
            eps = self.find_epsilon(which, eps_guess)

        # Compute the starting configuration and the manifold
        neps, nint = kwargs.get("neps", 40), kwargs.get("nint", 6)
        RZs = self.start_config(eps, rfp, lambda_, vector, neps, goes)
        logger.info(f"Computing {which} manifold...")
        manifoldpoints = self.integrate(RZs, nintersect=nint, direction=goes)
        orderedmanifoldpoints = np.concatenate(manifoldpoints)  # put points in order: first intersections, second intersections, etc)

        if which == "stable":
            self._stable_trajectory = orderedmanifoldpoints
        else:
            self._unstable_trajectory = orderedmanifoldpoints

        return orderedmanifoldpoints

    @property
    def stable(self):
        if self._stable_trajectory is None:
            logger.warning(
                "Stable manifold not computed. Using the computation method."
            )
            self.compute_manifold('stable', 1e-5)
        return self._stable_trajectory

    @property
    def unstable(self):
        if self._unstable_trajectory is None:
            logger.warning(
                "Unstable manifold not computed. Using the computation method."
            )
            self.compute_manifold('unstable', 1e-5)
        return self._unstable_trajectory

    def compute(self, eps_s=None, eps_u=None, **kwargs):
        """
        Computation of the stable and unstable manifolds.

        Args:
            eps_s (float): epsilon in the stable direction.
            eps_u (float): epsilon in the unstable direction

        Keyword Args:
            eps_guess_s (float): guess for epsilon in the stable direction (if eps_s is not given)
            eps_guess_u (float): guess for epsilon in the unstable direction (if eps_u is not given)
            neps_s (int): number of points in the starting configuration for the stable part
            neps_u (int): number of points in the starting configuration for the unstable part
            nint_s (int): number of evolutions of the initial segments for the stable part
            nint_u (int): number of evolutions of the initial segments for the unstable part

        Returns:
            tuple: A tuple containing two np.array of points, the first one for the stable manifold and the second one for the unstable manifold.
        """
        # Extract the keyword arguments
        kwargs_s = {
            key[:-2]: kwargs.pop(key)
            for key in ["eps_guess_s", "neps_s", "nint_s"]
            if key in kwargs
        }
        kwargs_u = {
            key[:-2]: kwargs.pop(key)
            for key in ["eps_guess_u", "neps_u", "nint_u"]
            if key in kwargs
        }
        if kwargs:
            logger.warning(f"Unused keyword arguments: {kwargs}")

        if eps_s is None and not self.clinics.is_empty:
            eps_s = self.clinics.stable_epsilons[0]
        if eps_u is None and not self.clinics.is_empty:
            eps_u = self.clinics.unstable_epsilons[0]


        # Compute the manifolds
        self.compute_manifold("stable", eps_s, **kwargs_s)
        self.compute_manifold("unstable", eps_u, **kwargs_u)

        return self._stable_trajectory, self._unstable_trajectory

    def get_lobe_boundary(self, lobe_number, neps=40, which_section: int= 1):
        """
        Compute the boundary of the lobe with the given number (counted from the unstable manifold).

        The Manifold class must have a computed at least one clinic, preferrably all of them.
        The boundary of the lobe is computed by mapping the part of the fundamental section of the unstable
        manifold between two clinic trajecories lobe_number times, and
        mapping the section of the stable manifold between two clinictrajectories in the reverse direction
        (clinic_points-lobe_number) times.

        which_section determines which of the two sections of the fundamental domain to use.
        """
        if len(self.clinics)<2:
            raise ValueError("Need at least two clinics to compute the lobe boundary.")
        clinic1 = self.clinics[which_section-1]
        clinic2 = self.clinics[which_section] if which_section < len(self.clinics) else self.clinics[0]

        total_number_of_points = self.clinics.total_number_of_points
        total_epsilons = len(self.clinics)+1

        unstable_fundamental_epsilons = self.clinics.unstable_epsilons
        unstable_start = unstable_fundamental_epsilons[which_section-1]
        unstable_end = unstable_fundamental_epsilons[which_section]
        unstable_epsilons = np.logspace(
            np.log(unstable_start) / np.log(self.lambda_u),
            np.log(unstable_end) / np.log(self.lambda_u),
            neps,
            base=self.lambda_u,
        )
        unstable_vectorsteps = np.outer(unstable_epsilons, self.vector_u)
        unstable_points = self.rfp_u + unstable_vectorsteps
        unstable_part_of_lobe = self.integrate(unstable_points, lobe_number, 1)[-1, :, :]


        # compute well-spaced epsilons for the stable manifold
        stable_fundamental_epsilons = self.clinics.stable_epsilons
        stable_start = stable_fundamental_epsilons[total_epsilons-which_section-1]
        stable_end = stable_fundamental_epsilons[total_epsilons-which_section]
        stable_epsilons = np.logspace(
            np.log(stable_start) / np.log(self.lambda_s),
            np.log(stable_end) / np.log(self.lambda_s),
            neps,
            base=self.lambda_s,
        )
        stable_vectorsteps = np.outer(stable_epsilons, self.vector_s)
        stable_points = self.rfp_s + stable_vectorsteps

        stable_part_of_lobe = self.integrate(stable_points, total_number_of_points - lobe_number-1, -1)[-1, :, :]
        return stable_part_of_lobe, unstable_part_of_lobe

    def plot_lobe_boundary(self, lobe_number, neps=40, which_section: int= 1, **kwargs):
        """
        Plot the boundary of the lobe with the given number (counted from the unstable manifold).

        The Manifold class must have a computed at least one clinic, preferrably all of them.
        The boundary of the lobe is computed by mapping the part of the fundamental section of the unstable
        manifold between two clinic trajecories lobe_number times, and
        mapping the section of the stable manifold between two clinictrajectories in the reverse direction
        (clinic_points-lobe_number) times.
        """
        stable_part_of_lobe, unstable_part_of_lobe = self.get_lobe_boundary(lobe_number, neps, which_section)
        fig, ax, kwargs = create_canvas(**kwargs)

        boundary_points = np.concatenate([unstable_part_of_lobe, stable_part_of_lobe])  # put points in order including clinic points. Nonething adds axis.

        ax.plot(boundary_points[:,0], boundary_points[:,1], **kwargs)
        return fig, ax

    def plot_filled_lobe(self, lobe_number, neps=40, which_section: int= 1, **kwargs):
        """
        Plot the filled lobe with the given number (counted from the unstable manifold).

        The Manifold class must have a computed at least one clinic, preferrably all of them.
        The boundary of the lobe is computed by mapping the part of the fundamental section of the unstable
        manifold between two clinic trajecories lobe_number times, and
        mapping the section of the stable manifold between two clinictrajectories in the reverse direction
        (clinic_points-lobe_number) times.
        """
        stable_part_of_lobe, unstable_part_of_lobe = self.get_lobe_boundary(lobe_number, neps, which_section)
        boundary_points = np.concatenate([unstable_part_of_lobe, stable_part_of_lobe])
        fig, ax, kwargs = create_canvas(**kwargs)
        # Set default value for lw if not provided
        if 'lw' not in kwargs:
            kwargs['lw'] = 0

        ax.fill(boundary_points[:,0], boundary_points[:,1], **kwargs)
        return fig, ax





    def plot(self, which="both", stepsize_limit=None, **kwargs):
        """
        Plot the stable and/or unstable manifolds.

        kwargs:
        which (str): which manifold to plot. Can be 'stable', 'unstable' or 'both'.
        stepsize_limit =

        Other kwargs are givent to the plot.

        Specific extra plots:
        *rm_points* (int): remove the last *rm_points* points of the manifold.

        """
        fig, ax, kwargs = create_canvas(**kwargs)

        labels = kwargs.pop("labels", ["stable manifold", "unstable manifold"])

        colors = kwargs.pop("colors", ["xkcd:royal blue", "xkcd:magenta"])
        markersize = kwargs.pop("markersize", 2)
        fmt = kwargs.pop("fmt", "-o")
        rm_points = kwargs.pop("rm_points", 0)
        final_index = -rm_points - 1

        for i, dir in enumerate(["stable", "unstable"]):
            if dir == which or which == "both":
                points = self.stable if dir == "stable" else self.unstable
                if stepsize_limit is not None:
                    points = clean_bigsteps(points, threshold=stepsize_limit)
                ax.plot(
                    points[:,0][:final_index],
                    points[:,1][:final_index],
                    fmt,
                    label=labels[i],
                    #label=f"{dir} manifold",
                    color=colors[i],
                    markersize=markersize,
                    **kwargs,
                )

        return fig, ax

    def plot_manifold_copies(self, which='both', stepsize_limit=None, **kwargs):
        """
        plot the images of the manifolds as they appear arount the other islands
        of the chain, using the periodicity of the fixed points.
        """
        fig, ax, kwargs = create_canvas(**kwargs)

        colors = kwargs.pop("colors", ["xkcd:royal blue", "xkcd:magenta"])
        markersize = kwargs.pop("markersize", 2)
        fmt = kwargs.pop("fmt", "-o")
        rm_points = kwargs.pop("rm_points", 0)
        final_index = -rm_points - 1

        number_of_copies = self.fixedpoint_1.m - 1
        for i, dir in enumerate(["stable", "unstable"]):
            if dir == which or which == "both":
                points = self.stable if dir == "stable" else self.unstable
                points = points[:final_index]
                for _ in range(number_of_copies):
                    points = self._map.f_many(1, points)
                    if stepsize_limit is not None:
                        points = clean_bigsteps(points, threshold=stepsize_limit)
                    ax.plot(
                        points[:,0],
                        points[:,1],
                        fmt,
                        label=f"{dir} manifold",
                        color=colors[i],
                        markersize=markersize,
                        **kwargs,
                    )

    ### Homo/Hetero-clinic methods

    def find_N(self, eps_s: float, eps_u: float):
        """
        Find the number of times the map needs to be applied for the stable and unstable points to cross.

        This method evolves the initial stable :math:`x_s = x^\\star + \\varepsilon_s\\textbf{e}_s` and unstable :math:`x_u = x^\\star + \\varepsilon_u\\textbf{e}_u` points until they cross. They are alternatively evolved once and when the initial direction is reversed, the number of iterations is returned.

        Args:
            eps_s (float, optional): Initial :math:`\\varepsilon_s` along the stable manifold direction. Defaults to 1e-3.
            eps_u (float, optional): Initial :math:`\\varepsilon_u` along the unstable manifold direction. Defaults to 1e-3.

        Returns:
            tuple: A tuple containing two integers:
                - n_s (int): Number of iterations for the stable manifold.
                - n_u (int): Number of iterations for the unstable manifold.
        """
        r_s = self.rfp_s + eps_s * self.vector_s
        r_u = self.rfp_u + eps_u * self.vector_u

        first_dir = r_u - r_s
        last_norm = np.linalg.norm(first_dir)

        n_s, n_u = 0, 0
        success, stable_evol = False, True
        while not success:
            if stable_evol:
                r_s = self._map.f(-1 * self.fixedpoint_1.m, r_s)
                n_s += 1
            else:
                r_u = self._map.f(+1 * self.fixedpoint_1.m, r_u)
                n_u += 1
            stable_evol = not stable_evol

            norm = np.linalg.norm(r_u - r_s)
            # logger.debug(f"{np.dot(first_dir, r_u - r_s)} / {last_norm} / {norm}")
            if np.sign(np.dot(first_dir, r_u - r_s)) < 0:  # and last_norm < norm:
                success = True
            last_norm = norm

        # if not success:
        #     raise ValueError("Could not find N")
        return n_s, n_u

    def find_clinic_single(self, guess_eps_s, guess_eps_u, n_s=None, n_u=None, reset_clinics=False, nretry=1, ERR=1e-3, **kwargs):
        """
        Search a homo/hetero-clinic point.

        This function attempts to find the intersection point of the stable and unstable manifolds by iteratively adjusting the provided epsilon guesses using scipy root-finding algorithm.

        Args:
            guess_eps_s (float): Initial guess for the stable manifold epsilon.
            guess_eps_u (float): Initial guess for the unstable manifold epsilon.
        *kwargs*: Additional keyword arguments.
            - n_s (int): Number of times the map needs to be applied for the stable manifold.
            - n_u (int): Number of times the map needs to be applied for the unstable manifold.
            - reset_clinics: replace all clinics and make this clinic the fundamental segment.
            - nretry: retry by jittering the epsilon guesses n times
            - ERR (float): Error tolerance for verifying the linear regime (default: 1e-3).
        **kwargs
            - root_args (dict): Arguments to pass to the root-finding function.
                suggested: {'jac':True/False} integrated jacobian or FD for step
                {'options':{'factor':1e-3}} takes smaller steps when jac is ill.

        Returns:
            tuple: A tuple containing the found epsilon values for the stable and unstable manifolds (eps_s, eps_u).
        """
        # Set the number of times the map needs to be applied (times the poloidal mode m)
        if n_s is None or n_u is None:
            n_s, n_u = self.find_N(guess_eps_s, guess_eps_u)

        # find the clinic, retry with a different guess if it fails
        this_eps_s = guess_eps_s
        this_eps_u = guess_eps_u
        newclinic = None
        for _ in range(nretry):
            try:
                newclinic = Clinic.from_guess(self, this_eps_s, this_eps_u, n_s, n_u, ERR=ERR, **kwargs)
                break
            except Exception as e:
                logger.warning(f"Failed to find clinic: {e}")
                stable_jitter = 1 + np.random.uniform(2*self.lambda_s, 1/(2*self.lambda_s))
                unstable_jitter = 1 + np.random.uniform(2/(self.lambda_u), self.lambda_u/2)
                this_eps_s = guess_eps_s * stable_jitter
                this_eps_u = guess_eps_u * unstable_jitter
                logger.warning(f"Failed to find clinic: {e}, re-trying with eps_s={this_eps_s}, eps_u={this_eps_u}")

        # if the clinic is not found, raise an error
        if newclinic is None:
            raise ValueError(f"Failed to find clinic after retrying {nretry} times.")

        # Recording the homo/hetero-clinic point
        if reset_clinics:
            self.clinics.reset()

        self.clinics.record_clinic(newclinic)

        return None
    
    def find_other_clinic_test(self, shift_in_stable:float, shift_in_unstable:float=None, nretry =1, **kwargs):
        """
        bla
        """
        if self.clinics.is_empty:
            raise ValueError('Need to have one clinic first before finding others')
        clinicnum = np.copy(self.clinics.size)
        total_number_of_points = self.clinics.total_number_of_points - 1
        n_s = total_number_of_points//2
        n_u = total_number_of_points - n_s

        stable_segment = self.clinics.stable_segment
        eps_s = stable_segment[0] + np.sqrt(shift_in_stable) * (stable_segment[1] - stable_segment[0])

        if shift_in_unstable is None:
            shift_in_unstable = 1-shift_in_stable
        unstable_segment = self.clinics.unstable_segment
        eps_u = unstable_segment[0] + np.sqrt(shift_in_unstable) * (unstable_segment[1] - unstable_segment[0])
        
        newclinic = Clinic.with_deflation(self, eps_s, eps_u, n_s, n_u, **kwargs)
        self.clinics.record_clinic(newclinic)

    def find_other_clinic(self, shift_in_stable:float, shift_in_unstable:float=None, nretry =1, **kwargs):
        """
        Find another clinic trajectory if the Manifold already has a fundamental segment.
        Starts the clinic trajectory finding with one less total step by shifting the
        start points
        Args:
        shift_in_stable: point in the stable fundamental segment to start the search from, 0 is from the same point
         as the first trajectory, 1 is shifted to the end.
        shift_in_unstable (optional): point in the unstable fundamental segment to start the search from,
        if not given same as shift_in_stable.
        """
        if self.clinics.is_empty:
            raise ValueError('Need to have one clinic first before finding others')
        clinicnum = np.copy(self.clinics.size)
        total_number_of_points = self.clinics.total_number_of_points - 1
        n_s = total_number_of_points//2
        n_u = total_number_of_points - n_s

        stable_segment = self.clinics.stable_segment
        eps_s = stable_segment[0] + np.sqrt(shift_in_stable) * (stable_segment[1] - stable_segment[0])

        if shift_in_unstable is None:
            shift_in_unstable = 1-shift_in_stable
        unstable_segment = self.clinics.unstable_segment
        eps_u = unstable_segment[0] + np.sqrt(shift_in_unstable) * (unstable_segment[1] - unstable_segment[0])

        this_eps_s = eps_s
        this_eps_u = eps_u
        newclinic = None
        for _ in range(nretry):
            try:
                newclinic = Clinic.from_guess(self, this_eps_s, this_eps_u, n_s, n_u, **kwargs)
                self.clinics.record_clinic(newclinic)
                if self.clinics.size == clinicnum + 1:
                    logger.info(f"clinic recorded after {_ +1} attempts")
                    break
                else: 
                    raise ValueError(f"Failed to find new clinic after {_ +1} attempts.") #triggers exception below
            except Exception as e:
                logger.warning(f"Failed to find other clinic: {e}")
                newshift = np.random.random()
                this_eps_s = stable_segment[0] + newshift * (stable_segment[1] - stable_segment[0])
                this_eps_u = unstable_segment[0] + newshift * (unstable_segment[1] - unstable_segment[0])
                logger.warning(f"Failed to find clinic: {e}, re-trying with eps_s={this_eps_s}, eps_u={this_eps_u}")

        # if the clinic is not found, raise an error
        if newclinic is None:
            raise ValueError(f"Failed to find clinic after retrying {nretry} times.")
        if self.clinics.size != clinicnum + 1:
            raise ValueError(f"Failed to find new clinic after retrying {nretry} times.")

    def find_clinics(
        self, first_guess_eps_s, first_guess_eps_u, n_points=None, reset_clinics=True, **kwargs
    ):
        """
        Args:

        """
        shift = kwargs.pop("shift", 0)

        if n_points is None:
            n_points = 2

        # Reset the clinic search
        if reset_clinics == True:
            self.clinics.reset()

        logger.info(
            f"Search {1}/{n_points} - initial guess for epsilon pair (eps_s, eps_u): {first_guess_eps_s, first_guess_eps_u}"
        )
        self.find_clinic_single(first_guess_eps_s, first_guess_eps_u, **kwargs)
        bounds = self.clinics.fundamental_segments

        stable_multiplicators = np.power(
            self.lambda_s, np.arange(n_points)[1:] / n_points
        )
        unstable_multiplicators = np.power(
            self.lambda_u, np.arange(n_points)[1:] / n_points
        )

        for i, (mult_s, mult_u) in enumerate(
            zip(stable_multiplicators, unstable_multiplicators)
        ):
            guess_i = [
                bounds["stable"][1] * mult_s,
                bounds["unstable"][0] * mult_u,
            ]

            logger.info(
                f"Search {i+2}/{n_points} - initial guess for epsilon pair (eps_s, eps_u): {guess_i}"
            )

            # Retrieve the
            n_s, n_u = self.clinics.nint_pair
            n_s += shift
            n_u += shift - 1

            self.find_clinic_single(*guess_i, n_s=n_s, n_u=n_u, **kwargs)

        # if len(self.onworking["clinics"]) != n_points:
        #     logger.warning("Number of clinic points is not the expected one.")

    def plot_clinics(self, **kwargs):
        """
        Plot the clinic trajectories.
        """
        markers = kwargs.get(
            "markers", ["o", "s", "*", "P", "p", "X", "D", "d", "^", "v", "<", ">"]
        )
        color = kwargs.pop("color", "royalblue")
        edgecolor = kwargs.pop("edgecolor", "cyan")

        fig, ax, kwargs = create_canvas(**kwargs)

        for i, clinic in enumerate(self.clinics):
            trajectory = clinic.trajectory
            ax.scatter(
                *trajectory.T,
                marker=markers[i],
                color=color,
                edgecolor=edgecolor,
                zorder=10,
                **kwargs
            )

        return fig, ax

    ### Calculating Island/Turnstile Flux

    @property
    def turnstile_areas(self):
        if self._areas is None:
            self.compute_turnstile_areas()
        return self._areas

    def compute_turnstile_areas(self, **kwargs):
        if isinstance(self._map, maps.CylindricalBfieldSection):
            return self._turnstile_areas_cylbfieldsection(**kwargs)
        else:
            raise NotImplementedError(
                "Turnstile area computation only implemented for CylindricalBfieldSection for now."
            )

    def _turnstile_areas_cylbfieldsection(self, n_joining=100):
        """
        Compute the turnstile area by integrating the vector potential along the trajectory
        of the clinics defined in the manifold.
        """
        if len(self.clinics) < 2:
            raise ValueError("Need at least two clinics to compute the turnstile area.")

        #create the integration trajectories for the clinics. each has to have the same number of elements, and the endpoints will be used for closing integrals.
        # Since the first clinic defines the fundamental segment, and is one mapping longer, the first integral skips the last point of this trajectory,
        #and returns over the next clinic. The last clinic is the same as the first, but the first point is skipped.

        integration_trajectories = []
        for num in range(len(self.clinics) + 1):
            if num == 0:
                integration_trajectories.append(self.clinics[num].trajectory[:-1])
            elif num == len(self.clinics):
                integration_trajectories.append(self.clinics[0].trajectory[1:])
            else:
                integration_trajectories.append(self.clinics[num].trajectory)

        # compute the integrals of the above-defined sections
        #lagrangians = [self._compute_lagrangian_in_sections(traj) for traj in integration_trajectories]

        turnstile_areas = []
        for turnstilenum in range(len(self.clinics)):
            traj1 = integration_trajectories[turnstilenum]
            traj2 = integration_trajectories[turnstilenum+1]
            int1 = np.sum(self._compute_lagrangian_in_sections(traj1))
            int2 = np.sum(self._compute_lagrangian_in_sections(traj2)) # this integral shold be in opposite direction, it will be subtracted.

            close_1_RZs = np.linspace(traj1[-1], traj2[-1], n_joining, endpoint=True)
            closing_integral_1 = self._AdL_integral_points(close_1_RZs, is_closed=False)
            logger.debug(f"Closing integral of turnstile {turnstilenum} stable end: {closing_integral_1}")

            close_2_RZs = np.linspace(traj2[0], traj1[0], n_joining, endpoint=True)
            closing_integral_2 = self._AdL_integral_points(close_2_RZs, is_closed=False)
            logger.debug(f"Closing integral of turnstile {turnstilenum} unstable end: {closing_integral_2}")

            turnstile_areas.append(int1 + closing_integral_1 - int2 + closing_integral_2)
            logger.info(f"Turnstile area {turnstilenum} computed: {turnstile_areas[-1]}")

        self._areas = np.array(turnstile_areas)
        return self._areas

    def _compute_lagrangian_in_sections(self, trajectory):
        """
        Compute the lagrangian value for a clinic trajectory, one map at a time.
        Skip the last point because that maps out of the fundamental segment.
        """
        logger.debug(f"starting Lagrangian integration for trajectory of length {len(trajectory)}")
        clinic_points = trajectory
        lagrangians = []
        for point in clinic_points[:-1]:
            lagrangians.append(self._map.lagrangian(point, self.fixedpoint_1.m))
        return np.array(lagrangians)
    
    def _AdL_integral_points(self, gamma, dl=None, is_closed=False):
        """
        approximate the integral of A.dl along a set of of points using a midpoint rule.
        the path is given by gamma, and dl is the distance between two points.
        If is_closed is True, the last point is connected to the first point.

        gamma is a path in the section, an Nx2 array.
        """
        if is_closed:
            gamma = np.vstack((gamma, gamma[0]))

        if dl is None:
            dl = np.diff(gamma, axis=0)

        midpoints = (gamma[:-1] + gamma[1:]) / 2
        midpointsRphiz = np.vstack(
            (
                midpoints[:, 0],
                self._map.phi0 * np.ones(midpoints.shape[0]),
                midpoints[:, 1],
            )
        ).T  # right form to evaluate vector potential
        A = np.array([self._map._mf.A(r)[0::2] for r in midpointsRphiz])  # Evaluate the vector potential
        return np.einsum("ij,ij->i", A, np.ones((A.shape[0], 1)) * dl).sum() # sum the dot product of A and dl



    def _turnstile_areas_cylbfieldsection_old(self, n_joining=100):
        """
        Compute the turnstile area by integrating the vector potential along the trajectory of the homo/hetero-clinics points.
        """
        lagrangian_values = np.NaN * np.zeros(len(self.clinics) + 1)

        # Could be put in the clinic trajectory directly. Open question.
        # Calculation of the lagrangian value (from the unstable to the stable fundamental segment)
        for i, clinic in enumerate([*self.clinics, self.clinics[0]]):
            x_s_0, x_u_0 = (
                clinic.trajectory[-1 if i != 0 else -2, :],
                clinic.trajectory[0 if i != len(self.clinics) else 1, :],
            )

            n_bwd = clinic.nint_s if i != 0 else clinic.nint_s - 1
            n_fwd = clinic.nint_u if i != len(self.clinics) else clinic.nint_u - 1

            # x_t_s = self._map.f(-n_bwd * self.fixedpoint_1.m, x_s_0)
            # x_t_u = self._map.f(n_fwd * self.fixedpoint_1.m, x_u_0)

            intA_s = self._map.lagrangian(x_s_0, -n_bwd * self.fixedpoint_1.m)
            intA_u = self._map.lagrangian(x_u_0, n_fwd * self.fixedpoint_1.m)

            lagrangian_values[i] = intA_u - intA_s

            logger.info(
                f"Lagrangian value obtained ({(lagrangian_values[i]):.3e}) for homo/hetero-clinic trajectory (eps_s, eps_u) : {clinic.eps_s, clinic.eps_u}"
            )

        # Computation of the turnstile area
        areas = np.empty(len(self.clinics))
        shifted_indices = [
            i for i in np.roll(np.arange(len(self.clinics), dtype=int), -1)
        ]

        # Loop on the L values : L_h current clinic point, L_m next clinic point (in term of >_u ordering)
        for i, shifted_i in enumerate(shifted_indices):
            # Area is the difference in the
            areas[i] = lagrangian_values[i] - lagrangian_values[i + 1]

            # Closure by joining integrals
            traj_h = self.clinics[i].trajectory
            traj_m = self.clinics[shifted_i].trajectory

            # Get the correct points to join
            r_h_u, r_m_u = traj_h[0, :], traj_m[0, :]
            r_h_s, r_m_s = traj_h[-1, :], traj_m[-1, :]
            if i == 0:
                r_h_s, r_m_s = traj_h[-2, :], traj_m[-1, :]
            elif i == len(shifted_indices) - 1:
                r_h_u, r_m_u = traj_h[0, :], traj_m[1, :]

            for j, (rA, rB) in enumerate(zip([r_m_u, r_h_s], [r_h_u, r_m_s])):
                # Create a segment between r1 and r2
                gamma, dl = np.linspace(rA, rB, n_joining, retstep=True)

                # Evaluate A at the middle point between (x_i, x_{i+1})
                mid_gamma = (gamma + dl / 2)[:-1]
                mid_gamma = np.vstack(
                    (
                        mid_gamma[:, 0],
                        self._map.phi0 * np.ones(mid_gamma.shape[0]),
                        mid_gamma[:, 1],
                    )
                ).T
                mid_A = np.array([self._map._mf.A(r)[0::2] for r in mid_gamma])

                # Discretize the A.dl integral and sum it
                closing_integral = np.einsum(
                    "ij,ij->i", mid_A, np.ones((mid_A.shape[0], 1)) * dl
                ).sum()
                logger.debug(f"Closing integral {i+1}, {j+1}/2 : {closing_integral}")
                areas[i] += closing_integral

        self._areas = areas
        self._lagrangian_values = lagrangian_values

        return areas

    ### Integration methods

    def integrate(self, x_many, nintersect, direction=1):
        """
        Integrate a set of points x_many for nintersect times in the direction specified.
        Robust to integration failures and has fixed return shape.

        Returns an array of shape (nintersect, len(x_many), _map.dimension).
        """

        x_many = np.atleast_2d(x_many)
        t = self.fixedpoint_1.m * direction
        res = []
        res.append(x_many)
        for _ in range(nintersect):
            x_many = self._map.f_many(t, x_many)
            res.append(x_many)
        return np.array(res)
    
    ### Save and loading

    def save(self, path):
        """
        save the manifold object to a .pkl file
        """

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """load the manifold object from a .pkl file"""

        with open(path, "rb") as f:
            return pickle.load(f)
    
    def save_mf_quasr(self, path):
        """
        save the manifold object for a Stellerator field in a .pkl file
        """
    
        payload = {
         'stable': np.asarray(getattr(self, 'stable', [])),
         'unstable': np.asarray(getattr(self, 'unstable', [])),
         'clinics': [np.asarray(getattr(c, 'trajectory', [])) for c in getattr(self, 'clinics', [])],
         'fp0_coords': np.asarray(getattr(self, 'fp0', getattr(self, 'fixed_point0', None)) and getattr(self.fp0, 'coords', None) or []),
         'fp1_coords': np.asarray(getattr(self, 'fp1', getattr(self, 'fixed_point1', None)) and getattr(self.fp1, 'coords', None) or []),
         'meta': {k: getattr(self, k) for k in ('nint_s', 'nint_u', 'eps_s', 'eps_u') if hasattr(self, k)}
        }
        with open(path, 'wb') as f:
         pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)  

    def load_mf_quasr(self, path):
        """
        load the payload saved by save_mf_quasr into the current instance.
        """

        with open(path, "rb") as f:
            payload = pickle.load(f)

        stable = payload.get("stable", None)
        unstable = payload.get("unstable", None)
        self._stable_trajectory = np.asarray(stable) if stable is not None and len(stable) else None
        self._unstable_trajectory = np.asarray(unstable) if unstable is not None and len(unstable) else None

        clinics = payload.get("clinics", [])

        self._clinics_payload = [np.asarray(c) for c in clinics]

        fp0 = payload.get("fp0_coords", None)
        fp1 = payload.get("fp1_coords", None)
        self.fp0_coords = np.asarray(fp0) if fp0 is not None and len(fp0) else None
        self.fp1_coords = np.asarray(fp1) if fp1 is not None and len(fp1) else None

        self._mf_quasr_meta = payload.get("meta", {})

        self._areas = None

        return payload


