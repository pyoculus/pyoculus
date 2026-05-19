from .integrated_map import IntegratedMap
from ..fields.cylindrical_bfield import CylindricalBfield
import pyoculus.solvers as solvers
import numpy as np

# Try to import numba and define a decorator
try:
    import numba
    njit = numba.njit
except ImportError:
    # Define a no-op decorator if numba is not available
    def njit(func):
        return func
    
@njit
def _jitted_f_RZ_tangent(y, Bfield, dBdRphiZ):
    """
    Returns the right-hand side (RHS) of the ODE with the calculation of the differential evolution. For an explanation, one could look at [S.R. Hudson](https://w3.pppl.gov/~shudson/Oculus/ga00aa.pdf).

    Args:
        y (array): The current :math:`R, Z` and :math:`d\\textbf{RZ}/dRZ` matrix. y = [dR/dphi, dZ/dphi, dR/dR_i, dZ/dR_i, dR/dZ_i, dZ/dZ_i] where i stands for the initial point in the phi0 plane (not the axis).
    Returns:
        array: 
    """
    dRZ = y[-4:].reshape(2, 2).T
    # dRZ = [[y[2], y[4]], [y[3], y[5]]]

    # R, Z evolution as in _rhs_RZ
    dRdphi = Bfield[0] / Bfield[1]
    dZdphi = Bfield[2] / Bfield[1]

    # Matrix of the derivatives of (B^R/B^phi, B^Z/B^phi) with respect to (R, Z)
    M00 = (
        dBdRphiZ[0, 0] / Bfield[1] - Bfield[0] / Bfield[1] ** 2 * dBdRphiZ[1, 0]
    )
    M01 = (
        dBdRphiZ[0, 2] / Bfield[1] - Bfield[0] / Bfield[1] ** 2 * dBdRphiZ[1, 2]
    )
    M10 = (
        dBdRphiZ[2, 0] / Bfield[1] - Bfield[2] / Bfield[1] ** 2 * dBdRphiZ[1, 0]
    )
    M11 = (
        dBdRphiZ[2, 2] / Bfield[1] - Bfield[2] / Bfield[1] ** 2 * dBdRphiZ[1, 2]
    )
    M = np.array([[M00, M01], [M10, M11]], dtype=np.float64)

    dRZ = M @ dRZ

    return np.array([dRdphi, dZdphi, dRZ[0, 0], dRZ[1, 0], dRZ[0, 1], dRZ[1, 1]])

class CylindricalBfieldSection(IntegratedMap):
    """
    Map given by following the a magnetic field in cylindrical system :math:`(R, \\phi, Z)` and recording the intersections with symmetry planes :math:`\\phi = \\phi_0, \\phi_0 + T, ...`.

    Attributes:
        mf (CylindricalBfield): The magnetic field to follow.
        phi0 (float): The cylindrical angle from which to start the field line tracing.
        R0 (float): The major radius of the magnetic axis in the :math:`\\phi_0` plane.
        Z0 (float): The vertical position of the magnetic axis in the :math:`\\phi_0` plane.
    """

    def __init__(self, cylindricalbfield : CylindricalBfield, phi0=0.0, R0=None, Z0=None, domain=None, finderargs=dict(), cache_size=None, **kwargs):
        """
        Initializes the CylindricalBfieldSection object.

        This constructor calls the CylindricalBfield and IntegrationMap constructor. If `R0` or `Z0` is not provided, the magnetic axis will be found using a FixedPoint solver.

        Args:
            cylindricalbfield (CylindricalBfield): The magnetic field to follow.
            phi0 (float): The cylindrical angle from which to start the field line tracing.
            R0 (float, optional): The major radius of the magnetic axis in the :math:`\\phi_0` plane. If None, the magnetic axis will be found.
            Z0 (float, optional): The vertical position of the magnetic axis in the :math:`\\phi_0` plane. If None, the magnetic axis will be found.
            domain (list of tuples, optional): The domain of the map. Each tuple should contain the lower and upper bounds for each dimension. If None, the domain is assumed to be :math:`(0, \\infty)` for the first dimension and :math:`(-\\infty, \\infty)` for the second dimension.
            finderargs (dict, optional): Additional arguments to pass to the FixedPoint solver.
            **kwargs: Additional parameters to be passed to the integrator.
        """
        if not isinstance(cylindricalbfield, CylindricalBfield):
            raise ValueError("The magnetic field should be an instance of CylindricalBfield.")
        else:
            self._mf = cylindricalbfield
        
        if domain is None:
            domain = [(0, np.inf), (-np.inf, np.inf)]

        super().__init__(dim=2, domain=domain, dzeta=2*np.pi/cylindricalbfield.Nfp, ode = self._ode_rhs_tangent, **kwargs)

        self.phi0 = phi0

        # Allow to cache the result of the field line tracing
        self.cache = Cache(cache_size)

        # Find the magnetic axis if not provided
        self.R0 = R0
        self.Z0 = Z0
        self._axis = None

        if R0 is None or Z0 is None:
            self.find_axis(**finderargs)

    @classmethod
    def without_axis(cls, cylindricalbfield : CylindricalBfield, guess=None, phi0=0.0, domain=None, finderargs=dict(), **kwargs):
        finderargs["guess"] = guess
        return cls(cylindricalbfield, phi0, None, None, domain, finderargs, **kwargs)

    def find_axis(self, guess=None, **kwargs):
        """
        Finds the magnetic axis of a magnetic field using a FixedPoint solver.

        This method attempts to locate the magnetic axis by solving a fixed-point problem, where the magnetic axis is a point where the magnetic field lines close on themselves after exactly one mapping.

        Args:
            guess (tuple, optional): An initial guess for the coordinates of the magnetic axis. If not provided, a default guess is used.
            **kwargs: Arbitrary keyword arguments passed directly to the FixedPoint solver's `find` method. This can be used to specify solver options such as tolerance levels, maximum iterations, etc.
        """
        axisfinder = solvers.FixedPoint(self)

        if guess is None and self.R0 is not None and self.Z0 is not None:         
            guess = [self.R0, self.Z0]

        axisfinder.find(1, guess, **kwargs)
        if axisfinder.successful:
            self.R0, self.Z0 = axisfinder.coords[0]
            self._axis = axisfinder.coords[0]
        else:
            raise ValueError("The magnetic axis could not be found.")
    
    @property 
    def axis(self):
        """
        Returns the coordinates of the magnetic axis.
        """
        if self._axis is None:
            axis = solvers.FixedPoint(self)
            axis.find(1, [self.R0, self.Z0])
            self._axis = axis
            if not np.allclose(self._axis.coords[0], [self.R0, self.Z0], atol=1e-4):
                raise ValueError("The map axis coordinates are not close to the actual axis.")
        return self._axis

    ## BaseMap methods

    def f(self, t, y0):
        """
        Trace the field line for a number of periods.

        Traces one field period at a time and caches the results if the number of periods is integer. 
        Otherwise, it traces the field line for the given number of fractional periods in one go. 
        """
        if self._integrator.rhs is not self._rhs_RZ:
            self._integrator.set_rhs(self._rhs_RZ)

        if type(t) is not int:
            cache_res = self.cache.retrieve(y0, "f_noninteger")
            if cache_res is not None and t in cache_res:
                return cache_res[t]
            else:
                ic = [*y0]
                res = self._integrate(t, ic)
                self.cache.save(y0, 'f_noninteger', t, res)
                return res
        
        # Check if the result is in the cache and use it if possible
        cache_res = self.cache.retrieve(y0, "f")
        if cache_res is None:
            self.cache.save(y0, 'f', 0, y0)
            cache_res = self.cache.retrieve(y0, "f")
        
        if t in cache_res:  # this point has been computed beforeA
            return cache_res[t]
        else:
            if t > 0:  # forward integration
                tstart = max(cache_res.keys())
                for t0 in range(tstart, t): # forward integration
                    cache_res[t0 + 1] = self._integrate(1, cache_res[t0])
            if t < 0:  # backward integration
                tstart = min(cache_res.keys())
                for t0 in range(tstart, t, -1): # backward integration 
                    cache_res[t0 - 1] = self._integrate(-1, cache_res[t0])
        
        return np.copy(cache_res[t])

    def df(self, t, y0):
        """
        Compute the Jacobian of the field line map for a number of periods.
        """
        if self._integrator.rhs is not self._rhs_RZ_tangent:
            self._integrator.set_rhs(self._rhs_RZ_tangent)
        
        if type(t) is not int:
            cache_res = self.cache.retrieve(y0, "df_noninteger")
            if cache_res is not None and t in cache_res:
                return cache_res[t]
            else: 
                ic = [*y0, 1., 0., 0., 1.]
                integrand = self._integrate(t, ic)
                f = integrand[:2]
                df = integrand[2:6].reshape(2, 2).T
                self.cache.save(y0, 'f_noninteger', t, f)
                self.cache.save(y0, 'df_noninteger', t, df)
                return df

        # Check if the result is in the cache and use it if possible
        cache_res = self.cache.retrieve(y0, "df")
        if cache_res is None:
            self.cache.save(y0, 'df', 0, [*y0, 1., 0., 0., 1.])  # the cache stores the ode rhs.
            cache_res = self.cache.retrieve(y0, "df")

        if t in cache_res:  # this point has been computed before
            df = cache_res[t][2:6].reshape(2, 2).T
            return np.copy(df)
        else:  # integrate one period at a time till you reach the desired time.
            if t > 0:  # forward integration
                tstart = max(cache_res.keys())
                for t0 in range(tstart, t):  # forward integration
                    cache_res[t0 + 1] = self._integrate(1, cache_res[t0])
            if t < 0:  # backward integration
                tstart = min(cache_res.keys())
                for t0 in range(tstart, t, -1):  # backward integration 
                    cache_res[t0 - 1] = self._integrate(-1, cache_res[t0])
        
        # update the cache of f with the new positions
        positions = {t0: cache_res[t0][:2] for t0 in cache_res}
        pos_cache = self.cache.retrieve(y0, 'f')
        if pos_cache is None:
            self.cache.save(y0, 'f', 0, y0)
            pos_cache = self.cache.retrieve(y0, 'f')
        pos_cache.update(positions)

        df = cache_res[t][2:6].reshape(2, 2).T
        return np.copy(df)
    
    def to_rhotheta(self, y, y0=None):
        """
        Converts the cylindrical coordinates to polar coordinates centered around the magnetic axis.
        """
        #ensure y is a 2D array:
        y = np.array(y)
        if y0 is None:
            y0 = np.array([self.R0, self.Z0])
        rho = np.linalg.norm(y - y0, axis=-1)
        theta = np.arctan2(y[1] - y0[1], y[0] - y0[0])
        return np.array([rho, theta])
    
    def to_RZ(self, rt):
        """
        Converts the polar coordinates centered around the magnetic axis to cylindrical coordinates.
        """
        rt = np.array(rt)
        R = rt[0] * np.cos(rt[1]) + self.R0
        Z = rt[0] * np.sin(rt[1]) + self.Z0
        return np.array([R, Z])
        
    def lagrangian(self, y0, t):
        """
        Set Meiss's Lagrangiat for the magnetic field.
        """
        if self._integrator.rhs is not self._rhs_RZ_A:
            self._integrator.set_rhs(self._rhs_RZ_A)

        if type(t) is not int:
            ic = [*y0, 0.0]
            return self._integrate(t, ic)[2]

        # Check if the result is in the cache and use it if possible
        cache_res = self.cache.retrieve(y0, "lagrangian")
        if cache_res is None:
            self.cache.save(y0, 'lagrangian', 0, [*y0, 0.0])
            cache_res = self.cache.retrieve(y0, "lagrangian")
        
        if t in cache_res:  # this point has been computed before
            return np.copy(cache_res[t][2])
        else:
            if t > 0:  # forward integration
                tstart = max(cache_res.keys())
                for t0 in range(tstart, t):  # forward integration
                    cache_res[t0 + 1] = self._integrate(1, cache_res[t0]) 
            if t < 0:  # backward integration
                tstart = min(cache_res.keys())
                for t0 in range(tstart, t, -1):  # backward integration
                    cache_res[t0 - 1] = self._integrate(-1, cache_res[t0])
        
        return np.copy(cache_res[t][2])

    def winding(self, t, y0, y1=None):
        """
        Calculates the winding number of the field line starting at :math:`y_0` around the one starting at :math:`y_1`. By going into the poloidal coordinates centered around the position of the field line given by :math:`y_0` and integrating the angle change along the trajectory.

        Args:
            t (float): The number of periods to integrate.
            y0 (array): The starting point of the field line.
            y1 (array, optional): Another field line around which winding is to be calculated. If not provided, the magnetic axis is used.
        
        Returns:
            np.ndarray: The radius difference between initial and final point and winding number between the two field lines.
        """
        #set integrator
        if self._integrator.rhs is not self._ode_rhs:
            self._integrator.set_rhs(self._ode_rhs)
        
        # Set the initial position in the phi0 plane in polar coordinates
        if y1 is None:
            y1 = np.array([self.R0, self.Z0])
        theta0 = np.arctan2(y0[1] - y1[1], y0[0] - y1[0])
        
        
        # Check if the result is in the cache and use it if possible
        cache_res = self.cache.retrieve(tuple(y0)+tuple(y1), "winding") # adding tuples lengthens, hased to create dict key
        if cache_res is None:
            self.cache.save(tuple(y0)+tuple(y1), 'winding', 0, [*y0, *y1, theta0])
            cache_res = self.cache.retrieve(tuple(y0)+tuple(y1), "winding")
        
        if t in cache_res:  # this point has been computed before
            output = cache_res[t]
        else:
            if t > 0:  # forward integration
                tstart = max(cache_res.keys())
                for t0 in range(tstart, t):  # forward integration
                    cache_res[t0 + 1] = self._integrate(1, cache_res[t0])
            if t < 0:  # backward integration
                tstart = min(cache_res.keys())
                for t0 in range(tstart, t, -1):  # backward integration
                    cache_res[t0 - 1] = self._integrate(-1, cache_res[t0])
            output = cache_res[t]
        
        theta1 = output[4]
        rho1 = np.sqrt((output[0] - output[2]) ** 2 + (output[1] - output[3]) ** 2)
        rhodtheta = np.array([rho1, theta1 - theta0])
        return rhodtheta

    
    def dwinding(self, t, y0, y1=None):
        """
        Calculates the Jacobian of the winding number 
        """
        #set integrator
        if self._integrator.rhs is not self._ode_rhs_tangent:
            self._integrator.set_rhs(self._ode_rhs_tangent)
        if y1 is None:
            y1 = np.array([self.R0, self.Z0])
        theta0 = np.arctan2(y0[1] - y1[1], y0[0] - y1[0])
        rho0 = np.sqrt((y0[0] - y1[0]) ** 2 + (y0[1] - y1[1]) ** 2)

        # Check if the result is in the cache and use it if possible
        cache_res = self.cache.retrieve(tuple(y0)+tuple(y1), "dwinding")
        if cache_res is None:
            self.cache.save(tuple(y0)+tuple(y1), 'dwinding', 0, [*y0, *y1, theta0, 1., 0., 0., 1.])
            cache_res = self.cache.retrieve(tuple(y0)+tuple(y1), "dwinding")
        
        if t in cache_res:  # this point has been computed before
            output = cache_res[t]
        else:
            if t > 0:
                tstart = max(cache_res.keys())
                for t0 in range(tstart, t):  # forward integration
                    cache_res[t0 + 1] = self._integrate(1, cache_res[t0])
            if t < 0:   
                tstart = min(cache_res.keys())
                for t0 in range(tstart, t, -1):
                    cache_res[t0 - 1] = self._integrate(-1, cache_res[t0])
            output = cache_res[t]
        
        # Retrieve the final position in the phi0 plane in polar coordinates
        theta1 = np.arctan2(output[1] - output[3], output[0] - output[2])
        rho1 = np.sqrt((output[0] - output[2]) ** 2 + (output[1] - output[3]) ** 2)

        # Jacobian of the map in R, Z coordinates
        dG = np.array([
                [output[5], output[7]],
                [output[6], output[8]]
            ], dtype=np.float64)
        
        # Jacobian of the change of coordinates from R, Z to polar at end point : dH = dH(G(R,Z)) 
        deltaRZ = output[:2] - y1
        dH = np.array([
                np.array([deltaRZ[0], deltaRZ[1]], dtype=np.float64) / np.sqrt(deltaRZ[0]**2 + deltaRZ[1]**2),
                np.array([-deltaRZ[1], deltaRZ[0]], dtype=np.float64) / (deltaRZ[0]**2 + deltaRZ[1]**2)
        ], dtype=np.float64)

        # Jacobian of the change of coordinates from R, Z to polar at starting point, dP = dH(R,Z)
        deltaRZ = y0 - y1
        dP = np.array([
            np.array([deltaRZ[0], deltaRZ[1]], dtype=np.float64) / np.sqrt(deltaRZ[0]**2 + deltaRZ[1]**2),
            np.array([-deltaRZ[1], deltaRZ[0]], dtype=np.float64) / (deltaRZ[0]**2 + deltaRZ[1]**2)
        ], dtype=np.float64)

        # Jacobian of the map W = H(G(R,Z)) - H(R,Z)
        dW = dH @ dG - dP
        return dW

    ## Integration methods

    def _integrate(self, t, y0):
        """
        Integrates the ODE for a number of periods.
        """
        dphi = t * self.dzeta
        y = np.array(y0)
        self._integrator.set_initial_value(self.phi0, y)
        return self._integrator.integrate(self.phi0 + dphi)

    def _ode_rhs(self, phi, y, *args):
        """
        Returns the right-hand side (RHS) of ODE.

        Args:
            phi (float): The cylindrical angle in the ODE.
            y (array): The cylindrical coordinates :math:`(R, Z, R_0, Z_0, \theta)` in the ODE.
            *args: Additional parameters for the magnetic field calculation.

        Returns:
            array: The RHS of the ODE.
        """
        R, Z, R0, Z0 = y[:4]
        dRZ = self._rhs_RZ(phi, np.array([R, Z]), *args)
        dRZ0 = self._rhs_RZ(phi, np.array([R0, Z0]), *args)

        # Calculating the change for the angle theta (poloidal angle with origin at the magnetic axis)
        deltaR = R - R0
        deltaZ = Z - Z0
        # dartan2(Z-Z0, R-R0)/d(R-R0) * d(R-R0)/dphi + dartan2(Z-Z0, R-R0)/d(Z-Z0) * d(Z-Z0)/dphi
        dtheta = (deltaR * (dRZ[1] - dRZ0[1]) - deltaZ * (dRZ[0] - dRZ0[0])) / (
            deltaR**2 + deltaZ**2
        )

        return np.array([*dRZ, *dRZ0, dtheta])

    def _rhs_RZ(self, phi, RZ, *args):
        """
        Calculates the right-hand side (RHS) of ODE following the magnetic field.

        Args:
            phi (float): The current cylindrical angle.
            RZ (array): The current R,Z coordinates, RZ = [R, Z].
            *args: Additional parameters for the ODE.

        Returns:
            array: An array containing the derivatives of R and Z with respect to phi, i.e., [dR/dphi, dZ/dphi].
        """
        RphiZ = np.array([RZ[0], phi, RZ[1]])

        Bfield = self._mf.B(RphiZ, *args)

        # R, Z evolution given by following the field
        # dR/dphi = B^R / B^phi and dZ/dphi = B^Z / B^phi
        dRdphi = Bfield[0] / Bfield[1]
        dZdphi = Bfield[2] / Bfield[1]

        return np.array([dRdphi, dZdphi])

    # Tangent ODE RHS

    def _ode_rhs_tangent(self, phi, y, *args):
        """
        Args:
            phi (float): The current cylindrical angle.
            y (array): The current R, Z, R0, Z0, theta, dR1, dZ1, dR2, dZ2.
        """
        R, Z, R0, Z0, theta, *dRZ = y

        dy1 = self._rhs_RZ_tangent(phi, np.array([R, Z, *dRZ]))
        dy2 = self._ode_rhs(phi, np.array([R, Z, R0, Z0, theta]))

        return np.array([*dy2, *dy1[2:]])

    def _rhs_RZ_tangent(self, phi, y, *args):
        """
        Returns the right-hand side (RHS) of the ODE with the calculation of the differential evolution. For an explanation, one could look at [S.R. Hudson](https://w3.pppl.gov/~shudson/Oculus/ga00aa.pdf).

        Args:
            phi (float): The current cylindrical angle :math:`\\phi`.
            y (array): The current :math:`R, Z` and :math:`d\\textbf{RZ}/dRZ` matrix. y = [dR/dphi, dZ/dphi, dR/dR_i, dZ/dR_i, dR/dZ_i, dZ/dZ_i] where i stands for the initial point in the phi0 plane (not the axis).
        Returns:
            array: 
        """
        rphiz = np.array([y[0], phi, y[1]])
        Bfield, dBdRphiZ = self._mf.dBdX(rphiz, *args)
        return _jitted_f_RZ_tangent(y, Bfield, dBdRphiZ)

    # dLangrangian ODE RHS

    def _rhs_RZ_A(self, phi, y, *args):
        """
        Returns RHS of the ODE for the integral of the vector potential along the field line.

        Args:
            phi (float): The current cylindrical angle.
            y (array): The current R, Z, integral of A.
            *args: Additional parameters to calculate the magnetic field.
        """

        # R, Z evolution
        dRdphi, dZdphi = self._rhs_RZ(phi, y[:2], *args)

        # magnetic potential at the current point
        RphiZ = np.array([y[0], phi, y[1]])
        A = self._mf.A(RphiZ, *args)

        # Integral of A, step
        dl = np.array([dRdphi, 1, dZdphi])
        dl = np.array([1, y[0] ** 2, 1]) * dl

        dintegralAdphi = np.dot(A, dl)

        return np.array([dRdphi, dZdphi, dintegralAdphi])
    
    # Cache methods
    
    def clear_cache(self):
        self.cache.clear()


class Cache:
    """
    A cache to store the results of integrations. 
    Entries are catalogued by the starting point of the field line and the type of result.
    the cache will return a dictionary of {t: result} where t is the time period and result is the result of the integration t times.
    The cache is limited in size and will remove the oldest entry if the limit is reached.
    The keys are a hash of the starting position and a string specifying the type of result. These are not specified in the cache itself, but
    by the functions that use the cache. 
    Currently implemented types are: 
    (y0, 'f') for the field line position at time t
    (y0, 'df') for the jacobian of the field line position at time t
    (y0, 'f_noninteger') for the field line position at time t for non-integer t
    (y0, 'df_noninteger') for the jacobian of the field line position at time t for non-integer t
    (y0, 'lagrangian') for the lagrangian form at time t
    (y0, 'winding') for the winding number at time t
    (y0, 'dwinding') for the jacobian of the winding number at time t
    [Note: the cache entries are based on *starting* point of a trajectory, so calculations starting at an intermediate point of a previous trajectory will not be retrieved from the cache.]
    non-integer integrations are always computed from phi=0, and do not continue from a previous result.

    Attributes:
        size (int): The maximum size of the cache.
        cache (dictionary): The cache storage.
    """

    def __init__(self, size=None):
        """
        Initializes the Cache object.
        Entries are dictionaries whose keys are a hash of the starting points, and the type of result (simple integraiton or gradient-based integration).
        the dictionaries are of the form {t: result} where t is the time period and result is the result of the integration t times from the starting point associated with the key. 

        Args:
            size (int, optional): The maximum size of the cache (number of trajectories to store
            for fast re-computing). Defaults to 4045.
        """
        self.size = size if size is not None else 4046
        self.cache = {}

    def retrieve(self, y0, what):
        """
        Retrieves a cached result if available.

        Args:
            y0 (array): The starting point of the field line.
            what (str): The type of result to retrieve.

        Returns:
            dict or None: The cached result if available, otherwise None.
        """
        key = (hash(tuple(y0)), what)
        if key in self.cache:
            return self.cache[key]
        else: 
            return None

    def save(self, y0, what, t, res):
        """
        Saves a result to the cache.

        Args:
            t (float): The time period.
            y0 (array): The starting point of the field line.
            dic (dict): The result to cache.
        """
        key = (hash(tuple(y0)), what)
        if key not in self.cache:
            self.cache[key] = {}
            if len(self.cache) >= self.size:
                self.cache.pop(next(iter(self.cache)))  # remove the oldest entry

        self.cache[key][t] = res
    

    def clear(self):
        """
        Clears the cache.
        """
        self.cache.clear()
