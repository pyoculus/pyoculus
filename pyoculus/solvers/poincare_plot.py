"""
poincare_plot.py
==================

Contains the class for generating the Poincare Plot and computing the winding number profile.

:authors:
    - Zhisong Qu (zhisong.qu@anu.edu.au)
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

import numpy as np
import matplotlib.pyplot as plt
from .base_solver import BaseSolver
import pyoculus.maps as maps
import concurrent.futures
import logging
import sys

logger = logging.getLogger(__name__)


class PoincarePlot(BaseSolver):
    """
    Class for generating the Poincare Plot and computing the winding number profile of a 2D map.

    Attributes:
        xs (np.ndarray): The initial points, shape (npoints, dimension).
        points_type (str): The type of points generation.
    """

    ## Computation methods

    def __init__(self, map, xs, points_type = "custom"):
        """
        Initialize the Poincare plot.

        Args:
            map (maps.base_map): The map to use for the computation. Should be 2D.
            xs (np.ndarray): The initial points, shape (npoints, dimension).
            points_type (str): The type of points generation. Default is "custom" for custom points.
        """
        # Check that the map is a 2D map
        if xs.shape[1] != map.dimension or map.dimension != 2:
            raise ValueError("The initial points should have the correct dimension and the map should be 2D.")
        
        # Check if the points are in the domain of the map
        for ii in range(map.dimension):
            if (
                xs[:, ii].any() < map.domain[ii][0]
                or xs[:, ii].any() > map.domain[ii][1]
            ):
                raise ValueError(
                    "The initial points should be in the domain of the map."
                )

        super().__init__(map)

        # Set the attributes
        self._xs = xs
        self._points_type = points_type

        # Flags for the computation
        self._successful = False
        self._iota_computed = False
        self._iota_successful = False

    # Classmethods for creating the Poincare plot

    @classmethod
    def with_linspace(cls, map, x0, x1, npts):
        """
        Creates a Poincare plot with points linearly spaced between x0 and x1.

        Args:
            map (maps.base_map): The map to use for the computation.
            x0 (np.array): The starting point.
            x1 (np.array): The ending point.
            npts (int): The number of points.

        Returns:
            PoincarePlot: The PoincarePlot object.
        """
        xs = np.linspace(x0, x1, npts)
        return cls(map, xs, points_type = "linear")
    
    @classmethod
    def with_segments(cls, map, xns, neps, connected = True):
        """
        Create a Poincare plot specifying the points to be along multiple segments.

        Can be used either to join the points passed as arguments when connected is True. Or, when connected is False, the points are taken to be the two by two extremities of eich segment.
        
        Args:
            map (maps.base_map): The map to use for the computation.
            xns (np.ndarray): The points defining the extremities of the segments.
            neps (list): The number of points per segment. The length of the list should be equal to : the number of segments minus one if connected is True, twice the number of segments otherwise.
            connected (bool): If True, then the segments are connected to each other. Default is True.

        Returns:
            PoincarePlot: The PoincarePlot object.
        """
        xs = np.empty((0, map.dimension))

        if connected:
            # Check for coherent lengths of xns and neps
            if len(xns) - 1 != len(neps):
                raise ValueError("The number of segments should be equal to the length of the number of points per segment minus one.")
            
            # Concatenate the points
            for i, eps in enumerate(neps):
                if eps < 2:
                    raise ValueError("The number of points per segment should be equal or greater than 2.")
                xs = np.concatenate((xs, np.linspace(xns[i], xns[i+1], eps)[:-1,:]))
            
            # Add the last point
            xs = np.concatenate((xs, xns[-1].reshape(1, 2)))
        else:
            # Check for coherent lengths of xns and neps
            if len(xns) != 2 * len(neps):
                raise ValueError("The number of segments should be equal to twice the length of the number of points per segment.")
            
            # Concatenate the points
            for i, eps in enumerate(neps):
                xs = np.concatenate((xs, np.linspace(xns[2*i], xns[2*i+1], eps)))

        return cls(map, xs, points_type = "segments")

    @classmethod
    def with_horizontal(cls, map, radius, ntraj):
        """
        Create a Poincare plot with horizontal points.

        Args:
            map (maps.base_map): The map to use for the computation.
            radius (float): The radius of the horizontal points.
            ntraj (int): The number of seeds, starting points for your trajectories.

        Returns:
            PoincarePlot: The PoincarePlot object.
        """

        if isinstance(map, maps.CylindricalBfieldSection):
            opoint = np.array([map.R0, map.Z0])
            xs = np.linspace(opoint + np.array([1e-8, 0]), opoint + np.array([radius, 0]), ntraj)
        elif isinstance(map, maps.ToroidalBfieldSection):
            xs = np.linspace(np.array([1e-8, 0]), np.array([radius, 0]), ntraj)
        else:
            raise ValueError("The map is not supported for horizontal points generation.")
        
        return cls(map, xs, points_type = "horizontal")

    # Properties

    @property
    def xs(self):
        return self._xs
    
    @xs.setter
    def xs(self, xs):
        self._points_type = "custom"
        self._xs = xs

    # Methods for computing the Poincare plot

    def compute(self, npts=200, compute_iota=False):
        """
        Computes the evolution of the initial points under the map for npts points.

        Args:
            npts (int): The number of points to compute for each initial point.
            nprocess (int): The number of processes to use for the computation. Default is 1.
            compute_iota (bool): If True, then it computes the winding number profile. Default is False.
            
        Returns:
            hits (np.ndarray): The position of the crossings, shape (npoints, npts+1, dimension)
        """
        xs = self.xs

        # Initialize the hits
        hits = np.nan * np.ones(
            (xs.shape[0], npts + 1, self._map.dimension), dtype=np.float64
        )
        hits[:, 0, :] = np.copy(xs)
        
        if compute_iota:
            self._iota_computed = True
            windings = np.nan * np.zeros_like(hits)

        # Define the function to compute a point evolution
        def compute_point(i, x):
            current_x = x.copy()
            for j in range(npts):
                logger.debug("Computing point %d, %d", i, j)
                try:
                    if compute_iota:
                        windings[i, j + 1, :] = self._map.winding(1, current_x)
                    current_x = self._map.f(1, current_x)
                except KeyboardInterrupt:
                    logger.info("KeyboardInterrupt detected. Stopping the program.")
                    sys.exit()
                except Exception as e:
                    logger.warning("The map failed to compute at point %s: %s", current_x, str(e))
                    break
                hits[i, j + 1, :] = current_x

        for i, x in enumerate(xs):
            compute_point(i, x)

        # Set the success flag to True and store the results
        self._hits = hits
        if compute_iota:
            self._windings = windings[:, 1:, :]
            self._iota_computed = True
        self._successful = True

        return hits

    def compute_iota(self, **kwargs):
        """
        Compute the winding number profile.

        Args:
            **kwargs: Parameters for the poincare hits computation.
        
        Returns:
            xs (np.ndarray): The initial points where to calculate the winding number profile, shape (npoints, dimension).
            winding (np.ndarray): The winding number of the points.
        """

        if not self._successful or not self._iota_computed:
            self.compute(**kwargs, compute_iota=True)

        if isinstance(self._map, maps.CylindricalBfieldSection) or isinstance(self._map, maps.ToroidalBfieldSection):
            
            theta = np.empty((self.xs.shape[0], self._windings.shape[1] + 1))
            if isinstance(self._map, maps.CylindricalBfieldSection):
                thetas_0 = np.arctan2(self.xs[:, 1] - self._map.Z0, self.xs[:, 0] - self._map.R0)
            else:
                thetas_0 = self.xs[:, 1]

            dtheta = np.cumsum(self._windings[:, :, 1], axis=1)
            theta[:, 0] = thetas_0
            theta[:, 1:] = dtheta + thetas_0[:, np.newaxis]

            self._iota = np.zeros(self.xs.shape[0], dtype=np.float64)

            # Uses the Reiman and Greenside least square fit optimization method
            for ii in range(self.xs.shape[0]):
                nlist = np.arange(theta.shape[1], dtype=np.float64)
                dzeta = self._map.dzeta
                leastfit = np.zeros(6, dtype=np.float64)
                leastfit[1] = np.sum((nlist * dzeta) ** 2)
                leastfit[2] = np.sum((nlist * dzeta))
                leastfit[3] = np.sum((nlist * dzeta) * theta[ii, :])
                leastfit[4] = np.sum(theta[ii, :])
                leastfit[5] = 1.0

                self._iota[ii] = (leastfit[5] * leastfit[3] - leastfit[2] * leastfit[4]) / (
                    leastfit[5] * leastfit[1] - leastfit[2] * leastfit[2]
                )
        else:
            # return the average change in angle around the axis because the smart method above is not working
            self._iota = np.array([np.mean(self._windings[i,:,1])/(2*np.pi) for i in range(self._windings.shape[0])])

        self._iota_successful = True

        return self.xs, self._iota
    
    @property
    def rho(self):
        """
        the rho values, calculated the axis of the map
        """
        distance_to_map_axis = self.xs - np.array([self._map.R0, self._map.Z0])
        return np.linalg.norm(distance_to_map_axis, axis=1)
        
    
    
    @property
    def iota(self):
        """
        the iota values, calculated from _windings which is calculated with a sucessful compute_iota
        """
        if not self._iota_successful:
            self.compute_iota()
        return self._iota
    
    @property
    def q(self):
        """
        the q values, calculated from _windings which is calculated with a sucessful compute_iota
        """
        if not self._iota_successful:
            self.compute_iota()
        return 1/self.iota
    
    

    ## Plotting methods

    def plot(self, plottype='RZ', xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs):
        """
        Plot the Poincare plot.

        Args:
            plottype (str): The type of plot to generate. Can be 'RZ' for R-Z coordinates or 'polar' (including pyoculus s-theta coordinates)
                            or None to directly plot in the coordinates of the map. 
            xlabel (str): The label of the x-axis.
            ylabel (str): The label of the y-axis.
            xlim (tuple): The range of the x-axis.
            ylim (tuple): The range of the y-axis.
            **kwargs: Additional parameters for the plotting routine "scatter".

        Returns:
            fig, ax: The figure and axis of the plot.
        """

        if not self._successful:
            raise Exception("A successful call of compute() is needed")

        if "fig" in kwargs.keys():
            fig = kwargs["fig"]
            ax = fig.gca()
            kwargs.pop("fig")
        elif "ax" in kwargs.keys():
            ax = kwargs["ax"]
            fig = ax.figure
            kwargs.pop("ax")
        elif plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
        else:
            fig, ax = plt.subplots()

        # set default plotting parameters
        if kwargs.get("marker") is None:
            kwargs.update({"marker": "."})
        if kwargs.get("color") is None:
            kwargs.update({"color": "black"})

        # set defaults for better pplots
        # line width 0 allows for smallest points.
        lw = kwargs.pop("lw", 0)
        s = kwargs.pop("s", 3)

        # plotting the points
        if plottype == 'RZ':
            plot_array = self.hits_rz
            for traj in plot_array:
                ax.scatter(traj[:, 0], traj[:, 1], lw=lw, s=s, **kwargs)
            ax.set_aspect('equal')
        elif plottype == 'polar':
            plot_array = self.hits_polar
            for traj in plot_array:
                ax.scatter(traj[:, 0], traj[:, 1], lw=lw, s=s, **kwargs)
        elif plottype is None:
            plot_array = self._hits
            for traj in plot_array:
                ax.scatter(traj[:, 0], traj[:, 1], lw=lw, s=s, **kwargs)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return fig, ax

    def plot_iota(self, xlim=None, ylim=None, **kwargs):
        """
        Plot the winding number profile.

        Args:
            xlim (tuple): The range of the x-axis.
            ylim (tuple): The range of the y-axis.
            **kwargs: Additional parameters for the plotting routine "plots".

        Returns:
            fig, ax: The figure and axis of the plot.   
        """
        
        if not self._iota_successful:
            try: 
                self.compute_iota()
            except:
                raise Exception("A successful call of compute_iota() is needed")

        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
        if "fig" in kwargs.keys():
            fig = kwargs.pop("fig")
            ax = fig.gca()
        elif "ax" in kwargs.keys():
            ax = kwargs.pop("ax")
            fig = ax.figure
        else:
            fig, ax = plt.subplots()

        ax.plot(self.rho, self.iota, **kwargs)

        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel(r"$\iota\!\!$-")

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return fig, ax

    def plot_q(self, xlim=None, ylim=None, **kwargs):
        """
        Plot the q-profile.

        Args:
            xlim (tuple): The range of the x-axis.
            ylim (tuple): The range of the y-axis.
            **kwargs: Additional parameters for the plotting routine "plots".

        Returns:
            fig, ax: The figure and axis of the plot.
        """
        if not self._iota_successful:
            try: 
                self.compute_iota()
            except:
                raise Exception("A successful call of compute_iota() is needed")

        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
        if "fig" in kwargs.keys():
            fig = kwargs.pop("fig")
            ax = fig.gca()
        elif "ax" in kwargs.keys():
            ax = kwargs.pop("ax")
            fig = ax.figure
        else:
            fig, ax = plt.subplots()

        ax.plot(self.rho, 1 / self.iota, **kwargs)

        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel(r"$q$")

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return fig, ax

    @property
    def hits_rz(self):
        """
        helper to return the rz coordnates of a call to 
        compute. 
        """
        if self._hits is None:
            raise Exception("A successful call of compute() is needed")
        if isinstance(self._map, maps.CylindricalBfieldSection):
            return self._hits
        elif isinstance(self._map, maps.ToroidalBfieldSection): # hits are stored in s-theta coordinates. 
            hits_rz = []
            for traj in self._hits: # first index
                traj_sthetazeta = np.insert(traj, 2, self._map.phi0, axis=1) #fill phi0 value
                traj_rz = []
                for point in traj_sthetazeta:
                    traj_rz.append(self._map._mf.convert_coords(point)[::2])
                hits_rz.append(traj_rz)
            return np.array(hits_rz)
        
    @property
    def hits_polar(self): 
        """
        helper to return the thetazeta or rho, theta coordnates of a call to compute. 
        """
        if self._hits is None:
            raise Exception("A successful call of compute() is needed")
        if isinstance(self._map, maps.CylindricalBfieldSection):
            hits_rho_theta = []
            for traj_RZ in self._hits: # first index
                traj_rhotheta = []
                for point in traj_RZ:
                    traj_rhotheta.append(self._map.to_rhotheta(point))
                hits_rho_theta.append(traj_rhotheta)
            return np.array(hits_rho_theta)
        elif isinstance(self._map, maps.ToroidalBfieldSection):
            return self._hits
    
    # Saving / Loading methods

    def save(self, filename : str = "poincare.npy"):
        """
        """
        np.save(filename, self._hits)

    def load(self, map : maps.base_map):
        """
        """
        
