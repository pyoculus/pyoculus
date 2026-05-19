import pyoculus.maps as maps
from .base_solver import BaseSolver
from .fixed_point import FixedPoint
from .manifold import Manifold, Clinic, ClinicSet
from ..utils.plot import create_canvas, clean_bigsteps
from scipy.optimize import root, minimize
from typing import Iterator, Literal, Union, Iterable
from numpy.typing import NDArray
# from functools import total_ordering
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

import logging

logger = logging.getLogger(__name__)




class ResonanceZone:
    """
    Class representing the resonnance zone formed by 2 stable and 2 unstable manifolds, intersected by 2 heteroclinic point and 2 fixed points.

    Args:
     - manifold1: The first manifold.
     - manifold2: The second manifold.

    Methods:
     - area: Compute the area of the resonance zone.
     - plot: Plot the stable and unstable manifolds.
     - contour: Compute the closed contour of the resonance zone.
    """

    def __init__(self, manifold1: Manifold, manifold2: Manifold):
        self.manifold1 = manifold1
        self.manifold2 = manifold2
        if manifold1._map != manifold2._map:
            raise ValueError("Manifolds must be defined on the same map.")
        self._map = manifold1._map
        self.fixedpoint_1 = manifold1.fixedpoint_1
        self.fixedpoint_2 = manifold1.fixedpoint_2
        if manifold1.fixedpoint_1 != manifold2.fixedpoint_2:
            raise ValueError("Fixed points must be the same and reversed.")
        if manifold1.fixedpoint_2 != manifold2.fixedpoint_1:
            raise ValueError("Fixed points must be the same and reversed.")

    def area(self, whole_chain=False):
        """
        Compute the area of the resonance zone using Meiss's action principle.
          whole_chain: if True, compute the area of the whole chain of islands, otherwise only one island.
        """

        if not hasattr(self.manifold1, 'clinics'): 
         self.manifold1.find_clinics(first_guess_eps_s=9e-6, first_guess_eps_u=8e-6,n_points=1)
        if not hasattr(self.manifold2, 'clinics'):
         self.manifold2.find_clinics(first_guess_eps_s=9e-6, first_guess_eps_u=8e-6,n_points=1)

        traj_lagrangian_1=self.manifold1.clinics[0].trajectory
        traj_lagrangian_2=self.manifold2.clinics[0].trajectory

        ### lagrangians of the clinic trajectories
        traj_lagrangian_1=self.manifold1._compute_lagrangian_in_sections(traj_lagrangian_1)
        traj_lagrangian_2=self.manifold2._compute_lagrangian_in_sections(traj_lagrangian_2)
        
        ### lagrangians of the fixed points
        h1=self._map.lagrangian(self.manifold1.rfp_s,self.fixedpoint_1.m)
        h2=self._map.lagrangian(self.manifold2.rfp_s,self.fixedpoint_2.m)

        traj_lagrangian_1[:]=traj_lagrangian_1[:]-h1
        traj_lagrangian_2[:]=traj_lagrangian_2[:]-h2

        ### sum of the lagrangian difference between the clinic trajectories and the fixed points
        int1= np.sum(traj_lagrangian_1)
        int2= np.sum(traj_lagrangian_2)
        if whole_chain:
            return self.fixedpoint_1.m*(int1+int2)
        else:
            return int1+int2

    def plot_manifolds(self, which="both", stepsize_limit=None, with_clinics=False, **kwargs):
        """
        Plot the 2 stable and the 2 unstable manifolds.
         which (str): which manifold to plot. Can be 'stable', 'unstable' or 'both'.
         stepsize_limit = threshold to remove big steps in the manifolds.
         with_clinics (bool): if True, plot the clinic points.
        """

        fig, ax, kwargs = create_canvas(**kwargs)
        
        self.manifold1.plot(ax=ax, which=which, stepsize_limit=stepsize_limit, **kwargs)
        self.manifold2.plot(ax=ax, which=which, stepsize_limit=stepsize_limit, **kwargs)

        if with_clinics:
            self.manifold1.plot_clinics(ax=ax)
            self.manifold2.plot_clinics(ax=ax)

        return fig, ax
    

    def plot_reso_zone(self):
        """
        Compute the closed contour of the resonnance zone, formed by the stable and unstable manifolds of the two fixed points.
        Returns the contour points as a Nx2 array.

        """

        def IndexClosestPoint(traj, point):

            """Find the index of the point in traj closest to the given point."""

            dist = np.linalg.norm(traj - point, axis=1)
            idx = np.argmin(dist)
            return idx

        #ResoManifold1=self.manifold1
        #ResoManifold2=self.manifold2

        if not hasattr(self.manifold1, 'clinics'): 
            self.manifold1.find_clinics(first_guess_eps_s=9e-6, first_guess_eps_u=8e-6,n_points=1)

        if not hasattr(self.manifold2, 'clinics'):
            self.manifold2.find_clinics(first_guess_eps_s=9e-6, first_guess_eps_u=8e-6,n_points=1)

        if not hasattr(self.manifold1, '_stable_trajectory'):
            self.manifold1.compute(
            eps_s=9e-6, eps_u=8e-6, nint_s=8, nint_u=8, neps_s=80, neps_u=80)

        if not hasattr(self.manifold2, '_stable_trajectory'):
            self.manifold2.compute(
            eps_s=9e-6, eps_u=8e-6, nint_s=8, nint_u=8, neps_s=80, neps_u=80)

        MF1s=self.manifold1._stable_trajectory  # first stable segment 
        MF1u=self.manifold1._unstable_trajectory #first unstable segment
        MF2s=self.manifold2._stable_trajectory #second stable segment
        MF2u=self.manifold2._unstable_trajectory #second unstable segment

        M2_0_idx=len(self.manifold2.clinics[0].trajectory)//2 
        M2_0 = self.manifold2.clinics[0].trajectory[M2_0_idx]
        idx_2s = IndexClosestPoint(MF2s, M2_0)
        idx_2u = IndexClosestPoint(MF2u, M2_0)
        MF2s= MF2s[:idx_2s]
        MF2u= MF2u[:idx_2u]
        MF2u= MF2u[::-1]  
        
        M1_0_idx=len(self.manifold1.clinics[0].trajectory)//2
        M1_0 = self.manifold1.clinics[0].trajectory[M1_0_idx]
        idx_1s = IndexClosestPoint(MF1s, M1_0)
        idx_1u = IndexClosestPoint(MF1u, M1_0)
        MF1u= MF1u[:idx_1u]  
        MF1s= MF1s[:idx_1s]         
        MF1u= MF1u[::-1]  

        contour_points = np.vstack([
         self.manifold1.rfp_s,      
         MF1s,              
         MF1u,      
         self.manifold2.rfp_s,
         MF2s,
         MF2u])      
      
        return contour_points[::-1] #reverse to have clockwise orientation
    
    def flux_approx_SL(self):
        """
        Approximate the flux through the resonnance zone using area computed with shoelace method, multiplied by the mean magnetic field inside the contour
        """
        
        contour_points=self.contour()
        
        # Compute grid inside contour
        xmin, xmax = contour_points[:,0].min(), contour_points[:,0].max()
        ymin, ymax = contour_points[:,1].min(), contour_points[:,1].max()
        nx, ny = 1000, 1000  # number of points in x and y direction
        X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
        points_grid = np.vstack((X.ravel(), Y.ravel())).T

        contour_path = Path(contour_points)

        inside = contour_path.contains_points(points_grid)
        inside_points = points_grid[inside]
        inside_points = np.column_stack([inside_points[:, 0], np.zeros(inside_points.shape[0]), inside_points[:, 1]])

        # Compute mean B field inside contour

        B_inside = np.array([self.manifold1._map._mf.B(pt)[1]*pt[0] for pt in inside_points])

    
        mean_B = np.mean(B_inside)

        x = contour_points[:, 0]
        y = contour_points[:, 1]

        return mean_B*0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) #compute area using shoelace