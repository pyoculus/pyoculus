"""
This module provides functions to make and plot a convergence domain.

For each pair in a grid of (x1, x2) points, it tries to find a cretain fixed point and the store the convergence behaviour.

:authors:
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

from pyoculus.solvers import FixedPoint
import pyoculus.maps as maps
from .plot import create_canvas
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.typing as npt
import numpy as np

# def join_convergence_domains(convdomlist, eps=1e-5):
#     """Join together a list of convergence domain results, returning a new tuple with the same format."""
#     convdomReturn = ([], [], [])

#     for convdom in convdomlist:
#         R_values, Z_values, _, _, all_fixed_points = convdom
#         convdomReturn[0].append(R_values)
#         convdomReturn[1].append(Z_values)
#         convdomReturn[2].append(all_fixed_points)

#     R_values = np.concatenate(convdomReturn[0])
#     Z_values = np.concatenate(convdomReturn[1])
#     all_fixed_points = np.concatenate(convdomReturn[2])

#     # Get the indices that would sort R_values and Z_values
#     R_sort_indices = np.argsort(R_values)
#     Z_sort_indices = np.argsort(Z_values)

#     # Use these indices to rearrange all_fixed_points
#     all_fixed_points = all_fixed_points[R_sort_indices][Z_sort_indices]

#     # # Reloop through the sorted arrays and assign the correct index to each point

#     # for fp in all_fixed_points:
#     #         fp_result = all_fixed_points[i]

#     #         if fp_result.successful is True:
#     #             fp_result_xyz = np.array([fp_result.x[0], fp_result.y[0], fp_result.z[0]])
#     #             assigned = False
#     #             for j, fpt in enumerate(fixed_points):
#     #                 fpt_xyz = np.array([fpt.x[0], fpt.y[0], fpt.z[0]])
#     #                 if np.isclose(fp_result_xyz, fpt_xyz, atol=kwargs["eps"]).all():
#     #                     assigned_to.append(j)
#     #                     assigned = True
#     #             if not assigned:
#     #                 assigned_to.append(len(fixed_points))
#     #                 fixed_points.append(fp_result)
#     #             all_fixed_points.append(fp_result)
#     #         else:
#     #             assigned_to.append(-1)
#     #             all_fixed_points.append(None)

#     # return R_values, Z_values, assigned_to, all_fixed_points


def compute_convergence_domain(
    map: maps.BaseMap,
    x1s: npt.ArrayLike,
    x2s: npt.ArrayLike,
    find_with_iota: bool = True,
    eps: float = 1e-3,
    **kwargs,
) -> tuple:
    """
    Compute where the FixedPoint solver converges to for each of the [x1, x2] point pair. If a point converges, it is assigned a number which is the index corresponding to the equal first fixedpoint converging to the same place, otherwise it is assigned -1. The number corresponds to the index of the fixed point in the returned list of fixed points.

    Args:
        map (pyoculus.maps): The problem to solve.
        x1s (np.ndarray): The x1 values of the meshgrid.
        x2s (np.ndarray): The x2 values of the meshgrid.
        find_with_iota (bool, optional): Whether to use the find_with_iota method. Defaults to True.
        eps (float, optional): The tolerance for the comparison with the fixed points. Defaults to 1e-3.

    Keyword Args:
        m (int): The poloidal mode to use.
        n (int): The toroidal mode to use.
        tol (float): The tolerance of the fixed point finder.
        t (float): The parameter t for the find method.
        nrestart (int): The number of restarts for the fixed point finder.
        niter (int): The number of iterations for the fixed point finder.

    Returns:
        tuple: A tuple containing:
        - np.ndarray: The meshgrid of x1s and x2s, the assigned number for each point in the meshgrid, and the list of all fixed points objects convergent or not.
        - np.ndarray: The list of fixed points.
    """
    X1s, X2s = np.meshgrid(x1s, x2s)

    assigned_to, fixed_points, all_fixed_points = list(), list(), list()

    for x1, x2 in zip(X1s.flatten(), X2s.flatten()):
        fp_result = FixedPoint(map)
        if find_with_iota:
            fp_result.find_with_iota(
                m=kwargs["m"],
                n=kwargs["n"],
                guess=[x1, x2],
                tol=kwargs["tol"],
            )
        else:
            fp_result.find(
                t=kwargs["t"],
                guess=[x1, x2],
                tol=kwargs["tol"],
            )

        if fp_result.successful is True:
            fp_result_x = fp_result.coords[0]
            assigned = False
            for j, fpt in enumerate(fixed_points):
                fpt_x = fpt.coords[0]
                if np.isclose(fp_result_x, fpt_x, atol=eps).all():
                    assigned_to.append(j)
                    assigned = True
            if not assigned:
                assigned_to.append(len(fixed_points))
                fixed_points.append(fp_result)
            all_fixed_points.append(fp_result)
        else:
            assigned_to.append(-1)
            all_fixed_points.append(fp_result)

    return np.array(
        [
            X1s,
            X2s,
            np.array(assigned_to).reshape(X1s.shape),
            np.array(all_fixed_points).reshape(X1s.shape),
        ]
    ), np.array(fixed_points, dtype=object)


def plot_convergence_domain(
    X1s: np.array,
    X2s: np.array,
    assigned_to: np.array,
    fixed_points: npt.ArrayLike,
    colors=None,
    **kwargs,
) -> tuple:
    """
    Plot the convergence domain for the FixedPoint solver in the X1s-X2s plane. If ax is None, a new figure is created,
    otherwise the plot is added to the existing figure.

    Args:
        X1s (np.ndarray): The x1 values of the meshgrid.
        X2s (np.ndarray): The x2 values of the meshgrid.
        assigned_to (np.ndarray): The assigned number for each point in the meshgrid.
        fixed_points (list): The list of fixed points object (FixedPoint.OutputData).
        colors (np.ndarray, optional): The colors to use. Defaults to None. Should be of dimension (k, 3 or 4) for RGB/RGBA with k at least the number of fixed points plus one.

    Keyword Args:
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object.
            - ax (matplotlib.axes.Axes): The axes object.
    """

    assigned_to = assigned_to.flatten() + 1
    assigned_to = assigned_to.astype(int)

    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, len(fixed_points)))
        colors[:, 3] = 0.8
        colors = np.vstack(([0.3, 0.3, 0.3, 0.15], colors))

    cmap = np.array([colors[j] for j in assigned_to])
    cmap = cmap.reshape(X1s.shape[0], X1s.shape[1], cmap.shape[1])

    fig, ax, kwargs = create_canvas(**kwargs)

    ax.pcolormesh(X1s, X2s, cmap, shading="nearest")

    # for r,z in zip(R, Z):
    #     ax.scatter(r, z, color = 'blue', s = 1)

    for i, fpt in enumerate(fixed_points):
        fpt.plot(
            color=colors[i + 1, :3],
            ax=ax,
            edgecolors="black",
            linewidths=1,
            label=f"[{fpt.coords[0][0]:.2f},{fpt.coords[0][1]:.2f}]",
        )

    # # Plot arrows from the meshgrid points to the fixed points they converge to
    # for r, z, a in zip(R.flat, Z.flat, assigned_to.flat):
    #     if a > 0:
    #         fpt = fixed_points[a - 1]
    #         dr = np.array([fpt.x[0] - r, fpt.z[0] - z])
    #         dr = 0.1*dr
    #         ax.arrow(r, z, dr[0], dr[1], color='blue')

    ax.set_aspect("equal")

    return fig, ax
