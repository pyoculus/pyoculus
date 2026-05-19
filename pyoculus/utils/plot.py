"""
This module provides a function to create a canvas for plotting using Matplotlib.

:authors:
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

import matplotlib.pyplot as plt
import numpy as np


def create_canvas(rcstyle=None, **kwargs) -> tuple:
    """
    Create a Matplotlib canvas with a figure and axes.

    This function checks if a figure or axes object is provided in the keyword arguments.
    If neither is provided, it creates a new figure and axes. It returns the figure, axes,
    and any remaining keyword arguments.

    Parameters:
        rcstyle (str): The Matplotlib style to use.
        **kwargs: Arbitrary keyword arguments. Can include `fig` for a Matplotlib figure or `ax` for axes.

    Returns:
        tuple: A tuple containing the figure, axes, and remaining keyword arguments.
    """

    if rcstyle is not None:
        plt.style.use(rcstyle)

    if "fig" in kwargs.keys():
        fig = kwargs["fig"]
        ax = fig.gca()
    elif "ax" in kwargs.keys():
        ax = kwargs["ax"]
        fig = ax.figure
    else:
        fig, ax = plt.subplots()

    kwargs.pop("fig", None)
    kwargs.pop("ax", None)

    return fig, ax, kwargs


def clean_bigsteps(array, threshold=0.1):
    """
    place nans where the difference between two consecutive values is greater than a threshold

    Useful for removing modulo-boundary crossing lines, that are very long
    """
    diff = np.linalg.norm(np.diff(array, axis=0), axis=1)
    idx = np.where(diff > threshold)[0]
    array[idx+1, :] = np.nan
    return array


