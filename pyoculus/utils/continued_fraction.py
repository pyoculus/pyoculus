"""
This module provides functions related to the continued fraction expansion. Reference in Ivan Niven, Irrational Numbers (Cambridge University Press, 2005).

:authors:
    - Zhisong Qu (zhisong.qu@anu.edu.au)
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

import numpy as np


def expandcf(realnumber, n=100, thres=1e-12):
    """
    Expands a positive real number in its continued fraction.

    Args:
        realnumber (float): The positive real number to expand. An absolute value will be taken if negative.
        n (int, optional): The maximum number of terms in the expansion. Default to 100.
        thres (float, optional): The threshold to stop the expansion. Default to 1e-6.

    Returns:
        np.ndarray: A NumPy array containing the continued fraction expansion of the real number up to the `nth` term or until the threshold is met.
    """
    ais = np.zeros(n, dtype=np.int64)
    residue = np.abs(realnumber)

    for i in range(n):
        int_part = np.ceil(residue)
        if np.abs(residue - int_part) > 1e-3:
            int_part -= 1

        ais[i] = int_part
        f = residue - int_part

        if f < thres:
            break
        residue = 1.0 / f

    return ais[: i + 1]


def fromcf(ai):
    """
    Obtains the fraction :math:`n/m` from the coefficients :math:`[a_0, a_1, ..., a_m]` of the continued fraction.

    Args:
        ai (list of int): An integer list containing ai for the continued fraction expansion.

    Returns:
        tuple: A tuple :math:`(n, m)`, representing the fraction.
    """

    # Use the relation of the Gaussian bracket to get the fraction
    h, k = np.zeros(len(ai) + 2), np.zeros(len(ai) + 2)

    h[0], h[1] = 0, 1
    k[0], k[1] = 1, 0

    for i, a in enumerate(ai):
        h[i + 2] = a * h[i + 1] + h[i]
        k[i + 2] = a * k[i + 1] + k[i]

    return int(h[-1]), int(k[-1])
