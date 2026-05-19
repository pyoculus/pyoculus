"""
This module provides functions to transform vectors and matrices between cylindrical and cartesian coordinates.

:authors:
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

import numpy as np
import numpy.typing as npt


import numpy as np
import numpy.typing as npt

# Try to import numba and define a decorator
try:
    import numba
    njit = numba.njit
except ImportError:
    # Define a no-op decorator if numba is not available
    def njit(func):
        return func


# Jacobian of the map (x, y, z) -> (r, phi, z) at (r, phi, z) and (x, y, z)

@njit
def xyz_jac(r: float, phi: float, z: float) -> np.array:
    """
    Jacobian of the map :math:`(x, y, z) \\to (r, \\phi, z)` at :math:`(r, \\phi, z)`.

    .. math::
        J(r, \\phi, z) = \\begin{bmatrix}
            \\partial_x r & \\partial_y r & \\partial_z r \\\\
            \\partial_x \\phi & \\partial_y \\phi & \\partial_z \\phi \\\\
            \\partial_x z & \\partial_y z & \\partial_z z \\\\
        \\end{bmatrix}
    """
    matrix = np.array(
        [
            [np.cos(phi), np.sin(phi), 0],
            [-1 * np.sin(phi) / r, np.cos(phi) / r, 0],
            [0*z, 0, 1],  # z is multiplied with zero otherwise numba cries on compilation
        ]
    )
    return matrix

@njit
def rphiz_jac(x: float, y: float, z: float) -> np.array:
    """
    Jacobian of the map :math:`(x, y, z) \\to (r, \\phi, z)` at :math:`(x, y, z)`.
    """
    return np.array(
        [
            [x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2), 0],
            [-y / (x**2 + y**2), x / (x**2 + y**2), 0],
            [0*z, 0, 1],  # z is multiplied with zero otherwise numba cries on compilation
        ]
    )


# Jacobian of the map (r, phi, z) -> (x, y, z) at (r, phi, z) and (x, y, z)

@njit
def xyz_inv_jac(r: float, phi: float, z: float) -> np.array:
    """
    Inverse Jacobian of the map :math:`(r, \\phi, z) \\to (x, y, z)` at :math:`(r, \\phi, z)`.
    """
    return np.array(
        [
            [np.cos(phi), -1 * np.sin(phi) * r, 0],
            [np.sin(phi), np.cos(phi) * r, 0],
            [0 *z, 0, 1],  # z is multiplied with zero otherwise numba cries on compilation
        ]
    )

@njit
def rphiz_inv_jac(x: float, y: float, z: float) -> np.array:
    """
    Inverse Jacobian of the map :math:`(r, \\phi, z) \\to (x, y, z)` at :math:`(x, y, z)`.
    """
    return np.array(
        [
            [x / np.sqrt(x**2 + y**2), -y * np.sqrt(x**2 + y**2), 0],
            [y / np.sqrt(x**2 + y**2), x * np.sqrt(x**2 + y**2), 0],
            [0*z, 0, 1],  # z is multiplied with zero otherwise numba cries on compilation
        ]
    )


# Coordinate transformations

@njit
def xyz(r: float, phi: float, z: float) -> np.array:
    """
    Transforms cylindrical coordinates :math:`(r, \\phi, z)` to cartesian coordinates :math:`(x, y, z)`.
    """
    return np.array([r * np.cos(phi), r * np.sin(phi), z])

@njit
def rphiz(x: float, y: float, z: float) -> np.array:
    """
    Transforms cartesian coordinates :math:`(x, y, z)` to cylindrical coordinates :math:`(r, \\phi, z)`.
    """
    return np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x), z])


# Vector transformations

@njit
def vec_cart2cyl(vec: npt.ArrayLike, r: float, phi: float, z: float) -> np.array:
    """
    Transforms the (contravariant) cartesian components of a vector to the contravariant cylindrical components at :math:`(r, \\phi, z)`.

    .. math:: 
        \\begin{bmatrix}
            v^r \\\\
            v^{\\phi} \\\\
            v^z
        \\end{bmatrix} =
        \\begin{bmatrix}
            \\partial_x r & \\partial_y r & \\partial_z r \\\\
            \\partial_x \\phi & \\partial_y \\phi & \\partial_z \\phi \\\\
            \\partial_x z & \\partial_y z & \\partial_z z \\\\
        \\end{bmatrix}
        \\begin{bmatrix}
            v^x \\\\
            v^y \\\\
            v^z
        \\end{bmatrix}.

    Args:
        vec (array): The cartesian components of a vector.
        r (float): The radial coordinate.
        phi (float): The azimuthal coordinate.
        z (float): The vertical coordinate.

    Returns:
        array: The contravariant cylindrical components of the vector.
    """
    return np.dot(xyz_jac(r, phi, z), np.atleast_2d(vec).T).T[0]

@njit
def vec_cyl2cart(vec: npt.ArrayLike, x: float, y: float, z: float) -> np.array:
    """
    Transforms the contravariant cylindrical components to the (contravariant) cartesian components of a vector at :math:`(x, y, z)`.
    """
    raise NotImplementedError("Not implemented yet.")


# Matrix transformations

@njit
def mat_cart2cyl(mat: np.array, r: float, phi: float, z: float) -> np.array:
    """
    Transforms a matrix :math:`A` from cartesian coordinates to cylindrical coordinates at :math:`(r, \\phi, z)`.

    .. math:: 
        \\begin{bmatrix}
            A^r_r & A^r_\\phi & A^r_z \\\\
            A^\\phi_r & A^\\phi_\\phi & A^\\phi_z \\\\
            A^z_r & A^z_\\phi & A^z_z \\\\
        \\end{bmatrix} =
        \\begin{bmatrix}
            \\partial_x r & \\partial_y r & \\partial_z r \\\\
            \\partial_x \\phi & \\partial_y \\phi & \\partial_z \\phi \\\\
            \\partial_x z & \\partial_y z & \\partial_z z \\\\
        \\end{bmatrix}
        \\begin{bmatrix}
            A^x_x & A^x_y & A^x_z \\\\
            A^y_x & A^y_y & A^y_z \\\\
            A^z_x & A^z_y & A^z_z \\\\
        \\end{bmatrix}
        \\begin{bmatrix}
            \\partial_r x & \\partial_\\phi x & \\partial_z x \\\\
            \\partial_r y & \\partial_\\phi y & \\partial_z y \\\\
            \\partial_r z & \\partial_\\phi z & \\partial_z z \\\\
        \\end{bmatrix}.

    Args:
        mat (array): The matrix :math:`A` in cartesian coordinates.
        r (float): The radial coordinate.
        phi (float): The azimuthal coordinate.
        z (float): The vertical coordinate.
    
    Returns:
        array: The matrix :math:`A` in cylindrical coordinates.
    """

    invjac = xyz_inv_jac(r, phi, z)
    jac = xyz_jac(r, phi, z)
    return jac @ mat @ invjac

@njit
def mat_cyl2cart(mat: np.array, x: float, y: float, z: float) -> np.array:
    """
    Transforms a matrix :math:`A` from cylindrical coordinates to cartesian coordinates at :math:`(x, y, z)`.
    """
    raise NotImplementedError("Not implemented yet.")

@njit
def dinvJ_matrix(vec: npt.ArrayLike, r: float, phi: float, z: float) -> np.array:
    """
    Computes the matrix due to the evolution of the basis in cylindrical coordinates.
    
    The Christoffel symbols :math:`\\Gamma^k_{ij} = \\frac{\\textbf{e}_i}{x^j}\\cdot g^{km}\\textbf{e}_m`, where :math:`g^{km}` is the inverse metric tensor, are used to compute the derivative of the basis vectors. Then for a vector :math:`X = X^i\\textbf{e}_i`, the covariant derivative is given by:

    .. math::
        \\nabla_i X^k = \\partial_i X^k - \\Gamma^k_{ij} X^j. 

    To get the partial derivatives such as :math:`\\partial_\\phi v^r`, we can use the above formula, the independance of basis of :math:`\\nabla v` and a change of basis to get the desired components. We can write the part we need to substract in matrix form as:
    
    .. math::
        \\text{to write}.
    
    Args:
        vec (array): The vector :math:`v` in cartesian coordinates.
        r (float): The radial coordinate.
        phi (float): The azimuthal coordinate.
        z (float): The vertical coordinate.

    Returns:
        array: The matrix due to the evolution of the basis in cylindrical coordinates.
    """

    vec = vec.flatten()
    return np.array(
        [
            [0, -vec[0] * np.sin(phi) + vec[1] * np.cos(phi), 0],
            [
                (vec[0] * np.sin(phi) - vec[1] * np.cos(phi)) / r**2,
                (-vec[0] * np.cos(phi) - vec[1] * np.sin(phi)) / r,
                0,
            ],
            [0*z, 0, 0],  # z is multiplied with zero otherwise numba cries on compilation
        ]
    )
