"""
This module provides functions to calculate equilibrium and perturbation fields in cylindrical coordinates.

The fields are calculated from the vector potential :math:`A` as :math:`B = \\nabla \\times A`.

:authors:
    - Ludovic Rais (ludovic.rais@epfl.ch)
"""

from functools import wraps

import logging
logger = logging.getLogger(__name__)

try:
    from jax import config
    config.update("jax_enable_x64", True)
    from jax import jacfwd
    from jax.lax import cond
    import jax.numpy as jnp
except ImportError as e:
    logger.warning("Could not import jax. Some functionalities will not be available.")
    raise e

import numpy as np

## Decorators


def psitob(f):
    """
    Decorator to calculate the contribution of a :math:`\\psi` function to a :math:`B` field
    using the relation :math:`B = \\nabla \\times A`, with :math:`A_\\phi \\textbf{e}_\\phi = \\psi / r \\textbf{e}_\\phi`, where
    :math:`g(\\textbf{e}_\phi, \\textbf{e}_\phi) = 1`.

    Args:
        f (function): Function that returns the :math:`\\psi` field. The function must take the position vector :math:`\\textbf{r}` as first argument.

    Returns:
        function: Function that returns the :math:`B` field.
    """

    @wraps(f)
    def dfun(rr, *args, **kwargs):
        deriv = jacfwd(f)(rr, *args, **kwargs)
        return jnp.array([-1 * deriv[2], 0.0, deriv[0]]) / rr[0]

    return dfun


def rot(f, from_holonomous=True):
    """
    Decorator that calculate the `curl` in cylindrical coordinates, useful to get
    the :math:`B` field from a vector potential :math:`A` using the relation :math:`B = \\nabla \\times A`.

    Args:
        f (function): Function that returns the vector potential :math:`A` field. The function must take the position vector :math:`\\textbf{r}` as first argument.
        from_holonomous (bool): If True, the input of the function is in holonomous coordinates :math:`\\{\\partial_r, \\partial_\\phi, \\partial_z\\}`. If False, the input is in non-holonomous coordinates :math:`\\{\\textbf{e}_r, \\textbf{e}_\\phi, \\textbf{e}_z\\}`.

    Returns:
        function: Function that returns the :math:`B` field.
    """

    @wraps(f)
    def dfun(rr, *args, **kwargs):
        if not from_holonomous:
            exp = 1
        else:
            exp = 2

        a = lambda rr, *args, **kwargs: jnp.multiply(
            jnp.array([1, rr[0] ** exp, 1]), jnp.array(f(rr, *args, **kwargs))
        )
        deriv = jacfwd(a)(rr, *args, **kwargs)

        return (
            jnp.array(
                [
                    deriv[2][1] - deriv[1][2],
                    deriv[0][2] - deriv[2][0],
                    deriv[1][0] - deriv[0][1],
                ]
            )
            / rr[0]
        )

    return dfun


## Equilibrium

# Equilibrium with q-profile = sf + shear / 2 * rho^2 with rho^2 = (rr[0]-R)^2 + (rr[2]-R)^2


def psi_squared(rr: jnp.array, R: float, Z: float) -> jnp.float64:
    """
    :math:`\\psi` flux function for the squared circle equilibrium field. The squared circle equilibrium field is defined as:

    .. math::
        \\psi(r, z) = (r - R)^2 + (z - Z)^2.

    Args:
        rr (array): Position vector in cylindrical coordinates.
        R (float): R coordinate of the center of the squared circle.
        Z (float): Z coordinate of the center of the squared circle.

    Returns:
        float: :math:`\\psi` flux at position `rr`.
    """
    return (Z - rr[2]) ** 2 + (R - rr[0]) ** 2


def A_r_squared(
    rr: jnp.array, R: float, Z: float, sf: float, shear: float
) -> jnp.float64:
    """
    Vector potential :math:`A_r` component (giving the poloidal flux :math:`F`) for the squared circle equilibrium field. Using it with the :code:`psi_squared` function makes sense of :code:`sf` and :code:`shear` as the :math:`q`-factor profile becomes:

    .. math::
        q(r, z) = sf + shear / 2 * ((r - R)^2 + (z - Z)^2).

    Args:
        rr (array): Position vector in cylindrical coordinates.
        R (float): R coordinate of the center of the squared circle.
        Z (float): Z coordinate of the center of the squared circle.
        sf (float): Safety factor at :math:`\\rho = 0`.
        shear (float): Shear of the :math:`q`-profile.

    Returns:
        float: :math:`A_r` component at position `rr`.
    """

    def a(rr):
        return jnp.real(
            (1 / (4 * rr[0]))
            * (
                (
                    4 * sf
                    + shear / 2
                    * (
                        5 * rr[0] ** 2
                        - 10 * rr[0] * R
                        + 4 * R**2
                        + 2 * (rr[2] - Z) ** 2
                    )
                )
                * jnp.sqrt(-rr[0] ** 2 + 2 * rr[0] * R - (rr[2] - Z) ** 2)
                * (rr[2] - Z)
                - 1j
                * rr[0]
                * (rr[0] - 2 * R)
                * (4 * sf + (3 * rr[0] ** 2 - 6 * rr[0] * R + 4 * R**2) * shear / 2)
                * jnp.log(
                    -1j * rr[2]
                    + jnp.sqrt(-rr[0] * (rr[0] - 2 * R) - (rr[2] - Z) ** 2)
                    + 1j * Z
                )
            )
        )

    return cond(
        R**2 - (R - rr[0]) ** 2 - (Z - rr[2]) ** 2 > 0.0,
        lambda rr: a(rr),
        lambda rr: 0.0,
        jnp.array(rr),
    )


def A_squared(
    rr: jnp.array, R: float, Z: float, sf: float, shear: float
) -> jnp.ndarray:
    """
    Holonomous component of the vector potential :math:`A` for the squared circle equilibrium field.

    Args:
        rr (array): Position vector in cylindrical coordinates.
        R (float): R coordinate of the center of the squared circle.
        Z (float): Z coordinate of the center of the squared circle.
        sf (float): Safety factor at :math:`\\rho = 0`.
        shear (float): Shear of the :math:`q`-profile.

    Returns:
        array: :math:`A` field at position `rr`.
    """
    return jnp.array([0.0, psi_squared(rr, R, Z) / rr[0] ** 2, 0.0]) + jnp.array(
        [A_r_squared(rr, R, Z, sf, shear), 0.0, 0.0]
    )


# # Equilibrium with q-profile = sf + shear * b^2 with b^2 = (rr[0]-R)^2/A^2 + (rr[2]-R)^2/B^2
#
# def psi_ellipse(rr: jnp.array, R: float, Z: float, A: float, B: float) -> jnp.float64:
#     """Psi flux function for the squared ellipse equilibrium field."""
#     return (Z - rr[2]) ** 2 / B**2 + (R - rr[0]) ** 2 / A**2
#
# def A_z_ellipse(rr, R, Z, sf, shear, A, B):
#     return -0.25 * (
#         (
#             (
#                 -5 * B**2 * (rr[0] - R) ** 2 * shear
#                 + A**2
#                 * (B**2 * (-4 * sf + R**2 * shear) - 2 * shear * (rr[2] - Z) ** 2)
#             )
#             * jnp.sqrt(
#                 ((rr[0] + (-1 + A) * R) * (-rr[0] + R + A * R)) / A**2 - (rr[2] - Z) ** 2 / B**2
#             )
#             * (rr[2] - Z)
#         )
#         / (A**2 * B**2)
#         + (
#             B
#             * (rr[0] + (-1 + A) * R)
#             * (rr[0] - (1 + A) * R)
#             * (3 * (rr[0] - R) ** 2 * shear + A**2 * (4 * sf + R**2 * shear))
#             * (jnp.pi/2 - jnp.arctan(
#                 (
#                     B
#                     * jnp.sqrt(
#                         ((rr[0] + (-1 + A) * R) * (-rr[0] + R + A * R)) / A**2
#                         - (rr[2] - Z) ** 2 / B**2
#                     )
#                 )
#                 / (rr[2] - Z)
#             ))
#         )
#         / A**4
#     ) / rr[0]
#
# def F_ellipse(rr, R, Z, sf, shear, A, B):
#     """F flux function for the squared ellipse equilibrium field."""
#     temp = jnp.maximum(
#         R**2 - (Z - rr[2]) ** 2 / B**2 - (R - rr[0]) ** 2 / A**2, 0.0
#     )
#     return (
#         2 * sf + 2 * shear * ((Z - rr[2]) ** 2 / B**2 + (R - rr[0]) ** 2 / A**2)
#     ) * jnp.sqrt(temp)

## Perturbations

# Maxwell-Boltzmann distributed perturbation


def psi_maxwellboltzmann(
    rr: jnp.array,
    R: float,
    Z: float,
    d: float,
    m: int,
    n: int,
    phase_poloidal: float = 0.0,
    phase_toroidal: float = 0.0,
    A: float = 1.0,
    B: float = 1.0,
) -> jnp.float64:
    """
    Maxwell-Boltzmann distributed :math:`\\psi` flux function.

    Args:
        rr (array): Position vector in cylindrical coordinates.
        R (float): R coordinate of the center of the Maxwell-Boltzmann distribution.
        Z (float): Z coordinate of the center of the Maxwell-Boltzmann distribution.
        d (float): Standard deviation of the Maxwell-Boltzmann distribution.
        m (int): Poloidal mode number.
        n (int): Toroidal mode number.
        phase_poloidal (float): Poloidal phase of the perturbation.
        phase_toroidal (float): Toroidal phase of the perturbation.
        A (float): Scaling factor for the R coordinate.
        B (float): Scaling factor for the Z coordinate.

    Returns:
        float: :math:`\\psi` flux at position `rr`.
    """

    rho2 = (R - rr[0]) ** 2 / A**2 + (Z - rr[2]) ** 2 / B**2

    def psi_mb(rr):
        return (
            jnp.sqrt(2)
            / (jnp.sqrt(jnp.pi) * d**3)
            * rho2
            * jnp.exp(-rho2 / (2 * d**2))
            * jnp.cos(
                jnp.arctan2((rr[2] - Z) / B, (rr[0] - R) / A) * m + phase_poloidal
            )
            * jnp.cos(rr[1] * n + phase_toroidal)
        )

    return cond(
        rho2**2 > jnp.finfo(jnp.float32).tiny,
        lambda rr: psi_mb(rr),
        lambda rr: 0.0,
        jnp.array(rr),
    )


def A_maxwellboltzmann(
    rr: jnp.array,
    R: float,
    Z: float,
    d: float,
    m: int,
    n: int,
    phase_poloidal: float = 0.0,
    phase_toroidal: float = 0.0,
    A: float = 1.0,
    B: float = 1.0,
) -> jnp.ndarray:
    """
    Holonomous component of the vector potential :math:`A` for the Maxwell-Boltzmann distributed perturbation.

    Args:
        rr (array): Position vector in cylindrical coordinates.
        R (float): R coordinate of the center of the Maxwell-Boltzmann distribution.
        Z (float): Z coordinate of the center of the Maxwell-Boltzmann distribution.
        d (float): Standard deviation of the Maxwell-Boltzmann distribution.
        m (int): Poloidal mode number.
        n (int): Toroidal mode number.
        phase_poloidal (float): Poloidal phase of the perturbation.
        phase_toroidal (float): Toroidal phase of the perturbation.
        A (float): Scaling factor for the R coordinate.
        B (float): Scaling factor for the Z coordinate.

    Returns:
        array: :math:`A` field at position `rr`.
    """
    return jnp.array(
        [
            0.0,
            psi_maxwellboltzmann(
                rr, R, Z, d, m, n, phase_poloidal, phase_toroidal, A, B
            )
            / rr[0] ** 2,
            0.0,
        ]
    )


# Gaussian distributed perturbation


def psi_gaussian(
    rr: jnp.array,
    R: float,
    Z: float,
    mu: float,
    sigma: float,
    m: int,
    n: int,
    phase_poloidal: float = 0.0,
    phase_toroidal: float = 0.0,
    A: float = 1.0,
    B: float = 1.0,
) -> jnp.float64:
    """
    Gaussian distributed :math:`\\psi` flux function.

    Args:
        rr (array): Position vector in cylindrical coordinates.
        R (float): R coordinate of the center of the Gaussian distribution.
        Z (float): Z coordinate of the center of the Gaussian distribution.
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        m (int): Poloidal mode number.
        n (int): Toroidal mode number.
        phase_poloidal (float): Poloidal phase of the perturbation.
        phase_toroidal (float): Toroidal phase of the perturbation.
        A (float): Scaling factor for the R coordinate.
        B (float): Scaling factor for the Z coordinate.

    Returns:
        float: :math:`\\psi` flux at position `rr`.
    """
    rho2 = (R - rr[0]) ** 2 / A**2 + (Z - rr[2]) ** 2 / B**2

    def psi_g(rr):
        return (
            jnp.sqrt(2)
            / (2 * jnp.sqrt(np.pi) * sigma)
            * jnp.exp(-((jnp.sqrt(rho2) - mu) ** 2) / (2 * sigma**2))
            * jnp.cos(
                jnp.arctan2((rr[2] - Z) / B, (rr[0] - R) / A) * m + phase_poloidal
            )
            * jnp.cos(rr[1] * n + phase_toroidal)
        )

    return cond(
        rho2**2 > jnp.finfo(jnp.float32).tiny,
        lambda rr: psi_g(rr),
        lambda rr: 0.0,
        jnp.array(rr),
    )


def A_gaussian(
    rr: jnp.array,
    R: float,
    Z: float,
    mu: float,
    sigma: float,
    m: int,
    n: int,
    phase_poloidal: float = 0.0,
    phase_toroidal: float = 0.0,
    A: float = 1.0,
    B: float = 1.0,
) -> jnp.ndarray:
    """
    Holonomous component of the vector potential :math:`A` for the Gaussian distributed perturbation.

    Args:
        rr (array): Position vector in cylindrical coordinates.
        R (float): R coordinate of the center of the Gaussian distribution.
        Z (float): Z coordinate of the center of the Gaussian distribution.
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        m (int): Poloidal mode number.
        n (int): Toroidal mode number.
        phase_poloidal (float): Poloidal phase of the perturbation.
        phase_toroidal (float): Toroidal phase of the perturbation.
        A (float): Scaling factor for the R coordinate.
        B (float): Scaling factor for the Z coordinate.

    Returns:
        array: :math:`A` field at position `rr`.
    """
    return jnp.array(
        [
            0.0,
            psi_gaussian(
                rr, R, Z, mu, sigma, m, n, phase_poloidal, phase_toroidal, A, B
            )
            / rr[0] ** 2,
            0.0,
        ]
    )


## Circular current loop perturbation


def ellpe(m):
    """
    Complete elliptic integral of the second kind.
    """
    P_coeffs = jnp.array(
        [
            1.53552577301013293365e-4,
            2.50888492163602060990e-3,
            8.68786816565889628429e-3,
            1.07350949056076193403e-2,
            7.77395492516787092951e-3,
            7.58395289413514708519e-3,
            1.15688436810574127319e-2,
            2.18317996015557253103e-2,
            5.68051945617860553470e-2,
            4.43147180560990850618e-1,
            1.00000000000000000299e0,
        ]
    )

    Q_coeffs = jnp.array(
        [
            3.27954898576485872656e-5,
            1.00962792679356715133e-3,
            6.50609489976927491433e-3,
            1.68862163993311317300e-2,
            2.61769742454493659583e-2,
            3.34833904888224918614e-2,
            4.27180926518931511717e-2,
            5.85936634471101055642e-2,
            9.37499997197644278445e-2,
            2.49999999999888314361e-1,
        ]
    )
    x = 1 - m
    # if x <= 0.0 or x > 1.0:
    #     if x == 0.0:
    #         return 1.0
    #     else:
    #         raise ValueError("ellpe: input out of domain")
    return jnp.polyval(P_coeffs, x) - jnp.log(x) * (x * jnp.polyval(Q_coeffs, x))


def ellpk(m):
    """
    Complete elliptic integral of the first kind.
    """
    P_coeffs = jnp.array(
        [
            1.37982864606273237150e-4,
            2.28025724005875567385e-3,
            7.97404013220415179367e-3,
            9.85821379021226008714e-3,
            6.87489687449949877925e-3,
            6.18901033637687613229e-3,
            8.79078273952743772254e-3,
            1.49380448916805252718e-2,
            3.08851465246711995998e-2,
            9.65735902811690126535e-2,
            1.38629436111989062502e0,
        ]
    )

    Q_coeffs = jnp.array(
        [
            2.94078955048598507511e-5,
            9.14184723865917226571e-4,
            5.94058303753167793257e-3,
            1.54850516649762399335e-2,
            2.39089602715924892727e-2,
            3.01204715227604046988e-2,
            3.73774314173823228969e-2,
            4.88280347570998239232e-2,
            7.03124996963957469739e-2,
            1.24999999999870820058e-1,
            4.99999999999999999821e-1,
        ]
    )

    x = 1 - m
    return jnp.polyval(P_coeffs, x) - jnp.log(x) * jnp.polyval(Q_coeffs, x)


def psi_circularcurrentloop(rr: jnp.array, R: float, Z: float) -> jnp.float64:
    """
    :math:`\\psi` flux function generated at :math:`(r, \\phi, z)` by a circular current loop located at position :math:`(R, Z)`.

    Args:
        rr (array): Position vector in cylindrical coordinates.
        R (float): R coordinate of the center of the circular current loop.
        Z (float): Z coordinate of the center of the circular current loop.

    Returns:
        float: :math:`\\psi` flux at position `rr`.
    """
    alpha2 = (R - rr[0]) ** 2 + (rr[2] - Z) ** 2
    beta2 = alpha2 + 4 * R * rr[0]
    k2 = 1 - alpha2 / beta2
    E = ellpe(k2)
    K = ellpk(k2)

    # Note psi = R*A^\phi
    return rr[0] * R * ((2 - k2) * K - 2 * E) / (jnp.sqrt(beta2) * k2 * jnp.pi)


def A_circularcurrentloop(rr: jnp.array, R: float, Z: float) -> jnp.ndarray:
    """
    Contravariant component of the vector potential :math:`A` for the circular current loop perturbation.

    Args:
        rr (array): Position vector in cylindrical coordinates.
        R (float): R coordinate of the center of the circular current loop.
        Z (float): Z coordinate of the center of the circular current loop.

    Returns:
        array: :math:`A` field at position `rr`.
    """
    return jnp.array([0.0, psi_circularcurrentloop(rr, R, Z) / rr[0] ** 2, 0.0])
