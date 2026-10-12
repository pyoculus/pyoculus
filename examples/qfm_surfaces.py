"""A rational quadratic-flux-minimizing surface in the two-wave magnetic field.

Run ``python examples/qfm_surfaces.py`` or add ``--output qfm.png``.

The two-wave Hamiltonian is H=rho**2/2-k[cos(2theta-phi)/2 +
cos(3theta-2phi)/3]. Its resonances lie at iota=1/2 and 2/3. We construct a
surface with winding 1/2, using families of periodic pseudo-orbits. At k=0 it
is exactly rho=1/2. With perturbations it is an approximate magnetic surface:
field lines need not remain on it, and it is not an invariant torus certified
by an orbit-closure test. See Hudson & Dewar, arXiv:0909.2096.

This uses the working QFM.action API directly. It does not use the separate
boundary-straightening or QFMBfield APIs, which currently have known defects.
"""

import argparse
import numpy as np
from pyoculus.fields import TwoWaves
from pyoculus.solvers import QFM
from pyoculus.solvers.qfm import irfft2D


def compute():
    """Return angle grids and reconstructed (rho, theta) for two strengths."""
    ntheta, nphi = 64, 64
    vartheta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    surfaces = {}
    for k in [0.0, 0.002]:
        solver = QFM(TwoWaves(k=k), pqMpol=4, pqNtor=4)
        # action returns Fourier coefficients of rho and theta-vartheta.
        sc, ts, ss, tc = solver.action(1, 2, sguess=0.5)
        rho = irfft2D(sc, ss, ntheta, nphi)
        theta = vartheta[:, None] + irfft2D(tc, ts, ntheta, nphi)
        if not (np.isfinite(rho).all() and np.isfinite(theta).all()):
            raise RuntimeError("Nonfinite QFM surface")
        if k == 0:
            np.testing.assert_allclose(rho, 0.5, atol=1e-10)
            np.testing.assert_allclose(
                theta, np.broadcast_to(vartheta[:, None], theta.shape), atol=1e-10
            )
        surfaces[k] = (rho, theta)
    return vartheta, phi, surfaces


def plot(vartheta, phi, surfaces):
    """Compare sections at phi=0 and pi; use the reconstructed physical angle."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, j in zip(axes, [0, len(phi) // 2]):
        for k, (rho, theta) in surfaces.items():
            # Close the angular curve by appending a 2*pi-shifted endpoint.
            ax.plot(
                np.r_[theta[:, j], theta[0, j] + 2 * np.pi],
                np.r_[rho[:, j], rho[0, j]],
                label=f"k={k:g}",
            )
        ax.set(
            xlabel=r"$\theta$ [rad]",
            ylabel=r"$\rho$",
            title=rf"$\phi={phi[j]:.2f}$ rad",
        )
        ax.legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Save a figure instead of displaying it")
    args = parser.parse_args()
    fig = plot(*compute())
    if args.output:
        fig.savefig(args.output, dpi=150)
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
