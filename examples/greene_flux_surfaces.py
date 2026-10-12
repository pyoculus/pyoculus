"""Rational approximants to an irrational magnetic surface: Greene's baseline.

Run ``python examples/greene_flux_surfaces.py`` to display the figure, or add
``--output greene.png`` to save it without opening a window.

The unperturbed two-wave Hamiltonian H=rho**2/2 gives rho'=0 and theta'=rho,
where prime denotes differentiation with respect to toroidal angle phi.
Thus each rho is an invariant surface with rotational transform iota=rho.
For a rational iota=n/m, the field line closes after m full toroidal turns.

Greene's method approaches an irrational surface through periodic orbits of
successive continued-fraction convergents. Their residue is (2-tr(M))/4, where
M is the m-turn tangent map. Here M is a shear, so every residue is zero.
This example validates that integrable limit; it does not demonstrate the
breakup threshold of a perturbed surface or turnstile transport flux.
"""

import argparse
import numpy as np
from pyoculus.fields import TwoWaves
from pyoculus.maps import ToroidalBfieldSection
from pyoculus.solvers import FixedPoint, FluxSurfaceGR


def compute():
    """Return the target transform and the convergent periodic-orbit objects."""
    target = (np.sqrt(5) - 1) / 2
    section = ToroidalBfieldSection(TwoWaves(k=0), rtol=1e-10)
    seeds = []
    # 1/2 and 2/3 are consecutive golden-mean convergents. The angle is a free
    # phase on an integrable surface, so choose the representative theta=0.
    for n, m in [(1, 2), (2, 3)]:
        fp = FixedPoint(section)
        fp.find_with_iota(n, m, np.array([n / m, 0.0]), tol=1e-9)
        if not fp.successful:
            raise RuntimeError(f"Could not locate the {n}/{m} seed orbit")
        seeds.append(fp)
    result = FluxSurfaceGR(section).GreeneMethod(target, *seeds, nexpand=6, tol=1e-9)
    for fp in result.fixedpoints:
        # For this Hamiltonian the exact answer is known without another solver.
        np.testing.assert_allclose(fp.coords[:, 0], fp.n / fp.m, atol=1e-9)
        np.testing.assert_allclose(fp.GreenesResidue, 0, atol=1e-9)
    return target, result.fixedpoints


def plot(target, fixedpoints):
    """Show the surfaces and their monodromy residues, with physical labels."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for fp in fixedpoints:
        axes[0].axhline(fp.n / fp.m, label=f"{fp.n}/{fp.m}")
        axes[0].scatter(fp.coords[:-1, 1], fp.coords[:-1, 0], s=12)
    axes[0].axhline(target, color="k", linestyle="--", label="irrational target")
    axes[0].set(xlabel=r"$\theta$ [rad]", ylabel=r"$\rho=\iota$", xlim=(0, 2 * np.pi))
    axes[0].legend()
    axes[1].plot(
        [fp.m for fp in fixedpoints], [fp.GreenesResidue for fp in fixedpoints], "o-"
    )
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set(
        xlabel="Orbit period [toroidal turns]",
        ylabel="Greene residue",
        title="Integrable shear: residue = 0",
    )
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
