from pyoculus.fields import AnalyticCylindricalBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint, Manifold, PoincarePlot
from matplotlib import pyplot as plt
import numpy as np
import logging

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
    }
)
logging.basicConfig(level=logging.DEBUG)

POINCARE_TRAJ = 5
POINCARE_ITS = 50

SHEAR = 1.2
SF = 1.16

fig, ax = plt.subplots(1, 1, figsize=(6, 6.6))

separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}

maxwellboltzmann = {
    "m": 6,
    "n": -1,
    "d": 2,
    "type": "maxwell-boltzmann",
    "amplitude": 0.32,
}


perturbedfield = AnalyticCylindricalBfield.with_new_axis(
    R=6, Z=0, sf=SF, shear=SHEAR, perturbations_args=[separatrix, maxwellboltzmann]
)
perturbedmap = CylindricalBfieldSection.without_axis(perturbedfield, guess=[6.0, 0])
axiscoords = np.array((perturbedmap.R0, perturbedmap.Z0))

xpointguess = [6.205, -4.50]
xpoint = FixedPoint(perturbedmap)
xpoint.find(t=1, guess=xpointguess)
xpoint.m = 1
xpointcoords = xpoint.coords[0]

opoint = FixedPoint(perturbedmap)
opoint.find(t=1, guess=[6., 0.])
opointcoords = opoint.coords[0]

separatrixcoords = np.array([6.0, -5.4])


pplot = PoincarePlot.with_segments(
    perturbedmap,
    [opointcoords, xpointcoords, xpointcoords, separatrixcoords, xpointcoords, xpointcoords + np.array([0.1, 0])],
    neps=[POINCARE_TRAJ, 5, 5],
    connected=False)

pplot.compute(POINCARE_ITS)
pplot.plot(ax=ax, color="xkcd:dark grey", s=1.4, linewidths=0)

perturbedmanifold = Manifold(perturbedmap, xpoint, xpoint)
perturbedmanifold.compute(
    eps_s=9e-6, eps_u=8e-6, nint_s=8, nint_u=8, neps_s=80, neps_u=80
)
perturbedmanifold.plot(ax=ax, markersize=0, lw=1.5)

ax.set_xlabel(r"$R$")
ax.set_ylabel(r"$Z$")

plt.savefig('victors_first_pplot.png', bbox_inches='tight', dpi=720)
