#!/usr/bin/env python

import os
import time
import logging
import numpy as np
import simsoptpp as sopp
from simsopt.field import BiotSavart
from simsopt.field import (
    InterpolatedField,
    SurfaceClassifier,
    particles_to_vtk,
    LevelsetStoppingCriterion,
    load_coils_from_makegrid_file,
    MinRStoppingCriterion,
    MaxRStoppingCriterion,
    MinZStoppingCriterion,
    MaxZStoppingCriterion,
)
from simsopt.geo import SurfaceRZFourier
from simsopt.util import proc0_print, comm_world
from simsopt._core.util import parallel_loop_bounds
from pyoculus.fields import SimsoptBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint
import matplotlib.pyplot as plt

###############################################################################
# Define the LHD-like coils. Coil set reproduce by M. Negalho from Y. Suzuki ‘Effect of
# Pressure Profile on Stochasticity of Magnetic Field in a Conventional Stellarator’,
# Plasma Physics and Controlled Fusion 62, no. 10 (August 2020): 104001,
# https://doi.org/10.1088/1361-6587/ab9a13
###############################################################################

coils = load_coils_from_makegrid_file("lhd.coils.txt", order=20, ppp=20)
bs = BiotSavart(coils)
R_major = 3.63  # Major radius of the helical coil
r_minor = 0.975  # Minor radius of the helical coil
nfp = 10  # Number of field periods

###############################################################################
# Set up the stopping criteria and initial conditions for field line tracing.
# Thanks to Matt Landreman for providing the useful script to generate the
# poincare plot as a tweaked version of simsopt routines.
#
# From: Matt Landreman <mattland@umd.edu>
# Sent: Tuesday, 28 May 2024 21:19
# To: Christopher Berg Smiet <christopher.smiet@epfl.ch>
# Cc: Todd Elder <telder1@umd.edu>; Ludovic Rais <ludovic.rais@epfl.ch>
# Subject: Re: Optimizing heliotron coils for a sharp separatrix
#
#
# Certainly, feel free.
#
#
# On Tue, May 28, 2024 at 11:55 AM Christopher Berg Smiet <christopher.smiet@epfl.ch> wrote:
#
# > Dear Matt, CC: Todd, Ludo
# >
# > Belatedly, I wanted to thank you very much for this script to generate the coils! I just had a meeting with Todd to get pyoculus running on this, and we are generating an example script building on your the coil generation code on our side to get things started.
# >
# > When our script is ready, can we include your code in it and upload it as a pyoculus example? That would be the easiest way to share with todd.
#
###############################################################################

tmax_fl = 100  # "Time" for field line tracing
tol = 1e-15
degree = 4  # Polynomial degree for interpolating the magnetic field

interpolant_n = 80  # Number of points in each dimension for the interpolant

# Set the range for the interpolant and the stopping conditions for field line tracing:
margin = 2
Zmax = r_minor * margin
Rmax = R_major + r_minor * margin
Rmin = R_major - r_minor * margin

# Set initial locations from which field lines will be traced:
# nfieldlines = 60
# p1 = np.array([4.8827, 0.1])
# p2 = np.array([4.8829, -0.1])
# Rs = np.linspace(p1[0], p2[0], nfieldlines)
# Zs = np.linspace(p1[1], p2[1], nfieldlines)

# Meshgrid close to the rotating X-point
# nfieldlines = 20
# Rs = np.linspace(4.8827, 4.8829, nfieldlines)
# Zs = np.linspace(-0.0001, 0.0001, nfieldlines)
# Rs, Zs = np.meshgrid(Rs, Zs)

# Meshgrid in the edge for phi=0.1*pi
nfieldlines = 20
Rs = np.linspace(4.5, 4.75, nfieldlines)
Zs = np.linspace(-0.1, 0.1, nfieldlines)
Rs, Zs = np.meshgrid(Rs, Zs)

initial_phi = 0.1 * np.pi
initial_conditions = np.array(
    [
        [r * np.cos(initial_phi), r * np.sin(initial_phi), z]
        for r, z in zip(Rs.flatten(), Zs.flatten())
    ]
)

bottom_str = (
    os.path.abspath(__file__)
    + f"  tol:{tol}  interpolant_n:{interpolant_n}  tmax:{tmax_fl}  nfieldlines: {nfieldlines} degree:{degree}"
)

# Create a rotating-ellipse surface to use as a stopping condition for field line tracing.
surf_classifier = SurfaceRZFourier.from_nphi_ntheta(
    mpol=1,
    ntor=1,
    nfp=nfp,
    nphi=400,
    ntheta=60,
    range="full torus",
)

classifier_aminor = 2
classifier_elongation = 1.5

classifier_elongation_inv = 1 / classifier_elongation
b = classifier_aminor / np.sqrt(classifier_elongation_inv)
surf_classifier.set_rc(0, 0, R_major)
surf_classifier.set_rc(1, 0, 0.5 * b * (classifier_elongation_inv + 1))
surf_classifier.set_zs(1, 0, -0.5 * b * (classifier_elongation_inv + 1))
surf_classifier.set_rc(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
surf_classifier.set_zs(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
surf_classifier.x = surf_classifier.x

# surf_classifier.to_vtk(__file__ + 'surf_classifier')
sc_fieldline = SurfaceClassifier(surf_classifier, h=0.2, p=2)

stopping_criteria = [
    MaxZStoppingCriterion(Zmax),
    MinZStoppingCriterion(-Zmax),
    MaxRStoppingCriterion(Rmax),
    MinRStoppingCriterion(Rmin),
    LevelsetStoppingCriterion(sc_fieldline.dist),
]

# Set domain for the interpolant:
rrange = (Rmin, Rmax, interpolant_n)
phirange = (0, 2 * np.pi / nfp, interpolant_n * 2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, Zmax, interpolant_n // 2)

proc0_print("Initializing InterpolatedField")
bsh = InterpolatedField(
    bs,
    degree,
    rrange,
    phirange,
    zrange,
    True,
    nfp=nfp,
    stellsym=True,  # skip=skip
)
proc0_print("Done initializing InterpolatedField.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simsopt.field.tracing")
logger.setLevel(1)


# Same as simsopt's compute_fieldlines, but take the initial locations as
# (x,y,z) values instead of (R, Z) values.
def compute_fieldlines(
    field, xyz_inits, tmax=200, tol=1e-7, phis=[], stopping_criteria=[], comm=None
):
    r"""
    Compute magnetic field lines by solving

    .. math::

        [\dot x, \dot y, \dot z] = B(x, y, z)

    Args:
        field: the magnetic field :math:`B`
        R0: list of radial components of initial points
        Z0: list of vertical components of initial points
        tmax: for how long to trace. will do roughly ``|B|*tmax/(2*pi*r0)`` revolutions of the device
        tol: tolerance for the adaptive ode solver
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, 4).
            Each row contains the time and
            the position, i.e.`[t, x, y, z]`.
        - ``res_phi_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the phi planes or
            one of the stopping criteria. Each row of the array contains
            `[time, idx, x, y, z]`, where `idx` tells us which of the `phis`
            or `stopping_criteria` was hit.  If `idx>=0`, then `phis[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
    """
    nlines = xyz_inits.shape[0]
    # assert len(R0) == len(Z0)
    # assert len(R0) == len(phi0)
    # nlines = len(R0)
    # xyz_inits = np.zeros((nlines, 3))
    # R0 = np.asarray(R0)
    # phi0 = np.asarray(phi0)
    # xyz_inits[:, 0] = R0 * np.cos(phi0)
    # xyz_inits[:, 1] = R0 * np.sin(phi0)
    # xyz_inits[:, 2] = np.asarray(Z0)
    res_tys = []
    res_phi_hits = []
    first, last = parallel_loop_bounds(comm, nlines)
    for i in range(first, last):
        res_ty, res_phi_hit = sopp.fieldline_tracing(
            field,
            xyz_inits[i, :],
            tmax,
            tol,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )
        res_tys.append(np.asarray(res_ty))
        res_phi_hits.append(np.asarray(res_phi_hit))
        dtavg = res_ty[-1][0] / len(res_ty)
        logger.debug(
            f"{i+1:3d}/{nlines}, t_final={res_ty[-1][0]}, average timestep {dtavg:.10f}s"
        )
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_phi_hits = [i for o in comm.allgather(res_phi_hits) for i in o]
    return res_tys, res_phi_hits


# Same as simsopt's plot_poincare_data, but with consistent colors between
# subplots, also plotting stellarator-symmetric points, and exploiting nfp
# symmetry to plot more points for the same length of field line tracing.
def plot_poincare_data(
    fieldlines_phi_hits,
    phis,
    filename,
    mark_lost=False,
    aspect="equal",
    dpi=300,
    xlims=None,
    ylims=None,
    s=2,
    marker="o",
):
    """
    Create a poincare plot. Usage:

    .. code-block::

        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')

    Requires matplotlib to be installed.

    """
    import matplotlib.pyplot as plt
    from math import ceil, sqrt

    plt.rc("font", size=7)
    # nrowcol = ceil(sqrt(len(phis)))
    nrowcol = 2
    plt.figure()
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    for ax in axs.ravel():
        ax.set_aspect(aspect)
    color = None
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for i in range(4):
        row = i // nrowcol
        col = i % nrowcol

        # if passed a surface, plot the plasma surface outline
        for surf in [surf_classifier]:
            cross_section = surf.cross_section(phi=phis[i])
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            g = 0.0
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c=(g, g, g))

        axs[row, col].grid(True, linewidth=0.5)

        if i != 4 - 1:
            axs[row, col].set_title(
                f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc="left", y=0.0
            )
        else:
            axs[row, col].set_title(
                f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc="right", y=0.0
            )
        if row == nrowcol - 1:
            axs[row, col].set_xlabel("$R$")
        if col == 0:
            axs[row, col].set_ylabel("$Z$")
        if col == 1:
            axs[row, col].set_yticklabels([])
        if xlims is not None:
            axs[row, col].set_xlim(xlims)
        if ylims is not None:
            axs[row, col].set_ylim(ylims)
        for j in range(len(fieldlines_phi_hits)):
            if fieldlines_phi_hits[j].size == 0:
                continue
            if mark_lost:
                lost = fieldlines_phi_hits[j][-1, 1] < 0
                color = "r" if lost else "g"
            phi_hit_codes = fieldlines_phi_hits[j][:, 1]
            condition = np.logical_and(
                phi_hit_codes >= 0, np.mod(phi_hit_codes, 4) == i
            )
            indices = np.where(condition)[0]
            data_this_phi = fieldlines_phi_hits[j][indices, :]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
            color = colors[j % len(colors)]
            axs[row, col].scatter(
                r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color
            )

            # stellarator-symmetric points:
            new_row = row
            if i == 1 or i == 3:
                new_row = 1 - row

            axs[new_row, col].scatter(
                r, -data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color
            )

        # plt.rc('axes', axisbelow=True)

    # plt.figtext(0.5, 0.995, os.path.abspath(coils_filename), ha="center", va="top", fontsize=4)
    plt.figtext(0.5, 0.005, bottom_str, ha="center", va="bottom", fontsize=6)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    return fig, axs


def trace_fieldlines(bfield, label):
    t1 = time.time()
    phis = [(i / 4) * (2 * np.pi / nfp) for i in range(4 * nfp)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield,
        initial_conditions,
        tmax=tmax_fl,
        tol=tol,
        comm=comm_world,
        phis=phis,
        stopping_criteria=stopping_criteria,
    )
    t2 = time.time()
    proc0_print(
        f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}",
        flush=True,
    )
    if comm_world is None or comm_world.rank == 0:
        # particles_to_vtk(fieldlines_tys, __file__ + f'fieldlines_{label}')
        radius = margin
        fig, axs = plot_poincare_data(
            fieldlines_phi_hits,
            phis,
            __file__ + f"_{label}.png",
            dpi=300,
            xlims=[R_major - radius, R_major + radius],
            ylims=[-radius, radius],
            s=1,
            # surf=surf1_Poincare,
        )
    return fig, axs


import datetime

label = "bsh_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
fig, axs = trace_fieldlines(bsh, label)

###############################################################################
# Searching the fixed points with pyoculus.
###############################################################################
if comm_world is not None and comm_world.rank != 0:
    comm_world.abort()

proc0_print("Setting up the problem")
simsoptfield = SimsoptBfield(nfp, bs)
pyoproblem = CylindricalBfieldSection.without_axis(simsoptfield, phi0=0.1*np.pi, guess=[3.6, 0.], rtol=1e-10)

proc0_print("Searching fixed points")
# Using checkonly we can search for fixed points only with the m=qq number
# so the value of pp is not used (and only checked <- in future implementation)
# This is why here the n=pp are conjectured from checking the angle evolution
# after qq*(2*pi/nfp) (using field periodicity)

# One of the two rotating fixed point seem to be really close to [4.8827, 0.]
# But the exponential growth is too high for the solver to converge
# fp_x1 = FixedPoint(pyoproblem)
# fp_x1.find(guess=[4.8827, 0.], pp=1, qq=2, sbegin=2, send=5.5, checkonly=True)

# Island m = 7, n = ? (10 ?)
fp_7o = FixedPoint(pyoproblem)
fp_7o.find(7, guess=[4.529755513450892, 0.0], tol=1e-13)
fp_7x = FixedPoint(pyoproblem)
fp_7x.find(7, guess=[4.450121349671787, -0.15772282315783062], tol=1e-13)

# Island m = 13, n = ? (20 ?)
fp_13o = FixedPoint(pyoproblem)
fp_13o.find(13, guess=[4.54492985895576, -0.07441064599290155], tol=1e-13)
fp_13x = FixedPoint(pyoproblem)
fp_13x.find(13, guess=[4.569643555967269, 0.0], tol=1e-13)

# Island m = 6, n = ? (10 ?)
fp_6o = FixedPoint(pyoproblem)
fp_6o.find(6, guess=[4.491994363204575, -0.1779668945266374], tol=1e-13)
fp_6x = FixedPoint(pyoproblem)
fp_6x.find(6, guess=[4.60062492610659, 0.0], tol=1e-13)

# Island m = 19, n = ? (30 ?)
fp_19o = FixedPoint(pyoproblem)
fp_19o.find(19, guess=[4.583935109909209, 0.0], tol=1e-13)
fp_19x = FixedPoint(pyoproblem)
fp_19x.find(19, guess=[4.5757101993549085, -0.038119626370894465], tol=1e-13)


colors = ["red", "green", "blue", "yellow"]
for i, fp in enumerate([fp_7o, fp_13o, fp_6o, fp_19o]):
    fp.plot(ax=axs[1, 0], edgecolors="black", linewidths=1, color=colors[i])

for i, fp in enumerate([fp_7x, fp_13x, fp_6x, fp_19x]):
    fp.plot(ax=axs[1, 0], edgecolors="black", linewidths=1, color=colors[i])
        
proc0_print(
    f"GreenesResidues for X-points:\n m=7 - {fp_7x.GreenesResidues[0]}\n m=13 - {fp_13x.GreenesResidues[0]}, \n m=6 - {fp_6x.GreenesResidues[0]}, \n m=19 - {fp_19x.GreenesResidues[0]}"
)

# Saving the whole figure with fixed points on the row 1, column 0 subplot
axs[1, 0].set_xlim(3.9, 4.7)
axs[1, 0].set_ylim(-0.5, 0.1)
fig.savefig(__file__ + f"_{label}_fixed_points.png", dpi=300)
