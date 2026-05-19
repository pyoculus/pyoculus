from pyoculus.fields import SimsoptBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
from simsopt._core import load
from simsopt.geo import SurfaceRZFourier
from simsopt.field.tracing import compute_fieldlines
from simsopt.util import proc0_print, comm_world
from simsopt.field import (
    particles_to_vtk,
    LevelsetStoppingCriterion,
    MinRStoppingCriterion,
    MaxRStoppingCriterion,
    MinZStoppingCriterion,
    MaxZStoppingCriterion,
)
import numpy as np
import logging
import time

######################################################################
# Define the configuration from the datafile of the 0928241 coilset. #
######################################################################

logging.basicConfig(level=logging.INFO)
surfaces, ma, coils = load(f'coils/serial0928241.json')
nfp = 3

s = SurfaceRZFourier.from_nphi_ntheta(
    mpol=5,
    ntor=5,
    stellsym=True,
    nfp=nfp,
    range="full torus",
    nphi=64,
    ntheta=24,
)
s.fit_to_curve(ma, 0.7, flip_theta=False)

simsoptfield = SimsoptBfield.from_coils(coils, Nfp=3, interpolate=True, surf=s)

##########################################
# Setup the poincare plotting functions. #
##########################################

surf_classifier = simsoptfield.surfclassifier
bsh = simsoptfield._mf_B if simsoptfield._interpolating else None

tmax_fl = 1000  # "Time" for field line tracing
tol = 1e-15

stopping_criteria = [
    MaxZStoppingCriterion(0.4),
    MinZStoppingCriterion(-0.4),
    MaxRStoppingCriterion(1.7),
    MinRStoppingCriterion(0.2),
    # LevelsetStoppingCriterion(surf_classifier.dist),
]

logger = logging.getLogger("simsopt.field.tracing")
logger.setLevel(1)

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
    fig, ax = plt.subplots()
    ax.set_aspect(aspect)
    color = None
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    ax.grid(True, linewidth=0.5)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    for j in range(len(fieldlines_phi_hits)):
        if fieldlines_phi_hits[j].size == 0:
            continue
        if mark_lost:
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            color = "r" if lost else "g"
        phi_hit_codes = fieldlines_phi_hits[j][:, 1]
        indices = np.where(phi_hit_codes >= 0)[0]
        data_this_phi = fieldlines_phi_hits[j][indices, :]
        if data_this_phi.size == 0:
            continue
        r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
        color = colors[j % len(colors)]
        ax.scatter(
            r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color
        )

    return fig, ax


def trace_fieldlines(bfield, phi, RZs, label):
    phis = [phi + i * (2 * np.pi / nfp) for i in range(nfp)]

    t1 = time.time()
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield,
        RZs[:, 0],
        RZs[:, 1],
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
        return fieldlines_tys, fieldlines_phi_hits

################################################################
# Setup and compute the poincare plot using simsopt fonctions. #
################################################################

phi = 0

# Line #1
nfieldlines = 10
Rs = np.linspace(0.884, 1.2, nfieldlines)
Zs = np.zeros_like(Rs)
RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

# Line #2
nfieldlines = 60
p1 = np.array([1.09955, 0.0712])
p2 = np.array([1.4016, 0.1072])
Rs = np.linspace(p1[0], p2[0], nfieldlines)
Zs = np.linspace(p1[1], p2[1], nfieldlines)
# Rs, Zs = np.meshgrid(Rs, Zs)
RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
RZs = np.concatenate((RZs, RZs2))

# Line #3
nfieldlines = 10
p1 = np.array([1.385, 0.])
p2 = np.array([1.526, 0.])
Rs = np.linspace(p1[0], p2[0], nfieldlines)
Zs = np.linspace(p1[1], p2[1], nfieldlines)
RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
RZs = np.concatenate((RZs, RZs2))

# Line #4
nfieldlines = 10
p1 = np.array([1.4446, 0.])
p2 = np.array([1.4822, 0.])
Rs = np.linspace(p1[0], p2[0], nfieldlines)
Zs = np.linspace(p1[1], p2[1], nfieldlines)
RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
RZs = np.concatenate((RZs, RZs2))

# Compute the fieldlines and plot
label = "0928241"
fieldlines_tys, fieldlines_phi_hits = trace_fieldlines(bsh, phi, RZs, label)
fig, ax = plot_poincare_data(
    fieldlines_phi_hits,
    phi,
    __file__ + f"_{label}",
    dpi=300,
    s=1,
)
ax.set_xlim(0.3, 1.6)
ax.set_ylim(-0.35, 0.35)

##########################################################################
# Setup the pyoculus map and solve for fixedpoints and turnstile fluxes. #
##########################################################################
if comm_world is not None and comm_world.rank != 0:
    comm_world.abort()

fieldmap = CylindricalBfieldSection.without_axis(simsoptfield, guess=ma.gamma()[0,::2], rtol=1e-13)

# Finding all fixedpoints
fp_o1 = FixedPoint(fieldmap)
fp_o1.find(6, guess=[1.4446355574662593, 0.0])
fp_o2 = FixedPoint(fieldmap)
fp_o2.find(6, guess=[1.40150403, 0.10815878])
fp_x1 = FixedPoint(fieldmap)
fp_x1.find(6, guess=[1.43378117, 0.05140443])
fp_x2 = FixedPoint(fieldmap)
fp_x2.find(6, guess=[1.43378117, -0.05140443])

for fp in [fp_o1, fp_o2, fp_x1, fp_x2]:
    fp.plot(ax=ax, linewidths=1)

# data = [
#     {'r': fp_x1.x[0], 'z': fp_x1.z[0], 'GreenesResidue': fp_x1.GreenesResidue},
#     {'r': fp_x2.x[0], 'z': fp_x2.z[0], 'GreenesResidue': fp_x2.GreenesResidue},
#     {'r': fp_o1.x[0], 'z': fp_o1.z[0], 'GreenesResidue': fp_o1.GreenesResidue},
#     {'r': fp_o2.x[0], 'z': fp_o2.z[0], 'GreenesResidue': fp_o2.GreenesResidue},
# ]
# df = pd.DataFrame(data)

# Provisional fix to set the poloidal mode number of the fixed points
fp_x1._m = 6
fp_x2._m = 6
fp_x1._found_by_iota = True
fp_x2._found_by_iota = True

# Inner manifold
inner_manifold = Manifold(fieldmap, fp_x1, fp_x2, '+', '+', False)
inner_manifold.compute(eps_s = 1e-3, eps_u = 1e-3, nint_s = 3, nint_u = 3, neps_s = 30, neps_u = 30)
inner_manifold.plot(ax=ax, rm_points=7)

inner_manifold.find_clinic_single(0.001276810579762792, 0.0012768113453997163, n_s=2, n_u=2)
inner_manifold.find_clinic_single(0.005129109370459298, 0.0051291087795083574, n_s=2, n_u = 1, tol=1e-8)
inner_manifold.plot_clinics(ax=ax)

inner_manifold.compute_turnstile_areas()

# Outer manifold
outer_manifold = Manifold(fieldmap, fp_x1, fp_x2, '-', '+', True)
outer_manifold.compute(eps_s = 1e-3, eps_u = 1e-3, nint_s = 3, nint_u = 3, neps_s = 30, neps_u = 30)
outer_manifold.plot(ax=ax, rm_points=20)

outer_manifold.find_clinic_single(0.0015488037705831256, 0.0015488037607238807, n_s=2, n_u=2)
outer_manifold.find_clinic_single(0.0006060200774938109, 0.0006060193763593331, n_s=3, n_u=2)
outer_manifold.plot_clinics(ax=ax)

outer_manifold.compute_turnstile_areas()

# Print the results
B_phi_axis = simsoptfield.B([fieldmap.R0, 0., fieldmap.Z0])[1]
print("Inner area (flux devided by B^\phi_axis in Tesla): ", inner_manifold.turnstile_areas / B_phi_axis / fieldmap.R0)
print("Outer area (flux devided by B^\phi_axis in Tesla): ", outer_manifold.turnstile_areas / B_phi_axis / fieldmap.R0)

# # Save the data
# np.save("inner_areas_0928241.npy", inner_manifold.turnstile_areas)
# np.save("outer_areas_0928241.npy", outer_manifold.turnstile_areas)

# Save the figure
fig.savefig("figs/simsopt_quasr_" + label + ".png", dpi=300)