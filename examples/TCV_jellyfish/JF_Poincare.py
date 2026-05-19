import os, sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))  # remonte jusqu'à "Travail master"
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
from pyoculus.fields import AxisymmetricCylindricalGridField
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint, Manifold, PoincarePlot
import numpy as np
from matplotlib import pyplot as plt
import logging
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Poincaré plot for a TCV (EPFL) plasma equilibrium configuration with a 
# non-axisymmetric perturbation from PF coil displacements, called "jellyfish" configuration (shot 75979). 
# The code computes the fixed points and their manifolds, and plots the Poincaré section with the manifolds.

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 9,
    }
)

logging.basicConfig(level=logging.DEBUG)

###### repository path #######

repository_path = './examples/TCV_jellyfish/'

###### folder to save manifolds #######

file_manifolds = 'manifolds_P/'

manifolds_dir = os.path.join(repository_path, file_manifolds)
os.makedirs(manifolds_dir, exist_ok=True)

###### LIUQE + non-axisymmetric perturbation from PF coil displacements #######

pert_mat_file = 'JF_75979_120.mat'

##################### field loading and fixed points searching #####################

JFField = AxisymmetricCylindricalGridField.from_matlab_file(f"{repository_path}{pert_mat_file}", with_perturbation=True)

section = CylindricalBfieldSection(JFField,phi0=1.9,R0=0.88, Z0=0)

top_o = FixedPoint(section)
top_o.find(1, [0.9,0.18], method='scipy.root')
top_o_coord = top_o.coords[0]

x_point1= FixedPoint(section)
x_point1.find(1, [0.8,-0.27], method='scipy.root')
x_point1_coord = x_point1.coords[0]


x_point2= FixedPoint(section)
x_point2.find(1, [0.8,-0.57], method='scipy.root')
x_point2_coord = x_point2.coords[0]

x_point3= FixedPoint(section)
x_point3.find(1, [1.05,-0.58], method='scipy.root')
x_point3_coord = x_point3.coords[0]

x_point4= FixedPoint(section) 
x_point4.find(1, [0.75,0.65], method='scipy.root')
x_point4_coord = x_point4.coords[0]

########################## poincaré computation ##########################

if not os.path.exists(f"{repository_path}JF_75979_pp_hits.npy"):
    pplot = PoincarePlot.with_linspace(section, x_point1_coord, top_o_coord , 40)
    pplot.compute(400)
    np.save(f"{repository_path}JF_75979_pp_hits", pplot._hits)

########################## manifolds computing and saving##########################

mf_list = ['mf_1T', 'mf_1B', 'mf_2T', 'mf_2B', 'mf_3L', 'mf_3R', 'mf_4T', 'mf_4B']

if not os.path.exists(f"{repository_path}{file_manifolds}{mf_list[0]}.pkl"):
    manifold_1T = Manifold(section, x_point1, x_point1,-x_point1_coord+top_o_coord, -x_point1_coord+top_o_coord)
    manifold_1T.compute(eps_s=9e-6, eps_u=8e-6, nint_s=24, nint_u=24, neps_s=80, neps_u=240) #8 14 240
    manifold_1T.save(f"{repository_path}{file_manifolds}{mf_list[0]}.pkl")

#####top fp bottom mf#########
if not os.path.exists(f"{repository_path}{file_manifolds}{mf_list[1]}.pkl"):
    manifold_1B = Manifold(section, x_point1, x_point1)
    manifold_1B.compute(eps_s=9e-6, eps_u=8e-6, nint_s=17, nint_u=18, neps_s=80, neps_u=80)
    manifold_1B.save(f"{repository_path}{file_manifolds}{mf_list[1]}.pkl")

### bottom fp top mf
if not os.path.exists(f"{repository_path}{file_manifolds}{mf_list[2]}.pkl"):
    manifold_2T = Manifold(section, x_point2, x_point2,-x_point2_coord+top_o_coord, -x_point2_coord+top_o_coord)
    manifold_2T.compute(eps_s=9e-6, eps_u=8e-6, nint_s=32, nint_u=30, neps_s=80, neps_u=80)
    manifold_2T.save(f"{repository_path}{file_manifolds}{mf_list[2]}.pkl")

#### bottom fp bottom mf
if not os.path.exists(f"{repository_path}{file_manifolds}{mf_list[3]}.pkl"):
    manifold_2B = Manifold(section, x_point2, x_point2,x_point2_coord-top_o_coord, x_point2_coord-top_o_coord)
    manifold_2B.compute(eps_s=9e-6, eps_u=8e-6, nint_s=20, nint_u=20, neps_s=80, neps_u=80)
    manifold_2B.save(f"{repository_path}{file_manifolds}{mf_list[3]}.pkl")

#### right fp left mf
if not os.path.exists(f"{repository_path}{file_manifolds}{mf_list[4]}.pkl"):
    manifold_3L = Manifold(section, x_point3, x_point3,-x_point3_coord+top_o_coord, -x_point3_coord+top_o_coord)
    manifold_3L.compute(eps_s=9e-6, eps_u=8e-6, nint_s=33, nint_u=30, neps_s=80, neps_u=80)
    manifold_3L.save(f"{repository_path}{file_manifolds}{mf_list[4]}.pkl")

# ##### right fp right mf
if not os.path.exists(f"{repository_path}{file_manifolds}{mf_list[5]}.pkl"):
    manifold_3R = Manifold(section, x_point3, x_point3,x_point3_coord-top_o_coord, x_point3_coord-top_o_coord)
    manifold_3R.compute(eps_s=9e-6, eps_u=8e-6, nint_s=20, nint_u=30, neps_s=80, neps_u=80)
    manifold_3R.save(f"{repository_path}{file_manifolds}{mf_list[5]}.pkl")

# top fp top mf
if not os.path.exists(f"{repository_path}{file_manifolds}{mf_list[6]}.pkl"):
    manifold_4T = Manifold(section, x_point4, x_point4,x_point4_coord-top_o_coord, x_point4_coord-top_o_coord)
    manifold_4T.compute(eps_s=9e-6, eps_u=8e-6, nint_s=8, nint_u=8, neps_s=80, neps_u=80)
    manifold_4T.save(f"{repository_path}{file_manifolds}{mf_list[6]}.pkl")

# # ##### over the top fp bottom mf
if not os.path.exists(f"{repository_path}{file_manifolds}{mf_list[7]}.pkl"):
    manifold_4B = Manifold(section, x_point4, x_point4,-x_point4_coord+top_o_coord, -x_point4_coord+top_o_coord)
    manifold_4B.compute(eps_s=9e-6, eps_u=8e-6, nint_s=8, nint_u=30, neps_s=80, neps_u=80)
    manifold_4B.save(f"{repository_path}{file_manifolds}{mf_list[7]}.pkl")

############################ POINCARé PLOT ###############################################################################

fig, ax = plt.subplots(1, 1, figsize=(5, 8))

### poincaré plotting ###

Hits=np.load(f"{repository_path}JF_75979_pp_hits.npy")
ax.scatter(Hits[:,:, 0], Hits[:,:, 1], color="xkcd:dark grey", s=0.6, linewidths=0)

##### fixed-point plotting ####

top_o.plot(ax=ax, marker='o', color="xkcd:black", label=None,zorder=10)
x_point1.plot(ax=ax, marker='x',s=80, color="xkcd:black", label='X-point',zorder=10)
x_point2.plot(ax=ax, marker='x',s=80, color="xkcd:black", label=None,zorder=10)
x_point3.plot(ax=ax, marker='x',s=80, color="xkcd:black", label=None,zorder=10)
x_point4.plot(ax=ax, marker='x',s=80, color="xkcd:black", label=None,zorder=10)

###  manifold plotting  ####

manifs = {name: {name} for name in mf_list}

for i in mf_list:
     manifs[i] = Manifold.load(f"{repository_path}{file_manifolds}{i}.pkl")
     manifs[i].plot(stepsize_limit=0.05, ax=ax, markersize=0, lw=0.5,colors=["rosybrown", "xkcd:red"],labels=['stable MF','unstable MF'] if i=='mf_1T' else [None,None])

### figure settings ###

ax.set_xlim(0.62, 1.15)
ax.set_ylim(-0.75, 0.75)
ax.set_xlabel(r"$R[m]$")
ax.set_ylabel(r"$Z[m]$")
ax.set_aspect('equal') 
ax.set_title('( 75979 / 1.2 s / 1.9 rad)')
ax.legend(loc='upper right', bbox_to_anchor=(1., 1), ncol=1, fontsize=9)
plt.show()

