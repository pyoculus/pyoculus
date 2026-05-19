#!/usr/bin/env python

import numpy as np
from simsopt.geo import SurfaceRZFourier, SurfaceClassifier # CurveHelical, CurveXYZFourier, curves_to_vtk
from simsopt.field import BiotSavart
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           LevelsetStoppingCriterion, load_coils_from_makegrid_file,
                           MinRStoppingCriterion, MaxRStoppingCriterion,
                           MinZStoppingCriterion, MaxZStoppingCriterion,
                           compute_fieldlines
                           )
from simsopt.field import coils_via_symmetries
from simsopt.configs import get_ncsx_data
from pyoculus.fields import SimsoptBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import FixedPoint
import matplotlib.pyplot as plt

###############################################################################
# Define the NCSX cpnfiguration and set up the pyoculus problem
###############################################################################

nfp = 3 # Number of field periods
curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(curves, currents, nfp, True)

surf = SurfaceRZFourier.from_nphi_ntheta(mpol=5, ntor=5, stellsym=True, nfp=nfp, range="full torus", nphi=64, ntheta=24)
surf.fit_to_curve(ma, 0.7, flip_theta=False)
surfclassifier = SurfaceClassifier(surf, h=0.1, p=2)

mf = SimsoptBfield.from_coils(coils, Nfp=nfp, interpolate=True, surf=surf)
