# coding: utf-8
from pyoculus.fields import SpecBfield, SpectreBfield
from pyoculus.maps import ToroidalBfieldSection
from pyoculus.solvers import PoincarePlot, FixedPoint
from spectre import SPECTRE_pyoculus_helper
from spectre.postprocess.main import SPECTREout
from matplotlib import pyplot as plt    
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

out = SPECTREout('tests/test_files/w7x_1vol_spectre.h5')

spectre = SpectreBfield(SPECTRE_pyoculus_helper(out, 1))
section = ToroidalBfieldSection(spectre)
fp = FixedPoint(section)
guess=[-0.1, 0.1]
fp.find_with_iota(n=-5, m=5, guess=guess, nrestart=10, method="scipy.root")

pplot = PoincarePlot(section, np.linspace(np.array([-1, 0]), np.array([1, 0]), 5))
pplot.compute(50)
pplot.plot()
