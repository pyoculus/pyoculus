# coding: utf-8
from pyoculus.fields import SpecBfield, SpectreBfield
from pyoculus.maps import ToroidalBfieldSection
from pyoculus.solvers import PoincarePlot, FixedPoint
from matplotlib import pyplot as plt
import numpy as np
from spectre.postprocess.main import SPECTREout
import logging

logging.basicConfig(level=logging.DEBUG)


#out = SPECout('tests/test_files/w7x_1vol_spectre.h5')
#out.input.physics.Istellsym = 1
#spec = SpecBfield(out, 2)
#sections = ToroidalBfieldSection(spec)
#pplots = PoincarePlot(sections, pp_starts)
#pplots.compute(100)
#pplots.plot(ax=ax, color=(1,0,0.))
fig, ax = plt.subplots(1,1)

spectre = SpectreBfield.from_h5_file('tests/test_files/w7x_1vol_spectre.h5', 0)
section = ToroidalBfieldSection(spectre)
pplot = PoincarePlot(section, np.linspace(np.array([-.9, 0]), np.array([0.9, 0]), 7))
pplot.compute(100)
pplot.plot(ax=ax, plottype='polar')

pp_starts = np.linspace(np.array([-.999, 0]), np.array([0.2, 0]), 10)
spectre = SpectreBfield.from_h5_file('tests/test_files/w7x_1vol_spectre.h5', 1)
section = ToroidalBfieldSection(spectre)
pplot = PoincarePlot(section, pp_starts)
pplot.compute(500)
pplot.plot(ax=ax, plottype='polar')

spectreout = SPECTREout('tests/test_files/w7x_1vol_spectre.h5')
#spectreout.plot_infaces(plot_axis=ax)

opoint = FixedPoint(section)
guess = [-0.1, 0.1]
opoint.find_with_iota(n=-5, m=5, guess=guess, nrestart=100)

opoint.plot(ax=ax, color=(1,0,0), s=10, plottype='polar')


xpoint = FixedPoint(section)
guess = [-0.3, 3.14]
xpoint.find_with_iota(n=-5, m=5, guess=guess, nrestart=100)
xpoint.plot(ax=ax, color=(1,0,0), s=10, plottype='polar')
plt.show()


