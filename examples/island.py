from pyoculus.fields import AnalyticCylindricalBfield
from pyoculus.maps import CylindricalBfieldSection
from pyoculus.solvers import PoincarePlot, FixedPoint
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")
    
    #a perturbation is a dictionary with elements that depend on what 'type' it is. the maxwell-boltzmann needs the parameter 'd'. 
    maxwellboltzmann = {"m": 3, "n": -2, "d": 0.4, "type": "maxwell-boltzmann", "amplitude": 0.012, "phase_poloidal": 0}

    # Creating a pyoculus field in cylindrical coordinates. Many are supported, 
    # also for example a function that reads simulation output. 
    # Here we use the class of analytical cylindrical fields we have implemented. 
    # The docstring explains the parameters
    R0=3.
    Z0=0.
    myfield = AnalyticCylindricalBfield(R0, Z0, 1.01, 1.0, perturbations_args = [maxwellboltzmann])

    # From a field, we define a Map, by integrating field lines. 
    # The CylindricalBfieldSection does this for  us. If we want to do winding numbers, 
    # such a map needs to know the location of the magnetic axis. 
    mymap = CylindricalBfieldSection(myfield, R0=R0, Z0=Z0, tol=1e-11)

    # with a map we can do fun things by giving them to a solver, for example finding fixed points
    # or computing a PoincarePlot. We will do the first here. 
    # a PoincarePlot needs a list of starting poitns, but there are some classmethods that 
    # make generating them easier
    pplot = PoincarePlot.with_horizontal(mymap, 1.1, 40)

    # We need to run the computation. This can take long and is not parallelized, so sorry for the wait. 
    pplot.compute(npts=300)

    fig, ax = pplot.plot(marker=".", s=0.5, xlim=[2., 4.], ylim=[-1.2, 1.2])
    fig.show()

    fig, ax = plt.subplots(1,1)

    # add another higher perturbation 
    maxwellboltzmann2 = {"m": 9, "n": -2, "d": 0.4, "type": "maxwell-boltzmann", "amplitude": 0.0001, "phase_poloidal": 0.1, "R":R0, "Z":Z0}
    myfield.add_perturbation(maxwellboltzmann2)

    pplot.compute(npts=600)

    fig, ax = pplot.plot(marker=".", s=0.5, xlim=[2., 4.], ylim=[-1.2, 1.2])
    fig.show()

    # If you like the configuration, you can evaluate the covariant field
    # at any point in space using the method myfield.B([R, phi, Z]). 
    myfield.B([3.0, 0.0, 0.1])
    #or use np.meshgrid  and loop over the values to have it on a grid. 

    
    # # R-only computation
    # pplot.compute()

#    # R-Z computation
#    for amplitude in np.linspace(0.003, 0.055, 100):
#        pyoproblem.set_amplitude(0, amplitude, find_axis=False) #this pert does not move axis
#        pplot.compute(RZs)
#
#        ### Plotting the results
#
#        fig, ax = pplot.plot(marker=".", s=0.5, xlim=[2, 4.], ylim=[-1.2, 1.2])
#        ax.scatter(pyoproblem._R0, pyoproblem._Z0, marker="o",s=1, edgecolors="black", linewidths=1)
#        plt.savefig(f'2_1_island_growth_amp_{amplitude:.4g}.png', bbox_inches='tight', dpi=200)
#        plt.clf()

