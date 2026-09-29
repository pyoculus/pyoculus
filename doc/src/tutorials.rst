Tutorials
=========

These examples use analytic fields and do not require an external equilibrium
file. Run them from the repository root after installing Pyoculus. Each script
has a computation function, physical checks, and a plotting function; importing
the script does not run it or open a window.

Greene's flux-surface method
---------------------------

:download:`greene_flux_surfaces.py <../../examples/greene_flux_surfaces.py>`
follows rational approximants to the golden-mean rotational transform in the
integrable two-wave field. The radius of each surface and its zero Greene
residue are known analytically. The example explains the distinction between
this baseline and evidence for surface breakup in a perturbed field.

.. code-block:: console

   python examples/greene_flux_surfaces.py
   MPLBACKEND=Agg python examples/greene_flux_surfaces.py --output /tmp/greene.png

Quadratic-flux-minimizing surfaces
---------------------------------

:download:`qfm_surfaces.py <../../examples/qfm_surfaces.py>` uses the working
``QFM.action`` interface to construct a surface with winding number 1/2. It
compares the unperturbed field with a weakly perturbed two-wave field and
reconstructs the returned Fourier coefficients at two toroidal sections.
A QFM surface in the perturbed field is an approximate magnetic surface;
its construction does not establish that field lines remain on it.

.. code-block:: console

   python examples/qfm_surfaces.py
   MPLBACKEND=Agg python examples/qfm_surfaces.py --output /tmp/qfm.png

Current limitations
-------------------

The examples use existing solver implementations without compatibility patches.
The separate QFM boundary-straightening path currently accesses an undefined
``Nfp`` attribute. The Lyapunov solver uses ``np.NaN``, which NumPy 2 removed,
and leaves ``d0`` undefined when an explicit displacement is supplied. There
is therefore no runnable Lyapunov example for the current environment.
The physics regression suite records these defects as strict expected failures.
