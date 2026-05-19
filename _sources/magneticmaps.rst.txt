.. |_| unicode:: 0xA0 
   :trim:

.. math:: \newcommand{\dpmap}{\mathcal{D}\mathcal{P}}

.. math:: \newcommand{\pmap}{\mathcal{P}}

Field Line map
==============

In the study of continuous dynamical systems, it is often convenient to consider a discrete map defined by the intersection of the trajectories with a lower-dimensional subspace and which captures some properties of the continuous case. In the case of magnetically confined plasmas, the particles loop around toroidally and the intersection of their trajectories with a constant :math:`\phi` cross-section reveals the distinct region of different field line behaviour |_| : closed surfaces, islands, chaotic regions. This section is known as a `Poincaré` section and to compute it the field lines need to be integrated and record their intersection with the desired cross-section. 

.. image:: _static/images/fieldlinemap_illustrated.png
   :width: 500
   :align: center


Flow of a vector field
----------------------

Given a magnetic field :math:`\textbf{B}`, or more generally any vector field in 3D, we can define a flow :math:`\Phi(x,t)` as the evolution of a point :math:`x` when following the field from time :math:`t_0` to :math:`t`. It means that the tangent vector for fixed :math:`x` is equal to :math:`\partial_t \Phi(x,t) = \mathbf{B}(\Phi(x,t),t)`. And we can define :math:`\Phi(x,t)` as |_| :

.. math::

   \Phi(x,t) = \int\limits_{t_0}^{t}\mathbf{B}(\Phi(x,s),s)ds.

This is a system of 3 ordinary differential equations (ODEs) that can be solved numerically. To record the intersection with a :math:`\phi=\phi_0` section, the choice of coordinate is relevant. In cartesian coordinates, the current :math:`\phi(x,y)` need to be verified at every step and when it crosses the desired value, one need to interpolate or integrate to look for a more precise intersection. In cylindrical :math:`(r,\phi,z)` and toroidal :math:`(\rho,\phi,\theta)` coordinates, the problem is much simpler as the :math:`\phi` coordinate is already present. In those two cases, we need to consider the :math:`\phi` coordinate as the time variable. Therefore the vector field, that can be written |_| :

.. math::

   \textbf{B}(\Phi(x,t),t) = B^i\partial_i + B^\phi\partial_\phi \quad \text{with} \quad i \in \{r,z\}/\{\rho,\theta\},

need to have a non-vanishing :math:`B^\phi` component. In this case, the map |_| :

.. math::

    t(\phi) = \int\limits_{\phi_0}^{\phi_0+\phi} \frac{dt}{d\phi}ds = \int\limits_{\phi_0}^{\phi_0+\phi} \frac{1}{B^\phi}ds

is a well-define change of variable. Indeed :math:`\Phi(x,t)` can be reparametrized as :math:`\Phi(x,\phi) = \Phi(x,t(\phi))` and |_| :

.. math::

    \partial_\phi{\Phi}^i(\phi) = (\frac{dx^1}{d\phi},\,1,\,\frac{dx^3}{d\phi}) = (\frac{dx^1}{dt}\frac{dt}{d\phi},\,1,\,\frac{dx^3}{dt}\frac{dt}{d\phi}) = (B^1/B^\phi,\, 1,\,B^3/B^\phi).

Due to the choice of coordinates the field will have at most a :math:`\phi` periodicity of :math:`2\pi`, yet in stellarator configurations have azymuthal redundancy. Their period is :math:`T = 2\pi/n_\text{fp}` where :math:`n_\text{fp}\in\mathbb{N}^\star` is the number of field period in a complete toroidal rotation. For example, W7X and LHD have 5 and 10 field periods respectively. If the field is not periodic then :math:`n_\text{fp} = 1`.

.. image:: _static/images/poincare-torus.png
  :width: 400
  :align: center

|

The `Poincaré` section is identical for :math:`\phi_0` and :math:`\phi_i + kT`, :math:`k\in\mathbb{Z}`. Writing :math:`\Omega` the set of initial points in the :math:`\phi_i` plane for which :math:`\Phi` is effectively re-parametrizable between :math:`\phi_i` and :math:`\phi_i + T`, allows to define the map :math:`\pmap : \Omega \rightarrow \mathbb{R}_+\times\mathbb{R}` as |_| :

.. math::

    (x^1, x^2) \mapsto \pmap(x^1, x^2) = \int_{\phi_i}^{\phi_i+T}(
        B^1/B^\phi,\,
        B^3/B^\phi
    )\,ds + (x^1, x^2)
 
The point :math:`(x^1, x^2) \in \Omega` are point in the initial section and should not be confused with the :math:`\Phi(x,\phi)^1` and :math:`\Phi(x,\phi)^3` components which are the evolution of the initial point after angle :math:`\phi`.

Flux conservation
-----------------

As the evolution is performed by following :math:`\mathbf{B}` and due to the magnetic field been divergence free :math:`\nabla\cdot\textbf{B} = 0`, the flux through any surface obtained by mapping a simple path in :math:`\phi_i` to :math:`\phi_i + T` will be zero. This result in the key property of :math:`\pmap` being flux-conserving ; the flux through any closed surface :math:`\Sigma \subset \Omega` is equal to the one through :math:`\pmap(\Sigma)` |_| :

.. math::

    \iint\limits_{\Sigma}\textbf{B}\cdot\textbf{dS} = \iint\limits_{\pmap(\Sigma)}\textbf{B}\cdot\textbf{dS}.

Jacobian of :math:`\mathcal{P}`
-------------------------------

The Jacobian of the field line map as a matrix form :math:`\dpmap := \partial \pmap^{\{R, Z\}}/{\partial \{R, Z\}} \in \mathbb{R}^{2\times2}`. Here we distinguish between :math:`R, Z` in the starting plane and the general evolution around the torus :math:`r = \Phi^r, z = \Phi^z`, which is a handy abuse of notation. For instance |_| :

.. math::

    \dpmap^{R}_{\,\:R} = \frac{\partial}{\partial R}\left[\int_{\phi_i}^{\phi_i+T}\frac{B^r}{B^\phi}d\phi\right] + 1 = \int_{\phi_i}^{\phi_i+T}\partial_{R}\left[\frac{B^r}{B^\phi}\right]d\phi + 1 =\, ...

with :math:`B^r` and :math:`B^\phi` being evaluated at |_| :

.. math::

    B^r &= B^r(r(R, Z, \phi), \phi, z(R, Z, \phi))\\
    B^\phi &= B^\phi(r(R, Z, \phi), \phi, z(R, Z, \phi)).

The integrand can then be developed using the chain rule |_| :

.. math::

    \partial_{R}\left[\frac{B^r}{B^\phi}\right] = \partial_{r}\left[\frac{B^r}{B^\phi}\right]\frac{\partial r}{\partial R} + \partial_{z}\left[\frac{B^r}{B^\phi}\right]\frac{\partial z}{\partial R}

and all the components can be written as a matrix multiplication by |_| :

.. math::

    \dpmap = \int_{\phi_i}^{\phi_i+T}\begin{pmatrix}
        \partial_{r}\left[B^r/B^\phi\right] & \partial_{z}\left[B^r/B^\phi\right]\\
        \partial_{r}\left[B^z/B^\phi\right] & \partial_{z}\left[B^z/B^\phi\right]
    \end{pmatrix}\cdot\begin{pmatrix}
        \partial_{R}r & \partial_{Z}r\\
        \partial_{R}z & \partial_{Z}z
    \end{pmatrix}d\phi + \mathbb{I}_2.

For example |_| :

.. math::
    \partial_{r}\left[\frac{B^r}{B^\phi}\right] = \frac{1}{B^\phi}\frac{\partial B^r}{\partial r} - \frac{B^r}{(B^\phi)^2}\frac{\partial B^\phi}{\partial r}.

Determinant of the Jacobian
---------------------------

Using differential forms, it can be shown a relation for the determinant of the Jacobian matrix :math:`\dpmap`. If we write the flux in the form formalism, then :math:`\beta = B^\phi dR \wedge dZ` and the integral becomes |_| :

.. math::

    \iint\limits_{\Sigma}\textbf{B}\cdot\textbf{dS} = \iint\limits_{\pmap(\Sigma)}\textbf{B}\cdot\textbf{dS} \Leftrightarrow \int\limits_{\Sigma}\beta = \int\limits_{\pmap(\Sigma)}\beta = \int\limits_{\Sigma}\pmap^\star\beta

with :math:`\pmap^\star\beta` the pullback of :math:`\beta` through the field line map :math:`\pmap`. Then the flux conservation becomes :math:`\pmap^\star\beta(\pmap(x)) = \beta(x)` and using the relation between the differential forms |_| :

.. math::

    \pmap^\star\beta &= \beta_{i'j'}
    d(\dpmap^{i'}_{\,\:i}x^i)\wedge d(\dpmap^{j'}_{\,\:j}x^j)\\ &= \beta_{i'j'}\left(\dpmap^{i'}_{\,\:i}\dpmap^{j'}_{\,\:j}-\dpmap^{i'}_{\,\:j}\dpmap^{j'}_{\,\:i}\right)dx^i\wedge dx^j = \beta_{ij}\det(\dpmap)dx^i\wedge dx^j

and we see that it implies here |_| :

.. math::

    \det(\dpmap) = \beta_{R\,Z}(x)/\beta_{R\,Z}(\pmap(x)) = B^\phi(x)/B^\phi(\pmap(x))

and we got the same formula back as a direct calculation.This shows the power of differential forms. This can be generalized to any number of iteration of the map.

Integration of a vector field along a field line
------------------------------------------------

The integral of a vector field, say $\textbf{A}$, along a curve is defined by~:

.. math::

    \int_\gamma g(\textbf{A},\textbf{dl}) = \int_\gamma g(\textbf{A}(\gamma(s)),\dot{\gamma}(s))\,ds.

To get the integral along a field line curve $\dot{\gamma}$ should be replaced by the field line curve tangent vector, the same as the one optain above~:

.. math::

    \int_\gamma g(\textbf{A},\textbf{dl}) = \int_0^\phi (A^r\dot{\gamma}^r + r^2A^\phi\dot{\gamma}^\phi + A^z\dot{\gamma}^z)\,ds.

The vector field can be the vector potential. Or, to get the arc length of the field line curve for instance, one must choose $A = \dot{\gamma}/\Vert\dot{\gamma}\Vert$, then~:

.. math::

    \int_\gamma g(\textbf{A},\textbf{dl}) = \int_\gamma g(\frac{\dot{\gamma}}{\Vert\dot{\gamma}\Vert},\dot{\gamma})\,ds = \int_\gamma \Vert\dot{\gamma}\Vert\,ds = \int_\gamma \sqrt{\pm g_{\mu\nu}\dot{\gamma}^\mu\dot{\gamma}^\nu}\,ds.
