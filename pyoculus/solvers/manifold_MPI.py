"""MPI-parallel versions of the expensive steps in :mod:`pyoculus.solvers.manifold`.

This module is optional and self-contained: it depends only on ``mpi4py`` (and
``numpy``/the rest of :mod:`pyoculus`), not on ``simsopt``, so ``pyoculus`` keeps
working without mpi4py installed (mirroring ``simsopt.util.mpi``'s own
self-contained-import style).

Every function here takes a plain ``comm`` argument: an ``mpi4py`` intracommunicator,
or ``None`` to run serially. In particular you can pass:

- ``None`` (default) -- pure serial execution, bit-identical to calling the
  corresponding :class:`~pyoculus.solvers.manifold.Manifold` methods directly. Works
  even without mpi4py installed.
- ``MPI.COMM_WORLD`` -- a flat MPI run.
- ``mpi.comm_groups`` from a ``simsopt.util.MpiPartition`` -- to nest this inside a
  simsopt optimization that also parallelizes across finite-difference workers
  (``MpiPartition(ngroups=W)`` with ``W`` less than the total number of ranks leaves
  spare ranks per worker group free to speed up a single turnstile-area evaluation).

**Precondition**: every rank in ``comm`` must already hold an independently
constructed, but numerically identical, :class:`Manifold` (same field, same fixed
points). This module never pickles or broadcasts the map/field/``Manifold`` object
itself -- only small, picklable numeric candidates -- matching how MPI is already used
elsewhere in this ecosystem (e.g. every rank rebuilding its own field map from the same
plain-Python inputs).
"""

import logging

import numpy as np

from .manifold import Clinic, find_zero_crossings

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

__all__ = [
    "find_clinics_exhaustive_mpi",
    "compute_turnstile_areas_mpi",
    "find_and_compute_turnstiles_mpi",
]

logger = logging.getLogger(__name__)


def _parallel_loop_bounds(comm, n):
    """Split ``range(n)`` into contiguous, per-rank slices.

    Vendored from ``simsopt._core.util.parallel_loop_bounds`` so this module does not
    need to depend on simsopt. With ``comm=None`` (or a size-1 communicator) the whole
    range ``(0, n)`` is returned, i.e. this reduces to the plain serial loop.
    """
    if comm is None:
        return 0, n
    size = comm.size
    idxs = [i * n // size for i in range(size + 1)]
    return idxs[comm.rank], idxs[comm.rank + 1]


def _gather_ordered(comm, local_list):
    """Gather ``local_list`` from every rank and concatenate in rank order.

    Because :func:`_parallel_loop_bounds` splits work contiguously (rank 0 gets the
    first slice, rank 1 the next, ...), concatenating gathered per-rank lists in rank
    order reconstructs the exact same ordered sequence a serial loop over the full
    range would have produced.
    """
    if comm is None:
        return list(local_list)
    return [item for chunk in comm.allgather(local_list) for item in chunk]


def find_clinics_exhaustive_mpi(
    manifold,
    comm=None,
    alpha=1e-3,
    first_guess_eps_s=None,
    first_guess_eps_u=None,
    reset_clinics=True,
    eps_findiff=1e-8,
    eps_res=10,
    **kwargs,
):
    """MPI-parallel equivalent of :meth:`Manifold.find_clinics_exhaustive`.

    The grid scan (stage 1) and the per-candidate root refinement (stage 2) are each
    distributed across ``comm``, since both consist of independent, expensive map
    evaluations. The stateful dedup/fundamental-domain bookkeeping in
    :meth:`ClinicSet.record_clinic` is order-dependent, so it is *not* parallelized:
    every rank replays it redundantly, in the same deterministic order (see module
    docstring / plan), giving every rank an identical, bit-for-bit reproducible
    :class:`ClinicSet` with no extra communication.

    Arguments and defaults match :meth:`Manifold.find_clinics_exhaustive` exactly;
    ``**kwargs`` is forwarded to ``Clinic.from_guess`` only, exactly as in the serial
    method (``record_clinic`` is called with no extra kwargs there too).
    """
    if reset_clinics:
        manifold.clinics.reset()

    logger.info("Starting MPI exhaustive clinic search...")

    if first_guess_eps_s is None:
        J = manifold._map.df(-1 * manifold.fixedpoint_1.m, manifold.rfp_s)
        J_plus = manifold._map.df(
            -1 * manifold.fixedpoint_1.m,
            manifold.rfp_s + eps_findiff * manifold.vector_s,
        )
        dJdv = (J_plus - J) / eps_findiff
        first_guess_eps_s = (
            alpha
            * 2
            * np.linalg.norm(J @ manifold.vector_s)
            / np.linalg.norm(dJdv @ manifold.vector_s)
        )
    if first_guess_eps_u is None:
        J = manifold._map.df(1 * manifold.fixedpoint_1.m, manifold.rfp_u)
        J_plus = manifold._map.df(
            1 * manifold.fixedpoint_1.m,
            manifold.rfp_u + eps_findiff * manifold.vector_u,
        )
        dJdv = (J_plus - J) / eps_findiff
        first_guess_eps_u = (
            alpha
            * 2
            * np.linalg.norm(J @ manifold.vector_u)
            / np.linalg.norm(dJdv @ manifold.vector_u)
        )

    guess_eps_s, guess_eps_u = first_guess_eps_s, first_guess_eps_u

    if manifold.lambda_s >= 1 or manifold.lambda_u <= 1:
        raise ValueError("The manifold is not hyperbolic: lambda_s >= 1 or lambda_u <= 1.")
    guess_eps_s *= manifold.lambda_s
    guess_eps_u /= manifold.lambda_u

    n_s, n_u = manifold.find_N(guess_eps_s, guess_eps_u, adaptive=True)

    eps_s_scan = np.geomspace(
        guess_eps_s * manifold.lambda_s, guess_eps_s / manifold.lambda_s, eps_res, endpoint=True
    )
    eps_u_scan = np.geomspace(
        guess_eps_u / manifold.lambda_u, guess_eps_u * manifold.lambda_u, eps_res, endpoint=True
    )
    eps_s_grid, eps_u_grid = np.meshgrid(eps_s_scan, eps_u_scan)
    grid_shape = eps_s_grid.shape
    eps_s_flat, eps_u_flat = eps_s_grid.ravel(), eps_u_grid.ravel()

    # --- Stage 1: grid fill, distributed by contiguous flat index range ---
    first, last = _parallel_loop_bounds(comm, eps_s_flat.size)
    local_F1, local_F2 = [], []
    for idx in range(first, last):
        eps_s, eps_u = eps_s_flat[idx], eps_u_flat[idx]
        try:
            r_s = manifold.rfp_s + eps_s * manifold.vector_s
            r_u = manifold.rfp_u + eps_u * manifold.vector_u
            r_s_final = manifold._map.f(-1 * manifold.fixedpoint_1.m * n_s, r_s)
            r_u_final = manifold._map.f(+1 * manifold.fixedpoint_1.m * n_u, r_u)
            local_F1.append(r_s_final[0] - r_u_final[0])
            local_F2.append(r_s_final[1] - r_u_final[1])
        except Exception as e:
            logger.warning(f"Error computing manifolds for eps_s={eps_s}, eps_u={eps_u}: {e}")
            local_F1.append(np.nan)
            local_F2.append(np.nan)

    F1_flat = np.array(_gather_ordered(comm, local_F1))
    F2_flat = np.array(_gather_ordered(comm, local_F2))
    F1_grid = F1_flat.reshape(grid_shape)
    F2_grid = F2_flat.reshape(grid_shape)

    # --- zero crossings: cheap, local numeric work, redundant on every rank ---
    zeros_eps_s, zeros_eps_u = find_zero_crossings(eps_s_grid, eps_u_grid, F1_grid, F2_grid)
    logger.info(f"Found {len(zeros_eps_s)} clinic points in the scan (can contain duplicates).")

    # --- Stage 2: candidate refinement, distributed by contiguous index range ---
    candidates = list(zip(zeros_eps_s, zeros_eps_u))
    first, last = _parallel_loop_bounds(comm, len(candidates))
    local_refined = []
    for eps_s, eps_u in candidates[first:last]:
        try:
            logger.info(f"Refining clinic at eps_s={eps_s}, eps_u={eps_u}")
            newclinic = Clinic.from_guess(manifold, eps_s, eps_u, n_s, n_u, expensive=False, **kwargs)
            local_refined.append((newclinic.eps_s, newclinic.eps_u, newclinic.nint_s, newclinic.nint_u))
        except Exception as e:
            logger.warning(f"Failed to refine clinic at eps_s={eps_s}, eps_u={eps_u}: {e}")

    refined = _gather_ordered(comm, local_refined)

    # --- Serial replay of the stateful dedup/fundamental-domain logic ---
    # Redundant but identical on every rank: cheap (no ODE integration), and it
    # keeps every rank's ClinicSet in sync without any extra communication.
    for eps_s, eps_u, nint_s, nint_u in refined:
        clinic = Clinic(manifold, eps_s, eps_u, nint_s, nint_u)
        manifold.clinics.record_clinic(clinic)

    logger.info(f"Number of clinics found ({len(manifold.clinics)}).")


def compute_turnstile_areas_mpi(manifold, comm=None, n_joining=100):
    """MPI-parallel equivalent of :meth:`Manifold._turnstile_areas_cylbfieldsection`.

    Requires ``manifold.clinics`` to already be populated identically on every rank in
    ``comm`` (guaranteed after :func:`find_clinics_exhaustive_mpi`, or after any
    serial clinic search performed identically on every rank). Each turnstile only
    depends on its two adjacent clinics, so the loop over turnstiles is distributed
    across ``comm`` and reassembled with an ordered gather.
    """
    if len(manifold.clinics) < 2:
        raise ValueError("Need at least two clinics to compute the turnstile area.")

    n_turnstiles = len(manifold.clinics)
    first, last = _parallel_loop_bounds(comm, n_turnstiles)

    def _integration_trajectory(num):
        if num == 0:
            return manifold.clinics[0].trajectory[:-1]
        elif num == n_turnstiles:
            return manifold.clinics[0].trajectory[1:]
        else:
            return manifold.clinics[num].trajectory

    local_areas = []
    for turnstilenum in range(first, last):
        traj1 = _integration_trajectory(turnstilenum)
        traj2 = _integration_trajectory(turnstilenum + 1)

        int1 = np.sum(manifold._compute_lagrangian_in_sections(traj1))
        int2 = np.sum(manifold._compute_lagrangian_in_sections(traj2))

        close_1_RZs = np.linspace(traj1[-1], traj2[-1], n_joining, endpoint=True)
        closing_integral_1 = manifold._AdL_integral_points(close_1_RZs, is_closed=False)

        close_2_RZs = np.linspace(traj2[0], traj1[0], n_joining, endpoint=True)
        closing_integral_2 = manifold._AdL_integral_points(close_2_RZs, is_closed=False)

        area = int1 + closing_integral_1 - int2 + closing_integral_2
        local_areas.append(area)
        logger.info(f"Turnstile area {turnstilenum} computed: {area}")

    manifold._areas = np.array(_gather_ordered(comm, local_areas))
    return manifold._areas


def find_and_compute_turnstiles_mpi(manifold, comm=None, turnstile_kwargs=None, **kwargs):
    """Convenience wrapper chaining :func:`find_clinics_exhaustive_mpi` and
    :func:`compute_turnstile_areas_mpi`, matching the common call pattern of running
    the exhaustive clinic search immediately followed by the turnstile-area
    calculation.

    ``turnstile_kwargs`` (e.g. ``n_joining``) is passed to the turnstile-area step;
    all other ``**kwargs`` are passed to the clinic search.
    """
    find_clinics_exhaustive_mpi(manifold, comm=comm, **kwargs)
    return compute_turnstile_areas_mpi(manifold, comm=comm, **(turnstile_kwargs or {}))
