# class for generating the (weighted) Quadratic Flux Minimising (QFM) surfaces

import pyoculus.fields as fields
import numpy as np

nax = np.newaxis


class QFM:
    def __init__(
        self,
        bfield: fields.ToroidalBfield,
        pqMpol=8,
        pqNtor=4,
        nfft_multiplier=2,
        stell_sym=True,
        ntheta=100,
    ):
        """
        Set up the QFM solver

        Args:
            bfield (ToroidalBfield): the toroidal bfield object
            pqMpol (int): the Fourier resolution multiplier for poloidal direction
            pqNtor (int): the Fourier resolution multiplier for toroidal direction
            nfft_multiplier (int): the extended (multiplier) resolution for FFT
            stell_sym (bool): stellarator symmetry
        """
        self._bfield = bfield
        self._MM = nfft_multiplier * 2
        self.nfft_multiplier = nfft_multiplier
        self._pqNtor = pqNtor
        self._pqMpol = pqMpol
        self.sym = stell_sym

    def construct_qfms(
        self,
        plist,
        qlist,
        sguesslist=None,
        bounding_surfaces=[0.0, 1.0],
        rho_label="s",
        niter=5,
        verbose=True,
    ):
        """! Construct the QFM surfaces for given list of p and q
        This subroutine will iteratively construct the QFM surfaces for the given list of p and q.
        If the system has boundaries with constants s that are flux surfaces, two bounding surfaces will be added.
        Otherwise, the outmost two QFM surfaces will be the new bounding surfaces.
        @param plist the list of p
        @param qlist the list of q
        @param sguesslist guess of the approximate s coordinate of each surface
        @param bounding_surfaces instruct to add the bounding surfaces for these s coordinates, maximum two
        @param rho_label what to use for the new radius label \f$\rho\f$. Can be 's', 'tflux', 'sqrttflux'.
        @param niter the number of iterations in constructing QFM based of straight field line coordinates
        @param verbose if True, print the progress

        @returns an instance of pyoculus.problems.SurfacesToroidal containing the intepolation coordinates.
        """
        from pyoculus.geo import SurfacesToroidal

        surfaces = SurfacesToroidal(
            mpol=self._pqMpol, ntor=self._pqNtor, nsurfaces=2, stellar_sym=self.sym
        )
        # adding bounding surfaces if needed
        if bounding_surfaces is not None:
            for i, s in enumerate(bounding_surfaces):
                surfaces.scn[i, 0, 0] = s
                if rho_label == "s":
                    surfaces.rhosurfs[i] = s
            self.straighten_boundary(surfaces)

        # now construct the given list of QFMs
        for i in range(len(plist)):
            if sguesslist is not None:
                scn, tsn, ssn, tcn = self.action(plist[i], qlist[i], sguesslist[i])
            else:
                scn, tsn, ssn, tcn = self.action(plist[i], qlist[i])

            if rho_label == "s":
                rho = scn[0, 0]
            else:
                raise ValueError("unsupported rho_label")

            surfaces.add_surface(rho, scn, tsn, ssn, tcn)

            if verbose:
                print(str(i + 1) + "/" + str(len(plist)) + " completed.")

        return surfaces

    def compute_tflux():
        pass

    def straighten_boundary(self, surfaces, tol=1e-9, niter=10):
        """! Convert a boundary surface to have straight field line
        For a boundary surface with rho=constant, find the function \f$\lambda(\vartheta, \zeta)\f$ with transformation
        \f[
            \theta = \vartheta + \lambda(\vartheta, \zeta),
        \f]
        such that \f$B^{\vartheta} / B^{\zeta} = 1 / q\f$ is a constant on the surface.

        The transformation gives
        \f[
            B^{\vartheta} = \frac{B^\theta - \lambda_\zeta B^\zeta}{1 + \lambda_\vartheta},
        \f]
        and no change to \f$B^\zeta\f$.
        Therefore, we get
        \f[
            \frac{B^\theta}{B^\zeta}(\theta = \vartheta + \lambda(\vartheta, \zeta), \zeta) = \frac{1}{q} (1 + \lambda_\vartheta) + \lambda_\zeta
        \f]
        A fixed-point iteration method will be used.
        For a initial guess of \f$\lambda \f$, we use it to compute the left-hand-side.
        An updated \f$\lambda \f$ is then computed by solving the right-hand-side assuming the left-hand-side is given.
        This new \f$\lambda \f$ is again substituted into the left hand side to generate a new right-hand-side.
        The process is repeated until converged (the difference between two iterations is lower than a threshold, or a given number of iteration is reached).
        @param surfaces class SurfacesToroidal, the surfaces for which the first and last surface are bounding surfaces
        @param tol the tolerance to stop iteration
        @param niter the number of iterations
        """
        rho0 = surfaces.scn[0, 0, 0]
        iota, lambda_sn, lambda_cn = self._straighten_boundary(
            rho0, tol=tol, niter=niter
        )
        surfaces.replace_surface(0, tsn=lambda_sn, tcn=lambda_cn)

        rho0 = surfaces.scn[-1, 0, 0]
        iota, lambda_sn, lambda_cn = self._straighten_boundary(
            rho0, tol=tol, niter=niter
        )
        surfaces.replace_surface(-1, tsn=lambda_sn, tcn=lambda_cn)

    def _straighten_boundary(self, rho=1, tol=1e-9, niter=10):
        """! Convert a boundary surface to have straight field line, internal function
        For a boundary surface with rho=constant, find the function \f$\lambda(\vartheta, \zeta)\f$ with transformation
        @param rho the boundary surface coordinate
        @param tol the tolerance to stop iteration
        @param niter the number of iterations
        @returns tsn, tcn  the sine and cosine coefficient of the map \f$\lambda\f$ in \f$\theta = \vartheta + \lambda(\vartheta, \zeta)\f$
        """
        mpol = self._pqMpol
        ntor = self._pqNtor

        nfft_theta = self._MM * mpol
        nfft_zeta = self._MM * ntor

        rarr = np.ones([1]) * rho
        zarr = np.linspace(0, np.pi * 2, nfft_zeta, endpoint=False)

        rarr = np.broadcast_to(
            rarr[:, np.newaxis, np.newaxis], [1, nfft_theta, nfft_zeta]
        )
        zarr = np.broadcast_to(
            zarr[np.newaxis, np.newaxis, :], [1, nfft_theta, nfft_zeta]
        )

        lambda_cn = np.zeros([mpol + 1, 2 * ntor + 1])
        lambda_sn = np.zeros([mpol + 1, 2 * ntor + 1])
        iota = 0

        mlist = np.arange(0, mpol + 1)
        nlist = np.concatenate([np.arange(0, ntor + 1), np.arange(-ntor, 0)]) * self.Nfp

        for i in range(niter):
            iota_old = iota
            lambda_cn_old = lambda_cn.copy()
            lambda_sn_old = lambda_sn.copy()

            lambda_real = irfft2D(lambda_cn, lambda_sn, nfft_theta, nfft_zeta)
            tarr = (
                np.linspace(0, np.pi * 2, nfft_theta, endpoint=False)[:, nax]
                + lambda_real
            )

            B = self._bfield.B_many(rarr.flatten(), tarr.flatten(), zarr.flatten(), input1D=True)

            Bs = np.reshape(B[:, 0], [nfft_theta, nfft_zeta])
            Bt = np.reshape(B[:, 1], [nfft_theta, nfft_zeta])
            Bz = np.reshape(B[:, 2], [nfft_theta, nfft_zeta])

            Bt_over_Bz = Bt / Bz

            cn, sn = rfft2D(Bt_over_Bz, mpol, ntor)

            iota = cn[0, 0]

            lambda_cn[0, 1:] = sn[0, 1:] / (-mlist[0, nax] * iota + nlist[nax, 1:])
            lambda_sn[0, 1:] = cn[0, 1:] / (+mlist[0, nax] * iota - nlist[nax, 1:])
            lambda_cn[1:, :] = sn[1:, :] / (-mlist[1:, nax] * iota + nlist[nax, :])
            lambda_sn[1:, :] = cn[1:, :] / (+mlist[1:, nax] * iota - nlist[nax, :])
            lambda_cn[0, 0] = 0
            lambda_sn[0, 0] = 0

            erriota = np.abs(iota - iota_old)
            errcn = np.max(np.abs(lambda_cn - lambda_cn_old))
            errsn = np.max(np.abs(lambda_sn - lambda_sn_old))

            if np.max([erriota, errcn, errsn]) < tol:
                break

        return iota, lambda_sn, lambda_cn

    def action(self, pp: int, qq: int, sguess=0.5, root_method="hybr", tol=1e-8):
        """! Construct a QFM surface based on Stuart's method, for a given orbit periodicity pp and qq.
        The surface in the old coordinate \f$(s, \theta, \zeta)\f$ will be expressed in Fourier harmonics of the reverse mapping
        \f[
            s(\vartheta, \zeta) = \sum_{m,n} s_{c, m, n} \cos(m \vartheta - n \zeta) + s_{s, m, n} \sin(m \vartheta - n \zeta),
        \f]
        \f[
            \theta(\vartheta, \zeta) = \vartheta + \sum_{m,n} t_{c, m, n} \cos(m \vartheta - n \zeta) + t_{s, m, n} \sin(m \vartheta - n \zeta),
        \f]
        @param pp the orbit peroidicity. The rotational transform of such a surface will be pp/qq.
        @param qq the orbit peroidicity. The rotational transform of such a surface will be pp/qq.
        @param sguess a guess of the s coordinate for the surfaces.
        @param root_method root finding method being used in scipy.optimize.root
        @param tol the tolarence of root finding
        @returns scn_surf, tsn_surf, ssn_surf, tcn_surf - the Fourier harmonics of the surface transformation.
        """
        from scipy.optimize import root

        # shorthand
        MM = self._MM
        pqNtor = self._pqNtor
        pqMpol = self._pqMpol
        iota = pp / qq

        ## The number of toroidal modes
        qN = qq * pqNtor
        ## The number of action curves with different area poloidally to be found, note that this has nothing to do with pqMpol
        fM = MM * pqNtor
        ## The number of points poloidally after folding the curves
        qfM = qq * MM * pqNtor
        ## The number of toroidal points for action gradient calculation
        Nfft = MM * qq * pqNtor
        ## The zeta distance between toroidal points
        dz = 2 * np.pi / (MM * pqNtor)
        self._dz = dz
        ## The theta distance between poloidal action curves
        dt = (np.pi * 2 / qq) / fM

        self._nlist = np.arange(0, qN + 1)
        self._zeta = np.arange(0, Nfft) * dz
        self._nzq = self._nlist[:, nax] * self._zeta[nax, :] / qq
        self._cnzq = np.cos(self._nzq)
        self._snzq = np.sin(self._nzq)

        rcnarr = np.zeros([fM, qN + 1])
        tcnarr = np.zeros([fM, qN + 1])
        rsnarr = np.zeros([fM, qN + 1])
        tsnarr = np.zeros([fM, qN + 1])
        nvarr = np.zeros(fM)

        # test if problem.dBdX_many is implemented
        try:
            zero_coords = np.zeros([1])
            B, dBdX = self._bfield.dBdX_many(zero_coords, zero_coords, zero_coords, input1D=True)
        except NotImplementedError:
            use_jacobi = False
        else:
            use_jacobi = True

        for jpq in range(fM):

            a = jpq * dt

            if jpq == 0:
                nv0 = 0
                rcn0 = np.zeros(qN + 1)
                tcn0 = np.zeros(qN + 1)
                rsn0 = np.zeros(qN + 1)
                tsn0 = np.zeros(qN + 1)
                rcn0[0] = sguess
                tcn0[0] = 0
            else:
                nv0 = nvarr[jpq - 1].copy()
                rcn0 = rcnarr[jpq - 1, :].copy()
                tcn0 = tcnarr[jpq - 1, :].copy()
                rsn0 = rsnarr[jpq - 1, :].copy()
                tsn0 = tsnarr[jpq - 1, :].copy()
                tcn0[0] += dt

            xx0 = self._pack_dof(nv0, rcn0, tsn0, rsn0, tcn0)
            if not use_jacobi:
                sol = root(
                    self.action_gradient,
                    xx0,
                    args=(pp, qq, a, qN, Nfft),
                    method=root_method,
                    tol=tol,
                )
            else:
                sol = root(
                    self.action_gradient,
                    xx0,
                    args=(pp, qq, a, qN, Nfft),
                    method=root_method,
                    jac=self.action_gradient_jacobi,
                    tol=tol,
                )
            success = sol.success
            if success:
                nv, rcn, tsn, rsn, tcn = self._unpack_dof(sol.x, qN)
            else:
                raise RuntimeError(
                    "QFM orbit for pp="
                    + str(pp)
                    + ",qq="
                    + str(qq)
                    + ",a="
                    + str(a)
                    + " not found."
                )

            rcnarr[jpq, :] = rcn
            tsnarr[jpq, :] = tsn
            tcnarr[jpq, :] = tcn
            rsnarr[jpq, :] = rsn
            nvarr[jpq] = nv

        # wrap the lines into a 2D surface in (alpha, zeta)
        r = irfft1D(rcnarr, rsnarr, self.nfft_multiplier)
        z = np.linspace(0, 2 * qq * np.pi, r.shape[-1], endpoint=False)
        # Note that we will remove the DC part ~ alpha and the p/q*zeta part which will be counted seperately
        tcnarr[:, 0] = 0
        t = irfft1D(tcnarr, tsnarr, self.nfft_multiplier)

        r2D_alpha = np.zeros([qfM, Nfft])
        t2D_alpha = np.zeros([qfM, Nfft])

        for i in range(qq):
            idx = np.mod(pp * i, qq)
            r2D_alpha[idx * fM : (idx + 1) * fM, 0 : (qq - i) * pqNtor * MM] = r[
                :, i * pqNtor * MM :
            ]
            r2D_alpha[idx * fM : (idx + 1) * fM, (qq - i) * pqNtor * MM :] = r[
                :, 0 : i * pqNtor * MM
            ]
            t2D_alpha[idx * fM : (idx + 1) * fM, 0 : (qq - i) * pqNtor * MM] = t[
                :, i * pqNtor * MM :
            ]
            t2D_alpha[idx * fM : (idx + 1) * fM, (qq - i) * pqNtor * MM :] = t[
                :, 0 : i * pqNtor * MM
            ]

        r2D_vartheta = np.zeros([qfM, MM * pqNtor])
        t2D_vartheta = np.zeros([qfM, MM * pqNtor])

        # convert (alpha, zeta) into (vartheta, zeta), knowing alpha + p/q * zeta = vartheta
        # Therefore, for the same theta, as we move in zeta, alpha should decrease p/q * zeta
        # the alpha angle inteval is 2pi / fM / q, the zeta inteval is 2pi / fM
        # therefore if we move one grid in zeta, we should move -p grids in alpha
        for i in range(MM * pqNtor):
            idx = np.mod(np.arange(0, qfM) - i * pp, qfM)
            r2D_vartheta[:, i] = r2D_alpha[idx, i]
            t2D_vartheta[:, i] = t2D_alpha[idx, i]

        scn_surf, ssn_surf = rfft2D(r2D_vartheta, pqMpol, pqNtor)
        tcn_surf, tsn_surf = rfft2D(t2D_vartheta, pqMpol, pqNtor)

        return scn_surf, tsn_surf, ssn_surf, tcn_surf

    def action_gradient(self, xx, pp, qq, a, qN, Nfft):
        """
        Compute the action gradient, used in root finding.

        Args:
            xx (np.array): The packed degrees of freedom. It should contain rcn, tsn, rsn, tcn, nv.
            pp (int): The poloidal periodicity of the island.
            qq (int): The toroidal periodicity of the island.
            a (float): The target area.

        Returns:
            np.array: The equations to find zeros.

        Notes:
            Construct the Fourier transform of :math:`B^\\vartheta_i / B^\\zeta_i` and :math:`B^\\rho_i / B^\\zeta_i + \\bar \\nu / (J B^\\zeta_i)`,

            .. math::
                B^\\vartheta / B^\\zeta = f^c_0 + \\sum_{n=1}^{qN} \\left[ f^c_n \\cos(n\\z/q) + f^s_n \\sin(n\\z/q) \\right]

            .. math::
                B^\\rho / B^\\zeta + \\bar \\nu / \\sqrt{g} B^\\zeta = g^c_0 + \\sum_{n=1}^{qN} \\left[ g^c_n \\cos(n\\z/q) + g^s_n \\sin(n\\z/q) \\right]

            The Fourier harmonics of :math:`\\dot\\rho` and :math:`\\dot\\vartheta` are given directly by,

            .. math::
                \\dot \\vartheta(\\z) = p/q + \\sum_{n=1}^{qN} \\left[ - \\vartheta^c_n \\sin(n\\z/q) + \\vartheta^s_n \\cos(n\\z/q) \\right](n/q)

            .. math::
                \\dot \\rho(\\z) = \\sum_{n=1}^{qN} \\left[ - \\rho^c_n \\sin(n\\z/q) + \\rho^s_n \\cos(n\\z/q) \\right] (n/q)
        """
        # shorthand
        iota = pp / qq

        # unpack dof
        nv, rcn, tsn, rsn, tcn = self._unpack_dof(xx, qN)

        r = irfft1D(rcn, rsn, self.nfft_multiplier)
        t = irfft1D(tcn, tsn, self.nfft_multiplier)
        z = self._zeta
        t += iota * z

        # area = ( np.sum( t ) + np.pi * pp) * self._dz / (qq*2*np.pi) - pp * np.pi
        area = tcn[0]

        B = self._bfield.B_many(r, t, z, input1D=True)

        Br = B[:, 0]
        Bt = B[:, 1]
        Bz = B[:, 2]

        # if not self._problem.has_jacobian:
        #     jacobian = self._problem.jacobian_many(r, t, z, input1D=True)
        #     Bz = Bz * jacobian

        rhs_tdot = Bt / Bz
        rhs_rdot = Br / Bz - nv / Bz

        rhs_rdot_fft_cos, rhs_rdot_fft_sin = rfft1D(rhs_rdot)
        rhs_tdot_fft_cos, rhs_tdot_fft_sin = rfft1D(rhs_tdot)

        # now pack the function values
        ff = np.zeros_like(xx)

        ff[0] = area - a
        ff[1 : qN + 2] = rsn * self._nlist / qq - rhs_rdot_fft_cos[0 : qN + 1]
        ff[qN + 2 : 2 * qN + 2] = (
            -rcn * self._nlist / qq - rhs_rdot_fft_sin[0 : qN + 1]
        )[1:]
        ff[2 * qN + 2 : 3 * qN + 3] = (
            tsn * self._nlist / qq - rhs_tdot_fft_cos[0 : qN + 1]
        )
        ff[2 * qN + 2] += iota
        ff[3 * qN + 3 :] = (-tcn * self._nlist / qq - rhs_tdot_fft_sin[0 : qN + 1])[1:]

        return ff

    def action_gradient_jacobi(self, xx, pp, qq, a, qN, Nfft):
        """
        Compute the Jacobian matrix of the action gradient, used in root finding.

        Args:
            xx (np.array): The packed degrees of freedom. It should contain rcn, tsn, rsn, tcn, nv.
            pp (int): The poloidal periodicity of the island.
            qq (int): The toroidal periodicity of the island.
            a (float): The target area.
            qN (int): Fourier resolution.
            Nfft (int): Number of FFT points.

        Returns:
            np.array: The Jacobian matrix.
        """

        # shorthand
        iota = pp / qq

        # unpack dof
        nv, rcn, tsn, rsn, tcn = self._unpack_dof(xx, qN)

        r = irfft1D(rcn, rsn, self.nfft_multiplier)
        t = irfft1D(tcn, tsn, self.nfft_multiplier)
        z = self._zeta
        t += iota * z

        # area = ( np.sum( t ) + np.pi * pp) * self._dz / (qq*2*np.pi) - pp * np.pi
        area = tcn[0]

        B, dBdX = self._bfield.dBdX_many(r, t, z, input1D=True)

        Br = B[:, 0]
        Bt = B[:, 1]
        Bz = B[:, 2]

        dBrdr = dBdX[:, 0, 0]
        dBrdt = dBdX[:, 1, 0]
        dBtdr = dBdX[:, 0, 1]
        dBtdt = dBdX[:, 1, 1]
        dBzdr = dBdX[:, 0, 2]
        dBzdt = dBdX[:, 1, 2]

        # if not self._problem.has_jacobian:
        #     jacobian = self._problem.jacobian_many(r, t, z, input1D=True)
        #     Bz = Bz * jacobian

        oBz = 1 / Bz

        # rhs_tdot = Bt / Bz
        # rhs_rdot = Br / Bz - nv / Bz

        rhs_tdot_dr = dBtdr / Bz - Bt / Bz ** 2 * dBzdr
        rhs_tdot_dt = dBtdt / Bz - Bt / Bz ** 2 * dBzdt
        rhs_rdot_dr = dBrdr / Bz - (Br - nv) / Bz ** 2 * dBzdr
        rhs_rdot_dt = dBrdt / Bz - (Br - nv) / Bz ** 2 * dBzdt

        oBz_cos, oBz_sin = rfft1D(oBz)
        tdot_drcn = rhs_tdot_dr[nax, :] * self._cnzq
        tdot_drsn = rhs_tdot_dr[nax, :] * self._snzq
        tdot_dtcn = rhs_tdot_dt[nax, :] * self._cnzq
        tdot_dtsn = rhs_tdot_dt[nax, :] * self._snzq
        rdot_drcn = rhs_rdot_dr[nax, :] * self._cnzq
        rdot_drsn = rhs_rdot_dr[nax, :] * self._snzq
        rdot_dtcn = rhs_rdot_dt[nax, :] * self._cnzq
        rdot_dtsn = rhs_rdot_dt[nax, :] * self._snzq

        dummy_nv = np.zeros([1, tdot_drcn.shape[-1]])
        dtdot = self._pack_dof(dummy_nv, tdot_drcn, tdot_dtsn, tdot_drsn, tdot_dtcn) - 1
        drdot = self._pack_dof(dummy_nv, rdot_drcn, rdot_dtsn, rdot_drsn, rdot_dtcn) - 1
        dtdot_cos, dtdot_sin = rfft1D(dtdot)
        drdot_cos, drdot_sin = rfft1D(drdot)
        dtdot_cos = dtdot_cos.T
        dtdot_sin = dtdot_sin.T
        drdot_cos = drdot_cos.T
        drdot_sin = drdot_sin.T

        # now pack the function values
        ff = np.zeros([xx.size, xx.size])

        ff[1 : qN + 2, :] = -drdot_cos[0 : qN + 1]
        ff[qN + 2 : 2 * qN + 2, :] = -drdot_sin[1 : qN + 1]
        ff[2 * qN + 2 : 3 * qN + 3, :] = -dtdot_cos[0 : qN + 1]
        ff[3 * qN + 3 :, :] = -dtdot_sin[1 : qN + 1]

        ff[0, 3 * qN + 2] += 1  # tcn[0]
        ff[np.arange(1, qN + 2), np.arange(2 * qN + 1, 3 * qN + 2)] += (
            self._nlist / qq
        )  # rsn
        ff[np.arange(qN + 2, 2 * qN + 2), np.arange(2, qN + 2)] += -(self._nlist / qq)[
            1:
        ]  # -rcn[1:]
        ff[np.arange(2 * qN + 2, 3 * qN + 3), np.arange(qN + 1, 2 * qN + 2)] += (
            self._nlist / qq
        )  # tsn
        ff[np.arange(3 * qN + 3, 4 * qN + 3), np.arange(3 * qN + 3, 4 * qN + 3)] += -(
            self._nlist / qq
        )[
            1:
        ]  # -tcn[1:]

        # derivative wrt nv
        ff[1 : qN + 2, 0] += oBz_cos[0 : qN + 1]
        ff[qN + 2 : 2 * qN + 2, 0] += oBz_sin[1 : qN + 1]

        return ff

    def _unpack_dof(self, xx, qN):
        """
        Unpack the degrees of freedom into Fourier harmonics.

        Args:
            xx (np.array): The packed degrees of freedom.
            qN (int): Fourier resolution.

        Returns:
            tuple: A tuple containing:
                - nv (int): The first degree of freedom.
                - rcn (np.array): The cosine components for r.
                - tsn (np.array): The sine components for t.
                - rsn (np.array): The sine components for r.
                - tcn (np.array): The cosine components for t.
        """
        nv = xx[0] - 1
        rcn = xx[1 : qN + 2] - 1
        tsn = np.concatenate([[0], xx[qN + 2 : 2 * qN + 2] - 1])
        rsn = np.concatenate([[0], xx[2 * qN + 2 : 3 * qN + 2] - 1])
        tcn = xx[3 * qN + 2 :] - 1

        return nv, rcn, tsn, rsn, tcn

    def _pack_dof(self, nv, rcn, tsn, rsn, tcn):
        """
        Pack the degrees of freedom from Fourier harmonics.

        Args:
            nv (int): The first degree of freedom.
            rcn (np.array): The cosine components for r.
            tsn (np.array): The sine components for t.
            rsn (np.array): The sine components for r.
            tcn (np.array): The cosine components for t.

        Returns:
            np.array: The packed degrees of freedom.
        """
        xx = np.concatenate([np.atleast_1d(nv), rcn, tsn[1:], rsn[1:], tcn], axis=0) + 1

        return xx

## Fourier transform functions

def rfft1D(f):
    """
    Perform 1D Fourier transform from real space to cosine and sine.

    Args:
        f (np.array): The data in real space. If f is 2D, then the last axis will be the axis along which FFT is computed.

    Returns:
        tuple: A tuple containing:
            - np.array: The cosine components.
            - np.array: The sine components.
    """
    Nfft = f.shape[-1]
    ffft = np.fft.rfft(f)
    cosout = np.real(ffft) / Nfft * 2
    cosout[..., 0] /= 2

    sinout = -np.imag(ffft) / Nfft * 2
    sinout[..., 0] = 0

    return cosout, sinout

def irfft1D(cos_in, sin_in, nfft_multiplier=1):
    """
    Perform 1D inverse Fourier transform from cosine and sine to real space.

    Args:
        cos_in (np.array): The cosine components. If cos_in is 2D, then the last axis will be the axis along which FFT was computed.
        sin_in (np.array): The sine components.
        nfft_multiplier (int, optional): The number of output points will be this*(cos_in.shape[-1] - 1). Defaults to 1.

    Returns:
        np.array: The function value in real space.
    """
    Nfft = nfft_multiplier * (cos_in.shape[-1] - 1)
    ffft = (cos_in - complex(0, 1) * sin_in) * Nfft
    ffft[..., 0] *= 2
    result = np.fft.irfft(ffft, 2 * Nfft, axis=-1)
    return result


def rfft2D(f, mpol=None, ntor=None):
    """
    Perform 2D Fourier transform from real space to cosine and sine.

    Args:
        f (np.array): The data in real space. If f is 2D, then the last axis will be the axis along which FFT is computed.
        mpol (int, optional): The number of theta points on output. Defaults to f.shape[-2] // 4.
        ntor (int, optional): The number of phi points on output. Defaults to f.shape[-1] // 4.

    Returns:
        np.array: Cosine components :math:`n \\in \\{0, ..., m_{pol}\\}`
        np.array: Sine components :math:`n \\in \\{-n_{tor}, ..., n_{tor}\\}`
    """

    Nfft1 = f.shape[-2]
    Nfft2 = f.shape[-1]

    fftout = np.fft.rfft2(f, axes=[-1, -2])
    fftcos = np.real(fftout) / Nfft1 / Nfft2 * 2
    fftcos[0, :] /= 2
    fftcos[-1, :] /= 2
    fftsin = -np.imag(fftout) / Nfft1 / Nfft2 * 2
    fftsin[0, :] /= 2
    fftsin[-1, :] /= 2

    if mpol is None:
        mpol = Nfft1 // 4
    if ntor is None:
        ntor = Nfft2 // 4

    mpol_data = Nfft1 // 2
    ntor_data = Nfft2 // 2 - 1

    cn = np.zeros([mpol + 1, 2 * ntor + 1])
    sn = np.zeros([mpol + 1, 2 * ntor + 1])

    if mpol < mpol_data and ntor < ntor_data:
        # 1. assuming mpol and ntor are lower than that of the data
        # so we just need to truncate it, otherwise we will need to pad it
        idxlist = np.concatenate([[0], -np.arange(1, ntor + 1), np.arange(ntor, 0, -1)])
        cn[0 : mpol + 1, :] = fftcos[0 : mpol + 1, idxlist]
        sn[0 : mpol + 1, :] = fftsin[0 : mpol + 1, idxlist]
    elif mpol >= mpol_data and ntor < ntor_data:
        # 2, mpol is higher than data, ntor is lower
        idxlist = np.concatenate([[0], -np.arange(1, ntor + 1), np.arange(ntor, 0, -1)])
        cn[0 : mpol_data + 1, :] = fftcos[0 : mpol_data + 1, idxlist]
        sn[0 : mpol_data + 1, :] = fftsin[0 : mpol_data + 1, idxlist]
    else:
        raise ValueError("A higher value of pqNtor is needed")

    return cn, sn


def irfft2D(cn, sn, nfft_theta=None, nfft_phi=None):
    """
    Perform 2D Fourier transform from real space to cosine and sine.

    Args:
        cn (np.array): The cosine components.
        sn (np.array): The sine components.
        nfft_theta (int): The number of theta points on output. Defaults to (cn.shape[0] - 1)*4.
        nfft_phi (int): The number of zeta points on output. Defaults to ((cn.shape[1] - 1) // 2)*4.
    """
    mpol = cn.shape[0] - 1
    ntor = (cn.shape[1] - 1) // 2

    if nfft_theta is None:
        nfft_theta = mpol * 4
    if nfft_phi is None:
        nfft_phi = ntor * 4

    mpol_new = nfft_theta // 2
    ntor_new = nfft_phi // 2

    # now we pad cn and sn with zeros
    cn_pad = np.zeros([mpol_new + 1, 2 * ntor_new])
    sn_pad = np.zeros([mpol_new + 1, 2 * ntor_new])

    idxlist = np.concatenate([[0], -np.arange(1, ntor + 1), np.arange(ntor, 0, -1)])

    cn_pad[0 : mpol + 1, idxlist] = cn
    sn_pad[0 : mpol + 1, idxlist] = sn

    cn_pad[0, :] *= 2
    cn_pad[-1, :] *= 2
    sn_pad[0, :] *= 2
    sn_pad[-1, :] *= 2

    fout = (
        np.fft.irfft2(cn_pad - complex(0, 1) * sn_pad, axes=[-1, -2])
        * nfft_theta
        * nfft_phi
        / 2
    )

    return fout