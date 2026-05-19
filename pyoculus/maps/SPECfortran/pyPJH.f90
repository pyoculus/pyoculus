!> @file pyPJH.f90
!> @brief Fortran module for SPEC Pressure Jump Hamiltonian problems
!> @author Zhisong Qu (zhisong.qu@anu.edu.au)
!> @author Stuart Hudson (shudson@pppl.gov)

!> Fortran module for SPEC Pressure Jump Hamiltonian problems
!>
!> @author Stuart Hudson (shudson@pppl.gov)
!> @author Adapted by Zhisong Qu (zhisong.qu@anu.edu.au)
!>
!> Interfaced to Python, see \ref pyoculus.problems.SPECPJH.SPECPJH
!>
!> @todo non-stellarator symmetric cases
!>
!> ## Pressure Jump Hamiltonian (PJH) Fortran module
!>
!> Please note that in the Python module \ref pyoculus.problems.SPECPJH.SPECPJH, \f$\zeta\f$ is equivalent to \f$\varphi\f$
!>
!> ### Usage in Python:
!> To use the pjh module, one needs to initialize the fortran module first in Python (using class pyoculus.problems.SPECProblem or classes derived from it), 
!> then initialize pjh in Python by calling its Python wrap in `pyoculus.problems.SPECfortran.fortran_module`
!>
!>     init_pjh(dp, inside_or_outside, plus_or_minus)
!>
!> @param dp \f$\delta p\f$, the pressure jump
!> @param inside_or_outside for the specified volume, we compute things on the inner interface or outer
!> @param plus_or_minus whether to take the plus or minus sign in computing \f$p_\varphi\f$
!>
!> After initialization, one can call
!>
!>     rhs = get_pjhfield(phi, ptt)
!>
!> or for rhs and tangent,
!> 
!>     rhs = get_pjhfield_tangent(phi, ptt)
!>
!> @param phi the \f$\varphi\f$ angle
!> @param ptt the state vector (\f$p_\theta\f$, \f$\theta\f$), or (\f$p_\theta\f$, \f$\theta\f$, dp1, dt1, dp2, dt2)
!> @returns array of size 2 (for get_pjhfield) or array of size 6 (for get_pjhfield_tangent)
!>
!> To estimate the initial value for \f$p_\theta\f$, sometimes we need \f$B_\theta\f$ on the known side of the interface.
!> This can be obtained by calling
!>
!>     (Btheta, Bphi) = get_covariant_field(theta, phi)
!>
!> @param theta \f$\theta\f$
!> @param phi \f$\varphi\f$
!> @returns the covariant component on the known side of the interface, \f$(B_{1,\theta}, B_{1,\varphi})\f$
!>
MODULE SPECpjh

  USE SPECtypedefns, only : REAL_KIND

  PRIVATE

  INTEGER                          :: innout       !< specifying field on which side (-1) inner side, (+1) outer side
  INTEGER                          :: ioi          !< specifying which interface
  INTEGER                          :: plusminus    !< +1 or -1 sign taken for calculating \f$p_\varphi$\
  REAL(KIND=REAL_KIND)             :: delta_p      !< the jump of pressure

  REAL(KIND=REAL_KIND),ALLOCATABLE :: gBtmne(:)    !< the contravariant components of magnetic field on the interface
  REAL(KIND=REAL_KIND),ALLOCATABLE :: gBzmne(:)    !< the contravariant components of magnetic field on the interface
  REAL(KIND=REAL_KIND),ALLOCATABLE :: gBtmno(:)    !< the contravariant components of magnetic field on the interface
  REAL(KIND=REAL_KIND),ALLOCATABLE :: gBzmno(:)    !< the contravariant components of magnetic field on the interface

  PUBLIC :: get_pjhfield, get_pjhfield_tangent, init_PJH, destroy_PJH, get_b2_interface, get_covariant_field

  CONTAINS

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
!> Compute the ODEs for PJH system, \f$dp_\theta/d\varphi\f$, \f$d\theta/d\varphi\f$
  SUBROUTINE get_pjhfield( phi , ptt , dptt )
!f2py threadsafe

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    USE SPECconstants
    USE SPECvariables
    USE SPECcoords, ONLY: get_metric_interface

    IMPLICIT NONE

    REAL(KIND=REAL_KIND),INTENT(IN) :: phi      !< the \f$\varphi\f$ angle
    REAL(KIND=REAL_KIND),INTENT(IN) :: ptt(2)   !< (\f$p_\theta\f$, \f$\theta\f$)
    REAL(KIND=REAL_KIND),INTENT(OUT) :: dptt(2) !< (\f$dp_\theta/d\varphi\f$, \f$d\theta/d\varphi\f$)

    INTEGER :: imn 

    REAL(KIND=REAL_KIND) :: cosarg(mn), sinarg(mn),arg,pt,pp,theta
    REAL(KIND=REAL_KIND) :: dBB1(0:2),dBB2(0:3),a,b,c,discrim !< test variable

    REAL(KIND=REAL_KIND) :: dR(0:3,0:3,0:3),dZ(0:3,0:3,0:3),gl(3,3,0:2),sg(0:2),dG(0:2),dA(0:1),db1(0:2),dabc(3,0:2),dP(0:2),db2(0:2)
    REAL(KIND=REAL_KIND) :: ftftptpt,ftfpptpp,fpfppppp
    REAL(KIND=REAL_KIND) :: dtdot(0:2),dpdot(0:2),dsdot(0:2)
  
    pt = ptt(1) ! shorthand; "momentum" coordinate;

    theta = ptt(2) ! theta
  
!> - We need the metrics \f$g_{\theta \theta}, g_{\theta \varphi}, g_{\varphi \varphi}\f$ and the Jacobian \f$J\f$, as well as their first \f$\theta\f$ derivatives
!! by calling the subroutine coords::get_metric_interface
    dR=zero ; dZ=zero; gl=zero; sg=zero
    call get_metric_interface(ioi, theta, phi, dR, dZ, gl, sg, 1)
  
!> - Compute \f$ G \f$ and its derivatives.
    dG(0)= gl(2,2,0)*gl(3,3,0)                                                                           -         gl(2,3,0)**2
    dG(1)= gl(2,2,1)*gl(3,3,0)                           + gl(2,2,0)*gl(3,3,1)                           - two *   gl(2,3,0)*gl(2,3,1)
!    dG(2)= gl(2,2,2,2)*gl(3,3,0,0) + gl(2,2,2,0)*gl(3,3,2,0) + gl(2,2,2,0)*gl(3,3,2,0) + gl(2,2,0,0)*gl(3,3,2,2) - two * ( gl(2,3,2,0)*gl(2,3,2,0) + gl(2,3,0,0)*gl(2,3,2,2) )

!> - Compute the squared magnetic field strength \f$B_1^2\f$ on the other side of the interface by calling the subroutine specpjh::get_b2_interface
    call get_b2_interface(dBB1, ioi, theta, phi, gl, sg, 1)

!> - Construct \f$a, b, c\f$ for the quadratic equation for solving \f$p_\varphi\f$, then solve it by applying plus_or_minus sign we specified earlier (stored in dP)
    ! this is a, b, c for computing p_phi

    dabc(1,0) = gl(2,2,0)
    dabc(2,0) = -two * gl(2,3,0) * pt
    dabc(3,0) = gl(3,3,0) * pt**2 - (dBB1(0) + two * delta_p) * dG(0)

    discrim=dabc(2,0)**2-four*dabc(1,0)*dabc(3,0)
   
    if( discrim.lt.zero ) stop "ph00aa :          : WF=? ; discrim.lt.zero ;"
 
    dP(0)  = ( -dabc(2,0) + plusminus *                                                                                   sqrt (discrim)                       ) / (two*dabc(1,0))

!> - Computes \f$b_2\f$
    db2(0)  = gl(3,3,0)*   pt**2 - two*gl(2,3,0)* pt*dP(0)            + gl(2,2,0)*    dP(0)**2

!> - Construct the denominator of \f$d\theta/d\varphi\f$

    dtdot(0) = gl(3,3,0)*two*pt - two*gl(2,3,0)*dP(0)
    dpdot(0) = -two*gl(2,3,0)*pt + gl(2,2,0)*two*dP(0)

!> - Construct the numerators

    ftftptpt= -pt**2 ; ftfpptpp= -pt*dP(0) ; fpfppppp=-dP(0)**2

    dsdot(0) = gl(3,3,1)* ftftptpt                 - two*gl(2,3,1)* ftfpptpp                                          + gl(2,2,1)*fpfppppp                              &
             + dBB1(1)*dG(0) + db2(0)*dG(1)/dG(0)
 
!> - Get right hand side

    dptt(1:2) = (/ dsdot(0) , dtdot(0) /) / dpdot(0)
 
!> .
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

  END SUBROUTINE get_pjhfield


!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
!> Compute the ODEs for PJH system, \f$dp_\theta/d\varphi\f$, \f$d\theta/d\varphi\f$, with tangent
  SUBROUTINE get_pjhfield_tangent( phi , ptt , dptt )
!f2py threadsafe

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    USE SPECconstants
    USE SPECvariables
    USE SPECcoords, ONLY: get_metric_interface

    IMPLICIT NONE

    REAL(KIND=REAL_KIND),INTENT(IN) :: phi      !< the \f$\varphi\f$ angle
    REAL(KIND=REAL_KIND),INTENT(IN) :: ptt(6)   !< (\f$p_\theta\f$, \f$\theta\f$, \f$\Delta p_{\theta,1}\f$, \f$\Delta \theta_1\f$, \f$\Delta p_{\theta,2}\f$, \f$\Delta \theta_2\f$)
    REAL(KIND=REAL_KIND),INTENT(OUT) :: dptt(6) !< (\f$dp_\theta/d\varphi\f$, \f$d\theta/d\varphi\f$)

    INTEGER :: imn 

    REAL(KIND=REAL_KIND) :: cosarg(mn), sinarg(mn),arg,pt,pp,theta
    REAL(KIND=REAL_KIND) :: dBB1(0:2),dBB2(0:3),a,b,c,discrim !< test variable

    REAL(KIND=REAL_KIND) :: dR(0:3,0:3,0:3),dZ(0:3,0:3,0:3),gl(3,3,0:2),sg(0:2),dG(0:2),dA(0:1),db1(0:2),dabc(3,0:2),dP(0:2),db2(0:2)
    REAL(KIND=REAL_KIND) :: ftftptpt,ftfpptpp,fpfppppp
    REAL(KIND=REAL_KIND) :: dtdot(0:2),dpdot(0:2),dsdot(0:2),TM(2,2)
  
! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

    pt = ptt(1) ! shorthand; "momentum" coordinate;

    theta = ptt(2) ! theta
  
! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

!> - We need the metrics \f$g_{\theta \theta}, g_{\theta \varphi}, g_{\varphi \varphi}\f$ and the Jacobian \f$J\f$, as well as their first \f$\theta\f$ derivatives
!! by calling the subroutine coords::get_metric_interface
    dR=zero ; dZ=zero; gl=zero; sg=zero
    call get_metric_interface(ioi, theta, phi, dR, dZ, gl, sg, 2)
  
!> - Compute \f$ G \f$ and its derivatives.
    dG(0)= gl(2,2,0)*gl(3,3,0)                                                                           -         gl(2,3,0)**2
    dG(1)= gl(2,2,1)*gl(3,3,0)                           + gl(2,2,0)*gl(3,3,1)                           - two *   gl(2,3,0)*gl(2,3,1)
    dG(2)= gl(2,2,2)*gl(3,3,0) + gl(2,2,1)*gl(3,3,1)     + gl(2,2,1)*gl(3,3,1) + gl(2,2,0)*gl(3,3,2)     - two * ( gl(2,3,1)*gl(2,3,1) + gl(2,3,0)*gl(2,3,2) )

!> - Compute the squared magnetic field strength \f$B_1^2\f$ on the other side of the interface by calling the subroutine specpjh::get_b2_interface
    call get_b2_interface(dBB1, ioi, theta, phi, gl, sg, 2)

!> - Construct \f$a, b, c\f$ for the quadratic equation for solving \f$p_\varphi\f$, then solve it by applying plus_or_minus sign we specified earlier (stored in dP)
    ! this is a, b, c for computing p_phi

    dabc(1,0) = gl(2,2,0)
    dabc(1,1) = gl(2,2,1)
    dabc(1,2) = zero
 
    dabc(2,0) = -two * gl(2,3,0) * pt
    dabc(2,1) = -two * gl(2,3,1) * pt
    dabc(2,2) = -two * gl(2,3,0)
 
    dabc(3,0) = gl(3,3,0) * pt**2 - (dBB1(0) + two * delta_p) * dG(0)
    dabc(3,1) = gl(3,3,1) * pt**2 - (dBB1(0) + two * delta_p) * dG(1) - dBB1(1) * dG(0)
    dabc(3,2) = gl(3,3,0) * pt*two
   
    discrim=dabc(2,0)**2-four*dabc(1,0)*dabc(3,0)
   
    if( discrim.lt.zero ) stop "ph00aa :          : WF=? ; discrim.lt.zero ;"
 
    dP(0)  = ( -dabc(2,0) + plusminus *                                                                                   sqrt (discrim)                       ) / (two*dabc(1,0))
    dP(1)  = ( -dabc(2,1) + plusminus * half * (two*dabc(2,0)*dabc(2,1)-four*(dabc(1,1)*dabc(3,0)+dabc(1,0)*dabc(3,1))) / sqrt (discrim) - dP(0)*two*dabc(1,1) ) / (two*dabc(1,0))
    dP(2)  = ( -dabc(2,2) + plusminus * half * (two*dabc(2,0)*dabc(2,2)-four*(                    dabc(1,0)*dabc(3,2))) / sqrt (discrim)                       ) / (two*dabc(1,0))
  
! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

!> - Computes \f$b_2\f$
    db2(0)  = gl(3,3,0)*   pt**2 - two*gl(2,3,0)* pt*dP(0)            + gl(2,2,0)*    dP(0)**2
    db2(1)  = gl(3,3,1)*   pt**2 - two*gl(2,3,1)* pt*dP(0)            + gl(2,2,1)*    dP(0)**2 & 
                                 - two*gl(2,3,0)* pt*dP(1)            + gl(2,2,0)*two*dP(0)*dP(1)
    db2(2)  = gl(3,3,0)*two*pt   - two*gl(2,3,0)*(   dP(0)+pt*dP(2) ) + gl(2,2,0)*two*dP(0)*dP(2)

!> - Construct the denominator of \f$d\theta/d\varphi\f$

    dtdot(0) = gl(3,3,0)*two*pt - two*gl(2,3,0)*dP(0)
    dtdot(1) = gl(3,3,1)*two*pt - two*gl(2,3,1)*dP(0) - two*gl(2,3,0)*dP(1)
    dtdot(2) = gl(3,3,0)*two                          - two*gl(2,3,0)*dP(2)
 
    dpdot(0) = -two*gl(2,3,0)*pt + gl(2,2,0)*two*dP(0)
    dpdot(1) = -two*gl(2,3,1)*pt + gl(2,2,1)*two*dP(0) + gl(2,2,0)*two*dP(1)
    dpdot(2) = -two*gl(2,3,0)                          + gl(2,2,0)*two*dP(2)

! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

!> - Construct the numerators

    ftftptpt= -pt**2 ; ftfpptpp= -pt*dP(0) ; fpfppppp=-dP(0)**2

! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

    dsdot(0) = gl(3,3,1)* ftftptpt                 - two*gl(2,3,1)* ftfpptpp                                          + gl(2,2,1)*fpfppppp                              &
             + dBB1(1)*dG(0)                       + db2(0)*dG(1)/dG(0)
 
    dsdot(1) = gl(3,3,2)* ftftptpt                 - two*gl(2,3,2)* ftfpptpp                                          + gl(2,2,2)*fpfppppp                              &
                                                   - two*gl(2,3,1)*(-pt*dP(1))                                        + gl(2,2,1)*two*(-dP(0)*dP(1))                    & 
             + dBB1(2)*dG(0) + dBB1(1)*dG(1)       + db2(1)*dG(1)/dG(0) + db2(0)*(dG(2)-dG(1)**2/dG(0))/dG(0)
 
    dsdot(2) = gl(3,3,1)*(-two*pt)                 - two*gl(2,3,1)*(-dP(0)-pt*dP(2))                                  + gl(2,2,1)*(-two*dP(0)*dP(2))                    &
                                                   +  db2(2)*dG(1)/dG(0) 

! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

!> - Get right hand side

    dptt(1:2) = (/ dsdot(0) , dtdot(0) /) / dpdot(0)
 
    TM(1,1) = ( dsdot(2) - dptt(1)*dpdot(2) ) / dpdot(0) ! d radial / d radial ;
    TM(1,2) = ( dsdot(1) - dptt(1)*dpdot(1) ) / dpdot(0) ! d radial / d angle  ;
    TM(2,1) = ( dtdot(2) - dptt(2)*dpdot(2) ) / dpdot(0) ! d angle  / d radial ;
    TM(2,2) = ( dtdot(1) - dptt(2)*dpdot(1) ) / dpdot(0) ! d angle  / d angle  ;
 
    dptt(3)=TM(1,1)*ptt(3) + TM(1,2)*ptt(4) ! standard format;
    dptt(4)=TM(2,1)*ptt(3) + TM(2,2)*ptt(4)
    dptt(5)=TM(1,1)*ptt(5) + TM(1,2)*ptt(6)
    dptt(6)=TM(2,1)*ptt(5) + TM(2,2)*ptt(6)

!> .
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

  END SUBROUTINE get_pjhfield_tangent

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
!> Get the covariant component of known side of magnetic field \f$\bf B_1\f$, i.e. \f$B_{1,\theta}, B_{1,\varphi}\f$.
!> Can be used to estimate the initial condition for \f$p_\theta\f$. 
  SUBROUTINE get_covariant_field(theta, phi, Bco)

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    USE SPECconstants
    USE SPECvariables
    USE SPECcoords, ONLY: get_metric_interface

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    IMPLICIT NONE

    REAL(KIND=REAL_KIND), INTENT(IN) :: theta  !< the \f$\theta\f$ angle
    REAL(KIND=REAL_KIND), INTENT(IN) :: phi    !< the \f$\phi\f$ angle
    REAL(KIND=REAL_KIND), INTENT(OUT):: Bco(2) !< the covariant components (\f$B_{1,\theta}, B_{1,\varphi}\f$)

    REAL(KIND=REAL_KIND) :: cosarg(mn), sinarg(mn),pt,pp
    REAL(KIND=REAL_KIND) :: dR(0:3,0:3,0:3),dZ(0:3,0:3,0:3),gl(3,3,0:2),sg(0:2),gB(2:3,0:2)

    call get_metric_interface(ioi, theta, phi, dR, dZ, gl, sg, 1)

    cosarg = COS(im*theta-in1*phi)
    sinarg = SIN(im*theta-in1*phi)

    gB(2,0) = SUM(gBtmne * cosarg)
    gB(3,0) = SUM(gBzmne * cosarg)

    Bco(1) = gB(2,0) * gl(2,2,0) + gB(3,0) * gl(2,3,0)
    Bco(2) = gB(2,0) * gl(2,3,0) + gB(3,0) * gl(3,3,0)

    Bco = Bco / sg(0)

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

  END SUBROUTINE get_covariant_field

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
!> An internal subroutine that computes the squared magnetic field strength on the interface \f$B_1^2\f$ and its first and second \f$\theta\f$ derivatives.
!> This is a private subroutine that is not interfaced with Python.
  SUBROUTINE get_b2_interface(db2, ioi, theta, phi, guvij, sg, ideriv)

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    USE SPECtypedefns, ONLY : REAL_KIND
    USE SPECconstants
    USE SPECvariables

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    IMPLICIT NONE
    REAL(KIND=REAL_KIND),    INTENT(IN) :: theta, phi ! the three coordinates (s,theta,varphi)
    INTEGER             ,    INTENT(IN) :: ioi ! on which interface
    INTEGER             ,    INTENT(IN) :: ideriv ! the level of derivatives. 0 for none, 1 for first, 2 for second

    REAL(KIND=REAL_KIND),    INTENT(IN):: guvij(3,3,0:2), sg(0:2)

    REAL(KIND=REAL_KIND),    INTENT(OUT):: db2(0:2)

    REAL(KIND=REAL_KIND) :: lss, rsign, tmpdb2, tmpdb21
    REAL(KIND=REAL_KIND) :: cosarg(mn), sinarg(mn), gB(2:3,0:2)

    INTEGER :: ii, jj

    db2 = zero

    cosarg = COS(im*theta-in1*phi)
    sinarg = SIN(im*theta-in1*phi)

    gB(2,0) = SUM(gBtmne * cosarg)
    gB(3,0) = SUM(gBzmne * cosarg)

    DO ii = 2, 3
      DO jj = 2, 3
        db2(0) = db2(0) + guvij(ii,jj,0) * gB(ii,0) * gB(jj,0)
      ENDDO
    ENDDO
    tmpdb2 = db2(0)
    db2(0) = db2(0) / sg(0)**2

    IF (ideriv .GE. 1) THEN
      gB(2,1) = SUM(gBtmne * (-im) * sinarg)
      gB(3,1) = SUM(gBzmne * (-im) * sinarg)

      DO ii = 2, 3
        DO jj = 2, 3
          db2(1) = db2(1) + guvij(ii,jj,1) * gB(ii,0) * gB(jj,0) + guvij(ii,jj,0) * (gB(ii,1) * gB(jj,0) + gB(ii,0) * gB(jj,1))
        ENDDO
      ENDDO
      tmpdb21 = db2(1)
      db2(1) = db2(1) / sg(0)**2 - 2 * db2(0) * sg(1) / sg(0)
    ENDIF

    IF (ideriv .GE. 2) THEN

      gB(2,2) = SUM(gBtmne * (-im) * (im) * cosarg)
      gB(3,2) = SUM(gBzmne * (-im) * (im) * cosarg)

      DO ii = 2, 3
        DO jj = 2, 3
          db2(2) = db2(2) + guvij(ii,jj,2) * gB(ii,0) * gB(jj,0) + guvij(ii,jj,0) * (gB(ii,2) * gB(jj,0) + gB(ii,0) * gB(jj,2) + two * gB(ii,1) * gB(jj,1)) &
                      +two* guvij(ii,jj,1) * (gB(ii,1) * gB(jj,0) + gB(ii,0) * gB(jj,1)) 
        ENDDO
      ENDDO
      db2(2) = (db2(2) * sg(0) - tmpdb21 * sg(1) - two * tmpdb2 * sg(2)) / sg(0)**3 - three * db2(1) * sg(1) / sg(0)

    ENDIF


!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

  END SUBROUTINE get_b2_interface

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
!> Initialize the PJH system.
!> Whenever \f$\delta p\f$ is modified, you will need to call this subroutine (in Python interface) again.
  SUBROUTINE init_pjh(dp, inside_or_outside, plus_or_minus)

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
  
    USE SPECconstants
    USE SPECvariables
    USE SPECbasefn

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    IMPLICIT NONE

    REAL(KIND=REAL_KIND), INTENT(IN) :: dp   !< the pressure jump \f$\delta p\f$
    INTEGER, INTENT(IN) :: inside_or_outside !< 0 for inside interface of the selected volume ivol (set in SPECvariables::ivol), +1 for outside
    INTEGER, INTENT(IN) :: plus_or_minus     !< set the sign (-1 or +1) for computing \f$p_\varphi\f$ using the quadratic root equation.

    INTEGER :: ii, jj, ll, mi, ni

    REAL(KIND=REAL_KIND) :: sbar, lss
    REAL(KIND=REAL_KIND) :: cheby(0:Lrad,0:1), zernike(0:Lrad,0:Mpol,0:1)
    
    REAL(KIND=REAL_KIND) :: TT(0:Lrad,0:1)

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    plusminus = plus_or_minus
    delta_p = dp
    ioi = ivol + inside_or_outside       ! the index of the interface

    IF (ALLOCATED(gBtmne)) THEN
      DEALLOCATE(gBtmne)
    ENDIF
    IF (ALLOCATED(gBzmne)) THEN
      DEALLOCATE(gBzmne)
    ENDIF
    ALLOCATE(gBtmne(1:mn))
    ALLOCATE(gBzmne(1:mn))

    IF (NOTstellsym) THEN
      IF (ALLOCATED(gBtmno)) THEN
        DEALLOCATE(gBtmno)
      ENDIF
      IF (ALLOCATED(gBzmno)) THEN
        DEALLOCATE(gBzmno)
      ENDIF
      ALLOCATE(gBtmno(1:mn))
      ALLOCATE(gBzmno(1:mn))
    ENDIF

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

!> The contravariant component of \f$\bf B_1\f$ is computed on the interface.
!> The Fourier components of \f$ J B^{\theta}_1\f$ and \f$ J B^{\varphi}_1\f$ are stored for further use.

    IF (Lcoordinatesingularity) THEN
      sbar = FLOAT(inside_or_outside)
      CALL get_zernike(sbar, Lrad, Mpol, zernike(:,:,0:1))
    ELSE
      lss = FLOAT(inside_or_outside) * two - one
      CALL get_cheby(lss, Lrad, cheby(0:Lrad,0:1))
    END IF


    DO ii = 1, mn ; mi = im(ii) ; ni = in1(ii) 


      IF( Lcoordinatesingularity ) THEN ! regularization factor depends on mi; 17 Dec 15;

        DO ll = 0, Lrad ; TT(ll,0:1) = (/        zernike(ll,mi,0),        zernike(ll,mi,1)*half                      /)
        ENDDO

      ELSE

        DO ll = 0, Lrad ; TT(ll,0:1) = (/             cheby(ll,0),             cheby(ll,1)                           /)
        ENDDO

      ENDIF ! end of IF( Lcoordinatesingularity ) ; 16 Jan 15;
  
      gBtmne(ii) = SUM( -Aze(1:Lrad+1,ii) * TT(0:Lrad, 1) )
      gBzmne(ii) = SUM( +Ate(1:Lrad+1,ii) * TT(0:Lrad, 1) )

      IF (NOTstellsym) THEN
        gBtmno(ii) = SUM( -Azo(1:Lrad+1,ii) * TT(0:Lrad, 1) )
        gBzmno(ii) = SUM( +Ato(1:Lrad+1,ii) * TT(0:Lrad, 1) )
      END IF

    ENDDO

  END SUBROUTINE init_pjh

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

!> This subroutine frees the memory space allocated by specpjh::init_pjh.
  SUBROUTINE destroy_pjh()

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    USE SPECvariables

    IMPLICIT NONE

    IF (ALLOCATED(gBtmne)) THEN
      DEALLOCATE(gBtmne)
    ENDIF
    IF (ALLOCATED(gBzmne)) THEN
      DEALLOCATE(gBzmne)
    ENDIF

    IF (NOTstellsym) THEN
      IF (ALLOCATED(gBtmno)) THEN
        DEALLOCATE(gBtmno)
      ENDIF
      IF (ALLOCATED(gBzmno)) THEN
        DEALLOCATE(gBzmno)
      ENDIF
    ENDIF

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

  END SUBROUTINE destroy_pjh

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

END MODULE SPECpjh

