!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
! Fortran module for SPEC problems
! outputs contravariant Bfield given (s, theta zeta) coordinates
! adapted bfield.f90 in SPEC main code
! written by Zhisong Qu (zhisong.qu@anu.edu.au)
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

MODULE SPECbfield

  CONTAINS

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
  SUBROUTINE get_bfield_many_1d( n, s, t, z, Bstz )
  !f2py threadsafe
    USE SPECtypedefns, ONLY : REAL_KIND

    IMPLICIT NONE

    INTEGER, INTENT(IN)                ::       n
    REAL(KIND=REAL_KIND), INTENT(IN)   ::       s(n), t(n), z(n)
    REAL(KIND=REAL_KIND), INTENT(OUT)  ::       Bstz(n, 3)

    INTEGER :: i
    REAL(KIND=REAL_KIND) :: stz(3)

    DO i = 1, n

      stz = (/s(i), t(i), z(i)/)

      CALL get_bfield(stz, Bstz(i, 1:3))
    
    ENDDO

  END SUBROUTINE get_bfield_many_1D

  SUBROUTINE get_bfield_tangent_many_1d( n, s, t, z, Bstz, dBstz )
  !f2py threadsafe
    USE SPECtypedefns, ONLY : REAL_KIND

    IMPLICIT NONE

    INTEGER, INTENT(IN)                ::       n
    REAL(KIND=REAL_KIND), INTENT(IN)   ::       s(n), t(n), z(n)
    REAL(KIND=REAL_KIND), INTENT(OUT)  ::       Bstz(n, 3), dBstz(n, 3,3)

    INTEGER :: i
    REAL(KIND=REAL_KIND) :: stz(3)

    DO i = 1, n

      stz = (/s(i), t(i), z(i)/)

      CALL get_bfield_tangent(stz, Bstz(i,1:3), dBstz(i,1:3,1:3))
    
    ENDDO

  END SUBROUTINE get_bfield_tangent_many_1D

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

  SUBROUTINE get_bfield( stz, Bstz )
!f2py threadsafe
    
  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
    
    USE SPECtypedefns, ONLY : REAL_KIND

    USE SPECconstants, ONLY : zero, one, half, two
    
    USE SPECvariables

    USE SPECbasefn
    
    IMPLICIT NONE
    
    REAL(KIND=REAL_KIND), INTENT(IN)   ::       stz(3)
    REAL(KIND=REAL_KIND), INTENT(OUT)  ::       Bstz(3)
    
    INTEGER              :: lvol, ii, ll, mi, ni
    REAL(KIND=REAL_KIND) :: teta, lss, sbar, arg, carg, sarg, dBu(1:3), zeta
    REAL(KIND=REAL_KIND) :: cheby(0:Lrad,0:1), zernike(0:Lrad,0:Mpol,0:1)
    
    REAL(KIND=REAL_KIND) :: TT(0:Lrad,0:1) ! this is almost identical to cheby; 17 Dec 15;

  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    lvol = ivol ;  ! short hand

    Bstz(1:3) = 0 ! set default intent out; this should cause a compilation error IF Node.ne.2;
  
  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
    
    lss = stz(1) ; teta = stz(2); zeta = stz(3)
      
    IF( Lcoordinatesingularity ) sbar = MAX( ( lss + one ) * half, zero )

  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
    
    IF (Lcoordinatesingularity) THEN
      CALL get_zernike(sbar, Lrad, Mpol, zernike(:,:,0:1))
    ELSE
      CALL get_cheby(lss, Lrad, cheby(0:Lrad,0:1))
    END IF

  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    dBu(1:3) = zero ! initialize summation;

  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
    
    DO ii = 1, mn ; mi = im(ii) ; ni = in1(ii) ; arg = mi * teta - ni * zeta ; carg = COS(arg) ; sarg = SIN(arg) ! shorthand; 20 Apr 13;
    
  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
    
      IF( Lcoordinatesingularity ) THEN ! regularization factor depends on mi; 17 Dec 15;

        DO ll = 0, Lrad ; TT(ll,0:1) = (/        zernike(ll,mi,0),        zernike(ll,mi,1)*half                      /)
        ENDDO

      ELSE

        DO ll = 0, Lrad ; TT(ll,0:1) = (/             cheby(ll,0),             cheby(ll,1)                           /)
        ENDDO

      ENDIF ! end of IF( Lcoordinatesingularity ) ; 16 Jan 15;

  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

      dBu(1) = dBu(1) + SUM(( - mi * Aze(1:Lrad+1,ii)- ni * Ate(1:Lrad+1,ii) ) * TT(0:Lrad, 0)) * sarg
      dBu(2) = dBu(2) + SUM((                        -      Aze(1:Lrad+1,ii) ) * TT(0:Lrad, 1)) * carg
      dBu(3) = dBu(3) + SUM((        Ate(1:Lrad+1,ii)                        ) * TT(0:Lrad, 1)) * carg

      IF( NOTstellsym ) THEN
        dBu(1) = dBu(1) + SUM(( + mi * Azo(1:Lrad+1,ii) + ni * Ato(1:Lrad+1,ii) ) * TT(0:Lrad,0)) * carg
        dBu(2) = dBu(2) + SUM((                         -      Azo(1:Lrad+1,ii) ) * TT(0:Lrad,1)) * sarg
        dBu(3) = dBu(3) + SUM((        Ato(1:Lrad+1,ii)                         ) * TT(0:Lrad,1)) * sarg
      ENDIF
  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
    
    ENDDO ! end of DO ii = 1, mn;
    
  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    Bstz= dBu
    
  !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

  END SUBROUTINE get_bfield

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

subroutine get_bfield_tangent( stz, Bstz, dBstz )
!f2py threadsafe 
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    USE SPECtypedefns, ONLY : REAL_KIND

    USE SPECconstants, ONLY : zero, one, half, two
    
    USE SPECvariables

    USE SPECbasefn
    
    IMPLICIT NONE
    
    REAL(KIND=REAL_KIND), INTENT(IN)   ::       stz(3)
    REAL(KIND=REAL_KIND), INTENT(OUT)  ::       Bstz(3)
    REAL(KIND=REAL_KIND), INTENT(OUT)  ::       dBstz(3,3)
    
    INTEGER              :: lvol, ii, ll, mi, ni
    REAL(KIND=REAL_KIND) :: teta, lss, sbar, arg, carg, sarg, dBu(1:3,0:3), zeta
    REAL(KIND=REAL_KIND) :: cheby(0:Lrad,0:2), zernike(0:Lrad,0:Mpol,0:2)
    
    REAL(KIND=REAL_KIND) :: TT(0:Lrad,0:2) ! this is almost identical to cheby; 17 Dec 15;

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    lvol = ivol ;

    Bstz = zero ! set default intent out

    dBstz = zero
 
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
  
    lss = stz(1) ; teta = stz(2) ; zeta = stz(3)
  
    IF( Lcoordinatesingularity ) sbar = MAX( ( lss + one ) * half, zero )

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
  
    IF (Lcoordinatesingularity) THEN
      CALL get_zernike_d2(sbar, Lrad, Mpol, zernike(0:Lrad,:,0:2))
    ELSE
      CALL get_cheby_d2(lss, Lrad, cheby(0:Lrad,0:2))
    END IF

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    dBu = zero ! initialize summation;
  
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
  
    DO ii = 1, mn ; mi = im(ii) ; ni = in1(ii) ; arg = mi * teta - ni * zeta ; carg = COS(arg) ; sarg = SIN(arg) ! shorthand; 20 Apr 13;
      
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
   
      IF( Lcoordinatesingularity ) THEN ! regularization factor depends on mi; 17 Dec 15;

        DO ll = 0, Lrad ; TT(ll,0:2) = (/        zernike(ll,mi,0),        zernike(ll,mi,1)*half , zernike(ll,mi,2)*half*half /)
        ENDDO

      ELSE

        DO ll = 0, Lrad ; TT(ll,0:2) = (/             cheby(ll,0),             cheby(ll,1)       ,cheby(ll,2)               /)
        ENDDO

      ENDIF ! end of IF( Lcoordinatesingularity ) ; 16 Jan 15;

! !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
   
      ! no derivative
      ;dBu(1,0) = dBu(1,0) + SUM(( - mi * Aze(1:Lrad+1,ii) - ni * Ate(1:Lrad+1,ii) ) * TT(0:Lrad,0)) * sarg
      ;dBu(2,0) = dBu(2,0) + SUM((                         -      Aze(1:Lrad+1,ii) ) * TT(0:Lrad,1)) * carg
      ;dBu(3,0) = dBu(3,0) + SUM((        Ate(1:Lrad+1,ii)                         ) * TT(0:Lrad,1)) * carg
      ! ds
      ;dBu(1,1) = dBu(1,1) + SUM(( - mi * Aze(1:Lrad+1,ii) - ni * Ate(1:Lrad+1,ii) ) * TT(0:Lrad,1)) * sarg
      ;dBu(2,1) = dBu(2,1) + SUM((                         -      Aze(1:Lrad+1,ii) ) * TT(0:Lrad,2)) * carg
      ;dBu(3,1) = dBu(3,1) + SUM((        Ate(1:Lrad+1,ii)                         ) * TT(0:Lrad,2)) * carg
      ! dtheta
      ;dBu(1,2) = dBu(1,2) + mi * SUM(( - mi * Aze(1:Lrad+1,ii) - ni * Ate(1:Lrad+1,ii) ) * TT(0:Lrad,0)) * carg
      ;dBu(2,2) = dBu(2,2) - mi * SUM((                         -      Aze(1:Lrad+1,ii) ) * TT(0:Lrad,1)) * sarg
      ;dBu(3,2) = dBu(3,2) - mi * SUM((        Ate(1:Lrad+1,ii)                         ) * TT(0:Lrad,1)) * sarg
      ! dzeta
      ;dBu(1,3) = dBu(1,3) - ni * SUM(( - mi * Aze(1:Lrad+1,ii) - ni * Ate(1:Lrad+1,ii) ) * TT(0:Lrad,0)) * carg
      ;dBu(2,3) = dBu(2,3) + ni * SUM((                         -      Aze(1:Lrad+1,ii) ) * TT(0:Lrad,1)) * sarg
      ;dBu(3,3) = dBu(3,3) + ni * SUM((        Ate(1:Lrad+1,ii)                         ) * TT(0:Lrad,1)) * sarg
      IF( NOTstellsym ) THEN ! include non-symmetric harmonics; 28 Jan 13;
        ! no derivative
        dBu(1,0) = dBu(1,0) + SUM(( + mi * Azo(1:Lrad+1,ii) + ni * Ato(1:Lrad+1,ii) ) * TT(0:Lrad,0)) * carg
        dBu(2,0) = dBu(2,0) + SUM((                         -      Azo(1:Lrad+1,ii) ) * TT(0:Lrad,1)) * sarg
        dBu(3,0) = dBu(3,0) + SUM((        Ato(1:Lrad+1,ii)                         ) * TT(0:Lrad,1)) * sarg
        ! ds
        dBu(1,1) = dBu(1,1) + SUM(( + mi * Azo(1:Lrad+1,ii) + ni * Ato(1:Lrad+1,ii) ) * TT(0:Lrad,1)) * carg
        dBu(2,1) = dBu(2,1) + SUM((                         -      Azo(1:Lrad+1,ii) ) * TT(0:Lrad,2)) * sarg
        dBu(3,1) = dBu(3,1) + SUM((        Ato(1:Lrad+1,ii)                         ) * TT(0:Lrad,2)) * sarg
        ! dtheta
        dBu(1,2) = dBu(1,2) - mi * SUM(( + mi * Azo(1:Lrad+1,ii) + ni * Ato(1:Lrad+1,ii) ) * TT(0:Lrad,0)) * sarg
        dBu(2,2) = dBu(2,2) + mi * SUM((                         -      Azo(1:Lrad+1,ii) ) * TT(0:Lrad,1)) * carg
        dBu(3,2) = dBu(3,2) + mi * SUM((        Ato(1:Lrad+1,ii)                         ) * TT(0:Lrad,1)) * carg
        ! dzeta
        dBu(1,3) = dBu(1,3) + ni * SUM(( + mi * Azo(1:Lrad+1,ii) + ni * Ato(1:Lrad+1,ii) ) * TT(0:Lrad,0)) * sarg
        dBu(2,3) = dBu(2,3) - ni * SUM((                         -      Azo(1:Lrad+1,ii) ) * TT(0:Lrad,1)) * carg
        dBu(3,3) = dBu(3,3) - ni * SUM((        Ato(1:Lrad+1,ii)                         ) * TT(0:Lrad,1)) * carg
      ENDIF

! !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
   
    ENDDO ! end of DO ii = 1, mn;
  
! !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    Bstz = dBu(1:3,0)
    dBstz(1:3,1:3) = TRANSPOSE(dBu(1:3,1:3))

! !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

  END SUBROUTINE get_bfield_tangent

! !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
END MODULE SPECbfield