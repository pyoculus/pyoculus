!> @file pybasefn.f90
!> @brief Fortran module that computes the radial basis functions
!> @author Zhisong Qu (zhisong.qu@anu.edu.au)

!> Fortran module that computes the radial basis functions, adapted basefn.f90 in SPEC main code
MODULE SPECbasefn

CONTAINS

!> Get the Chebyshev polynomials with zeroth, first derivatives
  SUBROUTINE get_cheby(lss, lrad, cheby)
!f2py threadsafe

    USE SPECconstants, ONLY : zero, one, two
    USE SPECtypedefns, ONLY : REAL_KIND

    IMPLICIT NONE

    REAL(KIND=REAL_KIND),INTENT(IN) :: lss !< coordinate input lss
    INTEGER, INTENT(IN) :: lrad            !< radial resolution
    REAL(KIND=REAL_KIND), INTENT(OUT) :: cheby(0:lrad,0:1) !< the value, first derivative of Chebyshev polynomial

    integer :: ll

    cheby = zero

      ;                 cheby( 0,0:1) = (/one                                      ,zero                                                           /)
      ;                 cheby( 1,0:1) = (/lss                                      ,one                                                            /)
      DO ll = 2, lrad ; cheby(ll,0:1) = (/two * lss * cheby(ll-1,0) - cheby(ll-2,0),two * cheby(ll-1,0) + two * lss * cheby(ll-1,1) - cheby(ll-2,1)/)
      ENDDO

    ! basis recombination
    DO ll = 1, lrad
      cheby(ll, 0) = cheby(ll, 0)  - (-1)**ll
    ENDDO

    DO ll = 0, lrad
      cheby(ll, 0:1) = cheby(ll, 0:1) / FLOAT(ll+1) ! scale for better conditioning
    ENDDO

    return
  END SUBROUTINE get_cheby

!> Get the Chebyshev polynomials with zeroth, first and second derivatives
  SUBROUTINE get_cheby_d2(lss, lrad, cheby)
!f2py threadsafe

    USE SPECconstants, ONLY : zero, one, two
    USE SPECtypedefns, ONLY : REAL_KIND

    IMPLICIT NONE

    REAL(KIND=REAL_KIND),INTENT(IN) :: lss !< coordinate input lss
    INTEGER, INTENT(IN) :: lrad            !< radial resolution
    REAL(KIND=REAL_KIND), INTENT(OUT) :: cheby(0:lrad,0:2) !< the value, first and second derivative of Chebyshev polynomial

    integer :: ll

    cheby = zero

    ;                     ; cheby( 0,0:2) = (/ one, zero, zero /) ! T_0: Chebyshev initialization; function, 1st-derivative, 2nd-derivative;
    ;                     ; cheby( 1,0:2) = (/ lss,  one, zero /) ! T_1: Chebyshev initialization; function, 1st-derivative, 2nd-derivative;
    DO ll = 2, lrad       
      cheby(ll,0:2) = (/ two * lss * cheby(ll-1,0)                                                         - cheby(ll-2,0) , &
                        two       * cheby(ll-1,0) + two * lss * cheby(ll-1,1)                             - cheby(ll-2,1) , &
                        two       * cheby(ll-1,1) + two       * cheby(ll-1,1) + two * lss * cheby(ll-1,2) - cheby(ll-2,2) /)
    ENDDO 

    DO ll = 1, lrad
      cheby(ll, 0) = cheby(ll, 0)  - (-1)**ll
    ENDDO

    DO ll = 0, lrad
      cheby(ll, 0:2) = cheby(ll, 0:2) / FLOAT(ll+1) ! scale for better conditioning
    ENDDO

    return
  END SUBROUTINE get_cheby_d2

!> Get the Zernike polynomials with zeroth, first derivatives
  SUBROUTINE get_zernike(r, lrad, mpol, zernike)
!f2py threadsafe

    USE SPECconstants, ONLY : zero, one, two
    USE SPECtypedefns, ONLY : REAL_KIND

    IMPLICIT NONE

    REAL(KIND=REAL_KIND),INTENT(IN) :: r !< coordinate input r
    INTEGER, INTENT(IN) :: lrad !< radial resolution
    INTEGER, INTENT(IN) :: mpol !< poloidal resolution
    REAL(KIND=REAL_KIND), INTENT(OUT) :: zernike(0:lrad,0:mpol,0:1) !< the value, first derivative of Zernike polynomial

    REAL(KIND=REAL_KIND) ::    rm, rm1  ! r to the power of m'th and m-1'th
    REAL(KIND=REAL_KIND) ::    factor1, factor2, factor3, factor4
    INTEGER :: m, n  ! Zernike R^m_n
    
    rm = one  ! r to the power of m'th
    rm1 = zero ! r to the power of m-1'th
    zernike(:,:,:) = zero
    DO m = 0, mpol
      IF (lrad >= m) then
        zernike(m,m,0:1) = (/ rm, FLOAT(m)*rm1 /)
      ENDIF

      IF (lrad >= m+2) then
        zernike(m+2,m,0) = FLOAT(m+2)*rm*r**2 - FLOAT(m+1)*rm
        zernike(m+2,m,1) = FLOAT((m+2)**2)*rm*r - FLOAT((m+1)*m)*rm1
      ENDIF

      DO n = m+4, lrad, 2
        factor1 = FLOAT(n)/FLOAT(n**2 - m**2)
        factor2 = FLOAT(4 * (n-1))
        factor3 = FLOAT((n-2+m)**2)/FLOAT(n-2) + FLOAT((n-m)**2)/FLOAT(n)
        factor4 = FLOAT((n-2)**2-m**2) / FLOAT(n-2)

        zernike(n, m, 0) = factor1 * ((factor2*r**2 - factor3)*zernike(n-2,m,0) - factor4*zernike(n-4,m,0))
        zernike(n, m, 1) = factor1 * (two*factor2*r*zernike(n-2,m,0) + (factor2*r**2 - factor3)*zernike(n-2,m,1) - factor4*zernike(n-4,m,1))
      ENDDO

      rm1 = rm
      rm = rm * r

    ENDDO

    DO n = 2, lrad, 2
      zernike(n,0,0) = zernike(n,0,0) - (-1)**(n/2)
    ENDDO

    IF (mpol >= 1) then
      DO n = 3, lrad, 2
        zernike(n,1,0) = zernike(n,1,0) - (-1)**((n-1)/2) * FLOAT((n+1)/2) * r
        zernike(n,1,1) = zernike(n,1,1) - (-1)**((n-1)/2) * FLOAT((n+1)/2)
      ENDDO
    END if

    DO m = 0, mpol
      DO n = m, lrad, 2
        zernike(n,m,:) = zernike(n,m,:) / FLOAT(n+1)
      END do
    END do
  END SUBROUTINE get_zernike

!> Get the Zernike polynomials with zeroth, first, second derivatives
  SUBROUTINE get_zernike_d2(r, lrad, mpol, zernike)
!f2py threadsafe

    USE SPECconstants, ONLY : zero, one, two
    USE SPECtypedefns, ONLY : REAL_KIND

    IMPLICIT NONE

    REAL(KIND=REAL_KIND),INTENT(IN) :: r !< coordinate input r
    INTEGER, INTENT(IN) :: lrad !< radial resolution
    INTEGER, INTENT(IN) :: mpol !< poloidal resolution
    REAL(KIND=REAL_KIND), INTENT(OUT) :: zernike(0:lrad,0:mpol,0:2) !< the value, first/second derivative of Zernike polynomial

    REAL(KIND=REAL_KIND) ::    rm, rm1, rm2  ! r to the power of m'th, m-1'th and m-2'th
    REAL(KIND=REAL_KIND) ::    factor1, factor2, factor3, factor4
    INTEGER :: m, n  ! Zernike R^m_n
    
    rm = one  ! r to the power of m'th
    rm1 = zero ! r to the power of m-1'th
    rm2 = zero ! r to the power of m-2'th
    zernike(:,:,:) = zero
    DO m = 0, mpol
      IF (lrad >= m) then
        zernike(m,m,0:2) = (/ rm, FLOAT(m)*rm1, FLOAT(m*(m-1))*rm2 /)
        !write(0, *) m, m, r, zernike(m,m,:)
      ENDIF

      IF (lrad >= m+2) then
        zernike(m+2,m,0) = FLOAT(m+2)*rm*r**2 - FLOAT(m+1)*rm
        zernike(m+2,m,1) = FLOAT((m+2)**2)*rm*r - FLOAT((m+1)*m)*rm1 
        zernike(m+2,m,2) = FLOAT((m+2)**2*(m+1))*rm - FLOAT((m+1)*m*(m-1))*rm2 
        !write(0, *) m+2, m, r, zernike(m+2,m,:)
      ENDIF

      DO n = m+4, lrad, 2
        factor1 = FLOAT(n)/FLOAT(n**2 - m**2)
        factor2 = FLOAT(4 * (n-1))
        factor3 = FLOAT((n-2+m)**2)/FLOAT(n-2) + FLOAT((n-m)**2)/FLOAT(n)
        factor4 = FLOAT((n-2)**2-m**2) / FLOAT(n-2)

        zernike(n, m, 0) = factor1 * ((factor2*r**2 - factor3)*zernike(n-2,m,0) - factor4*zernike(n-4,m,0))
        zernike(n, m, 1) = factor1 * (two*factor2*r*zernike(n-2,m,0) + (factor2*r**2 - factor3)*zernike(n-2,m,1) - factor4*zernike(n-4,m,1))
        zernike(n, m, 2) = factor1 * (two*factor2*(two*r*zernike(n-2,m,1) + zernike(n-2,m,0)) &
                          +(factor2*r**2 - factor3)*zernike(n-2,m,2) - factor4*zernike(n-4,m,2))
        !write(0, *) n, m, r, zernike(n,m,:)
      ENDDO

      rm2 = rm1
      rm1 = rm
      rm = rm * r

    ENDDO
    DO n = 2, lrad, 2
      zernike(n,0,0) = zernike(n,0,0) - (-1)**(n/2)
    ENDDO
    IF (mpol >= 1) then
      DO n = 3, lrad, 2
        zernike(n,1,0) = zernike(n,1,0) - (-1)**((n-1)/2) * FLOAT((n+1)/2) * r
        zernike(n,1,1) = zernike(n,1,1) - (-1)**((n-1)/2) * FLOAT((n+1)/2)
      ENDDO
    END if

    DO m = 0, mpol
      DO n = m, lrad, 2
        zernike(n,m,:) = zernike(n,m,:) / FLOAT(n+1)
      END do
    END do
  END SUBROUTINE get_zernike_d2

!> Get the Zernike polynomials Z(n,m,r)/r^m
  SUBROUTINE get_zernike_rm(r, lrad, mpol, zernike)
!f2py threadsafe

    USE SPECconstants, ONLY : zero, one, two
    USE SPECtypedefns, ONLY : REAL_KIND

    IMPLICIT NONE

    REAL(KIND=REAL_KIND),INTENT(IN) :: r !< coordinate input r
    INTEGER, INTENT(IN) :: lrad !< radial resolution
    INTEGER, INTENT(IN) :: mpol !< poloidal resolution
    REAL(KIND=REAL_KIND), INTENT(OUT) :: zernike(0:lrad,0:mpol) !< the value

    REAL(KIND=REAL_KIND) ::    factor1, factor2, factor3, factor4
    INTEGER :: m, n  ! Zernike R^m_n

    zernike(:,:) = zero
    DO m = 0, mpol
      IF (lrad >= m) then
        zernike(m,m) = one
      ENDIF

      IF (lrad >= m+2) then
        zernike(m+2,m) = FLOAT(m+2)*r**2 - FLOAT(m+1)
      ENDIF

      DO n = m+4, lrad, 2
        factor1 = FLOAT(n)/FLOAT(n**2 - m**2)
        factor2 = FLOAT(4 * (n-1))
        factor3 = FLOAT((n-2+m)**2)/FLOAT(n-2) + FLOAT((n-m)**2)/FLOAT(n)
        factor4 = FLOAT((n-2)**2-m**2) / FLOAT(n-2)

        zernike(n, m) = factor1 * ((factor2*r**2 - factor3)*zernike(n-2,m) - factor4*zernike(n-4,m))
      ENDDO

    ENDDO
    DO n = 2, lrad, 2
      zernike(n,0) = zernike(n,0) - (-1)**(n/2)
    ENDDO
    IF (mpol >= 1) then
      DO n = 3, lrad, 2
        zernike(n,1) = zernike(n,1) - (-1)**((n-1)/2) * FLOAT((n+1)/2)
      ENDDO
    END if

    DO m = 0, mpol
      DO n = m, lrad, 2
        zernike(n,m) = zernike(n,m) / FLOAT(n+1)
      END do
    END do
  END SUBROUTINE get_zernike_rm

END MODULE SPECbasefn