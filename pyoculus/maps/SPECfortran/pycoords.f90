!> @file pycoords.f90
!> @brief Fortran module for SPEC problems, computing geometry relevant quantities
!> @author Zhisong Qu (zhisong.qu@anu.edu.au)
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!


!> Fortran module for SPEC problems, computing geometry relevant quantities
!>
!> ### SPEC coordinates
!> - `Igeometry=1`, slab geometry
!>   \f$ \mathbf{x} = R(\theta, \zeta) \mathbf{i} + \mbox{rpol} \times \theta \mathbf{j} + \mbox{rtor} \times \zeta \mathbf{j} \f$
!>
!>   `get_xyz` returns \f$(R, \mbox{dummy}, \mbox{dummy}) \f$
!>
!> - `Igeometry=2`, cylindrical geometry
!>
!>   Something
!>
!> - `Igeometry=3`, toroidal geometry
!>
MODULE SPECcoords

  CONTAINS

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

!> Compute Cartisian coordinates based on toroidal coordinates \f$(s,\theta,\zeta)\f$.
  SUBROUTINE get_xyz(stz , RpZ )
!f2py threadsafe  
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
  
    USE SPECtypedefns, ONLY : REAL_KIND
    USE SPECconstants, ONLY : zero, one, half
    USE SPECvariables, ONLY : Igeometry, Ntor, mn, im, in1, &
                          iRbc, iZbs, iRbs, iZbc, &
                          Lcoordinatesingularity, &
                          NOTstellsym, &
                          Mvol, ivol
    
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
  
    IMPLICIT NONE  
  
    
    REAL(KIND=REAL_KIND),    INTENT(IN)  :: stz(1:3) !< \f$(s,\theta,\zeta)\f$
    REAL(KIND=REAL_KIND),    INTENT(OUT) :: RpZ(1:3) !< the Cartesian coordinates, depending on `Igeometry`
  
    INTEGER              :: ii, mi, ni, lvol
    REAL(KIND=REAL_KIND) :: Remn, Zomn, Romn, Zemn, RR, phi, ZZ, arg, carg, sarg, lss, alss, blss, sbar, sbarhim, fj
  
    lvol = ivol

    RpZ(1:3) = zero ; RR = zero ; phi = stz(3) ; ZZ = zero ! initialize intent(out), summations ; 17 Dec 15;
  
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
  
    lss = stz(1) ; alss = half - lss * half ; blss = half + lss * half ! shorthand; 17 Dec 15;
  
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    DO ii = 1, mn ; mi = im(ii) ; ni = in1(ii) ; arg = mi * stz(2) - ni * phi ; carg = COS(arg) ; sarg = SIN(arg) ! Fourier sum; 17 Dec 15;
   
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
   
      Remn = zero ; Zomn = zero
      Romn = zero ; Zemn = zero
   
      IF( Lcoordinatesingularity ) THEN
    
        sbar = ( lss + one ) * half
    
        IF( mi.EQ.0 ) THEN
          IF( Igeometry.EQ.2 ) THEN ; fj = sbar
          ELSE                      ; fj = sbar**2
          ENDIF
        ELSE
          IF( Igeometry.EQ.2 ) THEN ; fj = sbar**(im(ii)+1)
          ELSE                      ; fj = sbar**(im(ii))
          ENDIF                        ; 
        ENDIF
    
        Remn = iRbc(ii,1) + ( iRbc(ii,2) - iRbc(ii,1) ) * fj
        IF( NOTstellsym ) THEN
          Romn = iRbs(ii,1) + ( iRbs(ii,2) - iRbs(ii,1) ) * fj
        ENDIF
        IF( Igeometry.eq.3 ) THEN ! recall that for cylindrical geometry there is no need for Z; 20 Apr 13;
          Zomn = iZbs(ii,1) + ( iZbs(ii,2) - iZbs(ii,1) ) * fj
          IF( NOTstellsym ) THEN
            Zemn = iZbc(ii,1) + ( iZbc(ii,2) - iZbc(ii,1) ) * fj
          ENDIF
        ENDIF

      ELSE ! matches IF( Lcoordinatesingularity) ; 20 Apr 13;
    
        Remn =   alss * iRbc(ii,lvol) + blss * iRbc(ii,lvol+1)
        IF( NOTstellsym ) THEN
          Romn =   alss * iRbs(ii,lvol) + blss * iRbs(ii,lvol+1)
        ELSE
          Romn = zero
        ENDIF ! end of IF( NOTstellsym ) ; 22 Apr 13;
        IF( Igeometry.eq.3 ) THEN
          Zomn =   alss * iZbs(ii,lvol) + blss * iZbs(ii,lvol+1)
          IF( NOTstellsym ) THEN
            Zemn =   alss * iZbc(ii,lvol) + blss * iZbc(ii,lvol+1)
          ELSE
            Zemn = zero
          ENDIF ! end of IF( NOTstellsym ) ; 22 Apr 13; 
        ELSE    
          Zomn = zero
          Zemn = zero
        ENDIF ! end of IF( Igeometry.eq.3 ) ; 22 Apr 13;
    
      ENDIF ! end of IF( Lcoordinatesingularity ) ; 20 Feb 13;
   
      RR = RR + Remn * carg + Romn * sarg
      IF( Igeometry.eq.3 ) THEN
        ZZ = ZZ + Zemn * carg + Zomn * sarg
      ENDIF
   
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
   
    ENDDO ! end of do ii = 1, mn ; Fourier summation loop;

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    RpZ(1:3) = (/ RR, phi, ZZ /) ! return coordinate functions;
  
!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
  
  END SUBROUTINE get_xyz

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

! get the metric for PJH calculation on the interface

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

  SUBROUTINE get_metric_interface(ioi, theta, zeta, dR, dZ, guvij, sg, ideriv)

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    USE SPECtypedefns, ONLY : REAL_KIND
    USE SPECconstants
    USE SPECvariables

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    IMPLICIT NONE
    REAL(KIND=REAL_KIND),    INTENT(IN) :: theta, zeta ! the three coordinates (s,theta,zeta)
    INTEGER             ,    INTENT(IN) :: ioi ! on which interface
    INTEGER             ,    INTENT(IN) :: ideriv ! the level of derivatives. 0 for none, 1 for first, 2 for second

    REAL(KIND=REAL_KIND),    INTENT(OUT):: dR(0:3,0:3,0:3), dZ(0:3,0:3,0:3)
    REAL(KIND=REAL_KIND),    INTENT(OUT):: guvij(3,3,0:2), sg(0:2)

    REAL(KIND=REAL_KIND) :: lss, rsign
    REAL(KIND=REAL_KIND) :: cosarg(mn), sinarg(mn)

    INTEGER              :: iother ! the index of the other interface with in the same volume
    INTEGER              :: ii, jj
    
    IF (ivol .EQ. ioi) THEN
      ! inner interface, the other interface is the outer interface
      iother = ivol + 1
      rsign = -half
    ELSE
      ! outer interface, the other interface is the inner interface
      iother = ivol
      rsign = half
    ENDIF

    cosarg = COS(im*theta-in1*zeta)
    sinarg = SIN(im*theta-in1*zeta)

    dR(0,0,0) = SUM(iRbc(:,ioi)*cosarg)             !dR(0,0,0) + iRbc(ioi,imn)*cs(1)
    dR(2,0,0) = SUM(iRbc(:,ioi)*sinarg * (-im ))    !dR(2,0,0) + iRbc(ioi,imn)*cs(2)*(-im(imn))
    dR(3,0,0) = SUM(iRbc(:,ioi)*sinarg * (+in1))    !dR(3,0,0) + iRbc(ioi,imn)*cs(2)*(+in(imn))

    ! compute dR/ds, dZ/ds
    IF (Lcoordinatesingularity) THEN
      dR(1,0,0) = (SUM(iRbc(:,ioi) * im * cosarg) + two * SUM((iRbc(1:Ntor+1,ioi)-iRbc(1:Ntor+1,ioi))*cosarg(1:Ntor+1))) * half
    ELSE
      dR(1,0,0) = (dR(0,0,0) - SUM(iRbc(:,iother)*cosarg)) * rsign
    ENDIF

    IF (ideriv .ge. 1) THEN ! need 1st derivative w.r.t. theta

      dR(2,2,0) = SUM(iRbc(:,ioi)*cosarg * (-im ) * (+im )) !dR(2,2,0) + iRbc(ioi,imn)*cs(1)*(-im(imn))*(+im(imn))
      dR(3,2,0) = SUM(iRbc(:,ioi)*cosarg * (+in1) * (+im )) !dR(3,2,0) + iRbc(ioi,imn)*cs(1)*(+in(imn))*(+im(imn))

      IF (Lcoordinatesingularity) THEN
        dR(1,2,0) = (SUM(iRbc(:,ioi) * im * (-im) * sinarg)) * half
      ELSE
        dR(1,2,0) = (dR(2,0,0) - SUM(iRbc(:,iother) * (-im) *sinarg)) * rsign
      ENDIF

    END IF

    IF (ideriv .ge. 2) THEN ! need 2nd derivative w.r.t. theta
    
      dR(2,2,2) = SUM(iRbc(:,ioi) * sinarg * (-im) * (+im) * (-im)) !dR(2,2,2) + iRbc(ioi,imn)*cs(2)*(-im(imn))*(+im(imn))*(-im(imn))
      dR(3,2,2) = SUM(iRbc(:,ioi) * sinarg * (in1) * (+im) * (-im)) !dR(3,2,2) + iRbc(ioi,imn)*cs(2)*(+in(imn))*(+im(imn))*(-im(imn))

      IF (Lcoordinatesingularity) THEN
        dR(1,2,2) = (SUM(iRbc(:,ioi) * im * (-im) * (+im) * cosarg)) * half
      ELSE
        dR(1,2,2) = (dR(2,2,0) - SUM(iRbc(:,iother) * (-im) * (+im) * cosarg)) * rsign
      ENDIF
      
    END IF

    IF (Igeometry .eq. 3) THEN
      dZ(0,0,0) = SUM(iZbs(:,ioi)*sinarg)             !dZ(0,0,0) + iZbs(ioi,imn)*cs(2)
      dZ(2,0,0) = SUM(iZbs(:,ioi)*cosarg * (+im ))    !dZ(2,0,0) + iZbs(ioi,imn)*cs(1)*(+im(imn))
      dZ(3,0,0) = SUM(iZbs(:,ioi)*cosarg * (-in1))    !dZ(3,0,0) + iZbs(ioi,imn)*cs(1)*(-in(imn))

      ! compute dR/ds, dZ/ds
      IF (Lcoordinatesingularity) THEN
        dZ(1,0,0) = (SUM(iZbs(:,ioi) * im * sinarg) + two * SUM((iZbs(1:Ntor+1,ioi)-iZbs(1,1:Ntor+1))*sinarg(1:Ntor+1)))*half
      ELSE
        dZ(1,0,0) = (dZ(0,0,0) - SUM(iZbs(:,iother)*sinarg)) * rsign
      ENDIF

      IF (ideriv .ge. 1) THEN ! need 1st derivative w.r.t. theta

        dZ(2,2,0) = SUM(iZbs(:,ioi)*sinarg * (+im ) * (-im )) !dZ(2,2,0) + iZbs(ioi,imn)*cs(2)*(+im(imn))*(-im(imn))
        dZ(3,2,0) = SUM(iZbs(:,ioi)*sinarg * (-in1) * (-im )) !dZ(3,2,0) + iZbs(ioi,imn)*cs(2)*(-in(imn))*(-im(imn))

        IF (Lcoordinatesingularity) THEN
          dZ(1,2,0) = (SUM(iZbs(:,ioi) * im * (+im) * cosarg)) * half
        ELSE
          dZ(1,2,0) = (dZ(2,0,0) - SUM(iZbs(:,iother) * (im) *cosarg)) * rsign
        END IF

      END IF

      IF (ideriv .ge. 2) THEN ! need 2nd derivative w.r.t. theta
      
        dZ(2,2,2) = SUM(iZbs(:,ioi) * cosarg * (+im) * (-im) * (+im)) !dZ(2,2,2) + iZbs(ioi,imn)*cs(1)*(+im(imn))*(-im(imn))*(+im(imn))
        dZ(3,2,2) = SUM(iZbs(:,ioi) * cosarg * (-in1)* (-im) * (+im)) !dZ(3,2,2) + iZbs(ioi,imn)*cs(1)*(-in(imn))*(-im(imn))*(+im(imn))

        IF (Lcoordinatesingularity) THEN
          dZ(1,2,2) = (SUM(iZbs(:,ioi) * im * (+im) * (-im) * sinarg)) * half
        ELSE
          dZ(1,2,2) = (dZ(2,2,0) - SUM(iZbs(:,iother) * (+im) * (-im) * sinarg)) * rsign
        END IF
        
      END IF ! ideriv >= 2

    END IF ! Igeometry == 3

    ! assemble metric
    IF (Igeometry .eq. 1) THEN

      sg(0) = dR(1,0,0)*rpol*rtor

      DO ii = 2, 3
        DO jj = 2, 3
          guvij(ii,jj,0) = dR(ii,0,0) * dR(jj,0,0)
        ENDDO
      ENDDO

      guvij(2,2,0) = guvij(2,2,0) + rpol*rpol
      guvij(3,3,0) = guvij(3,3,0) + rtor*rtor

      IF (ideriv .ge. 1) THEN
        sg(1) = dR(1,2,0)*rpol*rtor
        DO ii = 2, 3
          DO jj = 2, 3
            guvij(ii,jj,1) = dR(ii,2,0) * dR(jj,0,0) + dR(ii,0,0) * dR(jj,2,0)
          ENDDO
        ENDDO
      ENDIF

      IF (ideriv .ge. 2) THEN
        sg(2) = dR(1,2,2)*rpol*rtor
        DO ii = 2, 3
          DO jj = 2, 3
            guvij(ii,jj,2) = two * dR(ii,2,0) * dR(jj,2,0) + dR(ii,2,2) * dR(jj,0,0) + dR(ii,0,0) * dR(jj,2,2)
          ENDDO
        ENDDO
      ENDIF

    ELSEIF (Igeometry .eq. 2) THEN

    ELSEIF (Igeometry .eq. 3) THEN

      sg(0) = (dR(2,0,0)*dZ(1,0,0) - dR(1,0,0)*dZ(2,0,0)) * dR(0,0,0)

      DO ii = 2, 3
        DO jj = 2, 3
          guvij(ii,jj,0) = dR(ii,0,0) * dR(jj,0,0) + dZ(ii,0,0) * dZ(jj,0,0)
        ENDDO
      ENDDO

      guvij(3,3,0) = guvij(3,3,0) + dR(0,0,0)**2

      IF (ideriv .ge. 1) THEN
        sg(1) = (dR(2,2,0)*dZ(1,0,0) + dR(2,0,0)*dZ(1,2,0) &
              -  dR(1,2,0)*dZ(2,0,0) - dR(1,0,0)*dZ(2,2,0)) * dR(0,0,0) &
              + (dR(2,0,0)*dZ(1,0,0) - dR(1,0,0)*dZ(2,0,0)) * dR(2,0,0)
        DO ii = 2, 3
          DO jj = 2, 3
            guvij(ii,jj,1) = dR(ii,2,0) * dR(jj,0,0) + dR(ii,0,0) * dR(jj,2,0) &
                           + dZ(ii,2,0) * dZ(jj,0,0) + dZ(ii,0,0) * dZ(jj,2,0)
          ENDDO
        ENDDO

        guvij(3,3,1) = guvij(3,3,1) + two * dR(2,0,0) * dR(0,0,0)
      ENDIF

      IF (ideriv .ge. 2) THEN
        sg(2) = (dR(2,2,2)*dZ(1,0,0) + two * dR(2,2,0)*dZ(1,2,0) + dR(2,0,0)*dZ(1,2,2) &
              -  dR(1,2,2)*dZ(2,0,0) - two * dR(1,2,0)*dZ(2,2,0) - dR(1,0,0)*dZ(2,2,2)) * dR(0,0,0) &
              + (dR(2,0,0)*dZ(1,0,0) - dR(1,0,0)*dZ(2,0,0)) * dR(2,2,0) &
              + two * (dR(2,2,0)*dZ(1,0,0) + dR(2,0,0)*dZ(1,2,0) &
              -  dR(1,2,0)*dZ(2,0,0) - dR(1,0,0)*dZ(2,2,0)) * dR(2,0,0)
        DO ii = 2, 3
          DO jj = 2, 3
            guvij(ii,jj,2) = dR(ii,2,2) * dR(jj,0,0) + dR(ii,0,0) * dR(jj,2,2) + two * dR(ii,2,0) * dR(jj,2,0) &
                           + dZ(ii,2,2) * dZ(jj,0,0) + dZ(ii,0,0) * dZ(jj,2,2) + two * dZ(ii,2,0) * dZ(jj,2,0)
          ENDDO
        ENDDO

        guvij(3,3,2) = guvij(3,3,2) + two * (dR(2,0,0) * dR(2,0,0) + dR(2,2,0) * dR(0,0,0))
      ENDIF

    ENDIF

  END SUBROUTINE get_metric_interface

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!
END MODULE SPECcoords