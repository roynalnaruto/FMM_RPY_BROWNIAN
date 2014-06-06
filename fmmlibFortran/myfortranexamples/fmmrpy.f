        program fmmrpy

        implicit real *8 (a-h,o-z)

cc //source vector, vector x
        real *8     source(3,1 000 000)

cc  //force vector, vector v
        complex *16 charge(1 000 000) 

cc //dipole strengths, p
        complex *16 dipstr(1 000 000)
        
cc //orientations of the dipole(of p)
	real *8     dipvec(3,1 000 000)

cc //potential at ith source, Pm(q,p,d) 
        complex *16 pot(1 000 000)
cc //gradient of potential gradient at ith source, 
        complex *16 fld(3,1 000 000)

       
cc //For rechecking
        complex *16 pot2(1 000 000)
        complex *16 fld2(3,1 000 000)

cc //temporaries
        complex *16 ptemp,ftemp(3)
c       
        complex *16 ima
        data ima/(0.0d0,1.0d0)/
c
        done=1
        pi=4*atan(done)
c
c     Initialize simple printing routines. The parameters to prini
c     define output file numbers using standard Fortran conventions.
c
c     Calling prini(6,13) causes printing to the screen and to 
c     file fort.13.     
c
        call prini(6,13)
c
        nsource= 16000
c
c     construct randomly located charge distribution on a unit sphere
c 
        d=hkrand(0)
        do i=1,nsource
           theta=hkrand(0)*pi
           phi=hkrand(0)*2*pi
           source(1,i)=.5d0*cos(phi)*sin(theta)
           source(2,i)=.5d0*sin(phi)*sin(theta)
           source(3,i)=.5d0*cos(theta)
        enddo
c
c     set precision flag
c
        iprec=1
        call prinf('iprec=*',iprec,1)
c       
c     set source type flags and output flags
c
        ifpot=1
        iffld=1
c
        ifcharge=1
        ifdipole=1
c
c       set source strengths
c
        if (ifcharge .eq. 1 ) then
           do i=1,nsource
              charge(i)=hkrand(0) + ima*hkrand(0)
           enddo
        endif
c
        if (ifdipole .eq. 1) then
           do i=1,nsource
              dipstr(i)=hkrand(0) + ima*hkrand(0)
              dipvec(1,i)=hkrand(0)
              dipvec(2,i)=hkrand(0)
              dipvec(3,i)=hkrand(0)
           enddo
        endif
c
c     initialize timing call
c
        t1=second()
C$        t1=omp_get_wtime()
c       
c     call FMM3D routine for sources and targets
c
        call lfmm3dparttarg(ier,iprec,
     $     nsource,source,ifcharge,charge,ifdipole,dipstr,dipvec,
     $     ifpot,pot,iffld,fld,ntarget,target,
     $     ifpottarg,pottarg,iffldtarg,fldtarg)


cc        call lfmm3dparttarg(ier,iprec,
cc     $     nsource,source,ifcharge,charge,ifdipole,dipstr,dipvec,
cc     $     ifpot,pot,iffld,fld)




c       
c     get time for FMM call
c
        t2=second()
C$        t2=omp_get_wtime()
c       
c       
        call prinf('nsource=*',nsource,1)
        call prin2('after fmm, time (sec)=*',t2-t1,1)
        call prin2('after fmm, speed (points/sec)=*',
     $     (nsource)/(t2-t1),1)
c       

