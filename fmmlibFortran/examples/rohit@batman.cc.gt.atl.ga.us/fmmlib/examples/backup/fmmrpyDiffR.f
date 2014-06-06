        program fmmrpy

        implicit real *8 (a-h,o-z)

cc //source vector, vector x
        real *8 source(3,1 000 000)
	real *8 radii(1 000 000)    	
	real *8 distance

cc //final thing to be computed
	complex *16 rpy(3,1 000 000)

cc  //force vector, vector v
        complex *16 charge(1 000 000) 

cc  //force vector, vector v_{i}
        complex *16 V1(1 000 000) 
        complex *16 V2(1 000 000) 
        complex *16 V3(1 000 000) 



cc //force vector for last call
        complex *16 V4(1 000 000) 

cc //force vector for last call
        complex *16 V5(1 000 000) 



cc //constants C1 and C2
	complex *16 C1
	complex *16 C2
	real *8 radius
	complex *16 scale_Kb
	real *8 test1
	complex *16 test2
	complex *16 temp
	

cc //dipole strengths, p
        complex *16 dipstr(1 000 000)
        
cc //orientations of the dipole(of p)
	real *8     dipvec(3,1 000 000)



cc //potential at ith source, P1m(q,p,d) for the three calls 
        complex *16 P1(1 000 000)
        complex *16 P2(1 000 000)
        complex *16 P3(1 000 000)


cc //potential at ith source, P1m(q,p,d) for the fourth call 
        complex *16 P4(1 000 000)
        complex *16 P5(1 000 000)




cc //gradient of potential gradient at ith source for all 3 calls, 
        complex *16 F1(3, 1 000 000)
        complex *16 F2(3, 1 000 000)
        complex *16 F3(3, 1 000 000)
        complex *16 F4(3, 1 000 000)
        complex *16 F5(3, 1 000 000)


cc //For rechecking
        complex *16 rpyActual(3,1 000 000)


cc //temporaries
        complex *16 ptemp,ftemp(3)
c       
	CHARACTER(LEN=100) :: dir

        complex *16 ima
        data ima/(0.0d0,1.0d0)/
c
        done=1
        pi=4*atan(done)


cc //Values of C1 and C2
	C1 = 0.75 
	C2 = 0.5


c
c     Initialize simple printing routines. The parameters to prini
c     define output file numbers using standard Fortran conventions.
c
c     Calling prini(6,13) causes printing to the screen and to 
c     file fort.13.     
c
        call prini(6,13)
c

cc //!! change this
cc   nsource= 16000




     


c
c
c     set precision flag
c
        iprec=0
        call prinf('iprec=*',iprec,1)
c       
c     set source type flags and output flags
c
        ifpot=1
        iffld=1
c
        ifcharge=1
        ifdipole=1



cc     read input from files

	dir = "20_diffR_far/"
	open(unit = 2, file = trim(dir)//"pos.csv.0000")  
	open(unit = 3, file = trim(dir)//"force.csv")	
	open(unit = 4, file = trim(dir)//"mf.csv")
	open(unit = 5, file = trim(dir)//"mf_final.csv")
	open(unit = 8, file = trim(dir)//"P1.csv")
	open(unit = 9, file = trim(dir)//"P2.csv")
	open(unit = 10, file = trim(dir)//"P3.csv")
	open(unit = 11, file = trim(dir)//"F1.csv")
	open(unit = 12, file = trim(dir)//"F2.csv")
	open(unit = 19, file = trim(dir)//"F3.csv")
	open(unit = 14, file = trim(dir)//"P4.csv")
	open(unit = 15, file = trim(dir)//"F4.csv")
	open(unit = 16, file = trim(dir)//"radius.csv")
	open(unit = 17, file = trim(dir)//"P5.csv")
	open(unit = 18, file = trim(dir)//"F5.csv")
	



	

	read (2,*) nsource

        do i=1,nsource
        enddo
cc	read(2,*)
c
       do i=1,nsource
                read (2,*) source(1,i), source(2,i), source(3,i)
		read (3,*) V1(i), V2(i), V3(i)
		read (4,*) rpyActual(1,i), rpyActual(2,i), rpyActual(3,i)
		read (16,*) radii(i)
	enddo
       

c     initialize timing call
c
        t1=second()

cc Ignore the effect of dipole moments for first three calls	
	ifdipole = 0
	ifpot = 1
	iffld = 1


	do i=1,nsource
		dipvec(1,i) = 0
		dipvec(2,i) = 0
		dipvec(3,i) = 0
		dipstr(i) = 0
	enddo


        call lfmm3dpartself(ier,iprec,
     $     nsource,source,ifcharge,V1,ifdipole,dipstr,dipvec,
     $     ifpot,P1,iffld,F1)

        call lfmm3dpartself(ier,iprec,
     $     nsource,source,ifcharge,V2,ifdipole,dipstr,dipvec,
     $     ifpot,P2,iffld,F2)

        call lfmm3dpartself(ier,iprec,
     $     nsource,source,ifcharge,V3,ifdipole,dipstr,dipvec,
     $     ifpot,P3,iffld,F3)



	do i=1,nsource
		write(8,*) P1(i)
		write(9,*) P2(i)
		write(10,*) P3(i)
		write(11,*) F1(1,i), F1(2,i), F1(3,i)
		write(12,*) F2(1,i), F2(2,i), F2(3,i)
		write(19,*) F3(1,i), F3(2,i), F3(3,i)
	enddo



CC The fourth call, no need to compute potential
	ifdipole = 1
	ifpot = 1
	iffld = 1

	do i=1,nsource
		dipvec(1,i) = (V1(i) * radii(i) * radii(i))/2
		dipvec(2,i) = (V2(i) * radii(i) * radii(i))/2
		dipvec(3,i) = (V3(i) * radii(i) * radii(i))/2
		dipstr(i) = C2
		V4(i) = source(1,i) * V1(i)
		V4(i) = V4(i) + (source(2,i) * V2(i))
		V4(i) = V4(i) + (source(3,i) * V3(i))
		V4(i) = C1 * V4(i)
	enddo

        call lfmm3dpartself(ier,iprec,
     $     nsource,source,ifcharge,V4,ifdipole,dipstr,dipvec,
     $     ifpot,P4,iffld,F4)


	do i=1,nsource
		write(14,*) P4(i)
		write(15,*) F4(1,i), F4(2,i), F4(3,i)
	enddo






CC The fifth call, no need to compute potential
	ifdipole = 1
	ifpot = 1
	iffld = 1

	do i=1,nsource
		dipvec(1,i) = V1(i)
		dipvec(2,i) = V2(i)
		dipvec(3,i) = V3(i)
		dipstr(i) = C2
		V5(i) = 0
	enddo

        call lfmm3dpartself(ier,iprec,
     $     nsource,source,ifcharge,V5,ifdipole,dipstr,dipvec,
     $     ifpot,P5,iffld,F5)


	do i=1,nsource
		write(17,*) P5(i)
		write(18,*) F5(1,i), F5(2,i), F5(3,i)
	enddo









c     get time for FMM calls
        t2=second()
c       

        call prinf('nsource=*',nsource,1)
        call prin2('after 5 fmm calls, time (sec)=*',t2-t1,1)
        call prin2('after 5 fmm calls, speed (points/sec)=*',
     $     (nsource)/(t2-t1),1)
       

	do m=1,nsource
	 do i=1,3
		rpy(i,m) = (source(1, m) * F1(i, m)) + 
     $                     (source(2, m) * F2(i, m)) + 
     $                     (source(3, m) * F3(i, m))
		
		rpy(i,m) = C1 * rpy(i,m)
		if(i .eq. 1) then	  	
			rpy(i, m) = ((C1 * P1(m)) + rpy(i,m))
		endif 	           
		if(i .eq. 2) then	  	
			rpy(i, m) = ((C1 * P2(m)) + rpy(i,m))
		endif 	           
		if(i .eq. 3) then	  	
			rpy(i, m) = ((C1 * P3(m)) + rpy(i,m))
		endif
		rpy(i,m) = rpy(i,m) - F4(i,m)
		rpy(i,m) = rpy(i,m) - (radii(m) * radii(m) * 0.5 * F5(i,m))
	enddo
	write(5,*) rpy(1,m), rpy(2,m), rpy(3,m)
	enddo




	call rpyerror(rpy,rpyActual,nsource,aerr,rerr)
	call prin2('relative error in rpy=*',rerr,1)
	call prin2('absolute error in rpy=*',aerr,1)


        stop
        end




	subroutine computeRPY(source, radii, charge, rpy)

	        implicit real *8 (a-h,o-z)

cc //final thing to be computed
		complex *16 rpy(3,1 000 000)

cc  //force vector, vector v
	        complex *16 charge(1 000 000) 

cc //source vector, vector x
		real *8 source(3,1 000 000)
		real *8 radii(1 000 000)    	
		real *8 distance



cc  //force vector, vector v_{i}
		complex *16 V1(1 000 000) 
		complex *16 V2(1 000 000) 
		complex *16 V3(1 000 000) 



cc //force vector for last call
	        complex *16 V4(1 000 000) 

cc //force vector for last call
	        complex *16 V5(1 000 000) 



cc //constants C1 and C2
		complex *16 C1
		complex *16 C2
		real *8 radius
		complex *16 scale_Kb
		real *8 test1
		complex *16 test2
		complex *16 temp
	

cc //dipole strengths, p
	        complex *16 dipstr(1 000 000)
        
cc //orientations of the dipole(of p)
		real *8     dipvec(3,1 000 000)



cc //potential at ith source, P1m(q,p,d) for the three calls 
		complex *16 P1(1 000 000)
		complex *16 P2(1 000 000)
		complex *16 P3(1 000 000)


cc //potential at ith source, P1m(q,p,d) for the fourth call 
		complex *16 P4(1 000 000)
		complex *16 P5(1 000 000)




cc //gradient of potential gradient at ith source for all 3 calls, 
		complex *16 F1(3, 1 000 000)
		complex *16 F2(3, 1 000 000)
		complex *16 F3(3, 1 000 000)
		complex *16 F4(3, 1 000 000)
		complex *16 F5(3, 1 000 000)


cc //For rechecking
        complex *16 rpyActual(3,1 000 000)


cc //temporaries
        complex *16 ptemp,ftemp(3)
c       
	CHARACTER(LEN=100) :: dir

        complex *16 ima
        data ima/(0.0d0,1.0d0)/
c


	return 
	end



        subroutine rpyerror(rpy1,rpy2,n,ae,re)
        implicit real *8 (a-h,o-z)
c
c       evaluate absolute and relative errors
c
        complex *16 rpy1(3,n),rpy2(3,n)
c
        d=0
        a=0
c       
        do i=1,n
	 do j=1,3
           d=d+abs(rpy1(j,i)-rpy2(j,i))**2
           a=a + abs(rpy1(j,i))**2
        enddo
	enddo
c       
        d=d/n
        d=sqrt(d)
        a=a/n
        a=sqrt(a)
c       
        ae=d
        re=d/a
c       
        return
        end
c







c
c
c
