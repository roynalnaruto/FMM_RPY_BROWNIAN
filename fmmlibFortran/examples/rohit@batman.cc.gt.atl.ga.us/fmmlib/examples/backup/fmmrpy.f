


        program fmmrpy

        implicit real *8 (a-h,o-z)

cc //source vector, vector x
        real *8     source(3,1 000 000)
	real *8 distance

cc //final thing to be computed
	complex *16 rpy(3,1 000 000)

cc  //force vector, vector v
        complex *16 charge(1 000 000) 

cc  //force vector, vector v_{i}
        complex *16 V1(1 000 000) 
        complex *16 V2(1 000 000) 
        complex *16 V3(1 000 000) 


	complex *16 C1
	complex *16 C2
	real *8 radius


cc //force vector for last call
        complex *16 V4(1 000 000) 



cc //constants C1 and C2
	complex *16 scale_Kb
	real *8 test1
	complex *16 test2
	complex *16 temp
	

cc //dipole strengths, p
        complex *16 dipstr(1 000 000)
        
cc //orientations of the dipole(of p)
	real *8     dipvec(3,1 000 000)

cc //potential at ith source, Pm(q,p,d) 
        complex *16 pot(1 000 000)


cc //potential at ith source, P1m(q,p,d) for the three calls 
        complex *16 P1(1 000 000)
        complex *16 P2(1 000 000)
        complex *16 P3(1 000 000)


cc //potential at ith source, P1m(q,p,d) for the fourth call 
        complex *16 P4(1 000 000)



cc //gradient of potential gradient at ith source, 
        complex *16 fld(3,1 000 000)



cc //gradient of potential gradient at ith source for all 3 calls, 
        complex *16 F1(3, 1 000 000)
        complex *16 F2(3, 1 000 000)
        complex *16 F3(3, 1 000 000)
        complex *16 F4(3, 1 000 000)


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
	radius = 0.5
	C1 = 0.75 
	C2 = 0.5 * radius * radius
	scale_Kb = 1


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

	dir = "20_0.5_close/"
	open(unit = 2, file = trim(dir)//"pos.csv.0000")  
	open(unit = 3, file = trim(dir)//"force.csv")	
	open(unit = 4, file = trim(dir)//"mf.csv")
	open(unit = 5, file = trim(dir)//"mf_final.csv")
	open(unit = 8, file = trim(dir)//"P1.csv")
	open(unit = 9, file = trim(dir)//"P2.csv")
	open(unit = 10, file = trim(dir)//"P3.csv")
	open(unit = 11, file = trim(dir)//"F1.csv")
	open(unit = 12, file = trim(dir)//"F2.csv")
	open(unit = 16, file = trim(dir)//"F3.csv")
	open(unit = 14, file = trim(dir)//"P4.csv")
	open(unit = 15, file = trim(dir)//"F4.csv")


	

	read (2,*) nsource

        do i=1,nsource
        enddo
cc	read(2,*)
c
       do i=1,nsource
                read (2,*) source(1,i), source(2,i), source(3,i)
		read (3,*) V1(i), V2(i), V3(i)
		read (4,*) rpyActual(1,i), rpyActual(2,i), rpyActual(3,i)
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
		write(16,*) F3(1,i), F3(2,i), F3(3,i)
	enddo





CC The fourth call, no need to compute potential
	ifdipole = 1
	ifpot = 1
	iffld = 1

	do i=1,nsource
		dipvec(1,i) = V1(i)
		dipvec(2,i) = V2(i)
		dipvec(3,i) = V3(i)
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



c     get time for FMM calls
        t2=second()
c       

        call prinf('nsource=*',nsource,1)
        call prin2('after 4 fmm calls, time (sec)=*',t2-t1,1)
        call prin2('after 4 fmm calls, speed (points/sec)=*',
     $     (nsource)/(t2-t1),1)
       

	do m=1,nsource
	 do i=1,3
		rpy(i,m) = (source(1, m) * F1(i, m)) + 
     $                     (source(2, m) * F2(i, m)) + 
     $                     (source(3, m) * F3(i, m))
		if(i .eq. 1) then	  	
			rpy(i, m) = ((C1 * P1(m)) + (C1 * rpy(i,m))) - F4(i, m)
		endif 	           
		if(i .eq. 2) then	  	
			rpy(i, m) = ((C1 * P2(m)) + (C1 * rpy(i,m))) - F4(i, m)
		endif 	           
		if(i .eq. 3) then	  	
			rpy(i, m) = ((C1 * P3(m)) + (C1 * rpy(i,m))) - F4(i, m)
		endif
	enddo
	enddo



cc POST CORRECTION 
	do i=1,nsource
		do j=1,nsource
		distance = (source(1,i) - source(1,j)) * 
     $                     (source(1,i) - source(1,j))

		distance = distance +
     $                     (source(2,i) - source(2,j)) * 
     $                     (source(2,i) - source(2,j))

		distance = distance +
     $                     (source(3,i) - source(3,j)) * 
     $                     (source(3,i) - source(3,j))

		if((i .ne. j) .AND. (distance < 4 * radius * radius)) then
c			write(*,*) i," :here ::::",j, " :here:::: ",nsource 		  	
			call postcorrection(i, j, source, V1, V2, V3, rpy,
     $ 				nsource, radius,C1,C2)  
	 	endif	
		enddo
	enddo
	
	do m=1,nsource
	write(5,*) rpy(1,m), rpy(2,m), rpy(3,m)
	enddo

	call rpyerror(rpy,rpyActual,nsource,aerr,rerr)
	call prin2('relative error in rpy=*',rerr,1)
	call prin2('absolute error in rpy=*',aerr,1)


        stop
        end










	subroutine postcorrection(i, j, source, V1, V2, V3, rpy,
     $			 nsource,radius,C1,C2)

	        real *8     source(3,nsource)
		complex *16 V1(nsource) 
		complex *16 V2(nsource) 
		complex *16 V3(nsource) 
		complex *16 rpy(3,nsource)
		complex *16 C1,C2
	  	real *8 radius
	
		real *8 pc(3,3)
		real *8 rCrossr(3,3)
		real *8 identity

	
		complex *16 C3, C4, temp
		C3 = (4 * C1)/(3 * radius)
		C4 = C3 * (3/(32 * radius))

		do m=1,3
		  do n=1,3
			rCrossr(m,n) = (source(m,i) - source(m,j)) * 
     $                                 (source(n,i) - source(n,j)) 
		enddo

		dist = 0.0
		  enddo
		do m=1,3
			dist = dist + rCrossr(m,m)
		enddo
		dist = sqrt(dist)


		do m=1,3
		 do n=1,3
			
			identity = 0
			if(m .eq. n) then
				identity = 1
			endif
			
			
			pc(m,n)=((1 - ((9 * dist)/(32 * radius))) * identity)
     $				* C3

			pc(m,n) = pc(m,n) + 
     $                             ((C4 * rCrossr(m,n))/(dist))


			pc(m,n) = pc(m,n) - 
     $                             ((C1/dist) * (identity  + 
     $                             (rCrossr(m,n)/(dist * dist))))
			
c			write(*,*) pc(m,n), identity, "1) pc(m,n),identity"

			pc(m,n) = pc(m,n) - ((c2/(dist * dist * dist))
     $              * (identity - ((3 * rCrossr(m,n))/(dist * dist))))
			
c			write(*,*) pc(m,n), identity, "2) pc(m,n),identity"

      		 enddo
		enddo
		

		do m=1,3
		     temp = (pc(m,1) * V1(j)) + 
     $                     (pc(m,2) * V2(j)) +
     $                     (pc(m,3) * V3(j))
		     rpy(m,i) = rpy(m,i) + temp
	      	 enddo
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










