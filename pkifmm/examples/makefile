all: ex0 ex1 ex2 ex3 ex4 ex5 ex6

include ../makefile.this_machine

DEP     = 	$(LIB_SRC:.cpp=.d)

CPPFLAGS+=-I../fmm3d_mpi

ex0: ex0.o 
	${CLINKER} -o ex0 ex0.o ../fmm3d_mpi/libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 

ex1: ex1.o 
	${CLINKER} -o ex1 ex1.o ../fmm3d_mpi/libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 

ex2: ex2.o 
	${CLINKER} -o ex2 ex2.o ../fmm3d_mpi/libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 

ex3: ex3.o 
	${CLINKER} -o ex3 ex3.o ../fmm3d_mpi/libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 

ex4: ex4.o
	${CLINKER} -o ex4 ex4.o ../fmm3d_mpi/libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 

ex5: ex5.o
	${CLINKER} -o ex5 ex5.o ../fmm3d_mpi/libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 

ex6: ex6.o
	${CLINKER} -o ex6 ex6.o ../fmm3d_mpi/libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 
