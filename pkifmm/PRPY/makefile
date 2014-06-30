all: rpy

include ../makefile.this_machine

DEP     = 	$(LIB_SRC:.cpp=.d)

CPPFLAGS+=-I../fmm3d_mpi

rpy: rpy.o
	${CLINKER} -o rpy rpy.o ../fmm3d_mpi/libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 
