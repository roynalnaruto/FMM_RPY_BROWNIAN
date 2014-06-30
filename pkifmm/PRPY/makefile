all: rpy

include ../makefile.this_machine

DEP     = 	$(LIB_SRC:.cpp=.d)

CPPFLAGS+=-I../fmm3d_mpi -I./normal-lib



# Project Paths
PROJECT_BASE = ../../$(CURDIR)
PROJECT_ROOT = $(CURDIR)
SRCDIR = $(PROJECT_ROOT)
OBJDIR = $(PROJECT_ROOT)/objs
DOCDIR = $(PROJECT_BASE)/doc
NORMDIR = $(PROJECT_ROOT)/normal-lib
NORMOBJS = $(NORMDIR)/normal.o $(NORMDIR)/sphere_grid.o 
# Main #########################################################################


$(OBJDIR):
	@mkdir -p $(OBJDIR)


OBJS = $(SRCS:%.cpp=$(OBJDIR)/%.o)
DEPS = $(SRCS:%.cpp=$(OBJDIR)/%.d)
SRCS = rpy.cpp utils.cpp interactions.cpp 


temp: $(OBJDIR)/$(OBJS)

rpy: rpy.o fmmrpy.o interactions.o utils.o
	${CLINKER} -o rpy rpy.o fmmrpy.o interactions.o utils.o $(NORMOBJS) ../fmm3d_mpi/libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 
