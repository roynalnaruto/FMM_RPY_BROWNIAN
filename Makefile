#source /opt/intel/bin/compilervars.sh intel64
#gcc -L/home/vipul/final sample.c /home/rohit/final/normal-lib/normal.o -lm -lfmmrpy -lgfortran -g 

# Configuration ################################################################




CC = g++
#CFLAGS = -Wall -Wextra
#CFLAGS += -DNDEBUG -O									# For production and benchmarks
#CFLAGS = -DDEBUG -g -L/opt/intel/mkl/lib/intel64		# For debugging
CPPFLAGS = -DDEBUG -g
#CFLAGS += -L/opt/intel/lib/intel64
#INCLUDES = -I. -I/opt/intel/mkl/include 
INCLUDES = -I. -I../fmm3d_mpi
LIBS = -lm -lfmmrpy -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5

# Project Paths

PROJECT_BASE = ../../$(CURDIR)
PROJECT_ROOT = $(CURDIR)
SRCDIR = $(PROJECT_ROOT)
OBJDIR = $(PROJECT_ROOT)/objs
DOCDIR = $(PROJECT_BASE)/doc
NORMDIR = $(PROJECT_ROOT)/normal-lib
# Main #########################################################################

OBJS = $(SRCS:%.cpp=$(OBJDIR)/%.o)
DEPS = $(SRCS:%.cpp=$(OBJDIR)/%.d)
SRCS = rpy.cpp utils.cpp interactions.cpp 
#	add lanczos
EXE = rpy

.PHONY: all clean doc

all: $(OBJDIR)/$(EXE)

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)/$(EXE): $(OBJS) | $(OBJDIR)
	$(CC) -L$(PROJECT_ROOT) $(CPPFLAGS) $(INCLUDES)  -o $@ $(NORMDIR)/normal.o $(OBJS) $(LIBS)


$(OBJDIR)/%.o : $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CC) $(CPPFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJDIR)/%.d: $(SRCDIR)/%.cpp | $(OBJDIR)
	@$(CC) $(CPPFLAGS) $(INCLUDES) -MM -MT $(@:%.d=%.o) $< > $@


doc:
	@latex $(DOCDIR)/report.tex
	@dvipdf $(PROJECT_ROOT)/report.dvi $(DOCDIR)/report.pdf
	@rm -rf $(PROJECT_ROOT)/report.*

%.pdf: %.dot
	@dot -Tpdf $< > $@

clean:
	@rm -rf $(OBJDIR) $(DOCDIR)/*.aux $(DOCDIR)/*.pdf $(DOCDIR)/*.dvi $(DOCDIR)/*.log $(DOCDIR)/*~
	@rm -rf $(PROJECT_ROOT)/*.aux $(PROJECT_ROOT)/*.pdf $(PROJECT_ROOT)/*.dvi $(PROJECT_ROOT)/*.log $(PROJECT_ROOT)/*~
	@rm -rf $(PROJECT_ROOT)/*.out

-include $(DEPS)

