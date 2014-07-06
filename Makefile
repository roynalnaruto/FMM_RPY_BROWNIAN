#source /opt/intel/bin/compilervars.sh intel64
#gcc -L/home/vipul/final sample.c /home/rohit/final/normal-lib/normal.o -lm -lfmmrpy -lgfortran -g 

# Configuration ################################################################

CC = icc
#CFLAGS = -Wall -Wextra
#CFLAGS += -DNDEBUG -O									# For production and benchmarks
CFLAGS = -DDEBUG -g -L/opt/intel/mkl/lib/intel64		# For debugging
CFLAGS += -L/opt/intel/lib/intel64
INCLUDES = -I. -I/opt/intel/mkl/include 
LIBS = -lm -lfmmrpy -lgfortran -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5

# Project Paths
PROJECT_ROOT ?= $(CURDIR)
SRCDIR = $(PROJECT_ROOT)
OBJDIR = $(PROJECT_ROOT)/objs
DOCDIR = $(PROJECT_ROOT)/doc
NORMDIR = $(PROJECT_ROOT)/normal-lib
FORTRANDIR = $(PROJECT_ROOT)/fmmlibFortran/examples
# Main #########################################################################

OBJS = $(SRCS:%.c=$(OBJDIR)/%.o)
DEPS = $(SRCS:%.c=$(OBJDIR)/%.d)
SRCS = new.c utils.c interactions.c lanczos.c
EXE = rpy

.PHONY: all clean doc

all: $(OBJDIR)/$(EXE)

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)/$(EXE): fortranlibrary $(OBJS) | $(OBJDIR)
	$(CC) -L$(PROJECT_ROOT) $(CFLAGS) $(INCLUDES)  -o $@ $(NORMDIR)/normal.o $(OBJS) $(LIBS)

$(OBJDIR)/%.o : $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJDIR)/%.d: $(SRCDIR)/%.c | $(OBJDIR)
	@$(CC) $(CFLAGS) $(INCLUDES) -MM -MT $(@:%.d=%.o) $< > $@


fortranlibrary: $(FORTRANDIR)/*.o
	rm -rf libfmmrpy.a
	@ar -cvq libfmmrpy.a $(FORTRANDIR)/*.o


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

