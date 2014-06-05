# Configuration ################################################################

CC = gcc
CFLAGS = -Wall -Wextra
#CFLAGS += -DNDEBUG -O		# For production and benchmarks
CFLAGS += -DDEBUG -g		# For debugging
INCLUDES = -I.
LIBS = -lm

# Project Paths
PROJECT_ROOT ?= $(CURDIR)
SRCDIR = $(PROJECT_ROOT)
OBJDIR = $(PROJECT_ROOT)/objs/
DOCDIR = $(PROJECT_ROOT)/doc/

# Main #########################################################################

OBJS = $(SRCS:%.c=$(OBJDIR)/%.o)
DEPS = $(SRCS:%.c=$(OBJDIR)/%.d)
SRCS = collection.c covtree.c main.c marking.c petrinet.c tree.c wnat.c
EXE = kmt

.PHONY: all clean doc

all: $(OBJDIR)/$(EXE)

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)/$(EXE): $(OBJS) | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LIBS)

$(OBJDIR)/%.o : $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJDIR)/%.d: $(SRCDIR)/%.c | $(OBJDIR)
	@$(CC) $(CFLAGS) $(INCLUDES) -MM -MT $(@:%.d=%.o) $< > $@
doc:
	@latex $(DOCDIR)/pda.tex
	@dvipdf $(PROJECT_ROOT)/pda.dvi $(DOCDIR)/pda.pdf
	@rm -rf $(PROJECT_ROOT)/pda.*

%.pdf: %.dot
	@dot -Tpdf $< > $@

clean:
	@rm -rf $(OBJDIR) $(DOCDIR)/*.aux $(DOCDIR)/*.pdf $(DOCDIR)/*.dvi $(DOCDIR)/*.log $(DOCDIR)/*~
	@rm -rf $(PROJECT_ROOT)/*.aux $(PROJECT_ROOT)/*.pdf $(PROJECT_ROOT)/*.dvi $(PROJECT_ROOT)/*.log $(PROJECT_ROOT)/*~
	@rm -rf $(PROJECT_ROOT)/*.out $(PROJECT_ROOT)/*.o

-include $(DEPS)

