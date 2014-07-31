#
# KS 2d
#
# Required PETSc.  To build, run make as, eg
#
#   make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-c-debug CC=mpicc
#

CFLAGS  += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
LDFLAGS += -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -lm

ks: ks.c
	$(CC) -o ks $(CFLAGS) ks.c $(LDFLAGS)
