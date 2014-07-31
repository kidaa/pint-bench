#
# KS 2d
#
# Required PETSc.  To build, run make as, eg
#
#   make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-c-debug CC=mpicc
#

CFLAGS  += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
LDFLAGS += -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -lm

backward-euler.exe: ks/ks.h ks/ks.c backward-euler/backward-euler.c
	$(CC) -o backward-euler.exe $(CFLAGS) -Iks ks/ks.c backward-euler/backward-euler.c $(LDFLAGS)
