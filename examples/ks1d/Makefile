#
# KS 2d
#
# Required PETSc.  To build, run make as, eg
#
#   make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-c-debug CC=mpicc
#

CFLAGS  += -Wall
CFLAGS  += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
LDFLAGS += -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -lm

all: backward-euler.exe petsc-ts.exe

backward-euler.exe: ks/ks.h ks/ks.c backward-euler/backward-euler.c
	$(CC) -o backward-euler.exe $(CFLAGS) -Iks ks/ks.c backward-euler/backward-euler.c $(LDFLAGS)

petsc-ts.exe: ks/ks.h ks/ks.c petsc-ts/petsc-ts.c
	$(CC) -o petsc-ts.exe $(CFLAGS) -Iks ks/ks.c petsc-ts/petsc-ts.c $(LDFLAGS)
