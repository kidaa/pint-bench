#ifndef _KS_H_
#define _KS_H_

#include <petscdmda.h>
#include <petscsnes.h>
#include <petscvec.h>

typedef struct {
  /* spatial discretization */
  DM     da;

  /* backward euler solver */
  PetscReal dt;
  SNES      snes;
  Vec       r;
  Mat       J;
} KSCtx;

#endif
