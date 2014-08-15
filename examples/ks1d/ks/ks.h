#ifndef _KS_H_
#define _KS_H_

#include <petscdmda.h>
#include <petscsnes.h>
#include <petscvec.h>

typedef struct {
  /* spatial discretization */
  DM        da;
  PetscReal L;
  char      form;

  /* backward euler solver */
  PetscReal dt;
  SNES      snes;
  Vec       r;
  Mat       J;
} KSCtx;


PetscErrorCode KSEvaluate(SNES, Vec, Vec, void*);
PetscErrorCode KSJacobian(SNES, Vec, Mat, Mat, void*);
PetscErrorCode KSBEEvaluate(SNES, Vec, Vec, void*);
PetscErrorCode KSBEJacobian(SNES, Vec, Mat, Mat, void*);
PetscErrorCode KSBESolve(Vec, double, Vec, Vec, KSCtx*);
PetscErrorCode KSCreate(MPI_Comm, KSCtx*);
void KSDestroy(KSCtx*);
PetscErrorCode KSInitial(Vec U, KSCtx* ks);
PetscErrorCode KSTestEvaluate(KSCtx*);

#endif
