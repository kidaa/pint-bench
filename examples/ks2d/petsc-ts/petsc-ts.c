/*
 * PETSc TS integrator for 2d KS.
 */

#include <assert.h>
#include <ks.h>

#include <petscts.h>

#undef __FUNCT__
#define __FUNCT__ "KSTSRHSFunction"
PetscErrorCode KSTSRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBeginUser;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = KSEvaluate(snes, X, F, ctx);CHKERRQ(ierr);
  VecScale(F, -1.0);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSTSRHSJacobian"
PetscErrorCode KSTSRHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat B, void* ctx)
{
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBeginUser;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = KSJacobian(snes, X, J, B, ctx);CHKERRQ(ierr);
  MatScale(J, -1.0);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSRun"
PetscErrorCode KSRun(double dt, double tend, KSCtx* ks)
{
  PetscErrorCode ierr;
  TS             ts;
  Vec            U;

  PetscFunctionBeginUser;
  ierr = VecDuplicate(ks->r, &U);CHKERRQ(ierr);
  ierr = KSInitial(U, ks);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,ks->da);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,ks->r,KSTSRHSFunction,ks);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,ks->J,ks->J,KSTSRHSJacobian,ks);
  ierr = TSSetType(ts,TSGL);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,1000,tend);CHKERRQ(ierr);

  ierr = TSSetInitialTimeStep(ts,0,dt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  VecDestroy(&U);
  PetscFunctionReturn(0);
}

int main(int argc, char** argv)
{
  PetscReal dt;
  KSCtx     ks;

  PetscInitialize(&argc, &argv, NULL, NULL);
  KSCreate(MPI_COMM_WORLD, &ks);

  dt = 5.e-12;
  PetscOptionsGetReal(NULL, "-dt", &dt, NULL);
  KSRun(dt, 1.0, &ks);

  KSDestroy(&ks);
  PetscFinalize();

  return 0;
}
