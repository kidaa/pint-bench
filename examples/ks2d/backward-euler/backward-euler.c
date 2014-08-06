/*
 * Simple backward-Euler integrator for 2d KS.
 */

#include <assert.h>
#include <ks.h>

#undef __FUNCT__
#define __FUNCT__ "KSBERun"
PetscErrorCode KSBERun(double dt, double tend, KSCtx* ks)
{
  PetscErrorCode ierr;
  Vec            U, Udot, RHS;

  PetscFunctionBeginUser;

  ierr = VecDuplicate(ks->r, &U);CHKERRQ(ierr);
  ierr = VecDuplicate(ks->r, &Udot);CHKERRQ(ierr);
  ierr = VecDuplicate(ks->r, &RHS);CHKERRQ(ierr);

  ierr = KSInitial(U, ks);CHKERRQ(ierr);

  PetscViewer viewer;
  PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,NULL,0,0,300,300,&viewer);
  PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_LG);

  double t = 0;
  while (t < tend) {
    VecView(U,viewer);
    VecCopy(U, RHS);
    ierr = KSBESolve(U, dt, Udot, RHS, ks);CHKERRQ(ierr);
    t += dt;
  }

  VecDestroy(&RHS);
  VecDestroy(&Udot);
  VecDestroy(&U);
  PetscFunctionReturn(0);
}

int main(int argc, char** argv)
{
  PetscReal dt;
  KSCtx ks;

  PetscInitialize(&argc, &argv, NULL, NULL);
  KSCreate(MPI_COMM_WORLD, &ks);

  /* KSTestEvaluate(&ks); */

  dt = 5.e-12;
  PetscOptionsGetReal(NULL, "-dt", &dt, NULL);
  KSBERun(dt, 1.0, &ks);

  KSDestroy(&ks);
  PetscFinalize();

  return 0;
}
