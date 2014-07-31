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
  PetscInt i, j, Mx, My, xs, ys, xm, ym;
  Vec U, Udot, RHS;
  PetscScalar    ***u;

  PetscFunctionBeginUser;

  ierr = VecDuplicate(ks->r, &U);
  ierr = VecDuplicate(ks->r, &Udot);
  ierr = VecDuplicate(ks->r, &RHS);

  ierr = DMDAGetInfo(ks->da,0,&Mx,&My,0,0,0,0,0,0,0,0,0,0); assert(Mx == My);
  ierr = DMDAVecGetArrayDOF(ks->da,U,&u);
  ierr = DMDAGetCorners(ks->da,&xs,&ys,NULL,&xm,&ym,NULL);

  /* compute function over the locally owned part of the grid */
  double const h = 1.0 / (double)(Mx);
  for (j=ys; j<ys+ym; j++) {
    double const y = 2 * M_PI * h * j;
    for (i=xs; i<xs+xm; i++) {
      double const x = 2 * M_PI * h * i;
      u[j][i][0] = (cos(x) + cos(8*x)) * (cos(y) + cos(16*y));
    }
  }
  ierr = DMDAVecRestoreArrayDOF(ks->da,U,&u);

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
}
