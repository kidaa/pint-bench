/*
 * Fourth order finite-difference discretisation of spatial operators
 * for 1d KS equation.
 */

#include <math.h>
#include <stdio.h>

#include <mpi.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <petscvec.h>

#include "ks.h"


/*******************************************************************************
 * spatial discretization
 */

#undef __FUNCT__
#define __FUNCT__ "KSEvaluate"
/*
 * Evaluate f(x)
 */
PetscErrorCode KSEvaluate(SNES snes, Vec X, Vec F, void *ctx)
{
  KSCtx          *ks = (KSCtx*) ctx;
  PetscErrorCode ierr;
  Vec            lX;
  PetscInt       i, Mx, xs, xm;
  PetscScalar    **x,**f;
  DM             da;

  PetscFunctionBeginUser;
  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  double const hinv = (double) Mx / ks->L;
  double const h2inv = hinv * hinv;
  double const h4inv = hinv * hinv * hinv * hinv;

  ierr = DMGetLocalVector(da,&lX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,lX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,lX);CHKERRQ(ierr);

  ierr = DMDAVecGetArrayDOF(da,lX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,F,&f);CHKERRQ(ierr);

  ierr = DMDAGetGhostCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "xg %d xm %d\n", xs, xm);

  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "xs %d xm %d\n", xs, xm);

  /* compute function over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {

    double const lap =
      + x[i-3][0] * (1.0 / 90)
      + x[i-2][0] * (-3.0 / 20)
      + x[i-1][0] * (3.0 / 2)
      + x[i+0][0] * (-49.0 / 18)
      + x[i+1][0] * (3.0 / 2)
      + x[i+2][0] * (-3.0 / 20)
      + x[i+3][0] * (1.0 / 90);

    double const hyplap =
      + x[i-3][0] * (-1.0 / 6)
      + x[i-2][0] * (2.0 / 1)
      + x[i-1][0] * (-13.0 / 2)
      + x[i+0][0] * (28.0 / 3)
      + x[i+1][0] * (-13.0 / 2)
      + x[i+2][0] * (2.0 / 1)
      + x[i+3][0] * (-1.0 / 6);

    double const u_x =
      + x[i-3][0] * (-1.0 / 60)
      + x[i-2][0] * (3.0 / 20)
      + x[i-1][0] * (-3.0 / 4)
      + x[i+1][0] * (3.0 / 4)
      + x[i+2][0] * (-3.0 / 20)
      + x[i+3][0] * (1.0 / 60);

    double const u = x[i][0];

    if (ks->form == 'u') {
      f[i][0] = h2inv * lap + h4inv * hyplap;
      /* f[i][0] = hinv * u * u_x + h2inv * lap + h4inv * hyplap; */
    } else {
      f[i][0] = 0.5 * h2inv * u_x * u_x + h2inv * lap + h4inv * hyplap;
    }
  }

  /* restore vectors */
  ierr = DMDAVecRestoreArrayDOF(da,lX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&lX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSJacobian"
/*
 * Evaluate J(x) := df(x)/dx
 */
PetscErrorCode KSJacobian(SNES snes, Vec X, Mat J, Mat B, void* ctx)
{
  KSCtx          *ks = (KSCtx*) ctx;
  PetscErrorCode ierr;
  Vec            lX;
  PetscInt       i, Mx, xs, xm;
  PetscScalar    **x;
  DM             da;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  //  assert(Mx == My);

  double const hinv = (double) Mx / ks->L;
  double const h2inv = hinv * hinv;
  double const h4inv = hinv * hinv * hinv * hinv;

  /* ghost exchange */
  ierr = DMGetLocalVector(da,&lX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,lX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,lX);CHKERRQ(ierr);

  /* get arrays */
  ierr = DMDAVecGetArrayDOF(da,lX,&x);CHKERRQ(ierr);

  /* local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  MatStencil row, cols[7];
  PetscReal  grad[7], lap[7], hyplap[7];

  lap[0] = h2inv * (1.0 / 90);
  lap[1] = h2inv * (-3.0 / 20);
  lap[2] = h2inv * (3.0 / 2);
  lap[3] = h2inv * (-49.0 / 18);
  lap[4] = h2inv * (3.0 / 2);
  lap[5] = h2inv * (-3.0 / 20);
  lap[6] = h2inv * (1.0 / 90);

  hyplap[0] = h4inv * (-1.0 / 6);
  hyplap[1] = h4inv * (2.0 / 1);
  hyplap[2] = h4inv * (-13.0 / 2);
  hyplap[3] = h4inv * (28.0 / 3);
  hyplap[4] = h4inv * (-13.0 / 2);
  hyplap[5] = h4inv * (2.0 / 1);
  hyplap[6] = h4inv * (-1.0 / 6);

  for (i=xs; i<xs+xm; i++) {
    row.i = i;

    cols[0].i = i-3;
    cols[1].i = i-2;
    cols[2].i = i-1;
    cols[3].i = i+0;
    cols[4].i = i+1;
    cols[5].i = i+2;
    cols[6].i = i+3;

    ierr = MatSetValuesStencil(J, 1, &row, 7, cols, lap, ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValuesStencil(J, 1, &row, 7, cols, hyplap, ADD_VALUES);CHKERRQ(ierr);

    cols[0].i = i-3;
    cols[1].i = i-2;
    cols[2].i = i-1;
    cols[3].i = i+1;
    cols[4].i = i+2;
    cols[5].i = i+3;
    cols[6].i = i;

    double const u_x =
      + x[i-3][0] * (-1.0 / 60)
      + x[i-2][0] * (3.0 / 20)
      + x[i-1][0] * (-3.0 / 4)
      + x[i+1][0] * (3.0 / 4)
      + x[i+2][0] * (-3.0 / 20)
      + x[i+3][0] * (1.0 / 60);

    double const u = x[i][0];

    if (ks->form == 'u') {

      double const a = hinv * u;

      grad[0] = a * (-1.0 / 60);
      grad[1] = a * (3.0 / 20);
      grad[2] = a * (-3.0 / 4);
      grad[3] = a * (3.0 / 4);
      grad[4] = a * (-3.0 / 20);
      grad[5] = a * (1.0 / 60);
      grad[6] = hinv * u_x;

      /* ierr = MatSetValuesStencil(J, 1, &row, 7, cols, grad, ADD_VALUES);CHKERRQ(ierr); */

    } else {

      PetscPrintf(PETSC_COMM_WORLD,"NOT IMPLEMENTED YET\n");
    }

  }

  /* assemble and restore */
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArrayDOF(da,lX,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&lX);CHKERRQ(ierr);

  if (J != B) {
    MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
  }

  PetscFunctionReturn(0);
}


/*******************************************************************************
 * backward euler solver
 */

#undef __FUNCT__
#define __FUNCT__ "KSBEEvaluate"
/*
 * Evaluate f(x) := x - dt xdot(x).
 */
PetscErrorCode KSBEEvaluate(SNES snes, Vec X, Vec F, void *ctx)
{
  KSCtx* ks = (KSCtx*) ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = KSEvaluate(snes, X, F, ctx);CHKERRQ(ierr);
  VecAXPBY(F,1.0,-ks->dt,X);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSBEJacobian"
/*
 * Evaluate J(x) := df(x)/dx = 1 - dt dxdot/dx.
 */
PetscErrorCode KSBEJacobian(SNES snes, Vec X, Mat J, Mat B, void* ctx)
{
  KSCtx* ks = (KSCtx*) ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = KSJacobian(snes, X, J, B, ctx);CHKERRQ(ierr);
  MatScale(J, -ks->dt);
  MatShift(J, 1.0);
  /* if (B) { */
  /*   MatScale(B, -ctx->dt); */
  /*   MatShift(B, 1.0); */
  /* } */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSBESolve"
/*
 * Solve f(x) := x - dt xdot(x) = rhs for x.
 */
PetscErrorCode KSBESolve(Vec x, double dt, Vec xdot, Vec rhs, KSCtx* ctx)
{
  PetscErrorCode ierr;
  PetscInt       its;
  SNESConvergedReason reason;

  PetscFunctionBeginUser;
  ctx->dt = dt;
  ierr = SNESSolve(ctx->snes, rhs, x);CHKERRQ(ierr);
  SNESGetIterationNumber(ctx->snes, &its);
  SNESGetConvergedReason(ctx->snes, &reason);
  PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D, %s\n",its,SNESConvergedReasons[reason]);
  PetscFunctionReturn(0);
}


/*******************************************************************************
 * context create/destroy
 */

#undef __FUNCT__
#define __FUNCT__ "KSCreate"
/*
 * Create KS context.
 */
PetscErrorCode KSCreate(MPI_Comm comm, KSCtx* ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  /* create da and work vectors */
  ierr = DMDACreate1d(comm, DM_BOUNDARY_PERIODIC,
                      -128, 1, 3, NULL, &ctx->da); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(ctx->da, &ctx->r);CHKERRQ(ierr);
  ierr = DMCreateMatrix(ctx->da, &ctx->J);CHKERRQ(ierr);

  ctx->L    = 100.0;
  ctx->form = 'u';

  /* create non-linear solver */
  ierr = SNESCreate(comm, &ctx->snes); CHKERRQ(ierr);
  SNESSetFunction(ctx->snes, ctx->r, KSBEEvaluate, ctx);
  SNESSetJacobian(ctx->snes, ctx->J, ctx->J, KSBEJacobian, ctx);
  SNESSetDM(ctx->snes,ctx->da);
  SNESSetFromOptions(ctx->snes);
  PetscFunctionReturn(0);
}

void KSDestroy(KSCtx* ctx)
{
  SNESDestroy(&ctx->snes);
  VecDestroy(&ctx->r);
  MatDestroy(&ctx->J);
  DMDestroy(&ctx->da);
}

#undef __FUNCT__
#define __FUNCT__ "KSInitial"
PetscErrorCode KSInitial(Vec U, KSCtx* ks)
{
  PetscErrorCode ierr;
  PetscInt       i, Mx, xs, xm;
  PetscScalar    **u;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(ks->da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(ks->da,U,&u);CHKERRQ(ierr);
  ierr = DMDAGetCorners(ks->da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  double const h = 2 * M_PI / Mx;
  for (i=xs; i<xs+xm; i++) {
    double const x = h * i;
    u[i][0] = (cos(x) + cos(3*x));
  }

  ierr = DMDAVecRestoreArrayDOF(ks->da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*******************************************************************************
 * tests
 */

#undef __FUNCT__
#define __FUNCT__ "KSTestEvaluate"
/*
 * Test KSEvaluate using manufactured solution.
 */
PetscErrorCode KSTestEvaluate(KSCtx *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    **u, **f;
  PetscInt Mx, xs, xm;
  Vec U, F1, F2;

  ierr = VecDuplicate(ctx->r, &U);
  ierr = VecDuplicate(ctx->r, &F1);
  ierr = VecDuplicate(ctx->r, &F2);

  /* create test state and exact evaluation */
  ierr = DMDAGetInfo(ctx->da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0);
  ierr = DMDAGetCorners(ctx->da,&xs,NULL,NULL,&xm,NULL,NULL);
  ierr = DMDAVecGetArrayDOF(ctx->da,U,&u);
  ierr = DMDAVecGetArrayDOF(ctx->da,F1,&f);

  ctx->dt = 1.e-4;

  int i;
  double const h = ctx->L / (double)(Mx);
  double const piL = M_PI / ctx->L;
  double const piL2 = piL * piL;
  double const piL4 = piL2 * piL2;
  for (i=xs; i<xs+xm; i++) {
    double const x = h * i;
    double const cosx = cos(2*piL*x);
    double const sinx = sin(2*piL*x);

    /* u[j][i][0] = 1.0; */
    /* f[j][i][0] = 0.0; */

    u[i][0] = cosx;
    f[i][0] = ( -2*piL*cosx*sinx ) - 4*piL2*cosx + 16*piL4*cosx;
  }

  ierr = DMDAVecRestoreArrayDOF(ctx->da,U,&u);
  ierr = DMDAVecRestoreArrayDOF(ctx->da,F1,&f);

  PetscReal norm1, norm2;
  VecNorm(F1, NORM_INFINITY, &norm1);
  PetscPrintf(PETSC_COMM_WORLD, "Test norm: max|F1|    = %g\n", norm1);

  KSEvaluate(ctx->snes, U, F2, ctx);
  VecNorm(F2, NORM_INFINITY, &norm2);
  PetscPrintf(PETSC_COMM_WORLD, "Test norm: max|F2|    = %g\n", norm2);

  VecAXPY(F1, -1.0, F2);
  VecNorm(F1, NORM_INFINITY, &norm2);
  PetscPrintf(PETSC_COMM_WORLD, "Test norm: max|F1-F2| = %g\n", norm2);///norm1);

  PetscViewer viewer;
  PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,NULL,0,0,300,300,&viewer);
  PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_LG);
  VecView(F1,viewer);
  /* VecView(F2,viewer); */

  /* PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,NULL,0,0,300,300,&viewer); */
  /* PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_LG); */
  /* VecView(F2,viewer); */

  PetscSleep(10);

  VecDestroy(&F2);
  VecDestroy(&F1);
  VecDestroy(&U);

  return 0;
}
