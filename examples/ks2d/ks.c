

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <petscvec.h>

typedef struct {
  double dt;
  DM     da;
  SNES   snes;
  Vec    r;
  Mat    J;
} KSCtx;

#undef __FUNCT__
#define __FUNCT__ "KSEvaluate"
/*
 * Evaluate f(x) := x - dt xdot(x).
 */
PetscErrorCode KSEvaluate(SNES snes, Vec X, Vec F, void *_ctx)
{
  KSCtx*         ctx = (KSCtx*) _ctx;
  PetscErrorCode ierr;
  Vec            lX;
  PetscInt       i, j, Mx, My, xs, ys, xm, ym;
  PetscScalar    ***x,***f;
  DM             da;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(ctx->da,0,&Mx,&My,0,0,0,0,0,0,0,0,0,0);
  assert(Mx == My);

  double const hinv = (double) Mx;
  double const h2inv = hinv * hinv;
  double const h4inv = hinv * hinv * hinv * hinv;

  /* ghost exchange */
  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&lX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,lX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,lX);CHKERRQ(ierr);

  /* get arrays */
  ierr = DMDAVecGetArrayDOF(da,lX,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,F,&f);CHKERRQ(ierr);

  /* local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /* compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {

      double const lap =
        + x[j+0][i-3][0] * (   0.011111111111)
        + x[j+0][i-2][0] * (  -0.150000000000)
        + x[j+0][i-1][0] * (   1.500000000000)
        + x[j-3][i+0][0] * (   0.011111111111)
        + x[j-2][i+0][0] * (  -0.150000000000)
        + x[j-1][i+0][0] * (   1.500000000000)
        + x[j+0][i+0][0] * (  -5.444444444444)
        + x[j+1][i+0][0] * (   1.500000000000)
        + x[j+2][i+0][0] * (  -0.150000000000)
        + x[j+3][i+0][0] * (   0.011111111111)
        + x[j+0][i+1][0] * (   1.500000000000)
        + x[j+0][i+2][0] * (  -0.150000000000)
        + x[j+0][i+3][0] * (   0.011111111111);

      double const hyplap =
        + x[j-3][i-3][0] * (   0.000246913580)
        + x[j-2][i-3][0] * (  -0.003333333333)
        + x[j-1][i-3][0] * (   0.033333333333)
        + x[j+0][i-3][0] * (  -0.227160493827)
        + x[j+1][i-3][0] * (   0.033333333333)
        + x[j+2][i-3][0] * (  -0.003333333333)
        + x[j+3][i-3][0] * (   0.000246913580)
        + x[j-3][i-2][0] * (  -0.003333333333)
        + x[j-2][i-2][0] * (   0.045000000000)
        + x[j-1][i-2][0] * (  -0.450000000000)
        + x[j+0][i-2][0] * (   2.816666666667)
        + x[j+1][i-2][0] * (  -0.450000000000)
        + x[j+2][i-2][0] * (   0.045000000000)
        + x[j+3][i-2][0] * (  -0.003333333333)
        + x[j-3][i-1][0] * (   0.033333333333)
        + x[j-2][i-1][0] * (  -0.450000000000)
        + x[j-1][i-1][0] * (   4.500000000000)
        + x[j+0][i-1][0] * ( -14.666666666667)
        + x[j+1][i-1][0] * (   4.500000000000)
        + x[j+2][i-1][0] * (  -0.450000000000)
        + x[j+3][i-1][0] * (   0.033333333333)
        + x[j-3][i+0][0] * (  -0.227160493827)
        + x[j-2][i+0][0] * (   2.816666666667)
        + x[j-1][i+0][0] * ( -14.666666666667)
        + x[j+0][i+0][0] * (  33.487654320988)
        + x[j+1][i+0][0] * ( -14.666666666667)
        + x[j+2][i+0][0] * (   2.816666666667)
        + x[j+3][i+0][0] * (  -0.227160493827)
        + x[j-3][i+1][0] * (   0.033333333333)
        + x[j-2][i+1][0] * (  -0.450000000000)
        + x[j-1][i+1][0] * (   4.500000000000)
        + x[j+0][i+1][0] * ( -14.666666666667)
        + x[j+1][i+1][0] * (   4.500000000000)
        + x[j+2][i+1][0] * (  -0.450000000000)
        + x[j+3][i+1][0] * (   0.033333333333)
        + x[j-3][i+2][0] * (  -0.003333333333)
        + x[j-2][i+2][0] * (   0.045000000000)
        + x[j-1][i+2][0] * (  -0.450000000000)
        + x[j+0][i+2][0] * (   2.816666666667)
        + x[j+1][i+2][0] * (  -0.450000000000)
        + x[j+2][i+2][0] * (   0.045000000000)
        + x[j+3][i+2][0] * (  -0.003333333333)
        + x[j-3][i+3][0] * (   0.000246913580)
        + x[j-2][i+3][0] * (  -0.003333333333)
        + x[j-1][i+3][0] * (   0.033333333333)
        + x[j+0][i+3][0] * (  -0.227160493827)
        + x[j+1][i+3][0] * (   0.033333333333)
        + x[j+2][i+3][0] * (  -0.003333333333)
        + x[j+3][i+3][0] * (   0.000246913580);

      double const grad_x =
        + x[j][i-3][0] * (  -0.016666666667)
        + x[j][i-2][0] * (   0.150000000000)
        + x[j][i-1][0] * (  -0.750000000000)
        + x[j][i+1][0] * (   0.750000000000)
        + x[j][i+2][0] * (  -0.150000000000)
        + x[j][i+3][0] * (   0.016666666667);

      double const grad_y =
        + x[j-3][i][0] * (  -0.016666666667)
        + x[j-2][i][0] * (   0.150000000000)
        + x[j-1][i][0] * (  -0.750000000000)
        + x[j+1][i][0] * (   0.750000000000)
        + x[j+2][i][0] * (  -0.150000000000)
        + x[j+3][i][0] * (   0.016666666667);

      double const grad_sq = grad_x * grad_x + grad_y * grad_y;

      f[j][i][0] = x[j][i][0] - ctx->dt * ( 0.5 * h2inv * grad_sq + h2inv * lap + h4inv * hyplap );
    }
  }

  /* restore vectors */
  ierr = DMDAVecRestoreArrayDOF(da,lX,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&lX);CHKERRQ(ierr);
  ierr = PetscLogFlops((2*49+2*6+8)*ym*xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSTestEvaluate"
/*
 * Test KSEvaluate using manufactured solution.
 */
PetscErrorCode KSTestEvaluate(KSCtx *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    ***u, ***f;
  PetscInt Mx, xs, ys, xm, ym;
  Vec U, F, F2;

  ierr = VecDuplicate(ctx->r, &U);
  ierr = VecDuplicate(ctx->r, &F);
  ierr = VecDuplicate(ctx->r, &F2);

  /* create test state and exact evaluation */
  ierr = DMDAGetInfo(ctx->da,0,&Mx,0,0,0,0,0,0,0,0,0,0,0);
  ierr = DMDAGetCorners(ctx->da,&xs,&ys,NULL,&xm,&ym,NULL);
  ierr = DMDAVecGetArrayDOF(ctx->da,U,&u);
  ierr = DMDAVecGetArrayDOF(ctx->da,F,&f);

  ctx->dt = 1.e-4;

  int i, j;
  double const h = 1.0 / (double)(Mx);
  for (j=ys; j<ys+ym; j++) {
    double const y = h * j;
    for (i=xs; i<xs+xm; i++) {
      double const x = h * i;
      u[j][i][0] = sin(2*M_PI*y);
      f[j][i][0] = sin(2*M_PI*y) - ctx->dt *
        (
         0.5 * (2*M_PI)*(2*M_PI)*cos(2*M_PI*y)*cos(2*M_PI*y)
         - (2*M_PI)*(2*M_PI)*sin(2*M_PI*y)
         + (2*M_PI)*(2*M_PI)*(2*M_PI)*(2*M_PI)*sin(2*M_PI*y)
          );
    }
  }

  ierr = DMDAVecRestoreArrayDOF(ctx->da,U,&u);
  ierr = DMDAVecRestoreArrayDOF(ctx->da,F,&f);

  KSEvaluate(ctx->snes, U, F2, ctx);

  VecAXPY(F, -1.0, F2);
  PetscReal norm;
  VecNorm(F, NORM_INFINITY, &norm);
  PetscPrintf(PETSC_COMM_WORLD, "Test norm: max|F-F1| = %g\n", norm);

  VecDestroy(&F2);
  VecDestroy(&F);
  VecDestroy(&U);
}


#undef __FUNCT__
#define __FUNCT__ "KSJacobian"
/*
 * Evaluate J(x) := df(x)/dx = 1 - dt dxdot/dx.
 */
PetscErrorCode KSJacobian(SNES snes, Vec X, Mat J, Mat B, void* _ctx)
{
  PetscErrorCode ierr;
  KSCtx* ctx = (KSCtx*) _ctx;
  Vec            lX;
  PetscInt       i, j, Mx, My, xs, ys, xm, ym;
  PetscScalar    ***x,***f;
  DM             da;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(J);CHKERRQ(ierr);

  ierr = SNESGetDM(snes,&da);
  ierr = DMDAGetInfo(da,0,&Mx,&My,0,0,0,0,0,0,0,0,0,0);
  assert(Mx == My);

  double const hinv = (double) Mx;
  double const h2inv = hinv * hinv;
  double const h4inv = hinv * hinv * hinv * hinv;

  /* ghost exchange */
  ierr = DMGetLocalVector(da,&lX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,lX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,lX);CHKERRQ(ierr);

  /* get arrays */
  ierr = DMDAVecGetArrayDOF(da,lX,&x);CHKERRQ(ierr);

  /* local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  MatStencil row, cols[49];
  PetscReal  one[1], grad_sq[6], lap[13], hyplap[49];

  one[0] = 1.0;

  lap[0] = -ctx->dt * h2inv * (   0.011111111111);
  lap[1] = -ctx->dt * h2inv * (  -0.150000000000);
  lap[2] = -ctx->dt * h2inv * (   1.500000000000);
  lap[3] = -ctx->dt * h2inv * (   0.011111111111);
  lap[4] = -ctx->dt * h2inv * (  -0.150000000000);
  lap[5] = -ctx->dt * h2inv * (   1.500000000000);
  lap[6] = -ctx->dt * h2inv * (  -5.444444444444);
  lap[7] = -ctx->dt * h2inv * (   1.500000000000);
  lap[8] = -ctx->dt * h2inv * (  -0.150000000000);
  lap[9] = -ctx->dt * h2inv * (   0.011111111111);
  lap[10] = -ctx->dt * h2inv * (   1.500000000000);
  lap[11] = -ctx->dt * h2inv * (  -0.150000000000);
  lap[12] = -ctx->dt * h2inv * (   0.011111111111);

  hyplap[0] = -ctx->dt * h4inv * (   0.000246913580);
  hyplap[1] = -ctx->dt * h4inv * (  -0.003333333333);
  hyplap[2] = -ctx->dt * h4inv * (   0.033333333333);
  hyplap[3] = -ctx->dt * h4inv * (  -0.227160493827);
  hyplap[4] = -ctx->dt * h4inv * (   0.033333333333);
  hyplap[5] = -ctx->dt * h4inv * (  -0.003333333333);
  hyplap[6] = -ctx->dt * h4inv * (   0.000246913580);
  hyplap[7] = -ctx->dt * h4inv * (  -0.003333333333);
  hyplap[8] = -ctx->dt * h4inv * (   0.045000000000);
  hyplap[9] = -ctx->dt * h4inv * (  -0.450000000000);
  hyplap[10] = -ctx->dt * h4inv * (   2.816666666667);
  hyplap[11] = -ctx->dt * h4inv * (  -0.450000000000);
  hyplap[12] = -ctx->dt * h4inv * (   0.045000000000);
  hyplap[13] = -ctx->dt * h4inv * (  -0.003333333333);
  hyplap[14] = -ctx->dt * h4inv * (   0.033333333333);
  hyplap[15] = -ctx->dt * h4inv * (  -0.450000000000);
  hyplap[16] = -ctx->dt * h4inv * (   4.500000000000);
  hyplap[17] = -ctx->dt * h4inv * ( -14.666666666667);
  hyplap[18] = -ctx->dt * h4inv * (   4.500000000000);
  hyplap[19] = -ctx->dt * h4inv * (  -0.450000000000);
  hyplap[20] = -ctx->dt * h4inv * (   0.033333333333);
  hyplap[21] = -ctx->dt * h4inv * (  -0.227160493827);
  hyplap[22] = -ctx->dt * h4inv * (   2.816666666667);
  hyplap[23] = -ctx->dt * h4inv * ( -14.666666666667);
  hyplap[24] = -ctx->dt * h4inv * (  33.487654320988);
  hyplap[25] = -ctx->dt * h4inv * ( -14.666666666667);
  hyplap[26] = -ctx->dt * h4inv * (   2.816666666667);
  hyplap[27] = -ctx->dt * h4inv * (  -0.227160493827);
  hyplap[28] = -ctx->dt * h4inv * (   0.033333333333);
  hyplap[29] = -ctx->dt * h4inv * (  -0.450000000000);
  hyplap[30] = -ctx->dt * h4inv * (   4.500000000000);
  hyplap[31] = -ctx->dt * h4inv * ( -14.666666666667);
  hyplap[32] = -ctx->dt * h4inv * (   4.500000000000);
  hyplap[33] = -ctx->dt * h4inv * (  -0.450000000000);
  hyplap[34] = -ctx->dt * h4inv * (   0.033333333333);
  hyplap[35] = -ctx->dt * h4inv * (  -0.003333333333);
  hyplap[36] = -ctx->dt * h4inv * (   0.045000000000);
  hyplap[37] = -ctx->dt * h4inv * (  -0.450000000000);
  hyplap[38] = -ctx->dt * h4inv * (   2.816666666667);
  hyplap[39] = -ctx->dt * h4inv * (  -0.450000000000);
  hyplap[40] = -ctx->dt * h4inv * (   0.045000000000);
  hyplap[41] = -ctx->dt * h4inv * (  -0.003333333333);
  hyplap[42] = -ctx->dt * h4inv * (   0.000246913580);
  hyplap[43] = -ctx->dt * h4inv * (  -0.003333333333);
  hyplap[44] = -ctx->dt * h4inv * (   0.033333333333);
  hyplap[45] = -ctx->dt * h4inv * (  -0.227160493827);
  hyplap[46] = -ctx->dt * h4inv * (   0.033333333333);
  hyplap[47] = -ctx->dt * h4inv * (  -0.003333333333);
  hyplap[48] = -ctx->dt * h4inv * (   0.000246913580);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;

      MatSetValuesStencil(J, 1, &row, 1, &row, one, ADD_VALUES);

      /* laplacian */
      cols[0].i = i-3; cols[0].j = j+0;
      cols[1].i = i-2; cols[1].j = j+0;
      cols[2].i = i-1; cols[2].j = j+0;
      cols[3].i = i+0; cols[3].j = j-3;
      cols[4].i = i+0; cols[4].j = j-2;
      cols[5].i = i+0; cols[5].j = j-1;
      cols[6].i = i+0; cols[6].j = j+0;
      cols[7].i = i+0; cols[7].j = j+1;
      cols[8].i = i+0; cols[8].j = j+2;
      cols[9].i = i+0; cols[9].j = j+3;
      cols[10].i = i+1; cols[10].j = j+0;
      cols[11].i = i+2; cols[11].j = j+0;
      cols[12].i = i+3; cols[12].j = j+0;

      MatSetValuesStencil(J, 1, &row, 13, cols, lap, ADD_VALUES);

      /* hyperlaplacian */
      cols[0].i = i-3; cols[0].j = j-3;
      cols[1].i = i-3; cols[1].j = j-2;
      cols[2].i = i-3; cols[2].j = j-1;
      cols[3].i = i-3; cols[3].j = j+0;
      cols[4].i = i-3; cols[4].j = j+1;
      cols[5].i = i-3; cols[5].j = j+2;
      cols[6].i = i-3; cols[6].j = j+3;
      cols[7].i = i-2; cols[7].j = j-3;
      cols[8].i = i-2; cols[8].j = j-2;
      cols[9].i = i-2; cols[9].j = j-1;
      cols[10].i = i-2; cols[10].j = j+0;
      cols[11].i = i-2; cols[11].j = j+1;
      cols[12].i = i-2; cols[12].j = j+2;
      cols[13].i = i-2; cols[13].j = j+3;
      cols[14].i = i-1; cols[14].j = j-3;
      cols[15].i = i-1; cols[15].j = j-2;
      cols[16].i = i-1; cols[16].j = j-1;
      cols[17].i = i-1; cols[17].j = j+0;
      cols[18].i = i-1; cols[18].j = j+1;
      cols[19].i = i-1; cols[19].j = j+2;
      cols[20].i = i-1; cols[20].j = j+3;
      cols[21].i = i+0; cols[21].j = j-3;
      cols[22].i = i+0; cols[22].j = j-2;
      cols[23].i = i+0; cols[23].j = j-1;
      cols[24].i = i+0; cols[24].j = j+0;
      cols[25].i = i+0; cols[25].j = j+1;
      cols[26].i = i+0; cols[26].j = j+2;
      cols[27].i = i+0; cols[27].j = j+3;
      cols[28].i = i+1; cols[28].j = j-3;
      cols[29].i = i+1; cols[29].j = j-2;
      cols[30].i = i+1; cols[30].j = j-1;
      cols[31].i = i+1; cols[31].j = j+0;
      cols[32].i = i+1; cols[32].j = j+1;
      cols[33].i = i+1; cols[33].j = j+2;
      cols[34].i = i+1; cols[34].j = j+3;
      cols[35].i = i+2; cols[35].j = j-3;
      cols[36].i = i+2; cols[36].j = j-2;
      cols[37].i = i+2; cols[37].j = j-1;
      cols[38].i = i+2; cols[38].j = j+0;
      cols[39].i = i+2; cols[39].j = j+1;
      cols[40].i = i+2; cols[40].j = j+2;
      cols[41].i = i+2; cols[41].j = j+3;
      cols[42].i = i+3; cols[42].j = j-3;
      cols[43].i = i+3; cols[43].j = j-2;
      cols[44].i = i+3; cols[44].j = j-1;
      cols[45].i = i+3; cols[45].j = j+0;
      cols[46].i = i+3; cols[46].j = j+1;
      cols[47].i = i+3; cols[47].j = j+2;
      cols[48].i = i+3; cols[48].j = j+3;

      MatSetValuesStencil(J, 1, &row, 49, cols, hyplap, ADD_VALUES);

      /* grad_x^2 */
      cols[0].i = i-3; cols[0].j = j;
      cols[1].i = i-2; cols[1].j = j;
      cols[2].i = i-1; cols[2].j = j;
      cols[3].i = i+1; cols[3].j = j;
      cols[4].i = i+2; cols[4].j = j;
      cols[5].i = i+3; cols[5].j = j;

      double const grad_x =
        + x[j][i-3][0] * (  -0.016666666667)
        + x[j][i-2][0] * (   0.150000000000)
        + x[j][i-1][0] * (  -0.750000000000)
        + x[j][i+1][0] * (   0.750000000000)
        + x[j][i+2][0] * (  -0.150000000000)
        + x[j][i+3][0] * (   0.016666666667);

      double const a = -ctx->dt * h2inv * grad_x;

      grad_sq[0] = a * (  -0.016666666667);
      grad_sq[1] = a * (   0.150000000000);
      grad_sq[2] = a * (  -0.750000000000);
      grad_sq[3] = a * (   0.750000000000);
      grad_sq[4] = a * (  -0.150000000000);
      grad_sq[5] = a * (   0.016666666667);

      MatSetValuesStencil(J, 1, &row, 6, cols, grad_sq, ADD_VALUES);

      /* grad_y^2 */
      cols[0].i = i; cols[0].j = j-3;
      cols[1].i = i; cols[1].j = j-2;
      cols[2].i = i; cols[2].j = j-1;
      cols[3].i = i; cols[3].j = j+1;
      cols[4].i = i; cols[4].j = j+2;
      cols[5].i = i; cols[5].j = j+3;

      double const grad_y =
        + x[j-3][i][0] * (  -0.016666666667)
        + x[j-2][i][0] * (   0.150000000000)
        + x[j-1][i][0] * (  -0.750000000000)
        + x[j+1][i][0] * (   0.750000000000)
        + x[j+2][i][0] * (  -0.150000000000)
        + x[j+3][i][0] * (   0.016666666667);

      double const b = -ctx->dt * h2inv * grad_y;

      grad_sq[0] = b * (  -0.016666666667);
      grad_sq[1] = b * (   0.150000000000);
      grad_sq[2] = b * (  -0.750000000000);
      grad_sq[3] = b * (   0.750000000000);
      grad_sq[4] = b * (  -0.150000000000);
      grad_sq[5] = b * (   0.016666666667);

      MatSetValuesStencil(J, 1, &row, 6, cols, grad_sq, ADD_VALUES);

    }
  }

  /* assemble and restore */
  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);

  DMDAVecRestoreArrayDOF(da,lX,&x);
  DMRestoreLocalVector(da,&lX);

  if (J != B) {
    MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
  }

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
  SNESGetIterationNumber(ctx->snes,&its);
  SNESGetConvergedReason(ctx->snes, &reason);
  PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D, %s\n",its,SNESConvergedReasons[reason]);
  PetscFunctionReturn(0);
}

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
  ierr = DMDACreate2d(comm, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX,
                      -128, -128, PETSC_DECIDE, PETSC_DECIDE,
                      1, 3, NULL, NULL, &ctx->da); CHKERRQ(ierr);

  DMCreateGlobalVector(ctx->da, &ctx->r);
  DMCreateMatrix(ctx->da, &ctx->J);

  /* create non-linear solver */
  ierr = SNESCreate(comm, &ctx->snes); CHKERRQ(ierr);
  SNESSetFunction(ctx->snes, ctx->r, KSEvaluate, ctx);
  SNESSetJacobian(ctx->snes, ctx->J, ctx->J, KSJacobian, ctx);
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
#define __FUNCT__ "KSRun"
PetscErrorCode KSRun(double dt, double tend, KSCtx* ctx)
{
  PetscErrorCode ierr;
  PetscInt i, j, Mx, My, xs, ys, xm, ym;
  Vec U, Udot, RHS;
  PetscScalar    ***u;

  PetscFunctionBeginUser;

  ierr = VecDuplicate(ctx->r, &U);
  ierr = VecDuplicate(ctx->r, &Udot);
  ierr = VecDuplicate(ctx->r, &RHS);

  ierr = DMDAGetInfo(ctx->da,0,&Mx,&My,0,0,0,0,0,0,0,0,0,0); assert(Mx == My);
  ierr = DMDAVecGetArrayDOF(ctx->da,U,&u);
  ierr = DMDAGetCorners(ctx->da,&xs,&ys,NULL,&xm,&ym,NULL);

  /* compute function over the locally owned part of the grid */
  double const h = 1.0 / (double)(Mx);
  for (j=ys; j<ys+ym; j++) {
    double const y = 2 * M_PI * h * j;
    for (i=xs; i<xs+xm; i++) {
      double const x = 2 * M_PI * h * i;
      u[j][i][0] = (cos(x) + cos(3*x)) * (cos(y) + cos(8*y));
    }
  }
  ierr = DMDAVecRestoreArrayDOF(ctx->da,U,&u);

  PetscViewer viewer;
  PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,NULL,0,0,300,300,&viewer);
  PetscViewerPushFormat(viewer,PETSC_VIEWER_DRAW_LG);

  double t = 0;
  while (t < tend) {
    VecView(U,viewer);
    VecCopy(U, RHS);
    ierr = KSBESolve(U, dt, Udot, RHS, ctx);CHKERRQ(ierr);
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
  KSCtx ctx;

  PetscInitialize(&argc, &argv, NULL, NULL);
  KSCreate(MPI_COMM_WORLD, &ctx);

  dt = 5.e-12;
  PetscOptionsGetReal(NULL, "-dt", &dt, NULL);
  KSRun(dt, 1.0, &ctx);

  /* KSTestEvaluate(&ctx); */

  KSDestroy(&ctx);
  PetscFinalize();
}
