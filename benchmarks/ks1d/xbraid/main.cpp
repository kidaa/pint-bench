
#include <complex>
#include <cstdio>

#include <mpi.h>
#include <ks1d.hpp>
#include <braid.h>

#include "braid_vector.hpp"

using namespace std;

#include <Eigen/Dense>
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<double, 1, Eigen::Dynamic>;

extern "C" {

  typedef struct _braid_App_struct
  {
    KS1D *ks;

    Vector rhs;
    Eigen::Array<Vector, 2, 6> fRK;
    Eigen::Array<Vector, 1, 6> qRK;

    Vector be;//(6);
    Matrix ae, ai; //(6,6), ai(6,6);

  } ks1d_app;

  unsigned int get_size(braid_App app)
  {
    return 2*app->ks->nx;
  }

  int
  ks1d_initial(braid_App     app,
          double        t,
          braid_Vector *u_ptr)
  {
    eigen_Vector *u = vector_create(get_size(app));
    app->ks->initial(u->vector->data());
    *u_ptr = u;
    return 0;
  }

  int
  ks1d_access(braid_App           app,
              braid_Vector        u,
              braid_AccessStatus  astatus)
  {

  }

  int ark4_phi(braid_App       app,
               braid_Vector    u,
               braid_PhiStatus status)
  {
    double tstart;
    double tstop;
    braid_PhiStatusGetTstartTstop(status, &tstart, &tstop);

    double dt = tstop - tstart;

    // cout << "ARK4 stepping: " << tstart << " " << tstop << endl;

    app->qRK(0) = *(u->vector);

    app->ks->expl_eval(app->qRK(0).data(), app->fRK(0,0).data());
    app->ks->impl_eval(app->qRK(0).data(), app->fRK(1,0).data());

    for (int k=1; k<6; k++) {
      app->rhs = app->qRK(0);
      for (int j=0; j<k; j++) {
        app->rhs += dt * app->ae(k,j) * app->fRK(0,j);
        app->rhs += dt * app->ai(k,j) * app->fRK(1,j);
      }
      app->ks->impl_solve(app->qRK(k).data(), app->fRK(1,k).data(), app->ai(k,k)*dt, app->rhs.data());
      app->ks->expl_eval(app->qRK(k).data(), app->fRK(0,k).data());
    }

    for (int j=0; j<6; j++) {
      app->qRK(0) += app->be(j) * dt * (app->fRK(0,j) + app->fRK(1,j));
    }

    *(u->vector) = app->qRK(0);

    braid_PhiStatusSetRFactor(status, 1);
    return 0;
  }

}

int main(int argc, char** argv)
{
  int    nsteps = 1000;
  double dt     = 0.1;
  unsigned int nx = 512;

  MPI_Init(&argc, &argv);

  braid_Core core;
  ks1d_app   app;

  app.ks = new KS1D(nx);

  for (int k=0; k<6; k++) {
    for (int i=0; i<2; i++) {
      app.fRK(i,k).resize(2*nx);
    }
    app.qRK(k).resize(2*nx);
  }

  app.rhs.resize(2*nx);

  app.be.resize(6);
  app.be.fill(0.0);
  app.be(0) = 82889.0/524892.0;
  app.be(2) = 15625.0/83664.0;
  app.be(3) = 69875.0/102672.0;
  app.be(4) =-2260.0/8211.0;
  app.be(5) = 1.0/4.0;

  app.ae.resize(6,5);
  app.ae.fill(0.0);
  app.ae(1,0) = 1.0/2.0;
  app.ae(2,0) = 13861.0/62500.0;
  app.ae(2,1) = 6889.0/62500.0;
  app.ae(3,0) =-116923316275.0/2393684061468.0;
  app.ae(3,1) =-2731218467317.0/15368042101831.0;
  app.ae(3,2) = 9408046702089.0/11113171139209.0;
  app.ae(4,0) =-451086348788.0/2902428689909.0;
  app.ae(4,1) =-2682348792572.0/7519795681897.0;
  app.ae(4,2) = 12662868775082.0/11960479115383.0;
  app.ae(4,3) = 3355817975965.0/11060851509271.0;
  app.ae(5,0) = 647845179188.0/3216320057751.0;
  app.ae(5,1) = 73281519250.0/8382639484533.0;
  app.ae(5,2) = 552539513391.0/3454668386233.0;
  app.ae(5,3) = 3354512671639.0/8306763924573.0;
  app.ae(5,4) = 4040.0/17871.0;

  app.ai.resize(6,6);
  app.ai.fill(0.0);
  app.ai(1,0) = 1.0/4.0;
  app.ai(1,1) = 1.0/4.0;
  app.ai(2,0) = 8611.0/62500.0;
  app.ai(2,1) =-1743.0/31250.0;
  app.ai(2,2) = 1.0/4.0;
  app.ai(3,0) = 5012029.0/34652500.0;
  app.ai(3,1) =-654441.0/2922500.0;
  app.ai(3,2) = 174375.0/388108.0;
  app.ai(3,3) = 1.0/4.0;
  app.ai(4,0) = 15267082809.0/155376265600.0;
  app.ai(4,1) =-71443401.0/120774400.0;
  app.ai(4,2) = 730878875.0/902184768.0;
  app.ai(4,3) = 2285395.0/8070912.0;
  app.ai(4,4) = 1.0/4.0;
  app.ai(5,0) = 82889.0/524892.0;
  app.ai(5,2) = 15625.0/83664.0;
  app.ai(5,3) = 69875.0/102672.0;
  app.ai(5,4) =-2260.0/8211.0;
  app.ai(5,5) = 1.0/4.0;

  braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, dt*nsteps, nsteps, &app, ark4_phi,
             ks1d_initial, vector_clone, vector_free, vector_sum, vector_norm0, ks1d_access,
             vector_bufsize, vector_bufpack, vector_bufunpack, &core);

  braid_Drive(core);

  MPI_Finalize();
}
