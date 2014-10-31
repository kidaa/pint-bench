

#include <ks1d.hpp>

#include <Eigen/Dense>

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<double, 1, Eigen::Dynamic>;

using namespace std;

/*
o'brien kennels
food water; $22/night per dog + $3 per play time, up to twice a day
mon, fri, $8-4:30pm tours, lunch 12-1pm.
*/

int main(int argc, char** argv)
{
  unsigned int nx = 512;
  unsigned int nsteps = 1000;
  double dt     = 0.1;

  KS1D ks(nx);

  Vector be(6);
  Matrix ae(6,6), ai(6,6);

  be.fill(0.0);
  be(0) = 82889.0/524892.0;
  be(2) = 15625.0/83664.0;
  be(3) = 69875.0/102672.0;
  be(4) =-2260.0/8211.0;
  be(5) = 1.0/4.0;

  ae.fill(0.0);
  ae(1,0) = 1.0/2.0;
  ae(2,0) = 13861.0/62500.0;
  ae(2,1) = 6889.0/62500.0;
  ae(3,0) =-116923316275.0/2393684061468.0;
  ae(3,1) =-2731218467317.0/15368042101831.0;
  ae(3,2) = 9408046702089.0/11113171139209.0;
  ae(4,0) =-451086348788.0/2902428689909.0;
  ae(4,1) =-2682348792572.0/7519795681897.0;
  ae(4,2) = 12662868775082.0/11960479115383.0;
  ae(4,3) = 3355817975965.0/11060851509271.0;
  ae(5,0) = 647845179188.0/3216320057751.0;
  ae(5,1) = 73281519250.0/8382639484533.0;
  ae(5,2) = 552539513391.0/3454668386233.0;
  ae(5,3) = 3354512671639.0/8306763924573.0;
  ae(5,4) = 4040.0/17871.0;

  ai.fill(0.0);
  ai(1,0) = 1.0/4.0;
  ai(1,1) = 1.0/4.0;
  ai(2,0) = 8611.0/62500.0;
  ai(2,1) =-1743.0/31250.0;
  ai(2,2) = 1.0/4.0;
  ai(3,0) = 5012029.0/34652500.0;
  ai(3,1) =-654441.0/2922500.0;
  ai(3,2) = 174375.0/388108.0;
  ai(3,3) = 1.0/4.0;
  ai(4,0) = 15267082809.0/155376265600.0;
  ai(4,1) =-71443401.0/120774400.0;
  ai(4,2) = 730878875.0/902184768.0;
  ai(4,3) = 2285395.0/8070912.0;
  ai(4,4) = 1.0/4.0;
  ai(5,0) = 82889.0/524892.0;
  ai(5,2) = 15625.0/83664.0;
  ai(5,3) = 69875.0/102672.0;
  ai(5,4) =-2260.0/8211.0;
  ai(5,5) = 1.0/4.0;

  Vector rhs;
  Eigen::Array<Vector, 2, 6> fRK;
  Eigen::Array<Vector, 1, 6> qRK;

  for (int k=0; k<6; k++) {
    for (int i=0; i<2; i++) {
      fRK(i,k).resize(2*nx);
    }
    qRK(k).resize(2*nx);
  }

  rhs.resize(2*nx);

  ks.initial(qRK(0).data());

  for (int nstep=0; nstep<nsteps; nstep++) {

    ks.expl_eval(qRK(0).data(), fRK(0,0).data());
    ks.impl_eval(qRK(0).data(), fRK(1,0).data());

    for (int k=1; k<6; k++) {
      rhs = qRK(0);
      for (int j=0; j<k; j++) {
        rhs += dt * ae(k,j) * fRK(0,j);
        rhs += dt * ai(k,j) * fRK(1,j);
      }
      ks.impl_solve(qRK(k).data(), fRK(1,k).data(), ai(k,k)*dt, rhs.data());
      ks.expl_eval(qRK(k).data(), fRK(0,k).data());
    }

    for (int j=0; j<6; j++) {
      qRK(0) += be(j) * dt * (fRK(0,j) + fRK(1,j));
    }


  }

  ks.dump(qRK(0).data(), "foo.dat");
}
