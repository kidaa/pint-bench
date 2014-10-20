
#include <complex>

#include <pfasst/pfasst.hpp>
#include <pfasst/mpi_communicator.hpp>
#include <pfasst/encap/imex_sweeper.hpp>
#include <pfasst/encap/mpi_vector.hpp>
#include <pfasst/encap/poly_interp.hpp>
#include <pfasst/encap/automagic.hpp>

#include <ks1d.hpp>

using namespace std;
using namespace pfasst;
using namespace pfasst::encap;
using namespace pfasst::mpi;

class KS1DSweeper : public IMEXSweeper<double>
{
  typedef shared_ptr<pfasst::encap::Encapsulation<double>> encap;

  KS1D ks;

public:
  KS1DSweeper(unsigned int ns) : ks(ns) { }

  void initial(encap y0)
  {
    auto& v = as_vector<complex<double>,double>(y0);
    this->ks.initial((double*) v.data());
  }

  void dump(encap y, string fname)
  {
    auto& v = as_vector<complex<double>,double>(y);
    this->ks.dump((double*) v.data(), fname.c_str());
  }

  void f_expl_eval(encap f_expl_encap, encap u_encap, double t) override
  {

  }

  void f_impl_eval(encap f_expl_encap, encap u_encap, double t) override
  {

  }

  void impl_solve(encap f_encap, encap u_encap, double t, double dt, encap rhs_encap) override
  {

  }

};

class SpectralTransfer1D : public PolyInterpMixin<double>
{
  typedef shared_ptr<pfasst::encap::Encapsulation<double>> encap;
  typedef shared_ptr<const pfasst::encap::Encapsulation<double>> const_encap;

  void interpolate(encap dst, const_encap src) override
  {
  }

  void restrict(encap dst, const_encap src) override
  {
  }
};


int main(int argc, char** argv)
{
  int    nsteps = 4;
  double dt     = 0.01;
  int    niters = 4;

  MPI_Init(&argc, &argv);

  vector<pair<size_t, pfasst::quadrature::QuadratureType>> nodes = {
    { 3, pfasst::quadrature::QuadratureType::GaussLobatto },
    { 5, pfasst::quadrature::QuadratureType::GaussLobatto }
  };

  vector<unsigned int> nx = { 64, 128 };

  auto build_level = [nx](size_t level) {
    auto factory  = make_shared<MPIVectorFactory<complex<double>>>(nx[level]);
    auto sweeper  = make_shared<KS1DSweeper>(nx[level]);
    auto transfer = make_shared<SpectralTransfer1D>();

    return AutoBuildTuple<>(sweeper, transfer, factory);
  };

  auto initial = [](shared_ptr<EncapSweeper<>> sweeper, shared_ptr<Encapsulation<>> q0) {
    auto swp = dynamic_pointer_cast<KS1DSweeper>(sweeper);
    swp->initial(q0);
    swp->dump(q0, "initial.dat");
  };

  MPICommunicator comm(MPI_COMM_WORLD);
  PFASST<double> pf;

  auto_build(pf, nodes, build_level);
  auto_setup(pf, initial);

  pf.set_comm(&comm);
  pf.set_duration(0.0, nsteps * dt, dt, niters);
  pf.set_nsweeps({2, 1});
  pf.run();

  //  fftw_cleanup();
  MPI_Finalize();
}
