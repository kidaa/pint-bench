
#include <complex>

#include <cstdio>

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

  void sweep() override
  {
    pfasst::encap::IMEXSweeper<double>::sweep();
    auto step = this->get_controller()->get_step();
    auto iter = this->get_controller()->get_iteration();
    char fname[128]; snprintf(fname, 128, "s%04dk%02d.dat", int(step), int(iter));
    this->dump(this->get_end_state(), fname);
  }

  void dump(encap y, string fname)
  {
    auto& v = as_vector<complex<double>,double>(y);
    this->ks.dump((double*) v.data(), fname.c_str());
  }

  void f_expl_eval(encap f_expl_encap, encap u_encap, double t) override
  {
    auto& u = as_vector<complex<double>,double>(u_encap);
    auto& f = as_vector<complex<double>,double>(f_expl_encap);
    this->ks.expl_eval((double*) u.data(), (double*) f.data());
  }

  void f_impl_eval(encap f_impl_encap, encap u_encap, double t) override
  {
    auto& u = as_vector<complex<double>,double>(u_encap);
    auto& f = as_vector<complex<double>,double>(f_impl_encap);
    this->ks.impl_eval((double*) u.data(), (double*) f.data());
  }

  void impl_solve(encap f_encap, encap u_encap, double t, double dt, encap rhs_encap) override
  {
    auto& u = as_vector<complex<double>,double>(u_encap);
    auto& f = as_vector<complex<double>,double>(f_encap);
    auto& rhs = as_vector<complex<double>,double>(rhs_encap);
    this->ks.impl_solve((double*) u.data(), (double*) f.data(), dt, (double*) rhs.data());
  }
};

class SpectralTransfer1D : public PolyInterpMixin<double>
{
  typedef shared_ptr<pfasst::encap::Encapsulation<double>> encap;
  typedef shared_ptr<const pfasst::encap::Encapsulation<double>> const_encap;

  void interpolate(encap dst, const_encap src) override
  {
    auto& fine = as_vector<complex<double>,double>(dst);
    auto& crse = as_vector<complex<double>,double>(src);

    unsigned int fi=0, ci=0, nf=fine.size(), nc=crse.size();

    for (    ; ci < nc/2;      ) fine[fi++] = crse[ci++];
    for (    ; fi < nf-nc/2+1; ) fine[fi++] = 0.0;
    for (ci++; fi < nf;        ) fine[fi++] = crse[ci++];
  }

  void restrict(encap dst, const_encap src) override
  {
    auto& fine = as_vector<complex<double>,double>(src);
    auto& crse = as_vector<complex<double>,double>(dst);

    unsigned int fi=0, ci=0, nf=fine.size(), nc=crse.size();

    for (    ; ci < nc/2;    ) crse[ci++] = fine[fi++];
    crse[ci++] = 0.0; fi += 1 + nc;
    for (    ; ci < nc;      ) crse[ci++] = fine[fi++];
  }
};


int main(int argc, char** argv)
{
  int    nsteps = 1000;
  double dt     = 0.1;
  int    niters = 4;

  MPI_Init(&argc, &argv);

  vector<pair<size_t, pfasst::QuadratureType>> nodes = {
    { 3, pfasst::QuadratureType::GaussLobatto },
    { 5, pfasst::QuadratureType::GaussLobatto }
  };

  vector<unsigned int> nx = { 256, 512 };

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
