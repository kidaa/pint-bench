

#include <complex>
#include <vector>

#include "ks1d.hpp"


#define FFTWPP_SINGLE_THREAD

#include <convolution.h>
#include <fftw++.h>

using namespace std;

#define PI 3.1415926535897932385

class KS1DPrivate {
public:

  fftwpp::fft1d forward;
  fftwpp::fft1d backward;
  fftwpp::ImplicitConvolution convolution;

  unsigned int nx;

  vector<complex<double> > ddx;
  vector<double> k2_k4;

  KS1DPrivate(unsigned int nx)
    : forward(nx, FFTW_FORWARD),
      backward(nx, FFTW_BACKWARD),
      convolution(nx, 1, 1),
      nx(nx) 
  { 
    this->ddx.resize(nx);
    this->k2_k4.resize(nx);
    for (size_t i = 0; i < nx; i++) {
      double kx = 2 * PI / 100.0 * ((i <= nx / 2) ? int(i) : int(i) - int(nx));
      this->ddx[i] = complex<double>(0.0, 1.0) * kx;
      this->k2_k4[i] = kx*kx*(1.0 - kx*kx);
    }
  }

};

KS1D::KS1D(unsigned int nx)
{
  this->nx = nx;
  this->d = new KS1DPrivate(nx);

}

KS1D::~KS1D()
{
  delete this->d;
}

void KS1D::initial(double *_y)
{
  complex<double> *y = (complex<double>*) _y;
  for (unsigned int i=0; i<this->nx; i++) {
    double x = 2 * PI * double(i) / this->nx;
    y[i] = (0.1*sin(3*x) + 0.2*sin(4*x) + 0.3*sin(7*x)) / this->nx;
  }
  this->d->forward.fft((Complex*) y);
}

void KS1D::dump(double const *_y, char const *fname)
{
  complex<double> *y = (complex<double>*) _y;

  ofstream file(fname, ios::out|ios::binary);
  if (file.is_open()) {
    vector<complex<double> > x(this->nx);
    for (unsigned int i=0; i<this->nx; i++) x[i] = y[i];
    this->d->backward.fft((Complex*) x.data());

    vector<double> r(this->nx);
    for (unsigned int i=0; i<this->nx; i++) r[i] = x[i].real();
    file.write((char *) r.data(), sizeof(double) * this->nx);
  }
}

void KS1D::expl_eval(double const *_y, double *_f)
{
  complex<double> *y = (complex<double>*) _y;
  complex<double> *f = (complex<double>*) _f;
  for (unsigned int i=0; i<this->nx; i++) {
    f[i] = y[i];
  }
  this->d->convolution.autoconvolve((Complex*) f);
  for (unsigned int i=0; i<this->nx; i++) {
    f[i] = -0.25 * this->d->ddx[i] * f[i];
  }
}

void KS1D::impl_eval(double const *_y, double *_f)
{
  complex<double> *y = (complex<double>*) _y;
  complex<double> *f = (complex<double>*) _f;
  for (unsigned int i=0; i<this->nx; i++) {
    f[i] = this->d->k2_k4[i] * y[i];
  }
}

void KS1D::impl_solve(double *_y, double *_f, double dt, double const *_rhs)
{
  complex<double> *y = (complex<double>*) _y;
  complex<double> *f = (complex<double>*) _f;
  complex<double> *rhs = (complex<double>*) _rhs;
  for (unsigned int i=0; i<this->nx; i++) {
    y[i] = rhs[i] / (1.0 - dt*this->d->k2_k4[i]);
    f[i] = (y[i] - rhs[i]) / dt;
  }
}



