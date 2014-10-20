

#include <complex>
#include <vector>

#include "ks1d.hpp"


#define FFTWPP_SINGLE_THREAD

#include <convolution.h>
#include <fftw++.h>

using namespace std;

class KS1DPrivate {
public:

  fftwpp::fft1d forward;
  fftwpp::fft1d backward;
  fftwpp::ImplicitConvolution convolution;

  unsigned int nx;

  KS1DPrivate(unsigned int nx)
    : forward(nx, FFTW_FORWARD),
      backward(nx, FFTW_BACKWARD),
      convolution(nx),
      nx(nx) { }

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

void KS1D::initial(double *y)
{
  complex<double> *y0 = (complex<double>*) y;
  for (unsigned int i=0; i<this->nx; i++) {
    double x = double(i) / this->nx;
    y0[i] = (sin(x) + sin(3*x)) / this->nx;
  }
  this->d->forward.fft((Complex*) y0);
}

void KS1D::dump(double const *y, char const *fname)
{

}
