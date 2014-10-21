
#ifndef _KS1D_H_
#define _KS1D_H_

class KS1DPrivate;

class KS1D
{
  KS1DPrivate* d;
  unsigned int nx;
public:
  KS1D(unsigned int nx);
  ~KS1D();
  void initial(double *y);
  void dump(double const *y, char const *fname);
  void expl_eval(double const *y, double *f);
  void impl_eval(double const *y, double *f);
  void impl_solve(double *y, double *f, double dt, double const *rhs);
};

#endif
