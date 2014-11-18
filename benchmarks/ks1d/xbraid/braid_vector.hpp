
#include <iostream>

#include <braid.h>
#include <Eigen/Dense>
using Vector = Eigen::Matrix<double, 1, Eigen::Dynamic>;

using namespace std;

extern "C" {

  unsigned int get_size(braid_App app);

  typedef struct _braid_Vector_struct
  {
    Vector *vector;
  } eigen_Vector;

  eigen_Vector *
  vector_create(unsigned int nx)
  {
    eigen_Vector *u = new eigen_Vector;
    u->vector = new Vector(nx);
    for (int i = 0; i < u->vector->size(); i++)
      (*u->vector)[i] = 0.0;
    return u;
  }

  int
  vector_clone(braid_App     app,
               braid_Vector  u,
               braid_Vector *v_ptr)
  {
    eigen_Vector *v = vector_create(u->vector->size());
    for (int i = 0; i < v->vector->size(); i++)
      (*v->vector)[i] = (*u->vector)[i];
    *v_ptr = v;
    return 0;
  }

  int
  vector_free(braid_App    app,
              braid_Vector u)
  {
    delete u->vector;
    delete u;
    return 0;
  }

  int
  vector_sum(braid_App     app,
             double        alpha,
             braid_Vector  x,
             double        beta,
             braid_Vector  y)
  {
    for (int i = 0; i < y->vector->size(); i++)
      (*y->vector)[i] = alpha * (*x->vector)[i] + beta * (*y->vector)[i];
    return 0;
  }

  int
  vector_norm0(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
  {
    double norm0;

    norm0 = 0.0;
    for (int i = 0; i < u->vector->size(); i++) {
      double a = fabs((*u->vector)[i]);
      if (a > norm0)
        norm0 = a;
    }
    *norm_ptr = norm0;
    return 0;
  }

  int
  vector_bufsize(braid_App  app,
                 int       *size_ptr)
  {
    *size_ptr = sizeof(double) * get_size(app);
    return 0;
  }

  int
  vector_bufpack(braid_App     app,
                 braid_Vector  u,
                 void         *buffer,
                 braid_Int    *size_ptr)
  {
    double *dbuffer = (double *) buffer;

    for (int i = 0; i < u->vector->size(); i++) {
      dbuffer[i] = (*u->vector)[i];
    }
    *size_ptr = u->vector->size() * sizeof(double);
    return 0;
  }

  int
  vector_bufunpack(braid_App     app,
                   void         *buffer,
                   braid_Vector *u_ptr)
  {
    eigen_Vector *u = vector_create(get_size(app));
    double *dbuffer = (double *) buffer;
    for (int i = 0; i < u->vector->size(); i++) {
      (*u->vector)[i] = dbuffer[i];
    }
    *u_ptr = u;
    return 0;
  }

}
