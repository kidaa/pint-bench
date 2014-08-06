"""Python script to generate centered finite-difference stencils for the KS eqn."""


import itertools
import numpy as np
#import mpmath
import sympy


def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)


def taylor_matrix(width, dim):
    """Compute the Taylor matrix T (and it's inverse) of width *width* and
    dimension *dim*.

    For 1d Taylor matrices, T[p,i] represents the coefficient of

       f^{p}(0) for p = 0 .. width

    in the Taylor series of

       f( (i-width/2)*dx) for i = 0 .. width.

    Similarly, for 2d Taylor matrices, the coefficient of

       f^{px,py}(0,0) for px, py in 0 .. width

    in the Taylor expansion of

       f( (i-width/2)*dx, (j-width/2)*dx ) for i, j in 0 .. width

    is given by T[].

    For example, to compute the centered finite-difference stencil for
    the 4th derivative of f(x) to 4th order

    >>> T, Tinv = taylor_matrix(7, 1)
    >>> stencil = np.dot(Tinv, np.asarray([ 0, 0, 0, 0, 1, 0, 0 ]))
    >>> print stencil
    [-0.16666667  2.         -6.5         9.33333333 -6.5         2.  -0.16666667]

    That is, we use a 7pt stencil and enforce 0 coefficients in front
    of all derivatives up to p=6 except for p=4.  The means that p=8
    is the first surviving term in the Taylor series (the coefficient
    of p=7 is zero by symmetry since we are considering centered
    stencils), and hence the stencil is 4th order accurate.

    """

    if width % 2 == 0:
        raise ValueError("width must odd")

    if dim not in [ 1, 2 ]:
        raise ValueError("dim %d not supported yet" % d)

    s = 2*[width]
    t = np.zeros(s, np.object)
    for mp in itertools.product(range(width)):
        for mi in itertools.product(range(width)):
            # t[mi+mp] = np.prod([ float(i-width/2)**p / factorial(p)
            #                      for i in mi for p in mp ])
            t[mi+mp] = np.prod([ sympy.Rational((i-width/2)**p, factorial(p))
                                 for i in mi for p in mp ])

    if dim == 2:
        t = np.kron(t, t)

    t = t.transpose()
    t = sympy.Matrix(width**dim, width**dim, t.flatten())

#    return t, np.linalg.inv(t)
    return t, t.inv()


def nstr(x):
    n, d = x.as_numer_denom()
    return "(%d.0 / %d)" % (n, d)


if __name__ == '__main__':
    np.set_printoptions(precision=18, linewidth=200)

    print "4th order stencils for 2d KS eqn"
    print ""
    print "lap"
    t, tinv = taylor_matrix(7, 2)
    s = tinv.dot(np.asarray([ [ 0, 0, 1, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 1, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ] ]).flatten())

    for ij, v in zip(itertools.product(range(-3, 4), repeat=2), s):
        i, j = ij
        if abs(v) > 0.0:
            print "+ x[j%+d][i%+d][0] * %s" % (j, i, nstr(v))

    k = 0
    for ij, v in zip(itertools.product(range(-3, 4), repeat=2), s):
        i, j = ij
        if abs(v) > 0.0:
            print "col[%d].i = i%+d; col[%d].j = j%+d;" % (k, i, k, j)
            k += 1

    k = 0
    for v in s:
        if abs(v) > 0.0:
            print "lap[%d] = h2inv * %s;" % (k, nstr(v))
            k += 1

    print "lap^2"
    t, tinv = taylor_matrix(7, 2)
    s = tinv.dot(np.asarray([ [ 0, 0, 0, 0, 1, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 2, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 1, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ] ]).flatten())

    for ij, v in zip(itertools.product(range(-3, 4), repeat=2), s):
        i, j = ij
        if abs(v) > 0.0:
            print "+ x[j%+d][i%+d][0] * %s" % (j, i, nstr(v))

    k = 0
    for ij, v in zip(itertools.product(range(-3, 4), repeat=2), s):
        i, j = ij
        if abs(v) > 0.0:
            print "col[%d].i = i%+d; col[%d].j = j%+d;" % (k, i, k, j)
            k += 1

    k = 0
    for v in s:
        if abs(v) > 0.0:
            print "hyplap[%d] = h4inv * %s;" % (k, nstr(v))
            k += 1


    print ""
    print "grad[0]"
    s = tinv.dot(np.asarray([ [ 0, 1, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0 ] ]).flatten())
    g = s[7*3:7*4]
    for i, v in zip(range(-3, 4), g):
        if abs(v) > 0:
            print "+ x[j%+d][i][0] * %s" % (i, nstr(v))
    for i, v in zip(range(-3, 4), g):
        if abs(v) > 0:
            print "+ x[j][i%+d][0] * %s" % (i, nstr(v))

    # for i, v in zip(range(-3, 4), g):
    #     print "+ x[j%+d][i][0] * %s" % (i, nstr(v))
    # for i, v in zip(range(-3, 4), g):
    #     print "+ x[j][i%+d][0] * %s" % (i, nstr(v))
