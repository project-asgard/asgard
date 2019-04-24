#include "apply_fmwt.hpp"

#include "connectivity.hpp"
#include "matlab_utilities.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

template<typename P>
fk::matrix<P> reshape(fk::matrix<P> mat, int const dim1, int const dim2)
{
  fk::vector<P> X(mat);
  fk::matrix<P> Xreshape(dim1, dim2);
  int count = 0;
  for (int i = 0; i < dim2; i++)
  {
    for (int j = 0; j < dim1; j++)
    {
      count          = i * dim1 + j;
      Xreshape(j, i) = X(count);
    }
  }
  return Xreshape;
}

template fk::matrix<double>
reshape(fk::matrix<double> mat, int const dim1, int const dim2);
template fk::matrix<float>
reshape(fk::matrix<float> mat, int const dim1, int const dim2);

template<typename P>
fk::matrix<P>
apply_fmwt(fk::matrix<P> fmwt, fk::matrix<P> matrix2, int const kdeg,
           int const lev, int const isLeft, int const isTrans, int const method)
{
  int const n = kdeg * pow(2, lev);
  fk::matrix<P> product(n, n);
  if (method == 1)
  {
    if (isLeft)
    {
      if (isTrans)
      {
        fk::matrix<P> fmwt_transpose = fk::matrix<P>(fmwt).transpose();
        product                      = fmwt_transpose * matrix2;
      }
      else
      {
        product = fmwt * matrix2;
      }
    }
    else
    {
      if (isTrans)
      {
        fk::matrix<P> fmwt_transpose = fk::matrix<P>(fmwt).transpose();
        product                      = matrix2 * fmwt_transpose;
      }
      else
      {
        product = matrix2 * fmwt;
      }
    }
  }
  if (method == 2)
  {
    int ip    = 0;
    int ipend = 2 * kdeg - 1;
    int col1  = 0;
    int col2  = n - 1;
    if (isLeft)
    {
      if (isTrans)
      {
        fk::matrix<P> fmwt_sub1 =
            fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1 + 1);
        fk::matrix<P> fmwt_sub1t = fk::matrix<P>(fmwt_sub1).transpose();
        fk::matrix<P> check1 =
            fmwt_sub1t * matrix2.extract_submatrix(ip, 0, ipend - ip + 1, n);
        product.set_submatrix(col1, 0, check1);
      }
      else
      {
        fk::matrix<P> check1 =
            fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1 + 1) *
            matrix2.extract_submatrix(col1, 0, col2 - col1 + 1, n);
        product.set_submatrix(ip, 0, check1);
      }
    }
    else
    {
      if (isTrans)
      {
        fk::matrix<P> fmwt_sub1 =
            fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1 + 1);
        fk::matrix<P> check1 =
            matrix2.extract_submatrix(0, col1, n, col2 - col1 + 1) *
            fmwt_sub1.transpose();
        product.set_submatrix(ip, 0, check1);
      }
      else
      {
        fk::matrix<P> check1 =
            matrix2.extract_submatrix(0, ip, n, ipend - ip + 1) *
            fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1 + 1);
        product.set_submatrix(ip, 0, check1);
      }
    }

    ip = 2 * kdeg;
    for (int iLev = 1; iLev < lev; iLev++)
    {
      int ncells = pow(2, iLev);
      int isize  = n / ncells;

      for (int icell = 0; icell < ncells; icell++)
      {
        ipend = ip + kdeg - 1;
        col1  = icell * isize;
        col2  = col1 + isize;
        if (isLeft)
        {
          if (isTrans)
          {
            fk::matrix<P> fmwt_sub1 =
                fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1);
            fk::matrix<P> fmwt_sub1t = fk::matrix<P>(fmwt_sub1).transpose();
            product.set_submatrix(
                col1, 0,
                product.extract_submatrix(col1, 0, col2 - col1, n) +
                    fmwt_sub1t *
                        matrix2.extract_submatrix(ip, 0, ipend - ip + 1, n));
          }
          else
          {
            product.set_submatrix(
                ip, 0,
                product.extract_submatrix(ip, 0, ipend - ip + 1, n) +
                    fmwt.extract_submatrix(ip, col1, ipend - ip + 1,
                                           col2 - col1) *
                        matrix2.extract_submatrix(col1, 0, col2 - col1, n));
          }
        }
        else
        {
          if (isTrans)
          {
            fk::matrix<P> fmwt_sub1 =
                fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1);
            product.set_submatrix(
                0, ip,
                product.extract_submatrix(0, ip, n, ipend - ip + 1) +
                    matrix2.extract_submatrix(0, col1, n, col2 - col1) *
                        fmwt_sub1.transpose());
          }
          else
          {
            product.set_submatrix(
                0, col1,
                product.extract_submatrix(0, col1, n, col2 - col1) +
                    matrix2.extract_submatrix(0, ip, n, ipend - ip + 1) *
                        fmwt.extract_submatrix(ip, col1, ipend - ip + 1,
                                               col2 - col1));
          }
        }

        ip = ipend + 1;
      }
    }
  }
  else if (method == 3)
  {
    int ip    = 0;
    int ipend = 2 * kdeg - 1;
    int col1  = 0;
    int col2  = n - 1;
    if (isLeft)
    {
      if (isTrans)
      {
        fk::matrix<P> fmwt_sub1 =
            fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1 + 1);
        fk::matrix<P> fmwt_sub1t = fk::matrix<P>(fmwt_sub1).transpose();
        fk::matrix<P> check1 =
            fmwt_sub1t * matrix2.extract_submatrix(ip, 0, ipend - ip + 1, n);
        product.set_submatrix(col1, 0, check1);
      }
      else
      {
        fk::matrix<P> check11 =
            fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1 + 1);
        fk::matrix<P> check12 =
            matrix2.extract_submatrix(col1, 0, col2 - col1 + 1, n);
        fk::matrix<P> check1 =
            fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1 + 1) *
            matrix2.extract_submatrix(col1, 0, col2 - col1 + 1, n);
        product.set_submatrix(ip, 0, check1);
      }
    }
    else
    {
      if (isTrans)
      {
        fk::matrix<P> fmwt_sub1 =
            fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1 + 1);
        fk::matrix<P> check1 =
            matrix2.extract_submatrix(0, col1, n, col2 - col1 + 1) *
            fmwt_sub1.transpose();
        product.set_submatrix(ip, 0, check1);
      }
      else
      {
        fk::matrix<P> check1 =
            matrix2.extract_submatrix(0, ip, n, ipend - ip + 1) *
            fmwt.extract_submatrix(ip, col1, ipend - ip + 1, col2 - col1 + 1);
        product.set_submatrix(0, col1, check1);
      }
    }

    ip = 2 * kdeg;
    for (int iLev = 1; iLev <= lev - 1; iLev++)
    {
      int ncells = pow(2, iLev);
      int isize  = n / ncells;

      int icell = 1;
      int ipend = ip + kdeg - 1;
      col1      = 1 + (icell - 1) * isize;
      col2      = col1 + isize - 1;

      fk::matrix<P> Fmat =
          fmwt.extract_submatrix(ip, col1 - 1, ipend - ip + 1, col2 - col1 + 1);
      int ncolX = n;
      int nrowX = n;
      int ncol  = (n * ncolX / isize);
      int nrows = (ncells * kdeg);
      ipend     = ip + (ncells * kdeg) - 1;

      if (isLeft)
      {
        if (isTrans)
        {
          fk::matrix<P> Xsub =
              matrix2.extract_submatrix(ip, 0, ipend - ip + 1, ncolX);
          fk::matrix<P> XsubReshape =
              reshape(Xsub, kdeg, ((ipend - ip + 1) * ncolX / kdeg));
          fk::matrix<P> FmatSub = Fmat.extract_submatrix(0, 0, kdeg, isize);
          fk::matrix<P> FmatSub_transpose = fk::matrix<P>(FmatSub).transpose();
          fk::matrix<P> FmatSubX          = FmatSub_transpose * XsubReshape;
          fk::matrix<P> FmatSubXreshape   = reshape(FmatSubX, nrowX, ncolX);
          product                         = product + FmatSubXreshape;
        }
        else
        {
          fk::matrix<P> Xreshape = reshape(matrix2, isize, ncol);
          fk::matrix<P> FmatSub  = Fmat.extract_submatrix(0, 0, kdeg, isize);
          fk::matrix<P> FmatSubX = FmatSub * Xreshape;
          fk::matrix<P> FSXreshape =
              reshape(FmatSubX, nrows, (kdeg * ncol) / nrows);
          product.set_submatrix(ip, 0, FSXreshape);
        }
      }
      else
      {
        if (isTrans)
        {
          for (int icell = 1; icell <= ncells; icell++)
          {
            int j1 = ip + (icell - 1) * kdeg;

            int jx1                = (icell - 1) * isize;
            int jx2                = jx1 + isize - 1;
            fk::matrix<P> FmatSubR = Fmat.extract_submatrix(0, 0, kdeg, isize);
            fk::matrix<P> XsubR =
                matrix2.extract_submatrix(0, jx1, nrowX, jx2 - jx1 + 1);
            fk::matrix<P> FmatX = XsubR * FmatSubR.transpose();
            product.set_submatrix(0, j1, FmatX);
          }
        }
        else
        {
          for (int icell = 1; icell <= ncells; icell++)
          {
            int j1 = (icell - 1) * isize;
            int j2 = j1 + isize - 1;

            int jx1                = ip + (icell - 1) * kdeg;
            int jx2                = jx1 + kdeg - 1;
            fk::matrix<P> FmatSubR = Fmat.extract_submatrix(0, 0, kdeg, isize);
            fk::matrix<P> XsubR =
                matrix2.extract_submatrix(0, jx1, nrowX, jx2 - jx1 + 1);
            fk::matrix<P> FmatX = XsubR * FmatSubR;
            fk::matrix<P> productPlus =
                product.extract_submatrix(0, j1, nrowX, j2 - j1 + 1) + FmatX;
            product.set_submatrix(0, j1, productPlus);
          }
        }
      }

      ip = ip + (ncells * kdeg);
    }
  }
  return product;
}

template fk::matrix<double>
apply_fmwt(fk::matrix<double> fmwt, fk::matrix<double> matrix2, int const kdeg,
           int const lev, int const isLeft, int const isTrans,
           int const method);
template fk::matrix<float> apply_fmwt(fk::matrix<float> fmwt,
                                      fk::matrix<float> matrix2, int const kdeg,
                                      int const lev, int const isLeft,
                                      int const isTrans, int const method);
