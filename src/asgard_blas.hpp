#pragma once

// wrappers for BLAS methods, use as internal header

extern "C" {
  // double precision
  double dnrm2_(int const *, double const[], int const *);
  void dscal_(int const *, double const *, double[], int const *);
  void drot_(int const *, double[], int const *, double[], int const *, double const *, double const *);
  void drotg_(double *, double *, double *, double *);
  void dtpsv_(char const *, char const *, char const *, int const *, double const[], double[], int const *);
  void dgemv_(char const *, int const *, int const *, double const *, double const[], int const*,
              double const[], int const *, double const *, double[], int const *);

  // single precision
  float snrm2_(int const *, float const[], int const *);
  void sscal_(int const *n, float const *, float[], int const *);
  void srot_(int const *, float[], int const *, float[], int const *, float const *, float const *);
  void srotg_(float *, float *, float *, float *);
  void stpsv_(char const *, char const *, char const *, int const *, float const[], float[], int const *);
  void sgemv_(char const *, int const *, int const *, float const *, float const[], int const*,
              float const[], int const *, float const *, float[], int const *);
}

namespace asgard {

// fast math
namespace fm {

template<typename P>
P nrm2(int n, P const x[]) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  int const one = 1;
  if constexpr (std::is_same_v<P, double>)
    return dnrm2_(&n, x, &one);
  else
    return snrm2_(&n, x, &one);
}

template<typename P>
void scal(int n, P alpha, P x[]) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  int const one = 1;
  if constexpr (std::is_same_v<P, double>)
    dscal_(&n, &alpha, x, &one);
  else
    sscal_(&n, &alpha, x, &one);
}

template<typename P>
void gemv(char trans, int m, int n, P alpha, P const A[], P const x[], P beta, P y[]) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  int const one = 1;
  if constexpr (std::is_same_v<P, double>)
    dgemv_(&trans, &m, &n, &alpha, A, &m, x, &one, &beta, y, &one);
  else
    sgemv_(&trans, &m, &n, &alpha, A, &m, x, &one, &beta, y, &one);
}

template<typename P>
void rot(int n, P x[], P y[], P c, P s) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  int const one = 1;
  if constexpr (std::is_same_v<P, double>)
    drot_(&n, x, &one, y, &one, &c, &s);
  else
    srot_(&n, x, &one, y, &one, &c, &s);
}

template<typename P>
void rotg(P *a, P *b, P *c, P *s) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  if constexpr (std::is_same_v<P, double>)
    drotg_(a, b, c, s);
  else
    srotg_(a, b, c, s);
}

template<typename P>
void tpsv(const char uplo, const char trans, const char diag, const int n,
          const P A[], P x[])
{
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  int const one = 1;
  if constexpr (std::is_same_v<P, double>)
    dtpsv_(&uplo, &trans, &diag, &n, A, x, &one);
  else
    stpsv_(&uplo, &trans, &diag, &n, A, x, &one);
}

} // namespace fm

} // namespace asgard
