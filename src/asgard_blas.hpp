#pragma once

// wrappers for BLAS methods, use as internal header

#if defined(ASGARD_ACCELERATE)
  #include <Accelerate/Accelerate.h>
#else
  #ifdef ASGARD_MKL
    #include <mkl_cblas.h>
  #else
    #include "cblas.h"
  #endif
#endif

namespace asgard {

// fast math
namespace fm {

inline CBLAS_TRANSPOSE cblas_transpose_enum(char trans)
{
  switch (trans) {
    case 'n':
    case 'N':
      return CblasNoTrans;
    case 't':
    case 'T':
      return CblasTrans;
    default:
      return CblasConjTrans;
  };
}

inline CBLAS_UPLO cblas_uplo_enum(char trans)
{
  return (trans == 'U' or trans == 'u') ? CblasUpper : CblasLower;
}

inline CBLAS_DIAG cblas_diag_enum(char trans)
{
  return (trans == 'U' or trans == 'u') ? CblasUnit : CblasNonUnit;
}
template<typename P>
P nrm2(int n, P const x[]) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  if constexpr (std::is_same_v<P, double>)
    return cblas_dnrm2(n, x, 1);
  else
    return cblas_snrm2(n, x, 1);
}

template<typename P>
void scal(int n, P alpha, P x[]) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  if constexpr (std::is_same_v<P, double>)
    cblas_dscal(n, alpha, x, 1);
  else
    cblas_sscal(n, alpha, x, 1);
}

template<typename P>
void gemv(char trans, int m, int n, P alpha, P const A[], P const x[], P beta, P y[]) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  if constexpr (std::is_same_v<P, double>)
    cblas_dgemv(CblasColMajor, cblas_transpose_enum(trans), m, n, alpha, A, m, x, 1, beta, y, 1);
  else
    cblas_sgemv(CblasColMajor, cblas_transpose_enum(trans), m, n, alpha, A, m, x, 1, beta, y, 1);
}

template<typename P>
void rot(int n, P x[], P y[], P c, P s) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  if constexpr (std::is_same_v<P, double>)
    cblas_drot(n, x, 1, y, 1, c, s);
  else
    cblas_srot(n, x, 1, y, 1, c, s);
}

template<typename P>
void rotg(P *a, P *b, P *c, P *s) {
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  if constexpr (std::is_same_v<P, double>)
    cblas_drotg(a, b, c, s);
  else
    cblas_srotg(a, b, c, s);
}

template<typename P>
void tpsv(const char uplo, const char trans, const char diag, const int n,
          const P A[], P x[])
{
  static_assert(std::is_same_v<P, double> or std::is_same_v<P, float>);
  if constexpr (std::is_same_v<P, double>)
    cblas_dtpsv(CblasColMajor, cblas_uplo_enum(uplo), cblas_transpose_enum(trans),
                cblas_diag_enum(diag), n, A, x, 1);
  else
    cblas_stpsv(CblasColMajor, cblas_uplo_enum(uplo), cblas_transpose_enum(trans),
                cblas_diag_enum(diag), n, A, x, 1);
}

} // namespace fm

} // namespace asgard
