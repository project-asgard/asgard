#pragma once

// wrappers for BLAS methods, use as internal header

#include "asgard_tools.hpp"

#ifdef ASGARD_USING_APPLEBLAS
  #include <Accelerate/Accelerate.h>
#else
  #ifdef ASGARD_USING_MKL
    #include <mkl_cblas.h>
  #else
    #include "cblas.h"
  #endif
#endif

namespace asgard {

// fast math
namespace fm {

//! converts N/n/T/t/C/c to a CBLAS_TRANSPOSE type, keeps this header private
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

//! converts U/u/L/l to a CBLAS_UPLO type, keeps this header private
inline CBLAS_UPLO cblas_uplo_enum(char trans)
{
  return (trans == 'U' or trans == 'u') ? CblasUpper : CblasLower;
}
//! converts U/u/N/n to a CBLAS_DIAG type, keeps this header private
inline CBLAS_DIAG cblas_diag_enum(char trans)
{
  return (trans == 'U' or trans == 'u') ? CblasUnit : CblasNonUnit;
}
//! computes norm L-2, BLAS snrm2()/dnrm2()
template<typename P>
P nrm2(int n, P const x[]) {
  static_assert(is_double<P> or is_float<P>);
  if constexpr (is_double<P>)
    return cblas_dnrm2(n, x, 1);
  else
    return cblas_snrm2(n, x, 1);
}
//! computes y += alpha * x, BLAS saxpy()/daxpy()
template<typename P>
void axpy(int n, P alpha, P const x[], P y[]) {
  static_assert(is_double<P> or is_float<P>);
  if constexpr (is_double<P>)
    cblas_daxpy(n, alpha, x, 1, y, 1);
  else
    cblas_daxpy(n, alpha, x, 1, y, 1);
}
//! scales vector by number, BLAS sscal()/dscal()
template<typename P>
void scal(int n, P alpha, P x[]) {
  static_assert(is_double<P> or is_float<P>);
  if constexpr (is_double<P>)
    cblas_dscal(n, alpha, x, 1);
  else
    cblas_sscal(n, alpha, x, 1);
}
template<typename P>
void scal_omp(int n, P alpha, P x[]) {
  static_assert(is_double<P> or is_float<P>);
  ASGARD_OMP_PARFOR_SIMD
  for (int i = 0; i < n; i++)
    x[i] *= alpha;
}
//! matrix vector product, BLAS sgemv()/dgemv()
template<typename P>
void gemv(char trans, int m, int n, no_deduce<P> alpha, P const A[],
          P const x[], no_deduce<P> beta, P y[]) {
  tools::time_event perf_("openblas gemv");
  static_assert(is_double<P> or is_float<P>);
  if constexpr (is_double<P>)
    cblas_dgemv(CblasColMajor, cblas_transpose_enum(trans), m, n, alpha, A, m, x, 1, beta, y, 1);
  else
    cblas_sgemv(CblasColMajor, cblas_transpose_enum(trans), m, n, alpha, A, m, x, 1, beta, y, 1);
}
template<typename P>
void gemv_omp(char trans, int m, int n, no_deduce<P> alpha, P const A[],
              P const x[], no_deduce<P> beta, P y[]) {
  static_assert(is_double<P> or is_float<P>);
  tools::time_event perf_("my gemv");

  if (trans == 't' or trans == 'T')
  {
    for (int j = 0; j < n; j++) {
      P sum = 0;
      ASGARD_OMP_PARFOR_SIMD_EXTRA(reduction(+:sum))
      for (int i = 0; i < m; i++) {
        sum += A[j * m + i] * x[i];
      }
      y[j] = alpha * sum + beta * y[j];
    }
  }
  else
  {
    int constexpr block_size = 128;
    int const num_blocks = [&]() ->
      int {
        int const b = m / block_size;
        return (block_size * b < m) ? b + 1 : b;
      }();

    #pragma omp parallel for
    for (int b = 0; b < num_blocks; b++) {
      int const begin = b * block_size;
      int const candidate_end = begin + block_size;
      int const end = (candidate_end < m) ? candidate_end : m;
      for (int i = begin; i < end; i++)
      {
        P sum = 0;
        ASGARD_OMP_SIMD
        for (int j = 0; j < n; j++) {
          sum += A[j * m + i] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
      }
    }
  }
}

//! apply the Givens rotation, BLAS srot()/drot()
template<typename P>
void rot(int n, P x[], P y[], P c, P s) {
  static_assert(is_double<P> or is_float<P>);
  if constexpr (is_double<P>)
    cblas_drot(n, x, 1, y, 1, c, s);
  else
    cblas_srot(n, x, 1, y, 1, c, s);
}
//! generate the parameters of a Givens rotation, BLAS srotg()/drotg()
template<typename P>
void rotg(P *a, P *b, P *c, P *s) {
  static_assert(is_double<P> or is_float<P>);
  if constexpr (is_double<P>)
    cblas_drotg(a, b, c, s);
  else
    cblas_srotg(a, b, c, s);
}
//! triangular matrix-vector solve using packed format, BLAS stpsv()/dtpsv()
template<typename P>
void tpsv(const char uplo, const char trans, const char diag, const int n,
          const P A[], P x[])
{
  static_assert(is_double<P> or is_float<P>);
  if constexpr (is_double<P>)
    cblas_dtpsv(CblasColMajor, cblas_uplo_enum(uplo), cblas_transpose_enum(trans),
                cblas_diag_enum(diag), n, A, x, 1);
  else
    cblas_stpsv(CblasColMajor, cblas_uplo_enum(uplo), cblas_transpose_enum(trans),
                cblas_diag_enum(diag), n, A, x, 1);
}

} // namespace fm

} // namespace asgard
