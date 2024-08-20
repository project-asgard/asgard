#pragma once
#include "asgard_resources.hpp"

namespace asgard
{
// -- precision/execution resource wrapper for blas --
namespace lib_dispatch
{
void initialize_libraries(int const local_rank);

template<resource resrc = resource::host, typename P>
void rot(const int n, P *x, const int incx, P *y, const int incy, const P c,
         const P s);

template<resource resrc = resource::host, typename P>
void rotg(P *a, P *b, P *c, P *s);

template<resource resrc = resource::host, typename P>
P nrm2(int n, P const x[], int incx);

template<resource resrc = resource::host, typename P>
void copy(int n, P const *x, int incx, P *y, int incy);

template<resource resrc = resource::host, typename P>
void copy(int64_t n, P const *x, P *y);

template<resource resrc = resource::host, typename P>
P dot(int n, P const *x, int incx, P const *y, int incy);

template<resource resrc = resource::host, typename P>
void axpy(int n, P alpha, P const *x, int incx, P *y, int incy);

template<resource resrc = resource::host, typename P>
void scal(int n, P alpha, P *x, int incx);

template<resource resrc = resource::host, typename P>
void gemv(char trans, int m, int n, P alpha, P const *A, int lda, P const *x,
          int incx, P beta, P *y, int incy);

template<resource resrc = resource::host, typename P>
void gemm(char transa, char transb, int m, int n, int k, P alpha, P const *A,
          int lda, P const *B, int ldb, P beta, P *C, int ldc);

template<resource resrc = resource::host, typename P>
int getrf(int m, int n, P *A, int lda, int *ipiv);

template<resource resrc = resource::host, typename P>
int getri(int n, P *A, int lda, int *ipiv, P *work, int lwork);

template<resource resrc = resource::host, typename P>
void batched_gemm(P **const &a, int lda, char transa, P **const &b, int ldb,
                  char transb, P **const &c, int ldc, int m, int n, int k,
                  P alpha, P beta, int num_batch);

template<typename P>
int gesv(int n, int nrhs, P *A, int lda, int *ipiv, P *b, int ldb);

template<resource resrc = resource::host, typename P>
void tpsv(const char uplo, const char trans, const char diag, const int n,
          const P *ap, P *x, const int incx);

template<typename P>
int getrs(char trans, int n, int nrhs, P const *A, int lda, int const *ipiv,
          P *b, int ldb);

template<typename P>
int pttrf(int n, P *D, P *E);

template<typename P>
int pttrs(int n, int nrhs, P const *D, P const *E, P *B, int ldb);

template<resource resrc = resource::host, typename P>
void sparse_gemv(char const trans, int rows, int cols, int nnz,
                 const int *row_offsets, const int *col_indices, const P *vals,
                 P alpha, const P *x, P beta, P *y);

} // namespace lib_dispatch
} // namespace asgard
