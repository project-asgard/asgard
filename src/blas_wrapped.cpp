#include "blas_wrapped.hpp"

template<typename P>
void copy(int *n, P *x, int *incx, P *y, int *incy, environment const environ)
{}

template<typename P>
P dot(int *n, P *X, int *incx, P *Y, int *incy, environment const environ)
{}

template<typename P>
void axpy(int *n, P *alpha, P *X, int *incx, P *Y, int *incy,
          environment const environ)
{}

template<typename P>
void scal(int *n, P *alpha, P *X, int *incx, environment const environ)
{}

template<typename P>
void gemv(char const *trans, int *m, int *n, P *alpha, P *A, int *lda, P *x,
          int *incx, P *beta, P *y, int *incy, environment const environ)
{}

template<typename P>
void gemm(char const *transa, char const *transb, int *m, int *n, int *k,
          P *alpha, P *A, int *lda, P *B, int *ldb, P *beta, P *C, int *ldc,
          environment const environ)
{}

template<typename P>
void getrf(int *m, int *n, P *A, int *lda, int *ipiv, int *info,
           environment const environ)
{}

template<typename P>
void getri(int *n, P *A, int *lda, int *ipiv, P *work, int *lwork, int *info,
           environment const environ)
{}

template void copy(int *n, float *x, int *incx, float *y, int *incy,
                   environment const environ);
template void copy(int *n, double *x, int *incx, double *y, int *incy,
                   environment const environ);
template void
copy(int *n, int *x, int *incx, int *y, int *incy, environment const environ);

template float dot(int *n, float *X, int *incx, float *Y, int *incy,
                   environment const environ);
template double dot(int *n, double *X, int *incx, double *Y, int *incy,
                    environment const environ);
template int
dot(int *n, int *X, int *incx, int *Y, int *incy, environment const environ);

template void axpy(int *n, float *alpha, float *X, int *incx, float *Y,
                   int *incy, environment const environ);
template void axpy(int *n, double *alpha, double *X, int *incx, double *Y,
                   int *incy, environment const environ);
template void axpy(int *n, int *alpha, int *X, int *incx, int *Y, int *incy,
                   environment const environ);

template void
scal(int *n, float *alpha, float *X, int *incx, environment const environ);
template void
scal(int *n, double *alpha, double *X, int *incx, environment const environ);
template void
scal(int *n, int *alpha, int *X, int *incx, environment const environ);

template void gemv(char const *trans, int *m, int *n, float *alpha, float *A,
                   int *lda, float *x, int *incx, float *beta, float *y,
                   int *incy, environment const environ);
template void gemv(char const *trans, int *m, int *n, double *alpha, double *A,
                   int *lda, double *x, int *incx, double *beta, double *y,
                   int *incy, environment const environ);
template void gemv(char const *trans, int *m, int *n, int *alpha, int *A,
                   int *lda, int *x, int *incx, int *beta, int *y, int *incy,
                   environment const environ);

template void gemm(char const *transa, char const *transb, int *m, int *n,
                   int *k, float *alpha, float *A, int *lda, float *B, int *ldb,
                   float *beta, float *C, int *ldc, environment const environ);
template void gemm(char const *transa, char const *transb, int *m, int *n,
                   int *k, double *alpha, double *A, int *lda, double *B,
                   int *ldb, double *beta, double *C, int *ldc,
                   environment const environ);
template void gemm(char const *transa, char const *transb, int *m, int *n,
                   int *k, int *alpha, int *A, int *lda, int *B, int *ldb,
                   int *beta, int *C, int *ldc, environment const environ);

template void getrf(int *m, int *n, float *A, int *lda, int *ipiv, int *info,
                    environment const environ);
template void getrf(int *m, int *n, double *A, int *lda, int *ipiv, int *info,
                    environment const environ);

template void getri(int *n, float *A, int *lda, int *ipiv, float *work,
                    int *lwork, int *info, environment const environ);
template void getri(int *n, double *A, int *lda, int *ipiv, double *work,
                    int *lwork, int *info, environment const environ);
