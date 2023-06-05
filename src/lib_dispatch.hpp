#pragma once
namespace asgard
{
enum class resource
{
  host,
  device
};

// -- precision/execution resource wrapper for blas --
namespace lib_dispatch
{
void initialize_libraries(int const local_rank);

template<resource resrc = resource::host, typename P>
void rotg(P *a, P *b, P *c, P *s);

template<resource resrc = resource::host, typename P>
P nrm2(int n, P const x[], int incx);

template<resource resrc = resource::host, typename P>
void copy(int n, P const *x, int incx, P *y, int incy);

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

template<typename P>
void getrf(int *m, int *n, P *A, int *lda, int *ipiv, int *info,
           resource const resrc = resource::host);

template<typename P>
void getri(int *n, P *A, int *lda, int *ipiv, P *work, int *lwork, int *info,
           resource const resrc = resource::host);

template<resource resrc = resource::host, typename P>
void batched_gemm(P **const &a, int lda, char transa, P **const &b, int ldb,
                  char transb, P **const &c, int ldc, int m, int n, int k,
                  P alpha, P beta, int num_batch);

template<typename P>
void gesv(int *n, int *nrhs, P *A, int *lda, int *ipiv, P *b, int *ldb,
          int *info);

template<typename P>
void getrs(char *trans, int *n, int *nrhs, P const *A, int *lda,
           int const *ipiv, P *b, int *ldb, int *info);

template<typename P>
void pttrf(int *n, P *D, P *E, int *info,
           resource const resrc = resource::host);

template<typename P>
void pttrs(int *n, int *nrhs, P const *D, P const *E, P *B, int *ldb, int *info,
           resource const resrc = resource::host);

#ifdef ASGARD_USE_SCALAPACK
template<typename P>
void scalapack_gesv(int *n, int *nrhs, P *A, int *descA, int *ipiv, P *b,
                    int *descB, int *info);

template<typename P>
void scalapack_getrs(char *trans, int *n, int *nrhs, P const *A, int *descA,
                     int const *ipiv, P *b, int *descB, int *info);
#endif
} // namespace lib_dispatch
} // namespace asgard
