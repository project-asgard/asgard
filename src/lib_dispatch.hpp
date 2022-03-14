#pragma once

enum class resource
{
  host,
  device
};

void initialize_libraries(int const local_rank);

// -- precision/execution resource wrapper for blas --
namespace lib_dispatch
{
template<typename P>
void rotg(P *a, P *b, P *c, P *s, resource const resrc = resource::host);

template<typename P>
P nrm2(int *n, P *x, int *incx, resource const resrc = resource::host);

template<typename P>
void copy(int *n, P *x, int *incx, P *y, int *incy,
          resource const resrc = resource::host);

template<typename P>
P dot(int *n, P *x, int *incx, P *y, int *incy,
      resource const resrc = resource::host);

template<typename P>
void axpy(int *n, P *alpha, P *x, int *incx, P *y, int *incy,
          resource const resrc = resource::host);

template<typename P>
void scal(int *n, P *alpha, P *x, int *incx,
          resource const resrc = resource::host);

template<typename P>
void gemv(char const *trans, int *m, int *n, P *alpha, P *A, int *lda, P *x,
          int *incx, P *beta, P *y, int *incy,
          resource const resrc = resource::host);

template<typename P>
void gemm(char const *transa, char const *transb, int *m, int *n, int *k,
          P *alpha, P *A, int *lda, P *B, int *ldb, P *beta, P *C, int *ldc,
          resource const resrc = resource::host);

template<typename P>
void getrf(int *m, int *n, P *A, int *lda, int *ipiv, int *info,
           resource const resrc = resource::host);

template<typename P>
void getri(int *n, P *A, int *lda, int *ipiv, P *work, int *lwork, int *info,
           resource const resrc = resource::host);

template<typename P>
void batched_gemm(P **const &a, int *lda, char const *transa, P **const &b,
                  int *ldb, char const *transb, P **const &c, int *ldc, int *m,
                  int *n, int *k, P *alpha, P *beta, int *num_batch,
                  resource const resrc = resource::host);

template<typename P>
void batched_gemv(P **const &a, int *lda, char const *transa, P **const &x,
                  P **const &y, int *m, int *n, P *alpha, P *beta,
                  int *num_batch, resource const resrc = resource::host);

template<typename P>
void gesv(int *n, int *nrhs, P *A, int *lda, int *ipiv, P *b, int *ldb,
          int *info);

template<typename P>
void getrs(char *trans, int *n, int *nrhs, P *A, int *lda, int *ipiv, P *b,
           int *ldb, int *info);

#ifdef ASGARD_USE_SCALAPACK
template<typename P>
void scalapack_gesv(int *n, int *nrhs, P *A, int *descA, int *ipiv, P *b,
                    int *descB, int *info);

template<typename P>
void scalapack_getrs(char *trans, int *n, int *nrhs, P *A, int *descA,
                     int *ipiv, P *b, int *descB, int *info);
#endif
} // namespace lib_dispatch
