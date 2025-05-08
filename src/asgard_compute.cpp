#include "asgard_compute.hpp"

// BLAS/LAPACK methods, using the common Fortran API
extern "C" {
  // general PLU factorize
  void dgetrf_(int const *m, int const *n, double *A, int const *lda, int *ipiv, int *info);
  void sgetrf_(int const *m, int const *n, float *A, int const *lda, int *ipiv, int *info);
  void dgetrs_(char const *trans, int const *n, int const *nrhs, double const *A,
               int const *lda, int const *ipiv, double *b, int const*ldb, int *info);
  void sgetrs_(char const *trans, int const *n, int const *nrhs, float const *A,
               int const *lda, int const *ipiv, float *b, int const *ldb, int *info);

  // tri-diagonal factorize
  void dpttrf_(int const *n, double *D, double *E, int *info);
  void spttrf_(int const *n, float *D, float *E, int *info);
  void dpttrs_(int const *n, int const *nrhs, double const *D, double const *E, double *B,
               int const *ldb, int *info);
  void spttrs_(int const *n, int const *nrhs, float const *D, float const *E, float *B,
               int const *ldb, int *info);
}

namespace asgard
{

#ifdef ASGARD_USE_CUDA
namespace gpu
{
std::string error_message(cudaError_t err) {
  return std::string("CUDA reported an error: '") + cudaGetErrorString(err) + std::string("'");
}

std::string error_message(cusolverStatus_t err) {
  std::string message = "cuSolver reported an error: '";
  switch (err) {
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      message += "library not initialized";
      break;
    case CUSOLVER_STATUS_ALLOC_FAILED:
      message += "resource allocation failed";
      break;
    case CUSOLVER_STATUS_INVALID_VALUE:
      message += "invalid value";
      break;
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      message += "architecture mismatch/missing hardware feature";
      break;
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      message += "execution failure, failure to launch a kernel";
      break;
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      message += "internal error, often caused by cudaMemcpyAsync()";
      break;
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      message += "matrix type not supported by this function";
      break;
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      message += "unsupported combination of parameters";
      break;
    case CUSOLVER_STATUS_MAPPING_ERROR:
      message += "CUSOLVER_STATUS_MAPPING_ERROR";
      break;
    case CUSOLVER_STATUS_ZERO_PIVOT:
      message += "CUSOLVER_STATUS_ZERO_PIVOT";
      break;
    case CUSOLVER_STATUS_INVALID_LICENSE:
      message += "CUSOLVER_STATUS_INVALID_LICENSE";
      break;
    default:
      message += "code " + std::to_string(err);
  }
  return message + "'";
}

} // namespace gpu
#endif

#ifdef ASGARD_USE_ROCM
namespace gpu
{
std::string error_message(hipError_t  err) {
  return std::string("ROCM reported an error: '") + hipGetErrorString(err) + std::string("'");
}

std::string error_message(rocblas_status err) {
  return std::string("rocBLAS reported an error: '") + rocblas_status_to_string(err) + std::string("'");
}

} // namespace gpu
#endif

compute_resources::compute_resources() {
  #ifdef ASGARD_USE_CUDA
  cudaGetDeviceCount(&num_gpus_);
  rassert(has_gpu(), "CUDA is enabled but there are no visible CUDA devices, maybe a driver problem");
  cusolver_check_error( cusolverDnCreate(&cusolverdn) );
  #endif
  #ifdef ASGARD_USE_ROCM
  rocm_check_error( hipGetDeviceCount(&num_gpus_) );
  rassert(has_gpu(), "ROCM is enabled but there are no visible ROCM devices, maybe a driver problem");
  rocblas_check_error( rocblas_create_handle(&rocblas) );
  #endif
}

compute_resources::~compute_resources() {
  #ifdef ASGARD_USE_CUDA
  cusolverDnDestroy(cusolverdn);
  #endif
  #ifdef ASGARD_USE_ROCM
  rocblas_destroy_handle(rocblas);
  #endif
}

// LAPACK factorize and solve for a general matrix, used by the direct solver
template<typename P>
void compute_resources::getrf(int M, std::vector<P> &A, std::vector<int> &ipiv) const
{
  expect(static_cast<size_t>(M) * static_cast<size_t>(M) == A.size());

  ipiv.resize(M);

  int info = 0;

  if constexpr (is_double<P>) {
    dgetrf_(&M, &M, A.data(), &M, ipiv.data(), &info);
  } else {
    sgetrf_(&M, &M, A.data(), &M, ipiv.data(), &info);
  }

  if (info != 0) {
    std::stringstream sout;
    if (info < 0)
    {
      sout << "getrf(): the " << -info << "-th parameter had an illegal value!\n";
    }
    else
    {
      sout << "getrf(): the diagonal element of the triangular factor of A,\n";
      sout << "U(" << info << ',' << info << ") is zero, so that A is singular;\n";
      sout << "the matrix could not be factorized.\n";
    }
    throw std::runtime_error(sout.str());
  }
}

template void
compute_resources::getrf<double>(int, std::vector<double> &A, std::vector<int> &ipiv) const;
template void
compute_resources::getrf<float>(int, std::vector<float> &A, std::vector<int> &ipiv) const;

template<typename P>
void compute_resources::getrs(int M, std::vector<P> const &A, std::vector<int> const &ipiv,
                              std::vector<P> &b) const
{
  expect(static_cast<size_t>(M) == ipiv.size());
  expect(ipiv.size() * ipiv.size() == A.size());
  expect(ipiv.size() == b.size());

  int info  = 0;
  int const nrhs = 1; // num right-hand-sides
  char const trans = 'N';

  if constexpr (is_double<P>) {
    dgetrs_(&trans, &M, &nrhs, A.data(), &M, ipiv.data(), b.data(), &M, &info);
  } else {
    sgetrs_(&trans, &M, &nrhs, A.data(), &M, ipiv.data(), b.data(), &M, &info);
  }

  // only check if arguments have illegal value
  expect(info == 0);
}

template void compute_resources::getrs<double>(
    int, std::vector<double> const &A, std::vector<int> const &ipiv, std::vector<double> &b) const;
template void compute_resources::getrs<float>(
    int, std::vector<float> const &A, std::vector<int> const &ipiv, std::vector<float> &b) const;

#ifdef ASGARD_USE_CUDA
template<typename P>
void compute_resources::getrf(int M, gpu::vector<P> &A, gpu::vector<int> &ipiv) const {
  expect(static_cast<int64_t>(M) * M == A.size());

  ipiv.resize(M);

  int lwork = 0;
  if constexpr (is_double<P>) {
    cusolver_check_error( cusolverDnDgetrf_bufferSize(cusolverdn, M, M, A.data(), M, &lwork) );
  } else {
    cusolver_check_error( cusolverDnSgetrf_bufferSize(cusolverdn, M, M, A.data(), M, &lwork) );
  }

  gpu::vector<P> workspace(lwork);
  gpu::vector<int> gpu_info(1);

  if constexpr (is_double<P>) {
    cusolver_check_error( cusolverDnDgetrf(cusolverdn, M, M, A.data(), M,
                                           workspace.data(), ipiv.data(), gpu_info.data()) );
  } else {
    cusolver_check_error( cusolverDnSgetrf(cusolverdn, M, M, A.data(), M,
                                           workspace.data(), ipiv.data(), gpu_info.data()) );
  }

  int info = 0;
  gpu_info.copy_to_host(&info);

  if (info != 0) {
    std::stringstream sout;
    if (info < 0)
    {
      sout << "cuda-getrf(): the " << -info << "-th parameter had an illegal value!\n";
    }
    else
    {
      sout << "cuda-getrf(): the diagonal element of the triangular factor of A,\n";
      sout << "U(" << info << ',' << info << ") is zero, so that A is singular;\n";
      sout << "the matrix could not be factorized.\n";
    }
    throw std::runtime_error(sout.str());
  }
}

template<typename P>
void compute_resources::getrs(int M, gpu::vector<P> const &A, gpu::vector<int> const &ipiv,
                              gpu::vector<P> &b) const
{
  expect(M == ipiv.size());
  expect(ipiv.size() * ipiv.size() == A.size());
  expect(ipiv.size() == b.size());

  gpu::vector<int> gpu_info(1);

  if constexpr (is_double<P>) {
    cusolver_check_error( cusolverDnDgetrs(cusolverdn, CUBLAS_OP_N, M, 1, A.data(), M,
                                           ipiv.data(), b.data(), M, gpu_info.data()) );
  } else {
    cusolver_check_error( cusolverDnSgetrs(cusolverdn, CUBLAS_OP_N, M, 1, A.data(), M,
                                           ipiv.data(), b.data(), M, gpu_info.data()) );
  }
}
#endif

#ifdef ASGARD_USE_ROCM
template<typename P>
void compute_resources::getrf(int M, gpu::vector<P> &A, gpu::vector<gpu::direct_int> &ipiv) const {
  expect(static_cast<int64_t>(M) * M == A.size());

  ipiv.resize(M);

  gpu::vector<gpu::direct_int> gpu_info(1);

  if constexpr (is_double<P>) {
    rocblas_check_error( rocsolver_dgetrf(rocblas, M, M, A.data(), M,
                                          ipiv.data(), gpu_info.data()) );
  } else {
    rocblas_check_error( rocsolver_sgetrf(rocblas, M, M, A.data(), M,
                                          ipiv.data(), gpu_info.data()) );
  }

  int info = gpu_info.copy_to_host()[0];

  if (info != 0) {
    std::stringstream sout;
    if (info < 0)
    {
      sout << "rocsolver-getrf(): the diagonal element of the triangular factor of A,\n";
      sout << "U(" << info << ',' << info << ") is zero, so that A is singular;\n";
      sout << "the matrix could not be factorized.\n";
    }
    throw std::runtime_error(sout.str());
  }
}

template<typename P>
void compute_resources::getrs(int M, gpu::vector<P> const &A,
                              gpu::vector<gpu::direct_int> const &ipiv,
                              gpu::vector<P> &b) const
{
  expect(M == ipiv.size());
  expect(ipiv.size() * ipiv.size() == A.size());
  expect(ipiv.size() == b.size());

  gpu::vector<int> gpu_info(1);

  if constexpr (is_double<P>) {
    rocblas_check_error( rocsolver_dgetrs(
        rocblas, rocblas_operation_none, M, 1, const_cast<P*>(A.data()), M,
        ipiv.data(), b.data(), M) );
  } else {
    rocblas_check_error( rocsolver_sgetrs(
        rocblas, rocblas_operation_none, M, 1, const_cast<P*>(A.data()), M,
        ipiv.data(), b.data(), M) );
  }
}
#endif

#ifdef ASGARD_USE_GPU
template void
compute_resources::getrf<double>(int, gpu::vector<double> &A, gpu::vector<int> &ipiv) const;
template void
compute_resources::getrf<float>(int, gpu::vector<float> &A, gpu::vector<int> &ipiv) const;

template void compute_resources::getrs<double>(
    int, gpu::vector<double> const &A, gpu::vector<int> const &ipiv, gpu::vector<double> &b) const;
template void compute_resources::getrs<float>(
    int, gpu::vector<float> const &A, gpu::vector<int> const &ipiv, gpu::vector<float> &b) const;
#endif

template<typename P>
void compute_resources::pttrf(std::vector<P> &diag, std::vector<P> &sub) const
{
  expect(sub.size() + 1 == diag.size());

  int const N = static_cast<int>(diag.size());
  int info = 0;

  if constexpr (is_double<P>)
    dpttrf_(&N, diag.data(), sub.data(), &info);
  else
    spttrf_(&N, diag.data(), sub.data(), &info);

  if (info < 0)
    throw std::runtime_error(std::string("pttrf() argument ") + std::to_string(info)
                             + " has illegal value");
}

template void
compute_resources::pttrf<double>(std::vector<double> &, std::vector<double> &) const;
template void
compute_resources::pttrf<float>(std::vector<float> &, std::vector<float> &) const;

template<typename P>
void compute_resources::pttrs(std::vector<P> const &diag, std::vector<P> const &sub,
                              std::vector<P> &b) const {
  expect(sub.size() + 1 == diag.size());

  int const N = static_cast<int>(diag.size());
  int const nrhs = 1;
  int info = 0;

  if constexpr (is_double<P>)
    dpttrs_(&N, &nrhs, diag.data(), sub.data(), b.data(), &N, &info);
  else
    spttrs_(&N, &nrhs, diag.data(), sub.data(), b.data(), &N, &info);

}

template void
compute_resources::pttrs<double>(std::vector<double> const &, std::vector<double> const &,
                                 std::vector<double> &) const;
template void
compute_resources::pttrs<float>(std::vector<float> const &, std::vector<float> const &,
                                std::vector<float> &) const;


} // namespace asgard
