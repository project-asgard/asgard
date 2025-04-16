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
}

namespace asgard
{

#ifdef ASGARD_USE_CUDA
namespace gpu
{
std::string error_message(cudaError_t err) {
  return std::string("CUDA reported an error: '") + cudaGetErrorString(err) + std::string("'");
}

} // namespace gpu
#endif

compute_resources::compute_resources() {
  // TODO: GPU logic goes here
}

// LAPACK factorize and solve for a general matrix, used by the direct solver
template<typename P>
void compute_resources::getrf(int M, std::vector<P> &A, std::vector<int> &ipiv)
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
compute_resources::getrf<double>(int, std::vector<double> &A, std::vector<int> &ipiv);
template void
compute_resources::getrf<float>(int, std::vector<float> &A, std::vector<int> &ipiv);

template<typename P>
void compute_resources::getrs(int M, std::vector<P> const &A, std::vector<int> const &ipiv,
                              std::vector<P> &b)
{
  expect(static_cast<size_t>(M) == ipiv.size());
  expect(ipiv.size() * ipiv.size() == A.size());
  expect(ipiv.size() == b.size());

  int info  = 0;
  int const one = 1;
  char const trans = 'N';

  if constexpr (is_double<P>) {
    dgetrs_(&trans, &M, &one, A.data(), &M, ipiv.data(), b.data(), &M, &info);
  } else {
    sgetrs_(&trans, &M, &one, A.data(), &M, ipiv.data(), b.data(), &M, &info);
  }

  // only check if arguments have illegal value
  expect(info == 0);
}

template void compute_resources::getrs<double>(
    int, std::vector<double> const &A, std::vector<int> const &ipiv, std::vector<double> &b);
template void compute_resources::getrs<float>(
    int, std::vector<float> const &A, std::vector<int> const &ipiv, std::vector<float> &b);

} // namespace asgard
