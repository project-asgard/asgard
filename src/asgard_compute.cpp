#include "asgard_compute.hpp"

// BLAS methods, using the common Fortran API
extern "C" {
  // general PLU factorize
  void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
  void sgetrf_(int *m, int *n, float *A, int *lda, int *ipiv, int *info);
}

namespace asgard
{

template<typename P>
void compute_resources::getrf(int M, std::vector<P> &A, std::vector<int> &ipiv)
{
  expect(static_cast<size_t>(M) * static_cast<size_t>(M) == A.size());

  ipiv.resize(M);

}


} // namespace asgard
