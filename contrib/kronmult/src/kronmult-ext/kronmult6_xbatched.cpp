#include "kronmult6_xbatched.hpp"

void kronmult6_xbatched(
                       int const n,
                       double const * const Aarray_[],
		       int const lda,
                       double* Xarray_[],
                       double* Yarray_[],
                       double* Warray_[],
                       int const batchCount )
{
#ifdef USE_GPU
        int constexpr warpsize = 32;
        int constexpr nwarps = 8;
        int constexpr nthreads = nwarps * warpsize;

        kronmult6_xbatched<double><<< batchCount, nthreads >>>( n, 
           Aarray_, lda,
	   Xarray_, Yarray_, Warray_, batchCount);
#else
        kronmult6_xbatched<double>( n, 
           Aarray_, lda,
	   Xarray_, Yarray_, Warray_, batchCount);
#endif

}


