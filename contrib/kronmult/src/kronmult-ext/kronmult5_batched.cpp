#include "kronmult5_batched.hpp"

void kronmult5_batched(
                       int const n,
                       double const Aarray_[],
                       double Xarray_[],
                       double Yarray_[],
                       double Warray_[],
                       int const batchCount )
{
#ifdef USE_GPU
        int constexpr warpsize = 32;
        int constexpr nwarps = 8;
        int constexpr nthreads = nwarps * warpsize;

        kronmult5_batched<double><<< batchCount, nthreads >>>( n, 
           Aarray_, Xarray_, Yarray_, Warray_, batchCount);
#else
        kronmult5_batched<double>( n, 
           Aarray_, Xarray_, Yarray_, Warray_, batchCount);
#endif

}


