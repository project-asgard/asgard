#include "kronmult1_batched.hpp"

void kronmult1_batched(
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

        kronmult1_batched<double><<< batchCount, nthreads >>>( n, 
           Aarray_, Xarray_, Yarray_, Warray_, batchCount);
#else
        kronmult1_batched<double>( n, 
           Aarray_, Xarray_, Yarray_, Warray_, batchCount);
#endif

}


