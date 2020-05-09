#ifndef KGEMM_NN_BATCHED_H
#define KGEMM_NN_BATCHED_H 1

#include "kroncommon.hpp"
#include "kgemm_nn.hpp"


template<typename T>
GLOBAL_FUNCTION
void kgemm_nn_batched( int const mm, int const nn, int const kk, 
                       T const alpha, 
                       T* const Aarray_[], 
                       int const ldAarray_[], 
                       T* const Barray_[], 
                       int const ldBarray_[], 
                       T const beta,  
                       T* const Carray_[], 
                       int const ldCarray_[], 
                       int const batchCount)
{
// ----------------------------
// use Fortran 1-based indexing
// ----------------------------


        auto Aarray = [&] (int const i) -> T* const & {
                return(  Aarray_[ (i) - 1] );
        };

        auto Barray = [&] (int const i) -> T* const & {
                return(  Barray_[ (i) - 1] );
        };

        auto Carray = [&] (int const i) -> T* const & {
                return(  Carray_[ (i) - 1] );
        };

        auto ldAarray = [&] (int const i) -> int const & {
                return( ldAarray_[ (i) - 1] );
        };

        auto ldBarray = [&] (int const i) -> int const & {
                return( ldBarray_[ (i) - 1] );
        };

        auto ldCarray = [&] (int const i) -> int const & {
                return( ldCarray_[ (i) - 1] );
        };



#ifdef USE_GPU
        int const iz_start = blockIdx.x + 1;
        int const iz_size =  gridDim.x;

        assert( gridDim.y == 1);
        assert( gridDim.z == 1);
#else
        int const iz_start = 1;
        int const iz_size = 1;
#endif

        for(int ibatch=iz_start; ibatch <= batchCount; ibatch += iz_size) {
                T const * const A_ = Aarray(ibatch);
                T const * const B_ = Barray(ibatch);
                T*        const C_ = Carray(ibatch);
                int const ldA = ldAarray(ibatch);
                int const ldB = ldBarray(ibatch);
                int const ldC = ldCarray(ibatch);

                kgemm_nn( mm,nn,kk,  alpha, A_, ldA, B_, ldB, 
                                     beta,  C_, ldC );
        };
}




#endif
