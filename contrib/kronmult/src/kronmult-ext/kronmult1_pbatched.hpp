#ifndef KRONMULT1_PBATCHED_HPP
#define KRONMULT1_PBATCHED_HPP 1

#include "kroncommon.hpp"

#include "kronmult1.hpp"



// --------------------------------------------------------------------
// Performs  Y(:,k) = kron(A1(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult1_pbatched(
                       int const n,
                       T const Aarray_[],
                       T* pX_[],
                       T* pY_[],
                       T* pW_[],
                       int const batchCount)
//
// conceptual shape of Aarray is  (n,n,1,batchCount)
// pX_[] is array of pointers, each X_ is n^1
// pY_[] is array of pointers, each Y_ is n^1
// pW_[] is array of pointers, each W_ is n^1
// Y_ is output
// X_ is input but can be modified
// W_ is work space
{
#ifdef USE_GPU
        // -------------------------------------------
        // note 1-based matlab convention for indexing
        // -------------------------------------------
        int const iz_start = blockIdx.x + 1;
        int const iz_size =  gridDim.x;
        assert( gridDim.y == 1 );
        assert( gridDim.z == 1 );
#else
        int const iz_start = 1;
        int const iz_size = 1;
#endif



        auto Aarray = [&] (int const i1,
                           int const i2,
                           int const i3,
                           int const i4) -> T const & {
                return( Aarray_[ indx4f(i1,i2,i3,i4, n,n,1 ) ] );
        };

        for(int ibatch=iz_start; ibatch <= batchCount; ibatch += iz_size) {
                T* const Xp = pX_[ (ibatch-1) ];
                T* const Yp = pY_[ (ibatch-1) ];
                T* const Wp = pW_[ (ibatch-1) ];

                T const * const A1 = &(Aarray(1,1,1,ibatch));
                int const nvec = 1;
                kronmult1( n, nvec, A1, Xp, Yp, Wp );
        };

}



#endif
