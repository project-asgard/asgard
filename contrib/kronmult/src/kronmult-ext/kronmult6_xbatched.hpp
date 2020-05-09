#ifndef KRONMULT6_XBATCHED_HPP
#define KRONMULT6_XBATCHED_HPP 1

#include "kroncommon.hpp"

#include "kronmult6.hpp"



// --------------------------------------------------------------------
// Performs  Y(:,k) += kron(A1(k),...,A6(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult6_xbatched(
                       int const n,
                       T const * const Aarray_[],
		       int const lda,
                       T* pX_[],
                       T* pY_[],
                       T* pW_[],
                       int const batchCount)
//
// conceptual shape of Aarray is  (ndim,batchCount)
//
// pX_[] is array of pointers to X[], each of size n^6
// pY_[] is array of pointers to Y[], each of size n^6
// pW_[] is array of pointers to Z[], each of size n^6
//
// Y is the output
// X is the input (but may be modified)
// W is workspace
//
//
{
#ifdef USE_GPU
        // -------------------------------------------
        // note 1-based matlab convention for indexing
        // -------------------------------------------
        int const iz_start = blockIdx.x + 1;
        int const iz_size =  gridDim.x;
        assert( gridDim.y == 1);
        assert( gridDim.z == 1);
#else
        int const iz_start = 1;
        int const iz_size = 1;
#endif

	int const ndim = 6;

        auto Aarray = [&] (int const i1,
                           int const i2
                           ) -> T const * {
                return( Aarray_[ indx2f(i1,i2,ndim ) ] );
        };



#ifndef USE_GPU
#pragma omp parallel for
#endif
        for(int ibatch=iz_start; ibatch <= batchCount; ibatch += iz_size) {
                T* const Xp =  pX_[ (ibatch-1) ];
                T* const Yp =  pY_[ (ibatch-1) ];
                T* const Wp =  pW_[ (ibatch-1) ];

                T const * const A1 = (Aarray(1,ibatch));
                T const * const A2 = (Aarray(2,ibatch));
                T const * const A3 = (Aarray(3,ibatch));
                T const * const A4 = (Aarray(4,ibatch));
                T const * const A5 = (Aarray(5,ibatch));
                T const * const A6 = (Aarray(6,ibatch));
                int const nvec = 1;
                kronmult6( n, nvec, A1,A2,A3,A4,A5,A6, Xp, Yp, Wp, lda);
        };

}



#endif
