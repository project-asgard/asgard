#ifndef KRONMULT6_BATCHED_HPP
#define KRONMULT6_BATCHED_HPP 1

#include "kroncommon.hpp"

#include "kronmult6.hpp"



// --------------------------------------------------------------------
// Performs  Y(:,k) = kron(A1(k),...,A6(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult6_batched(
                       int const n,
                       T const Aarray_[],
                       T X_[],
                       T Y_[],
                       T W_[],
                       int const batchCount)
//
// conceptual shape of Aarray is  (n,n,6,batchCount)
// X_ is (n^6, batchCount)
// Y_ is (n^6, batchCount)
// W_ is (n^6, batchCount)
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

        int const n2 = n*n;
        int const n4 = n2*n2;
        int const n6 = n2*n4;



        auto X = [&] (int const i,
                      int const j) -> T& {
                return(  X_[ indx2f(i,j,n6) ] );
        };

        auto Y = [&] (int const i,
                      int const j) -> T& {
                return(  Y_[ indx2f(i,j,n6) ] );
        };

        auto W = [&] (int const i,
                      int const j) -> T& {
                return(  W_[ indx2f(i,j,n6) ] );
        };

        auto Aarray = [&] (int const i1,
                           int const i2,
                           int const i3,
                           int const i4) -> T const & {
                return( Aarray_[ indx4f(i1,i2,i3,i4, n,n,6 ) ] );
        };



#ifndef USE_GPU
#pragma omp parallel for
#endif
        for(int ibatch=iz_start; ibatch <= batchCount; ibatch += iz_size) {
                T* const Xp = &( X(1,ibatch) );
                T* const Yp = &( Y(1,ibatch) );
                T* const Wp = &( W(1,ibatch) );
                T const * const A1 = &(Aarray(1,1,1,ibatch));
                T const * const A2 = &(Aarray(1,1,2,ibatch));
                T const * const A3 = &(Aarray(1,1,3,ibatch));
                T const * const A4 = &(Aarray(1,1,4,ibatch));
                T const * const A5 = &(Aarray(1,1,5,ibatch));
                T const * const A6 = &(Aarray(1,1,6,ibatch));
                int const nvec = 1;
                kronmult6( n, nvec, A1,A2,A3,A4,A5,A6, Xp, Yp, Wp );
        };

}



#endif
