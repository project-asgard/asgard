#include <iostream>
#include <cassert>
#include <chrono>
#include <unistd.h>

#include "kroncommon.hpp"
#include "kronmult6_xbatched.hpp"
#include "kronmult5_xbatched.hpp"
#include "kronmult4_xbatched.hpp"
#include "kronmult3_xbatched.hpp"
#include "kronmult2_xbatched.hpp"
#include "kronmult1_xbatched.hpp"


#ifdef USE_GPU
#include <cuda_runtime.h>
#else
#include <stdlib.h>
#include <string.h>
#endif


static inline
void host2gpu( void *dest, void *src, size_t nbytes )
{
#ifdef USE_GPU
        cudaError_t istat = cudaMemcpy( dest, 
                                        src, 
                                        nbytes,  
                                        cudaMemcpyHostToDevice );
        assert( istat == cudaSuccess );
#else
        memcpy( dest, src, nbytes );
#endif
}

static inline
void gpu2host( void *dest, void *src, size_t nbytes )
{
#ifdef USE_GPU
        cudaError_t istat = cudaMemcpy( dest,
                                        src,
                                        nbytes,
                                        cudaMemcpyDeviceToHost);
        assert( istat == cudaSuccess );
#else
        memcpy( dest, src, nbytes );
#endif

}

static inline
void *myalloc( size_t nbytes ) {
              void *devPtr = nullptr;
#ifdef USE_GPU
              cudaError_t istat = cudaMalloc( &devPtr, nbytes );
              assert( istat == cudaSuccess );
#else
              devPtr = malloc( nbytes );
#endif
              assert( devPtr != nullptr );
              return(devPtr);
}

static inline
void myfree( void * devPtr ) {
#ifdef USE_GPU
                cudaError_t istat = cudaFree( devPtr);
                assert( istat == cudaSuccess );
#else
                free( devPtr );
#endif
}
     

template<typename T>
T test_kronmult_xbatched(  int const idim,
                          int const n, int const batchCount, 
                          int const idebug = 0, 
                          bool const do_check  = true,
                          bool const use_overlap_in_Y = true )
        
{

	int const lda = n + 3 ;



        // -------------------------
        // Aarray is (lda,n,idim,batchCount)
	// Aparray is (idim,batchCount)
        // Xarray is (n^idim by batchCount)
        // Yarray is (n^idim by batchCount)
        // Zarray is (n^idim by batchCount)
        // Warray is (n^idim by batchCount)
        // ----------------------------

        int const Xsize = std::pow(n,idim);

	size_t const Aarray_nbytes = sizeof(T)*lda*n*idim*batchCount;
	size_t const Aparray_nbytes = sizeof(T*) * idim * batchCount;

        T *Aarray_   = (T *)  malloc( Aarray_nbytes );
        T **Aparray_ = (T **) malloc( Aparray_nbytes );

        T *Xarray_ = (T *) malloc( sizeof(T)*Xsize * batchCount);
        T *Yarray_ = (T *) malloc( sizeof(T)*Xsize * batchCount);
        T *Y2array_ = (T *) malloc( sizeof(T)*Xsize * batchCount);

        T *Zarray_ = (T *) malloc( sizeof(T)*Xsize * batchCount);
        T *Warray_ = (T *) malloc( sizeof(T)*Xsize * batchCount);

        assert( Aarray_ != nullptr );
        assert( Aparray_ != nullptr );

        assert( Xarray_ != nullptr );
        assert( Yarray_ != nullptr );
        assert( Y2array_ != nullptr );


        assert( Zarray_ != nullptr );
        assert( Warray_ != nullptr );

        T *dAarray_   = (T *)  myalloc( Aarray_nbytes );
	T **dAparray_ = (T **) myalloc( Aparray_nbytes );

        T *dXarray_ = (T *) myalloc( sizeof(T)*Xsize * batchCount );
        T *dZarray_ = (T *) myalloc( sizeof(T)*Xsize * batchCount );
        T *dYarray_ = (T *) myalloc( sizeof(T)*Xsize * batchCount );
        T *dWarray_ = (T *) myalloc( sizeof(T)*Xsize * batchCount );

        assert( dAarray_  != nullptr );
        assert( dAparray_ != nullptr );

        assert( dXarray_ != nullptr );
        assert( dYarray_ != nullptr );
        assert( dZarray_ != nullptr );
        assert( dWarray_ != nullptr );

        T** pdXarray_ = (T**) malloc( sizeof(T*) * batchCount );
        T** pdYarray_ = (T**) malloc( sizeof(T*) * batchCount );
        T** pdZarray_ = (T**) malloc( sizeof(T*) * batchCount );
        T** pdWarray_ = (T**) malloc( sizeof(T*) * batchCount );

        T** dpdXarray_ = (T**) myalloc( sizeof(T*) * batchCount );
        T** dpdZarray_ = (T**) myalloc( sizeof(T*) * batchCount );
        T** dpdYarray_ = (T**) myalloc( sizeof(T*) * batchCount );
        T** dpdWarray_ = (T**) myalloc( sizeof(T*) * batchCount );

        assert( dpdXarray_ != nullptr );
        assert( dpdYarray_ != nullptr );
        assert( dpdZarray_ != nullptr );
        assert( dpdWarray_ != nullptr );

        auto dAarray = [&] (int const i, 
                           int const j, 
                           int const k, 
                           int const ibatch ) -> T& {
                return(  dAarray_[ indx4f(i,j,k,ibatch, lda,n,idim) ] );
        };

        auto Aarray = [&] (int const i, 
                           int const j, 
                           int const k, 
                           int const ibatch ) -> T& {
                return(  Aarray_[ indx4f(i,j,k,ibatch, lda,n,idim) ] );
        };

	auto Aparray = [&] (int const i,
			    int const ibatch ) -> T* & {
		return( Aparray_[ indx2f(i,ibatch,idim) ] );
	};

        auto Xarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Xarray_[ indx2f(i,ibatch,Xsize) ] );
        };

        auto Yarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Yarray_[ indx2f(i,ibatch,Xsize) ] );
        };

        auto Y2array = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Y2array_[ indx2f(i,ibatch,Xsize) ] );
        };

        auto Zarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Zarray_[ indx2f(i,ibatch,Xsize) ] );
        };

        auto Warray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Warray_[ indx2f(i,ibatch,Xsize) ] );
        };


        auto dXarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( dXarray_[ indx2f(i,ibatch,Xsize) ] );
        };

        auto dYarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( dYarray_[ indx2f(i,ibatch,Xsize) ] );
        };

        auto dZarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( dZarray_[ indx2f(i,ibatch,Xsize) ] );
        };

        auto dWarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( dWarray_[ indx2f(i,ibatch,Xsize) ] );
        };


        //  ---------------------
        //  initialize the arrays
        //  save a copy of Xarray in Z
        //  ---------------------
        #pragma omp parallel for
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
        for(int i=1; i <= Xsize; i++) {
              T const r1 = (i + (ibatch-1)*Xsize );
              T const r2 = Xsize*batchCount;

              // --------------------------------
              // note Zarray is a copy of Xarray
              // --------------------------------
              Xarray(i,ibatch) = r1/r2;
              Zarray(i,ibatch) = Xarray(i,ibatch);
              Yarray(i,ibatch) = 0;
              Warray(i,ibatch) = 0;
              };
              };
        #pragma omp parallel for 
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
            for(int k=1; k <= idim; k++) {
            for(int j=1; j <= n; j++) {
            for(int i=1; i <= n; i++) {
                T const r1 = i + (j-1)*n + (k-1)*n*n + (ibatch-1)*batchCount;
                T const r2 = n*n*idim*batchCount;
                Aarray(i,j,k,  ibatch) = r1/r2;
            };
            };
            };
        };

        #pragma omp parallel for
	for(int ibatch=1; ibatch <= batchCount; ibatch++) {
	   for(int k=1; k <= idim; k++) {
		Aparray(k,ibatch) = &(dAarray(1,1,k,ibatch));
	   };
	};


        // ---------------------
        // copy from host to GPU
        // interface is host2gpu( dest, src, nbytes )
        // ---------------------
        host2gpu( dAarray_,  Aarray_,  Aarray_nbytes );
        host2gpu( dAparray_, Aparray_, Aparray_nbytes );

        host2gpu( dXarray_, Xarray_, sizeof(T)*Xsize*batchCount );
        host2gpu( dYarray_, Yarray_, sizeof(T)*Xsize*batchCount );
        host2gpu( dZarray_, Zarray_, sizeof(T)*Xsize*batchCount );
        host2gpu( dWarray_, Warray_, sizeof(T)*Xsize*batchCount );

        for(int ibatch=1; ibatch <= batchCount;  ibatch++) {
                pdXarray_[ (ibatch-1) ] = &(dXarray(1,ibatch));
                
                if (use_overlap_in_Y) {
                  pdYarray_[ (ibatch-1) ] = &(dYarray(1,1));
                }
                else {
                  pdYarray_[ (ibatch-1) ] = &(dYarray(1,ibatch));
                };

                pdZarray_[ (ibatch-1) ] = &(dZarray(1,ibatch));
                pdWarray_[ (ibatch-1) ] = &(dWarray(1,ibatch));
        };

        host2gpu( dpdXarray_, pdXarray_, sizeof(T*)*batchCount );
        host2gpu( dpdYarray_, pdYarray_, sizeof(T*)*batchCount );
        host2gpu( dpdZarray_, pdZarray_, sizeof(T*)*batchCount );
        host2gpu( dpdWarray_, pdWarray_, sizeof(T*)*batchCount );



        auto time_start = std::chrono::steady_clock::now();
#ifdef USE_GPU
        {
        int constexpr warpsize = 32;
        int constexpr nwarps = 8;
        int constexpr nthreads = nwarps * warpsize;

        // --------------------------------------------
        // note  the input Zarray will be over-written
        // --------------------------------------------
        switch(idim) { 
        case 1:  kronmult1_xbatched<T><<< batchCount, nthreads >>>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 2:  kronmult2_xbatched<T><<< batchCount, nthreads >>>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 3:  kronmult3_xbatched<T><<< batchCount, nthreads >>>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 4:  kronmult4_xbatched<T><<< batchCount, nthreads >>>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 5:  kronmult5_xbatched<T><<< batchCount, nthreads >>>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 6:  kronmult6_xbatched<T><<< batchCount, nthreads >>>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
         default: 
            assert( false );
        };

        // -------------------------------------------
        // note important to wait for kernel to finish
        // -------------------------------------------
        cudaError_t istat = cudaDeviceSynchronize();
        assert( istat == cudaSuccess );
        }
#else

        {

        // --------------------------------------------
        // note  the input Zarray will be over-written
        // --------------------------------------------
        switch(idim) { 
        case 1:  kronmult1_xbatched<T>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 2:  kronmult2_xbatched<T>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 3:  kronmult3_xbatched<T>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 4:  kronmult4_xbatched<T>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 5:  kronmult5_xbatched<T>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
        case 6:  kronmult6_xbatched<T>( n,
                           dAparray_, lda,
                           dpdZarray_,
                           dpdYarray_,
                           dpdWarray_,
                           batchCount );
            break;
         default: 
            assert( false );
        };

     }




#endif
        auto time_end = std::chrono::steady_clock::now();
        auto elapsed_time_us = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count();
        auto elapsed_time_sec = elapsed_time_us * 0.001 * 0.001;

        // ------------------------------------------
        // copy from gpu to host
        // interface is gpu2host( dest, src, nbytes )
        // ------------------------------------------
        gpu2host( Yarray_, dYarray_,  sizeof(T)*Xsize*batchCount);



        {
          double const giga = 1000.0*1000.0*1000.0;
          double const flops = 12.0*(std::pow(n,(idim+1))) * batchCount;
          double const gflops = flops/giga;
          double const gflops_per_sec = gflops  /elapsed_time_sec;
          if (flops > giga) {
                  std::cout << " idim = " << idim
                            << " n = " << n 
                            << " batchCount = " << batchCount
                            << " elapsed_time = " << elapsed_time_sec << " seconds "
                            << " Gflops/sec = " << gflops_per_sec
                            << "\n";
          };
        };


   T max_abserr = 0;
   if (do_check) {
        // -------------
        // check results
        // -------------

        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
                T const * const A1_ = &(Aarray(1,1,1,ibatch));
                T const * const A2_ = &(Aarray(1,1,2,ibatch));
                T const * const A3_ = &(Aarray(1,1,3,ibatch));
                T const * const A4_ = &(Aarray(1,1,4,ibatch));
                T const * const A5_ = &(Aarray(1,1,5,ibatch));
                T const * const A6_ = &(Aarray(1,1,6,ibatch));
                T const * const X_ = &(Xarray(1,ibatch));

                auto X = [&] (int const i) -> T const & {
                        return( X_[ (i)-1 ]);
                };

                auto A1 = [&](int const i,
                              int const j) -> T const & {
                        return( A1_[ indx2f(i,j,lda) ] );
                };

                auto A2 = [&](int const i,
                              int const j) -> T const & {
                        return( A2_[ indx2f(i,j,lda) ] );
                };

                auto A3 = [&](int const i,
                              int const j) -> T const & {
                        return( A3_[ indx2f(i,j,lda) ] );
                };

                auto A4 = [&](int const i,
                              int const j) -> T const & {
                        return( A4_[ indx2f(i,j,lda) ] );
                };

                auto A5 = [&](int const i,
                              int const j) -> T const & {
                        return( A5_[ indx2f(i,j,lda) ] );
                };

                auto A6 = [&](int const i,
                              int const j) -> T const & {
                        return( A6_[ indx2f(i,j,lda) ] );
                };


                int const max_i1 = (idim >= 1) ? n : 1;
                int const max_i2 = (idim >= 2) ? n : 1;
                int const max_i3 = (idim >= 3) ? n : 1;
                int const max_i4 = (idim >= 4) ? n : 1;
                int const max_i5 = (idim >= 5) ? n : 1;
                int const max_i6 = (idim >= 6) ? n : 1;

                int const max_j1 = (idim >= 1) ? n : 1;
                int const max_j2 = (idim >= 2) ? n : 1;
                int const max_j3 = (idim >= 3) ? n : 1;
                int const max_j4 = (idim >= 4) ? n : 1;
                int const max_j5 = (idim >= 5) ? n : 1;
                int const max_j6 = (idim >= 6) ? n : 1;

                #pragma omp parallel for collapse(6)  reduction(max:max_abserr)
                for(int i1=1; i1 <= max_i1; i1++) 
                for(int i2=1; i2 <= max_i2; i2++) 
                for(int i3=1; i3 <= max_i3; i3++) 
                for(int i4=1; i4 <= max_i4; i4++) 
                for(int i5=1; i5 <= max_i5; i5++) 
                for(int i6=1; i6 <= max_i6; i6++) {

                   int const ic = 1+indx6f( i6,i5,i4,i3,i2,i1,
                                            max_i6, max_i5, max_i4, 
                                            max_i3, max_i2 );
                   T Y_ic = 0;


                   for(int j1=1; j1 <= max_j1; j1++) {
                   for(int j2=1; j2 <= max_j2; j2++) {
                   for(int j3=1; j3 <= max_j3; j3++) {
                   for(int j4=1; j4 <= max_j4; j4++) {
                   for(int j5=1; j5 <= max_j5; j5++) {
                   for(int j6=1; j6 <= max_j6; j6++) {

                      // -------------------------------
                      // note last index i6 goes fastest
                      // -------------------------------
                      int const jc = 1+indx6f( j6,j5,j4,j3,j2,j1,
                                               max_j6, max_j5, max_j4,
                                               max_j3, max_j2 );


                      T C_ic_jc =  1;
                      C_ic_jc *= (idim >= 1) ? A1(i1,j1) : 1;
                      C_ic_jc *= (idim >= 2) ? A2(i2,j2) : 1;
                      C_ic_jc *= (idim >= 3) ? A3(i3,j3) : 1;
                      C_ic_jc *= (idim >= 4) ? A4(i4,j4) : 1;
                      C_ic_jc *= (idim >= 5) ? A5(i5,j5) : 1;
                      C_ic_jc *= (idim >= 6) ? A6(i6,j6) : 1;




                      T const X_jc = X(jc);

                      Y_ic += C_ic_jc * X_jc;
                   };
                   };
                   };
                   };
                   };
                   };

                   Y2array(ic,ibatch) = Y_ic;
                                    

                
                
                
                
                
                };
          }; // end for ibatch

                int const max_ic = std::pow( n, idim );
                for(int ic=1; ic <= max_ic; ic++) { 
                   T Y_ic = 0;
                   T Yval = 0;
                   T abs_err = 0;

                   if (use_overlap_in_Y) {
                        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
                                Yval += Y2array(ic,ibatch);
                        };
                        abs_err = std::abs( Yval - Yarray(ic,1) );
                   }
                   else {
                       for(int ibatch=1; ibatch <= batchCount; ibatch++) {
                               Yval = Y2array(ic,ibatch);
                               Y_ic  = Yarray(ic,ibatch);
                               abs_err = std::abs(Yval - Y_ic);
                       };
                   };
                   max_abserr = std::max( max_abserr,abs_err);



                   if (idebug >= 1) {
                       T const tol = 1.0/(1000.0 * 1000.0);
                       if (abs_err > tol ) {
                             std::cout  << " idim = " << idim
                                        << " ic = " << ic 
                                        << " Y_ic = " << Y_ic
                                        << " Yval =  " << Yval
                                        << " abs_err = " << abs_err << "\n";
                       };
                     };

                   }; // end for ic


      };



        // -------
        // cleanup
        // -------

        myfree( dAarray_ ); dAarray_ = nullptr;
        myfree( dAparray_ ); dAparray_ = nullptr;

        myfree( dXarray_ ); dXarray_ = nullptr;
        myfree( dYarray_ ); dYarray_ = nullptr;
        myfree( dZarray_ ); dZarray_ = nullptr;
        myfree( dWarray_ ); dWarray_ = nullptr;

        free( Aarray_ ); Aarray_ = nullptr;
        free( Aparray_ ); Aparray_ = nullptr;

        free( Xarray_ ); Xarray_ = nullptr;
        free( Yarray_ ); Yarray_ = nullptr;
        if (use_overlap_in_Y) {
          free( Y2array_ ); Y2array_ = nullptr;
        };

        free( Zarray_ ); Zarray_ = nullptr;
        free( Warray_ ); Warray_ = nullptr;

        return(max_abserr);

}


                      
int main() {

        int const idebug = 0;

        int batch_table[] = {1,16,128};
        int const size_batch_table = sizeof(batch_table)/sizeof(batch_table[0]);

        int n_table[] = {1, 2, 3, 4 };
        int const size_n_table = sizeof(n_table)/sizeof(n_table[0]);


        int nerrors = 0;

        for (int idim =1; idim <= 6; idim++) {
        for (int ibatch_table=0; ibatch_table < size_batch_table; ibatch_table++) {
        for (int in_table = 0;  in_table < size_n_table; in_table++) {
                int const n = n_table[in_table];
                int const batchCount = batch_table[ibatch_table];

                double const max_abserr =  test_kronmult_xbatched<double>( idim, n, batchCount, idebug );
                double const tol = 1.0/(1000.0 * 1000.0);
                bool const isok = (max_abserr <= tol);
                if (!isok) {
                        nerrors += 1;
                };

                if ((idebug >= 1) || (!isok)) {
                        std::cout << " idim = "  << idim
                                  << " n = " << n 
                                  << " batchCount = " << batchCount
                                  << " max_abserr= " << max_abserr << "\n";
                };
        };
        };
        };


        if (nerrors == 0) {
                std::cout << "ALL PASSED" << "\n";
        }
        else {
                std::cout << "There are " << nerrors << " errors" << "\n";
        };

        if (nerrors == 0) {
               // ---------------------
               // try performance test
               // ---------------------
               int const batchCount = 256;
               bool const do_check = 0;
               int const idebug = 0;
               int const idim = 6;


               for(int n=7; n <= 8; n++) {
                test_kronmult_xbatched<double>(idim,n, batchCount, idebug, do_check );
               };
        };




  return(0);
}


                     


