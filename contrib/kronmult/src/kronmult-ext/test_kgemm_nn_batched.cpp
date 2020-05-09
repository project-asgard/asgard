#include <iostream>
#include <cassert>
#include <chrono>
#include <unistd.h>

#ifdef USE_GPU
#include <cuda_runtime.h>
#else
#include <stdlib.h>
#include <string.h>
#endif

#include "kroncommon.hpp"
#include "kgemm_nn_batched.hpp"

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
void gpu2host( void * dest, void * src, size_t nbytes )
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
void *myalloc( size_t const nbytes ) {
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
T test_kgemm_nn_batched( int const mm,
                              int const nn,
                              int const kk,
                              int const batchCount,
                              int const idebug = 1
                              )
{
        // ----------------------------------------
        // simple program to test kgemm_nn_batched
        // ----------------------------------------

        T const alpha = 1.3;
        T const beta = 1.2;


        int const nrowA = mm; 
        int const ncolA = kk; 

        int const nrowB = kk; 
        int const ncolB = nn; 

        int const nrowC = mm; 
        int const ncolC = nn; 



        int ldA = nrowA;
        int ldB = nrowB;
        int ldC = nrowC;

        bool const need_align = false;
        if (need_align) {
           int const wsize = 32;
           ldA = wsize * (( nrowA + (wsize-1))/wsize );
           ldB = wsize * (( nrowB + (wsize-1))/wsize );
           ldC = wsize * (( nrowC + (wsize-1))/wsize );
        };

        T *Aarray_[batchCount];
        T *Barray_[batchCount];
        T *Carray_[batchCount];

        // -------------
        // device arrays
        // -------------
        T *hdAarray_[batchCount];
        T *hdBarray_[batchCount];
        T *hdCarray_[batchCount];

        T **ddAarray_ = nullptr;
        T **ddBarray_ = nullptr;
        T **ddCarray_ = nullptr;

        {
                size_t nbytes = sizeof(T *) * batchCount;
                ddAarray_ = (T **) myalloc( nbytes );
                ddBarray_ = (T **) myalloc( nbytes );
                ddCarray_ = (T **) myalloc( nbytes );

        };

        

        // ----------------
        // initialize array
        // ----------------
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                T * const A_ = new T[ldA*ncolA];
                T * const B_ = new T[ldB*ncolB];
                T * const C_ = new T[ldC*ncolC];

                assert( A_ != nullptr);
                assert( B_ != nullptr);
                assert( C_ != nullptr);

                Aarray_[ibatch] = A_;
                Barray_[ibatch] = B_;
                Carray_[ibatch] = C_;

        };






        for(int ibatch=0; ibatch < batchCount; ibatch++) {
             T *A_ = Aarray_[ibatch];
             T *B_ = Barray_[ibatch];
             T *C_ = Carray_[ibatch];




             auto A = [&](int const i,
                          int const j) -> T& {
                     return( A_[ indx2f(i,j,ldA) ] );
             };
     
             auto B = [&](int const i,
                          int const j) -> T& {
                     return( B_[ indx2f(i,j,ldB) ] );
             };
     
             auto C = [&](int const i,
                          int const j) -> T& {
                     return( C_[ indx2f(i,j,ldC) ] );
             };

             for(int j=1; j <= ncolA; j++) {
             for(int i=1; i <= nrowA; i++) {
                A(i,j) = 1.0 + i + j*nrowA + ibatch;
             };
             };


             for(int j=1; j <= ncolB; j++) {
             for(int i=1; i <= nrowB; i++) {
                B(i,j) = 1.0 /(1.0 + i + j*nrowB + ibatch);
             };
             };

             for(int j=1; j <= ncolC; j++) {
             for(int i=1; i <= nrowC; i++) {
                C(i,j) = 1;
             };
             };
        };

        // --------------------------
        // allocate storage on device
        // --------------------------
        for(int ibatch=0; ibatch < batchCount; ibatch++) {

                size_t const nbytes_A = sizeof(T)*ldA*ncolA;
                T *dA = (T *) myalloc( nbytes_A );

                size_t const nbytes_B = sizeof(T)*ldB*ncolB;
                T *dB = (T *) myalloc( nbytes_B );

                size_t const nbytes_C = sizeof(T)*ldC*ncolC;
                T *dC = (T *) myalloc( nbytes_C );;

                assert( dA != nullptr );
                assert( dB != nullptr );
                assert( dC != nullptr );

                hdAarray_[ibatch] = dA;
                hdBarray_[ibatch] = dB;
                hdCarray_[ibatch] = dC;
        };

        // ----------------------
        // copy matices to device
        // ----------------------

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                {
                size_t const nbytes = sizeof(T)*ldA*ncolA;
                void * dest = hdAarray_[ibatch];
                void * src =  Aarray_[ibatch];
                host2gpu( dest, src, nbytes );
                };

                {
                size_t nbytes = sizeof(T)*ldB*ncolB;
                void * dest = hdBarray_[ibatch];
                void * src =  Barray_[ibatch];
                host2gpu( dest, src, nbytes );
                };

                {
                size_t nbytes = sizeof(T)*ldC*ncolC;
                void * dest = hdCarray_[ibatch];
                void * src =  Carray_[ibatch];
                host2gpu( dest,src,nbytes );
                };
        };

        // -----------------------
        // copy pointers to device
        // -----------------------
        {
                size_t nbytes = sizeof( T *) * batchCount;
                void *dest = ddAarray_;
                void *src = &(hdAarray_[0]);
                host2gpu( dest, src, nbytes );
        }
        {
                size_t nbytes = sizeof( T *) * batchCount;
                void *dest = ddBarray_;
                void *src = &(hdBarray_[0]);
                host2gpu( dest, src, nbytes );
        }

        {
                size_t nbytes = sizeof( T *) * batchCount;
                void *dest = ddCarray_;
                void *src = &(hdCarray_[0]);
                host2gpu( dest, src, nbytes );
        }


        // --------------------------
        // setup ldA, ldB, ldC arrays
        // --------------------------
        int ldAarray_[batchCount];
        int ldBarray_[batchCount];
        int ldCarray_[batchCount];
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                ldAarray_[ibatch] = ldA;
                ldBarray_[ibatch] = ldB;
                ldCarray_[ibatch] = ldC;
        };

        int *dldAarray_ = nullptr;
        int *dldBarray_ = nullptr;
        int *dldCarray_ = nullptr;

        {
         size_t nbytes = sizeof(int) * batchCount;
         dldAarray_ = (int *) myalloc( nbytes );
         dldBarray_ = (int *) myalloc( nbytes );
         dldCarray_ = (int *) myalloc( nbytes );
        }

        // -------------------------------------------
        // copy array  ldAarray_, ldBarray_, ldCarray_
        // -------------------------------------------

        {
                size_t nbytes = sizeof(int) * batchCount;
                void *src = &(ldAarray_[0]);
                void *dest = dldAarray_;
                host2gpu( dest, src, nbytes );
        }

        {
                size_t nbytes = sizeof(int) * batchCount;
                void *src = &(ldBarray_[0]);
                void *dest = dldBarray_;
                host2gpu( dest, src, nbytes );
        }

        {
                size_t nbytes = sizeof(int) * batchCount;
                void *src = &(ldCarray_[0]);
                void *dest = dldCarray_;
                host2gpu( dest, src, nbytes );
        }

        auto time_start = std::chrono::steady_clock::now();

#ifdef USE_GPU
        {


        int constexpr warpsize = 32;
        int constexpr nwarps = 8;
        int constexpr nthreads = nwarps * warpsize;

        cudaError_t istat_sync_start = cudaDeviceSynchronize();
        assert( istat_sync_start == cudaSuccess );


        kgemm_nn_batched<T><<< batchCount, nthreads >>>( mm,nn,kk, 
                          alpha,
                          ddAarray_, dldAarray_,
                          ddBarray_, dldBarray_,
                          beta, 
                          ddCarray_, dldCarray_,
                          batchCount);

        cudaError_t istat_sync_end = cudaDeviceSynchronize();
        assert( istat_sync_end == cudaSuccess );
        }
#else
        {
        kgemm_nn_batched<T>( mm,nn,kk, 
                          alpha,
                          ddAarray_, dldAarray_,
                          ddBarray_, dldBarray_,
                          beta, 
                          ddCarray_, dldCarray_,
                          batchCount);
        }
#endif

        auto time_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(time_end- time_start).count();

        T elapsed_time_in_sec = elapsed_time * 0.001 * 0.001;
        T flops = (2.0*mm*nn)*kk*batchCount;
        T gflops_per_sec = flops/(1000.0*1000.0*1000.0) / elapsed_time_in_sec;

        if (idebug >= 1) {
          if (elapsed_time_in_sec > 0) {
            std::cout << " mm = " << mm  
                      << " nn = " << nn 
                      << " kk = " << kk 
                      << " batchCount = " << batchCount 
                      << "\n";
            std::cout << " elapsed time is " << elapsed_time_in_sec << " seconds " 
                    << gflops_per_sec << " Gflops/s" << "\n";
          };
          };




        // -------------
        // check results
        // -------------
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t const nbytes = sizeof(T) * ldC*ncolC;
                void * const dest = Carray_[ibatch];
                void * const src = hdCarray_[ibatch];
                gpu2host( dest, src, nbytes );
        };


        T max_abserr = 0;
        #pragma omp parallel for reduction(max:max_abserr)
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
              T const * const A_ = Aarray_[ibatch];
              T const * const B_ = Barray_[ibatch];
              T const * const C_ = Carray_[ibatch];




               auto A = [&](int const i,
                            int const j) -> T const & {
                       return( A_[ indx2f(i,j,ldA) ] );
               };
       
               auto B = [&](int const i,
                            int const j) -> T const & {
                       return( B_[ indx2f(i,j,ldB) ] );
               };
       
               auto C = [&](int const i,
                            int const j) -> T const & {
                       return( C_[ indx2f(i,j,ldC) ] );
               };

              T const cij0 = 1;

              for(int j=1; j <= nn; j++) {
              for(int i=1; i <= mm; i++) {
                      T cij = 0;
                      for(int k=1; k <= kk; k++) {
                             // ------------
                             // Note C = A*B
                             // ------------
                             cij += A(i,k) * B(k,j);
                      };
                      cij = alpha * cij + beta * cij0;

                      T const abserr = std::abs( cij  - C(i,j) );
                      max_abserr = std::max( max_abserr, abserr );
              };
              };
        }; 

        if (idebug >= 2) {
          std::cout << " batchCount = " << batchCount << "\n";
          std::cout << "max_abserr = " << max_abserr << "\n";
        };

        // --------
        // clean up
        // --------

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                delete [] Aarray_[ibatch];
                delete [] Barray_[ibatch];
                delete [] Carray_[ibatch];
        };

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                        myfree( hdAarray_[ibatch] );
                        myfree( hdBarray_[ibatch] );
                        myfree( hdCarray_[ibatch] );
             };


        {
                myfree( ddAarray_ );
                myfree( ddBarray_ );
                myfree( ddCarray_ );

                myfree( dldAarray_ );
                myfree( dldBarray_ );
                myfree( dldCarray_ );

        }





        return( max_abserr);
}



int main()
{
        int const idebug = 0;
        int const inc = 7;
        int const kk_max = 65;
        int const mm_max = 65;
        int const nn_max = 65;
        int const batchCount_max = 2*inc + 1;
        double const tol = 1.0/(1000.0*1000.0);

        int nerrors = 0;
        for(int batchCount=1; batchCount <= batchCount_max; batchCount += inc) {
        for(int kk=1; kk <= kk_max; kk += inc) {
        for(int nn=1; nn <= nn_max; nn += inc) {
        for(int mm=1; mm <= mm_max; mm += inc) {
                double const max_abserr = test_kgemm_nn_batched<double>(mm,nn,kk,batchCount,idebug);
                double const isok = (max_abserr < tol);

                if (!isok) {
                        nerrors += 1;
                };

                if ((!isok) || (idebug >= 2)) {
                        std::cout << " mm = " << mm 
                                  << " nn = " << nn
                                  << " kk = " << kk
                                  << " batchCount = " << batchCount 
                                  << " max_abserr = " << max_abserr
                                  << "\n";
                };
        };
        };
        };
        };

        if (nerrors == 0) {
                std::cout << "ALL PASSED" << "\n";
        }
        else {
                std::cout << "There are " << nerrors << " errors " << "\n";
        };

        //  -----------------
        //  performance tests
        //  -----------------
        if (nerrors == 0) {
                const int idebug = 1;
                const int batchCount = 2*64;
                for(int n=1; n <= 10; n++) {
                  int const n2 = n*n;
                  int const n5 = n2*n2*n;
                  int const mm = n;
                  int const nn = n5;
                  int const kk = n;
                  test_kgemm_nn_batched<double>(mm,nn,kk,batchCount,idebug);
                  };
        };



        return(0);
}
