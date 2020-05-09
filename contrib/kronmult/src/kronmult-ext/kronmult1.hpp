#ifndef KRONMULT1_HPP
#define  KRONMULT1_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nn.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1)*X as
//  which is just matrix multiply  Y = A1 * X
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION
void kronmult1( int const n, 
                int const nvec,
                T   const A1_[],
                T   X_[],
                T   Y_[],
                T   W_[],
	        int const lda_in = 0 )
// -----------------
// note A1 is n by n
//      X is (n by nvec)
// -----------------
{

    // used to suppress warnings in unused variables
    auto const ignore = [](T* ignored) { (void)ignored; };
    ignore(W_);

    int const lda = (lda_in == 0) ? n : lda_in;
    int const mm = n;
    int const nn = nvec;
    int const kk = n;
    T const * const Ap = &(A1_[0]);
    T const * const Bp = &(X_[0]);
    T       * const Cp = &(Y_[0]);
    int const ld1 = lda;
    int const ld2 = n;
    int const ld3 = n;
    T const alpha = 1;
    T const beta = 1;

    kgemm_nn( mm,nn,kk,
              alpha,  Ap, ld1,
                      Bp, ld2,
              beta,   Cp, ld3 );

}



#endif
