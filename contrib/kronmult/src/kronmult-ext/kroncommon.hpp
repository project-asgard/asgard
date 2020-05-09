#ifndef KRONCOMMON_HPP
#define KRONCOMMON_HPP 1



#ifdef USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#define GLOBAL_FUNCTION  __global__ 
#define SYNCTHREADS __syncthreads()
#define SHARED_MEMORY __shared__
#define DEVICE_FUNCTION __device__
#define HOST_FUNCTION __host__
#else
#define GLOBAL_FUNCTION
#define SYNCTHREADS 
#define SHARED_MEMORY 
#define DEVICE_FUNCTION
#define HOST_FUNCTION
#endif

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

#ifndef USE_GPU

static inline
double atomicAdd(double volatile *p, double dvalue)
{
        double oldvalue = 0;
        #pragma omp atomic capture
        {
        oldvalue = (*p);
        (*p) += dvalue;
        }
        return(oldvalue);
}

static inline
float atomicAdd( float volatile *p, float dvalue)
{
        float oldvalue = 0;
        #pragma omp atomic capture
        {
        oldvalue = (*p);
        (*p) += dvalue;
        }
        return(oldvalue);
}


#endif


static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx2f( int const i, 
            int const j, 
            int const ld )
{
        return( (((i)-1) + ((j)-1)*(ld)) );
}


static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx3f( int const i1, 
            int const i2, 
            int const i3,

            int const n1,
            int const n2 )
{
   return(indx2f(i1,i2,n1) + 
          ((i3)-1)*((n1)*(n2)) );
}


static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx4f( int const i1,
            int const i2,
            int const i3,
            int const i4,

            int const n1,
            int const n2,
            int const n3 )
{
        return(indx3f(i1,i2,i3,n1,n2) + 
               ((i4)-1)*((n1)*(n2)*(n3)) );
}



static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx5f( int const i1,
            int const i2,
            int const i3,
            int const i4,
            int const i5,

            int const n1,
            int const n2,
            int const n3,
            int const n4 ) 
{
        return(indx4f(i1,i2,i3,i4,n1,n2,n3) + 
               ((i5)-1)*((n1)*(n2)*(n3)*(n4)) );
}
        

static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx6f(int const i1,
           int const i2,
           int const i3,
           int const i4,
           int const i5,
           int const i6,

           int const n1,
           int const n2,
           int const n3,
           int const n4,
           int const n5)
{
        return(indx5f(i1,i2,i3,i4,i5,n1,n2,n3,n4) + 
               ((i6)-1)*((n1)*(n2)*(n3)*(n4)*(n5)) );
}







#endif
