#pragma once

#include "asgard_block_matrix.hpp"

// GPU kernels and algorithms for keeping as much of the workload as possible
// on the GPU device and avoid moving data between kronmult operations.

namespace asgard::gpu
{

//! apply the jacobi preconditioner
template<typename P>
void jacobi_apply(gpu::vector<P> const &jacobi, P x[]);

//! compute, p = r + beta * (p - omega * v), used by bicgstab
template<typename P>
void compute_last_bicgstab(P beta, P omega, gpu::vector<P> const &r,
                           gpu::vector<P> const &v, gpu::vector<P> &p);

//! computes, y = x + beta * y, which is different from axpy
template<typename P>
void xpby(gpu::vector<P> const &x, P beta, P y[]);

//! computes, y = alpha * x + beta * y, which is different from axpy
template<typename P>
void axpby(int64_t num, P alpha, P const x[], P beta, P y[]);

}
