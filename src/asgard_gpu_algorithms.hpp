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

//! computes, z = alpha * x + beta * y + gamma * z, which is different from axpy
template<typename P>
void axpbygz(int64_t num, no_deduce<P> alpha, P const x[], no_deduce<P> beta, P const y[], no_deduce<P> gamma, P z[]);

//! computes, y = alpha * x
template<typename P>
void set_scal(int64_t num, P alpha, P x[]);

//! y = x + a1 * x1
template<typename P>
void sum2(gpu::vector<P> const &x, no_deduce<P> a1, gpu::vector<P> const &x1, gpu::vector<P> &y);
//! y = x + a1 * x1 + a2 * x2
template<typename P>
void sum3(gpu::vector<P> const &x, no_deduce<P> a1, gpu::vector<P> const &x1, no_deduce<P> a2, gpu::vector<P> const &x2,
          gpu::vector<P> &y);
//! y = x + a1 * x1 + a2 * x2 + a3 * x3
template<typename P>
void sum4(gpu::vector<P> const &x, no_deduce<P> a1, gpu::vector<P> const &x1, no_deduce<P> a2, gpu::vector<P> const &x2,
          no_deduce<P> a3, gpu::vector<P> const &x3, gpu::vector<P> &y);
//! y = x + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4
template<typename P>
void sum5(gpu::vector<P> const &x, no_deduce<P> a1, gpu::vector<P> const &x1, no_deduce<P> a2, gpu::vector<P> const &x2,
          no_deduce<P> a3, gpu::vector<P> const &x3, no_deduce<P> a4, gpu::vector<P> const &x4, gpu::vector<P> &y);

//! returns the number of inf/nan entries in a vector
template<typename P>
int num_non_finite(int64_t num, P const x[]);

//! returns the number of inf/nan entries in a vector
template<typename P>
int num_non_finite(gpu::vector<P> const &data) { return num_non_finite(data.size(), data.data()); }

//! x = x + alpha * p, r = r - alpha * q
template<typename P>
void cg_update_x_r(int64_t n, P const *rho, P const *p_dot_q, P const p[], P const q[], P x[], P r[]);

//! p = r + beta * p
template<typename P>
void cg_update_p(int64_t n, P const *rho_new, P const *rho, P const r[], P p[]);

}
