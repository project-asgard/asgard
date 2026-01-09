#pragma once

#include "asgard_block_matrix.hpp"

// GPU kernels and algorithms for special terms, such as Lenard-Bernstein
// or BGK collision operators

namespace asgard::gpu
{

//! computing the ratio of two moments
template<typename P>
void moment_ratio(P nu, vector<P> const &momX, vector<P> const &mom0,
                  P const f[], P vals[]);

//! computing Lenard-Bernstein for 1 velocity dimension
template<typename P>
void lbc_vel1(P nu, vector<P> const &mom0, vector<P> const &mom1,
              vector<P> const &mom2, P const f[], P vals[]);

}
