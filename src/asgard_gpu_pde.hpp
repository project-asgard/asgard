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

//! computing Lenard-Bernstein for 2 velocity dimensions
template<typename P>
void lbc_vel2(P nu, vector<P> const &mom0, vector<P> const &mom10, vector<P> const &mom01,
              vector<P> const &mom20, vector<P> const &mom02, P const f[], P vals[]);

//! computing Lenard-Bernstein for 3 velocity dimensions
template<typename P>
void lbc_vel3(P nu, vector<P> const &mom0, vector<P> const &mom100, vector<P> const &mom010,
              vector<P> const &mom001, vector<P> const &mom200, vector<P> const &mom020,
              vector<P> const &mom002, P const f[], P vals[]);

//! computing Bhatnagar-Gross-Krook for 1 velocity dimension
template<typename P>
void bgk_vel1(P nu, int num_pos, P const nodes[], vector<P> const &mom0,
              vector<P> const &mom1, vector<P> const &mom2, P vals[]);

//! computing Bhatnagar-Gross-Krook for 2 velocity dimensions
template<typename P>
void bgk_vel2(P nu, int num_pos, P const nodes[], vector<P> const &mom0,
              vector<P> const &mom10, vector<P> const &mom01,
              vector<P> const &mom20, vector<P> const &mom02, P vals[]);

//! computing Bhatnagar-Gross-Krook for 3 velocity dimensions
template<typename P>
void bgk_vel3(P nu, int num_pos, P const nodes[], vector<P> const &mom0,
              vector<P> const &mom100, vector<P> const &mom010, vector<P> const &mom001,
              vector<P> const &mom200, vector<P> const &mom020, vector<P> const &mom002,
              P vals[]);
}
