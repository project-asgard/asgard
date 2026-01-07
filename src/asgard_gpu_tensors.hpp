#pragma once

#include "asgard_gpu_algorithms.hpp"

// Additional GPU kernels, related to more advanced tensor operations, e.g.,
// associated with sources and moments

namespace asgard::gpu
{

/*!
 * \brief Tensors the vectors into the output according to the multi-indexes
 *
 * Used in the construction of the sources, the constant vectors (separable) in
 * each direction are given by c1 ... c6 (only the first num_dim are used).
 * The sparse grid is given by num_indexes and indexes[] and n is the polynomial
 * degrees of freedom, e.g., n = 1 for constant basis, n = 3 for quadratic basis, etc.
 * The output is stored in x.
 *
 * The size of indexes[] must be num_dims * num_indexes.
 * The sizes of the c1 ... c(num_dims) must be n * max-level, the rest can be nullptr.
 * The size of x must be fm::ipow(n, num_dims) * num_indexes.
 */
template<typename P>
void tensor_by_index(int n, int num_dims, int num_indexes, int const indexes[],
                     P const c1[], P const c2[], P const c3[], P const c4[], P const c5[], P const c6[],
                     P x[]);

}
