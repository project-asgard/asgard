#pragma once

#include "chunk.hpp"
#include "pde.hpp"

std::vector<element_chunk>
get_my_chunks(std::vector<element_chunk> const &all_chunks, int const my_rank,
              int const all_ranks);

using rank_to_inputs = std::map<int, limits<int64_t>>;
rank_to_inputs get_input_map(std::vector<element_chunk> const &my_chunks,
                             int const my_rank, int const all_ranks);

// simplifying assumptions made here: the set of chunks assigned to a rank is
// contiguous that is, the set represents a rectangular sub-area of the element
// grid matrix
//
// we assert for this condition in the implementation
template<typename P>
limits<int64_t>
get_input_range(std::vector<element_chunk> const &my_chunks, PDE<P> const &pde);
template<typename P>
limits<int64_t> get_output_range(std::vector<element_chunk> const &my_chunks,
                                 PDE<P> const &pde);

extern template limits<int64_t>
get_input_range(std::vector<element_chunk> const &my_chunks,
                PDE<float> const &pde);
extern template limits<int64_t>
get_input_range(std::vector<element_chunk> const &my_chunks,
                PDE<double> const &pde);

extern template limits<int64_t>
get_output_range(std::vector<element_chunk> const &my_chunks,
                 PDE<float> const &pde);
extern template limits<int64_t>
get_output_range(std::vector<element_chunk> const &my_chunks,
                 PDE<double> const &pde);
