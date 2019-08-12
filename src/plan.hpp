#pragma once

#include "chunk.hpp"

std::vector<element_chunk>
get_my_chunks(std::vector<element_chunk> const &all_chunks, int const my_rank,
              int const all_ranks);

using rank_to_inputs = std::map<int, limits<int64_t>>;
rank_to_inputs get_my_inputs(std::vector<element_chunk> const &my_chunks,
                             int const my_rank, int const all_ranks);
