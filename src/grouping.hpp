#pragma once
#include "element_table.hpp"
#include "pde.hpp"
#include "tensors.hpp"

using element_group = std::map<int, std::pair<int, int>>;

// convenience functions when working with element groups
int num_elements_in_group(element_group const &g);
int max_connected_in_group(element_group const &g);

template<typename P>
class rank_workspace
{
public:
  rank_workspace(PDE<P> const &pde, std::vector<element_group> const &groups);
  fk::vector<P> const &get_unit_vector() const;
  // input, output, workspace for batched gemm/reduction
  // (unit vector below also falls under this category)
  fk::vector<P> batch_input;
  fk::vector<P> reduction_space;
  fk::vector<P> batch_intermediate;
  fk::vector<P> batch_output;
  double size_MB() const
  {
    int64_t num_elems = batch_input.size() + reduction_space.size() +
                        batch_intermediate.size() + batch_output.size() +
                        unit_vector_.size();
    double const bytes     = static_cast<double>(num_elems) * sizeof(P);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

private:
  fk::vector<P> unit_vector_;
};

template<typename P>
int get_num_groups(element_table const &table, PDE<P> const &pde,
                   int const num_ranks, int const rank_size_MB);

std::vector<element_group>
assign_elements(element_table const &table, int const num_groups);

extern template int get_num_groups(element_table const &table,
                                   PDE<float> const &pde, int const num_ranks,
                                   int const rank_size_MB);
extern template int get_num_groups(element_table const &table,
                                   PDE<double> const &pde, int const num_ranks,
                                   int const rank_size_MB);
