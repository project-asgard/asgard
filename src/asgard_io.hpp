#pragma once
#include "asgard_moment.hpp"
#include "asgard_solver.hpp"

namespace asgard
{
template<typename P>
void generate_initial_moments(
    PDE<P> &pde, std::vector<moment<P>> &moments,
    adapt::distributed_grid<P> const &adaptive_grid,
    asgard::basis::wavelet_transform<P, resource::host> const &transformer,
    fk::vector<P> const &initial_condition);

// the method expects either root or fixed name, one must be empty and one not
// the root is appended with step-number and .h5 extension
// the fixed filename is used "as-is" without any changes
template<typename P>
void write_output(PDE<P> const &pde, std::vector<moment<P>> const &moments,
                  fk::vector<P> const &vec, P const time, int const file_index,
                  int const dof, elements::table const &hash_table,
                  std::string const &output_dataset_root  = "asgard",
                  std::string const &output_dataset_fixed = "");

template<typename P>
struct restart_data
{
  fk::vector<P> solution;
  P const time;
  int step_index;
  std::vector<int64_t> active_table;
  int max_level;
};

template<typename P>
restart_data<P> read_output(PDE<P> &pde, elements::table const &hash_table,
                            std::vector<moment<P>> &moments,
                            std::string const &restart_file);

} // namespace asgard
