#pragma once
#include "asgard_moment.hpp"
#include "asgard_solver.hpp"

namespace asgard
{
// the method expects either root or fixed name, one must be empty and one not
// the root is appended with step-number and .h5 extension
// the fixed filename is used "as-is" without any changes
template<typename P>
void write_output(PDE<P> const &pde,
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
restart_data<P> read_output(PDE<P> &pde, std::string const &restart_file);

} // namespace asgard
