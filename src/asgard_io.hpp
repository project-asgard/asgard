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

/*!
 * \internal
 * \brief I/O manager that can save to and restart from file
 *
 * The manager is friend to most classes which allows it to directly access
 * the internal data-structures. This manager handles most of the restart logic
 * in handling default or overriding the current settings, e.g., restart and
 * change the final time or the time-step.
 * \endinternal
 */
template<typename P>
class h5manager {
public:
  //! write to file
  static void write(PDEv2<P> const &pde, int degree, sparse_grid const &grid,
                    time_data<P> const &tdata, std::vector<P> const &state,
                    std::string const &filename);

  //! read from file
  static void read(std::string const &filename, bool silent, PDEv2<P> &pde,
                   sparse_grid &grid, time_data<P> &tdata, std::vector<P> &state);

  //! indicator for the asgard "safe-file" version (mostly a future feature)
  static int constexpr asgard_file_version = 1;
};

} // namespace asgard
