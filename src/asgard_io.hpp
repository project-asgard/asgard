#pragma once
#include "asgard_time_data.hpp"

namespace asgard
{
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
  static void write(prog_opts const &options, pde_domain<P> const &domain, int degree,
                    sparse_grid const &grid, time_data const &tdata,
                    std::vector<P> const &state,
                    std::vector<aux_field_entry<P>> const &aux_fields,
                    std::string const &filename);

  //! read from file
  static void read(std::string const &filename, bool silent,
                   prog_opts &options, pde_domain<P> &domain,
                   sparse_grid &grid, time_data &tdata,
                   std::vector<aux_field_entry<P>> &aux_fields,
                   std::vector<P> &state);

  //! indicator for the asgard "save-file" version (mostly a future feature)
  static int constexpr asgard_file_version = 1;
};

} // namespace asgard
