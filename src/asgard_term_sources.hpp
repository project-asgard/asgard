#pragma once

#include "asgard_interp.hpp"

namespace asgard
{

//! holds data associated with with either a source term of boundary condition
template<typename P>
struct source_entry
{
  //! mode indicating when to recompute the coefficients
  enum class time_mode {
    //! interior source that is constant in time
    constant = 0,
    //! interior source that is separable in time, i.e., constant in space with time multiplier
    separable,
    //! interior source that is non-separable in time, still separable in space for fixed time
    time_dependent
  };
  //! default source entry, must be reinitialized before use
  source_entry() = default;
  //! create a new source entry
  source_entry(time_mode mode_in) : tmode(mode_in) {}

  //! when should we recompute the sources and when can we reuse existing data
  time_mode tmode = time_mode::constant;
  //! resource (GPU/MPI-rank) assigned to this source
  resource rec;

  bool is_constant() const { return tmode == time_mode::constant; }
  bool is_separable() const { return tmode == time_mode::separable; }
  bool is_time_dependent() const { return tmode == time_mode::time_dependent; }

  //! if the function is separable or time-dependent, handle the extra data
  std::variant<int, scalar_func<P>, separable_func<P>> func;

  //! vector for the current grid
  std::vector<P> val;
  //! constant components of the source vector
  std::array<std::vector<P>, max_num_dimensions> consts;
  //! index if lumped with other sources
  int ilump = -1;
};

/*!
 * \brief Manages the terms and matrices, also holds the mass-matrices and kronmult-workspace
 *
 * This is the core of the spatial discretization of the terms.
 */
template<typename P>
struct boundary_entry {
  //! mode indicating when to recompute the coefficients
  enum class time_mode {
    //! boundary condition that is constant in time
    constant = 0,
    //! boundary condition that is separable in time, i.e., constant in space with time multiplier
    separable,
    //! boundary condition that is non-separable in time, still separable in space for fixed time
    time_dependent
  };
  //! default source entry, must be reinitialized before use
  boundary_entry() = default;
  //! create a new source entry
  boundary_entry(boundary_flux<P> f) : flux(std::move(f)) {}
  //! defines the flux, moved out of the term
  boundary_flux<P> flux;

  //! when should we recompute the sources and when can we reuse existing data
  time_mode tmode = time_mode::constant;

  bool is_constant() const { return tmode == time_mode::constant; }
  bool is_separable() const { return tmode == time_mode::separable; }
  bool is_time_dependent() const { return tmode == time_mode::time_dependent; }

  //! the term associated with this boundary entry
  int term_index = -1;
  //! vector for the current grid
  std::vector<P> val;
  //! constant components of the source vector
  std::array<std::vector<P>, max_num_dimensions> consts;
  //! index if lumped with other sources
  int ilump = -1;
};

/*!
 * \brief Combines information about regular and boundary source groups
 */
struct group_combo {
  //! range for the regular sources
  irange source_range;
  //! boundary sources range
  irange bc_range;
  //! number of sources lumped into a gemv
  irange lump_range;
};

}
