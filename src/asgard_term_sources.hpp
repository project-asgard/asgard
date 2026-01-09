#pragma once

#include "asgard_interp.hpp"
#include "asgard_moment_manager.hpp"

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
  std::variant<std::monostate, scalar_func<P>, separable_func<P>> func;

  //! vector for the current grid
  std::vector<P> val;
  //! constant components of the source vector
  std::array<std::vector<P>, max_num_dimensions> consts;
  //! index if lumped with other sources
  int ilump = -1;

  //! computes approximate memory usage by the object
  size_t used_bytes() const {
    size_t t = val.size() * sizeof(P);
    for (auto const &v : consts) t += v.size() * sizeof(P);
    return t;
  }

  #ifdef ASGARD_USE_GPU
  //! constant components on the gpu
  std::array<gpu::vector<P>, max_num_dimensions> gpu_consts;
  //! vector for the current grid
  gpu::vector<P> gpu_val;
  #endif
};

//! holds the data for an interpolatory source entry
template<typename P>
struct source_entry_interp
{
  //! resource (GPU/MPI-rank) assigned to this source
  resource rec;
  //! calls the moment variant, if set for moments
  void operator() (P t, vector2d<P> const &x, momentset<P> const &moments,
                   std::vector<P> &vals) const
  {
    expect(not uses_gpu());
    if (std::holds_alternative<moment_source<P>>(func)) {
      std::get<moment_source<P>>(func)(t, x, moments, vals);
    } else {
      std::get<md_func<P>>(func)(t, x, vals);
    }
  }
  //! calls the moment variant, if set for moments
  void operator() (int64_t const num, P t, P const x[], momentset_gpu<P> const &moments,
                   P vals[]) const
  {
    expect(uses_gpu());
    if (std::holds_alternative<moment_source<P>>(func)) {
      std::get<moment_source<P>>(func)(num, t, x, moments, vals);
    } else {
      std::get<md_gpu_func<P>>(func)(num, t, x, vals);
    }
  }
  //! returns the moment source, use only if is_moment()
  moment_source<P> const &get_mom_md() const { return std::get<moment_source<P>>(func); }
  //! indicates whether the entry contains a moment function
  bool is_moment() const {
    return std::holds_alternative<moment_source<P>>(func);
  }
  //! indicates whether the entry contains a non-moment function
  bool is_non_moment() const {
    return std::holds_alternative<md_func<P>>(func)
           or std::holds_alternative<md_gpu_func<P>>(func);
  }
  //! indicates whether the entry contains a moment function
  bool uses_gpu() const {
    return std::holds_alternative<md_gpu_func<P>>(func)
           or (std::holds_alternative<moment_source<P>>(func)
               and std::get<moment_source<P>>(func).uses_gpu());
  }
  //! indicates whether the entry contains any function of any kind
  operator bool () const { return not std::holds_alternative<std::monostate>(func); }
  //! interpolatory function for the source entry
  md_source_var<P> func;
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

  #ifdef ASGARD_USE_GPU
  //! constant components on the gpu
  std::array<gpu::vector<P>, max_num_dimensions> gpu_consts;
  //! vector for the current grid
  gpu::vector<P> gpu_val;
  #endif
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
