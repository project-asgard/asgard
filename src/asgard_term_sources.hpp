#pragma once

#include "asgard_interp.hpp"
#include "asgard_moment_manager.hpp"

namespace asgard
{

//! holds data associated with with either a source term of boundary condition
template<typename P>
struct source_entry
{
  //! default source entry, must be reinitialized before use
  source_entry() = default;
  //! new source entry
  source_entry(separable_func<P> f) : func(std::move(f)) {}

  //! resource (GPU/MPI-rank) assigned to this source
  resource rec;

  bool is_time_const() const { return func.is_time_const(); }
  bool is_time_sep() const { return func.is_time_sep(); }
  bool is_time_non_sep() const { return func.is_time_non_sep(); }

  //! if the function is separable or time-dependent, handle the extra data
  separable_func<P> func;

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
    assert(not is_gpu());
    if (std::holds_alternative<md_mom_and_idx_func<P>>(func)) {
    // This signature requires indexes, so this overload cannot call it.
    // It is intended to be called only from eval_posonly_with_idx().
    throw std::runtime_error("md_mom_and_idx_func requires indexes (use eval_posonly_with_idx)");
    }
    else if (std::holds_alternative<md_mom_func<P>>(func)) {
      std::get<md_mom_func<P>>(func)(t, x, moments, vals);
    } else {
      std::get<md_func<P>>(func)(t, x, vals);
    }
  }
  //! calls the moment variant, if set for moments
  void operator() (int64_t const num, P t, P const x[], momentset_gpu<P> const &moments,
                   P vals[]) const
  {
    assert(is_gpu());
    if (std::holds_alternative<md_gpu_mom_func<P>>(func)) {
      std::get<md_gpu_mom_func<P>>(func)(num, t, x, moments, vals);
    } else {
      std::get<md_gpu_func<P>>(func)(num, t, x, vals);
    }
  }
  //! calls the moment+indexes variant
  void operator() (P t, vector2d<P> const &x, momentset<P> const &moments,
                  std::vector<int> const &indexes, std::vector<P> &vals) const
  {
    std::fprintf(stderr, "HIT source_entry_interp::operator()(with indexes)\n");
    std::fflush(stderr);

    assert(not is_gpu());
    if (std::holds_alternative<md_mom_and_idx_func<P>>(func)) {
      std::get<md_mom_and_idx_func<P>>(func)(t, x, moments, indexes, vals);
    } else if (std::holds_alternative<md_mom_func<P>>(func)) {
      std::get<md_mom_func<P>>(func)(t, x, moments, vals);
    } else {
      std::get<md_func<P>>(func)(t, x, vals);
    }
  }


  //! indicates whether the entry contains a moment function
  bool is_moment() const {
    return std::visit([](auto const &v) -> bool {
        return uses_moments<std::decay_t<decltype(v)>>;
    }, func);
  }
  //! indicates whether the entry contains a moment function
  bool is_gpu() const {
    return std::visit([](auto const &v) -> bool {
        return uses_gpu<std::decay_t<decltype(v)>>;
    }, func);
  }
  //! indicates whether the entry contains any function of any kind
  operator bool () const { return not std::holds_alternative<std::monostate>(func); }
  //! interpolatory function for the source entry
  md_source_func<P> func;
};

/*!
 * \brief Manages the terms and matrices, also holds the mass-matrices and kronmult-workspace
 *
 * This is the core of the spatial discretization of the terms.
 */
template<typename P>
struct boundary_entry {
  //! default source entry, must be reinitialized before use
  boundary_entry() = default;
  //! create a new source entry
  boundary_entry(boundary_flux<P> f) : flux(std::move(f)) {}
  //! defines the flux, moved out of the term
  boundary_flux<P> flux;

  bool is_time_const() const { return flux.func().is_time_const(); }
  bool is_time_sep() const { return flux.func().is_time_sep(); }
  bool is_time_non_sep() const { return flux.func().is_time_non_sep(); }

  //! vector for the current grid
  std::vector<P> val;
  //! constant components of the source vector
  std::array<std::vector<P>, max_num_dimensions> consts;
  //! the term associated with this boundary entry
  int term_index = -1;
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
