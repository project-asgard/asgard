#include "asgard_refinement.hpp"

namespace asgard
{

template<typename P>
refinement_manager<P>::refinement_manager(prog_opts const &options, pde_scheme<P> &pde)
{
  if (options.adapt_relative or options.adapt_threshold) {
    atol = static_cast<P>(options.adapt_threshold.value_or(0));
    rtol = static_cast<P>(options.adapt_relative.value_or(0));

    finterp_     = std::move(pde.ref_interp_);
    finterp_mom_ = std::move(pde.ref_interp_mom_);
    moments_     = std::move(pde.ref_moments_);
  }
}

template<typename P>
void refinement_manager<P>::refine_(
    connection_patterns const &conns, term_manager<P> const &terms,
    std::vector<P> const &state, strategy mode, sparse_grid &grid) const
{
  int64_t const num_indexes = grid.num_indexes();
  int64_t const block_size  = fm::ipow(terms.basis.pdof, grid.num_dims());

  P maxw = 0;

  weights.resize(num_indexes);
  #pragma omp parallel
  {
    P sumall = 0;

    #pragma omp for
    for (int64_t i = 0; i < num_indexes; i++) {
      P sum = 0;
      // ASGARD_OMP_SIMD
      for (int64_t j = 0; j < block_size; j++) {
        P const s = state[i * block_size + j];
        sum += s * s;
      }
      sumall += sum;
      weights[i] = sum;
    }

    #pragma omp atomic
    maxw += sumall;
  }

  P tol = rtol * std::sqrt(maxw) + atol;

  stats.resize(num_indexes);
  ASGARD_OMP_PARFOR_SIMD
  for (int64_t i = 0; i < num_indexes; i++) {
    stats[i] = (std::sqrt(weights[i]) >= tol) ? istatus::refine : istatus::clear;
  }

  grid.refine(conns[connect_1d::hierarchy::volume], mode, stats);

  ignore(terms);

//   if (finterp_mom_) {
//
//   }
}

#ifdef ASGARD_ENABLE_DOUBLE
template class refinement_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class refinement_manager<float>;
#endif

} // namespace asgard
