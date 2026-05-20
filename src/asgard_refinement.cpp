#include "asgard_refinement.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_tensors.hpp"
#endif

// the number beyond which numbers will be considered non-finite,
// e.g., if we get this number as one of the hierarchical interpolatory coefficients,
// then the grid is too coarse to produce a meaningful approximation and we must refine everywhere
#define ASGARD_INFINITE_TRESHOLD 1.E+100

namespace asgard
{

template<typename P>
refinement_manager<P>::refinement_manager(prog_opts const &options, pde_scheme<P> &pde)
{
  if (options.adapt_relative or options.adapt_threshold) {
    atol = static_cast<P>(options.adapt_threshold.value_or(0));
    rtol = static_cast<P>(options.adapt_relative.value_or(0));

    iweights_.interp_ = std::move(pde.ref_interp_);

    moments_ = std::move(pde.ref_moments_);

    if (iweights_) {
      iplan.enable();
      iplan.stop_hier();
    }
  }
}

template<typename P>
void refinement_manager<P>::refine_(std::vector<P> const &state, strategy mode,
                                    term_manager<P> &terms) const
{
  int64_t const num_indexes = terms.grid.num_indexes();
  int64_t const block_size  = fm::ipow(terms.basis.pdof, terms.grid.num_dims());

  P l2 = 0;

  weights.resize(num_indexes);
  #pragma omp parallel
  {
    P sumall = 0;

    #pragma omp for
    for (int64_t i = 0; i < num_indexes; i++) {
      P sum = 0;
      ASGARD_OMP_SIMD
      for (int64_t j = 0; j < block_size; j++) {
        P const s = state[i * block_size + j];
        sum += s * s;
      }
      sumall += sum;
      weights[i] = sum;
    }

    #pragma omp atomic
    l2 += sumall;
  }

  P const tol = rtol * std::sqrt(l2) + atol;

  stats.resize(num_indexes);
  ASGARD_OMP_PARFOR_SIMD
  for (int64_t i = 0; i < num_indexes; i++) {
    stats[i] = (std::sqrt(weights[i]) >= tol) ? istatus::refine : istatus::clear;
  }

  auto update_stats = [&](std::vector<P> const &vals) -> void
    {
      P wmax = 0;

      #pragma omp parallel
      {
        P maxall = 0;

        #pragma omp for
        for (int64_t i = 0; i < num_indexes; i++) {
          P m = 0;
          ASGARD_OMP_SIMD
          for (int64_t j = 0; j < block_size; j++)
            m = std::max(m, std::abs(vals[i * block_size + j]));

          maxall     = std::max(m, maxall);
          weights[i] = m;
        }

        #pragma omp critical
        if (maxall > wmax) {
          wmax = maxall;
        }
      }

      if (wmax > ASGARD_INFINITE_TRESHOLD) { // This is kind of AD-HOC
        // cannot compute a stable wmax, assume worst case scenario "refine all"
        std::fill(stats.begin(), stats.end(), istatus::refine);
      } else {
        P const ctol = rtol * wmax + atol;

        ASGARD_OMP_PARFOR_SIMD
        for (int64_t i = 0; i < num_indexes; i++) {
          if (stats[i] == istatus::clear and weights[i] >= ctol)
            stats[i] = istatus::refine;
        }
      }
    };

  // add the correction due to the interpolation terms
  if (iplan.is_enabled()) {
    if (iweights_.is_moment()) {
      terms.moms.compute_interps(moments_, terms.grid, state, terms.interp, terms.kwork);
      iplan.use_moments(true);
      terms.interp(iplan, terms.grid, terms.conn, terms.moms.get_cached_interps(), 0, state.data(),
                   1, iweights_, 0, terms.t1.data(), terms.kwork);
      update_stats(terms.t1);
    } else { // no moments in the adaptive weight
      iplan.use_moments(false);
      terms.interp(iplan, terms.grid, terms.conn, terms.moms.get_cached_interps(), 0, state.data(),
                   1, iweights_, 0, terms.t1.data(), terms.kwork);
      update_stats(terms.t1);
    }
  }

  terms.grid.refine(terms.conn[connect_1d::hierarchy::volume], mode, stats);
}

#ifdef ASGARD_USE_GPU
template<typename P>
void refinement_manager<P>::refine_(gpu::vector<P> const &state, strategy mode,
                                    term_manager<P> &terms) const
{
  int64_t const num_indexes = terms.grid.num_indexes();
  int64_t const block_size  = fm::ipow(terms.basis.pdof, terms.grid.num_dims());

  P wmax = 0;
  gpu::compute_l2_weights<P>(block_size, num_indexes, state, gweight, wmax);

  P const tol = rtol * std::sqrt(wmax) + atol;
  gstats.resize(num_indexes);
  gpu::set_istatus(num_indexes, tol, gweight, gstats);

  if (iplan.is_enabled())
  {
    if (iweights_.is_moment()) {
      iplan.use_moments(true);

      terms.moms.compute_moments(moments_, terms.grid, terms.interp, terms.kwork,
                                 state, not iweights_.is_gpu());
    } else {
      iplan.use_moments(false);
    }

    iplan.use_gpu_func(iweights_.is_gpu());

    ghier.resize(state.size());
    terms.interp(gpu::device{0}, iplan, terms.grid, terms.conn, terms.moms, 0, state.data(),
                 1, iweights_, 0, ghier.data(), terms.kwork);

    wmax = 0;
    gpu::compute_max_weights<P>(block_size, num_indexes, ghier, gweight, wmax);

    if (wmax > ASGARD_INFINITE_TRESHOLD) {
      // cannot compute wmax, probably the mesh is too coarse to compute moments
      // assume worst case and "refine everywhere"
      stats.resize(num_indexes);
      std::fill(stats.begin(), stats.end(), istatus::refine);
    } else {
      P const ctol = rtol * wmax + atol;
      gpu::update_istatus(num_indexes, ctol, gweight, gstats);
      gstats.copy_to_host(stats);
    }
  }
  else // no adapt weight, use the current gstats
    gstats.copy_to_host(stats);

  terms.grid.refine(terms.conn[connect_1d::hierarchy::volume], mode, stats);

  // - later add an option to do this without the function values, i.e., using source signatures
}
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template class refinement_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class refinement_manager<float>;
#endif

} // namespace asgard
