
#include "asgard_term_manager.hpp"

#include "asgard_blas.hpp"

namespace asgard
{

template<typename P>
template<data_mode dmode>
void term_manager<P>::apply_sources(
    int groupid, pde_domain<P> const &domain, sparse_grid const &grid, connection_patterns const &conns,
    hierarchy_manipulator<P> const &hier, P time, P alpha, P y[])
{
  int64_t const block_size  = hier.block_size();
  int64_t const num_entries = grid.num_indexes() * block_size;

  static std::vector<P> lumped_sources(num_lumped * num_entries);
  lumped_sources.resize(num_lumped * num_entries);

  // if a boundary entry is at a lower link of a chain, go back and apply the previous links
  auto rechain = [&, this](boundary_entry<P> &bc) -> void
    {
      // not in a chain or last link, then nothing to do
      if (terms[bc.term_index].num_chain > 0)
        return;

      // otherwise we have to push the vectors through the term_md chain

      t1.resize(num_entries); // workspace

      bool keep_working = true;
      int tid = bc.term_index;
      while (keep_working) {
        --tid;

        // TODO: move this to the GPU with the rest of the sources/bc terms
        kron_term(grid, conns, terms[tid], 1, bc.val, 0, t1);
        std::swap(bc.val, t1);

        keep_working = (terms[tid].num_chain < 0);
      }
    };

  // update the const-components of the sources, if the grid has updated
  if (grid.generation() != sources_grid_gen)
  {
    int const pdof = hier.degree() + 1;

    auto tensor_consts = [&](auto &entry) -> void
      {
        P *data = nullptr;
        if (entry.ilump == -1) {
          entry.val.resize(num_entries);
          data = entry.val.data();
        } else {
          data = lumped_sources.data() + entry.ilump * num_entries;
        }

        #pragma omp parallel
        {
          std::array<P const *, max_num_dimensions> data1d;

          #pragma omp for
          for (int64_t j = 0; j < grid.num_indexes(); j++)
          {
            P *proj = data + j * block_size;

            int const *idx = grid[j];
            for (int d = 0; d < num_dims; d++)
              data1d[d] = entry.consts[d].data() + idx[d] * pdof;

            std::array<int, max_num_dimensions> v;
            std::fill_n(v.begin(), num_dims, 0);

            int i = 0;

            bool is_in = true;
            int c = 0;
            while (is_in or c > 0)
            {
              if (is_in)
              {
                P val = 1;
                for (int d = 0; d < num_dims; d++)
                  val *= data1d[d][ v[d] ];

                c = num_dims - 1;
                v[c]++;

                proj[i++] = val;
              }
              else
              {
                std::fill(v.begin() + c, v.begin() + num_dims, 0);
                v[--c]++;
              }

              is_in = (v[c] < pdof);
            }
          }
        }
      }; // end of tensor_consts lambda

    // update the constant components
    for (auto &src : sources)
    {
      #ifdef ASGARD_USE_MPI
      if (not resources.owns(src.rec))
        continue;
      #endif

      // the time-dependent case will construct both the 1D can mD vector for each t
      // the rest of the cases will have constant components in space and a time variable
      // this handles the space vector 1D -> mD tensoring

      if (src.is_time_dependent())
        continue;

      tensor_consts(src);
    }

    // update the constant components
    for (auto &bc : bcs)
    {
      #ifdef ASGARD_USE_MPI
      if (not resources.owns(terms[bc.term_index].rec))
        continue;
      #endif

      if (bc.is_time_dependent())
        continue;

      // In addition to the tensoring, the boundary condition case
      // may require application of the chain operators

      tensor_consts(bc);

      rechain(bc);
    }

    if (sources_have_time_dep or bcs_have_time_dep)
      rebuild_mass_matrices(grid);

    sources_grid_gen = grid.generation();
  }

  if constexpr (dmode == data_mode::replace or dmode == data_mode::scal_rep)
    std::fill_n(y, num_entries, P{0});

  // NOTE: when adding the "boundary" and "edge" sources, the sign is flipped

  indexrange irng = (groupid == -1) ? indexrange(sources)
                                    : source_groups[groupid].source_range;

  for (int is : irng) {
    auto const &src = sources[is];

    #ifdef ASGARD_USE_MPI
    if (not resources.owns(src.rec))
      continue;
    #endif

    switch (src.tmode) {
      case source_entry<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] += src.val[i];
        else
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] += alpha * src.val[i];
        break;
      case source_entry<P>::time_mode::separable: {
          tools::time_event perf_("separable source");
          P t = std::get<scalar_func<P>>(src.func)(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            t *= alpha;
          // do not handle sources 1-by-1, if may use gemv
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] += t * src.val[i];
        }
        break;
      case source_entry<P>::time_mode::time_dependent:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          hier.template project_separable<data_mode::increment>
              (std::get<separable_func<P>>(src.func), domain, grid, lmass, time, alpha, y);
        else
          hier.template project_separable<data_mode::scal_inc>
              (std::get<separable_func<P>>(src.func), domain, grid, lmass, time, alpha, y);
        break;
      default:
        // unreachable here
        break;
    }
  }

  if (groupid == -1) {
    #ifdef ASGARD_USE_MPI
    if (resources.is_leader())
    #endif
    for (auto const &s : sources_md)
      if (s) {
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          interp(grid, conns, time, 1, s, 1, y, kwork, it1);
        else
          interp(grid, conns, time, alpha, s, 1, y, kwork, it1);
      }
  } else {
    #ifdef ASGARD_USE_MPI
    if (sources_md[groupid] and resources.is_leader()) {
    #else
    if (sources_md[groupid]) {
    #endif
      if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
        interp(grid, conns, time, 1, sources_md[groupid], 1, y, kwork, it1);
      else
        interp(grid, conns, time, alpha, sources_md[groupid], 1, y, kwork, it1);
    }
  }

  irng = (groupid == -1) ? indexrange(bcs) : source_groups[groupid].bc_range;

  for (int ib : irng) {
    auto &bc = bcs[ib]; // non-const for the time-dependent case

    #ifdef ASGARD_USE_MPI
    if (not resources.owns(terms[bc.term_index].rec))
      continue;
    #endif

    switch (bc.tmode) {
      case boundary_entry<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= bc.val[i];
        else
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= alpha * bc.val[i];
        break;
      case boundary_entry<P>::time_mode::separable: {
          P t = bc.flux.func().ftime(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            t *= alpha;
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= t * bc.val[i];
        }
        break;
      case boundary_entry<P>::time_mode::time_dependent:
        bc.val.resize(num_entries);
        hier.template project_separable<data_mode::replace>
            (bc.flux.func(), domain, grid, lmass, time, alpha, bc.val.data());

        rechain(bc);

        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= bc.val[i];
        else
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < num_entries; i++)
            y[i] -= alpha * bc.val[i];
        break;
      default:
        // unreachable here
        break;
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template void term_manager<double>::apply_sources<data_mode::replace>(
    int, pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::increment>(
    int, pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_inc>(
    int, pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_rep>(
    int, pde_domain<double> const &, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void term_manager<float>::apply_sources<data_mode::replace>(
    int, pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::increment>(
    int, pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_inc>(
    int, pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_rep>(
    int, pde_domain<float> const &, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
#endif

}
