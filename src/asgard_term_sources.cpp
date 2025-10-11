
#include "asgard_term_manager.hpp"

#include "asgard_blas.hpp"

namespace asgard
{

template<typename P>
template<data_mode dmode>
void term_manager<P>::apply_sources(
    int groupid, sparse_grid const &grid, connection_patterns const &conns,
    hierarchy_manipulator<P> const &hier, P time, P alpha, P y[])
{
  // make all sources/bc lumped, except the time-dependent ones
  // if lumped size is small, use the addition for-loop
  // move the lumped workspace (sources/weights) into the terms class

  tools::time_event perf_("inside apply sources");

  int64_t const block_size  = hier.block_size();
  int64_t const num_entries = grid.num_indexes() * block_size;

  // if a boundary entry is at a lower link of a chain, go back and apply the previous links
  auto rechain = [&, this](boundary_entry<P> &bc, P al, P data[]) -> void
    {
      // push the vectors through the term_md chain
      // assuming the current data is in t1, using t1/t2 as workspace

      // rechain until the top link
      int tid = bc.term_index - 1;
      while (tid > 0 and terms[tid - 1].is_chain_link()) {
        // TODO: move this to the GPU with the rest of the sources/bc terms
        kron_term(grid, conns, terms[tid], 1, t1, 0, t2);
        std::swap(t1, t2);

        --tid;
      }
      // apply the top chain and put the result in the final place
      kron_term(grid, conns, terms[tid - 1], al, t1.data(), 0, data);
    };

  // update the const-components of the sources, if the grid has updated
  if (grid.generation() != sources_grid_gen)
  {
    tools::time_event perf2_("updating grid");
    swork.resize(num_lumped * num_entries);

    int const pdof = hier.degree() + 1;

    auto tensor_consts = [&, this](auto &entry, P *data = nullptr) -> void
      {
        if (data == nullptr) {
          if (entry.ilump == -1) {
            entry.val.resize(num_entries);
            data = entry.val.data();
          } else {
            data = swork.data() + entry.ilump * num_entries;
          }
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

      if (terms[bc.term_index].is_chain_link()) { // if chain (not top link)
        // tensor into a temp, rechain and put the final result into swork
        tensor_consts(bc, t1.data());
        rechain(bc, P{1}, swork.data() + bc.ilump * num_entries);
      } else
        tensor_consts(bc);
    }

    if (sources_have_time_dep or bcs_have_time_dep)
      rebuild_mass_matrices(grid);

    sources_grid_gen = grid.generation();
  }

  if constexpr (dmode == data_mode::replace or dmode == data_mode::scal_rep)
    std::fill_n(y, num_entries, P{0});

  // NOTE: when adding the "boundary" and "edge" sources, the sign is flipped

  sweights.resize(0);

  indexrange isrng = (groupid == -1) ? indexrange(sources)
                                     : source_groups[groupid].source_range;

  for (int is : isrng) {
    auto const &src = sources[is];

    #ifdef ASGARD_USE_MPI
    if (not resources.owns(src.rec))
      continue;
    #endif

    switch (src.tmode) {
      case source_entry<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          sweights.push_back(P{1});
        else
          sweights.push_back(alpha);
        break;
      case source_entry<P>::time_mode::separable: {
          P t = std::get<scalar_func<P>>(src.func)(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            sweights.push_back(alpha * t);
          else
            sweights.push_back(t);
        }
        break;
      case source_entry<P>::time_mode::time_dependent:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          hier.template project_separable<data_mode::increment>
              (std::get<separable_func<P>>(src.func), grid, lmass, time, alpha, y);
        else
          hier.template project_separable<data_mode::scal_inc>
              (std::get<separable_func<P>>(src.func), grid, lmass, time, alpha, y);
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
          interp(grid, conns, time, 1, s, 1, y, kwork, it1, it2);
        else
          interp(grid, conns, time, alpha, s, 1, y, kwork, it1, it2);
      }
  } else {
    #ifdef ASGARD_USE_MPI
    if (sources_md[groupid] and resources.is_leader()) {
    #else
    if (sources_md[groupid]) {
    #endif
      if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
        interp(grid, conns, time, 1, sources_md[groupid], 1, y, kwork, it1, it2);
      else
        interp(grid, conns, time, alpha, sources_md[groupid], 1, y, kwork, it1, it2);
    }
  }

  indexrange ibrng = (groupid == -1) ? indexrange(bcs)
                                     : source_groups[groupid].bc_range;

  for (int ib : ibrng) {
    auto &bc = bcs[ib]; // non-const for the time-dependent case

    #ifdef ASGARD_USE_MPI
    if (not resources.owns(terms[bc.term_index].rec))
      continue;
    #endif

    switch (bc.tmode) {
      case boundary_entry<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          sweights.push_back(-P{1});
        else
          sweights.push_back(-alpha);
        break;
      case boundary_entry<P>::time_mode::separable: {
          P t = bc.flux.func().ftime(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            sweights.push_back(-alpha * t);
          else
            sweights.push_back(-t);
        }
        break;
      case boundary_entry<P>::time_mode::time_dependent:
        if (terms[bc.term_index].is_chain_link()) {
          hier.template project_separable<data_mode::replace>
              (bc.flux.func(), grid, lmass, time, 1, t1.data());
          if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
            rechain(bc, P{-1}, y);
          else
            rechain(bc, -alpha, y);
        } else {
          if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
            hier.template project_separable<data_mode::increment>
                (bc.flux.func(), grid, lmass, time, P{-1}, y);
          else
            hier.template project_separable<data_mode::scal_inc>
                (bc.flux.func(), grid, lmass, time, -alpha, y);
        }
        break;
      default:
        // unreachable here
        break;
    }
  }

  indexrange irng = (groupid == -1) ? indexrange(0, num_lumped)
                                    : source_groups[groupid].lump_range;

  // using BLAS level 2 gemv operation is more efficient when we are dealing
  // with a sufficiently large number of sources
  // however, using gemv here results in much slower cpu-kronmult operations
  //   suspected aggressive use of CPU cache by gemv leading to cache misses in gemv
  //   the problem happens even when OpenMP is off, but does not appear with CUDA
  #ifdef ASGARD_USE_GPU
  int constexpr gemv_threshold = 2;
  if (irng.size() >= gemv_threshold) {
    fm::gemv('N', num_entries, irng.size(), 1,
             swork.data() + num_entries * irng.ibegin(),
             sweights.data(), 1, y);
  } else
  #endif
    for (int i = 0; i < irng.size(); i++) {
      P const w = sweights[i];
      P const *s = swork.data() + (i + irng.ibegin()) * num_entries;
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t j = 0; j < num_entries; j++)
        y[j] += w * s[j];
    }
}

#ifdef ASGARD_ENABLE_DOUBLE
template void term_manager<double>::apply_sources<data_mode::replace>(
    int, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::increment>(
    int, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_inc>(
    int, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_rep>(
    int, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void term_manager<float>::apply_sources<data_mode::replace>(
    int, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::increment>(
    int, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_inc>(
    int, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_rep>(
    int, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
#endif

}
