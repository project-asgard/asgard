
#include "asgard_term_manager.hpp"

#include "asgard_blas.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_tensors.hpp"
#endif

namespace asgard
{

template<typename P>
template<data_mode dmode>
void term_manager<P>::apply_sources(
    group_id group, sparse_grid const &grid, connection_patterns const &conns,
    hierarchy_manipulator<P> const &hier, P time, P alpha, P y[])
{
  // make all sources/bc lumped, except the time-dependent ones
  // if lumped size is small, use the addition for-loop
  // move the lumped workspace (sources/weights) into the terms class

  tools::time_event perf_("sources apply");

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
    tools::time_event perf2_("sources grid update");
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
      if (src.is_time_dependent() or not resources.owns(src.rec))
        continue;

      // the time-dependent case will construct both the 1D can mD vector for each t
      // the rest of the cases will have constant components in space and a time variable
      // this handles the space vector 1D -> mD tensoring
      tensor_consts(src);
    }

    // update the constant components
    for (auto &bc : bcs)
    {
      if (bc.is_time_dependent() or not resources.owns(terms[bc.term_index].rec))
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

  indexrange isrng = (group == group_id::all()) ? indexrange(sources)
                                                : source_groups[group()].source_range;

  for (int is : isrng) {
    auto const &src = sources[is];
    if (not resources.owns(src.rec)) continue;

    switch (src.tmode) {
      case source_entry<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          sweights.push_back(P{1});
        else
          sweights.push_back(alpha);
        break;
      case source_entry<P>::time_mode::separable: {
          P t = src.func.time_at(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            sweights.push_back(alpha * t);
          else
            sweights.push_back(t);
        }
        break;
      case source_entry<P>::time_mode::time_dependent:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          hier.template project_separable<data_mode::increment>
              (src.func, grid, lmass, time, alpha, y);
        else
          hier.template project_separable<data_mode::scal_inc>
              (src.func, grid, lmass, time, alpha, y);
        break;
      default:
        // unreachable here
        break;
    }
  }

  if (group == group_id::all()) {
    for (auto const &src : sources_md) {
      if (not src or not resources.owns(src.rec)) continue;

      if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
        interp(grid, conns, moms.get_cached_interps(), time, 1, src, 1, y, kwork, it1, it2);
      else
        interp(grid, conns, moms.get_cached_interps(), time, alpha, src, 1, y, kwork, it1, it2);
    }
  } else {
    if (resources.owns(sources_md[group()].rec) and !!sources_md[group()])
    {
      if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
        interp(grid, conns, moms.get_cached_interps(), time, 1, sources_md[group()],
               1, y, kwork, it1, it2);
      else
        interp(grid, conns, moms.get_cached_interps(), time, alpha, sources_md[group()],
               1, y, kwork, it1, it2);
    }
  }

  indexrange ibrng = (group == group_id::all()) ? indexrange(bcs)
                                                : source_groups[group()].bc_range;

  for (int ib : ibrng) {
    auto &bc = bcs[ib]; // non-const for the time-dependent case
    if (not resources.owns(terms[bc.term_index].rec)) continue;

    switch (bc.tmode) {
      case boundary_entry<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          sweights.push_back(-P{1});
        else
          sweights.push_back(-alpha);
        break;
      case boundary_entry<P>::time_mode::separable: {
          P t = bc.flux.func().time_at(time);
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

  indexrange irng = (group == group_id::all()) ? indexrange(0, num_lumped)
                                               : source_groups[group()].lump_range;

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

#ifdef ASGARD_USE_GPU
template<typename P>
template<data_mode dmode>
void term_manager<P>::apply_sources_gpu(
    group_id group, sparse_grid const &grid, connection_patterns const &conns,
    hierarchy_manipulator<P> const &hier, P time, P alpha, P y[])
{
  // make all sources/bc lumped, except the time-dependent ones
  // if lumped size is small, use the addition for-loop
  // move the lumped workspace (sources/weights) into the terms class

  tools::time_event perf_("sources gpu-apply");

  compute->set_device(gpu::device{0}); // rework for multi-GPU

  int64_t const block_size  = hier.block_size();
  int64_t const num_entries = grid.num_indexes() * block_size;

  // if a boundary entry is at a lower link of a chain, go back and apply the previous links
  auto rechain = [&, this](gpu::device dev, boundary_entry<P> &bc, P al, P data[]) -> void
    {
      // push the vectors through the term_md chain
      // assuming the current data is in t1, using t1/t2 as workspace

      P *gt1 = gpu_t1[dev.id].data();
      P *gt2 = gpu_t2[dev.id].data();

      // rechain until the top link
      int tid = bc.term_index - 1;
      while (tid > 0 and terms[tid - 1].is_chain_link()) {
        // TODO: move this to the GPU with the rest of the sources/bc terms
        kron_term(dev, grid, conns, terms[tid], 1, gt1, 0, gt2);
        std::swap(gt1, gt2);

        --tid;
      }
      // apply the top chain and put the result in the final place
      kron_term(grid, conns, terms[tid - 1], al, gt1, 0, data);
    };

  // update the const-components of the sources, if the grid has updated
  if (grid.generation() != sources_gpu_grid_gen)
  {
    tools::time_event perf2_("sources grid gpu-update");
    gpu_swork.resize(num_lumped * num_entries);

    int const pdof = hier.degree() + 1;

    auto tensor_consts = [&, this](auto &entry, P *data = nullptr) -> void
      {
        if (data == nullptr) {
          if (entry.ilump == -1) {
            entry.gpu_val.resize(num_entries);
            data = entry.gpu_val.data();
          } else {
            data = gpu_swork.data() + entry.ilump * num_entries;
          }
        }

        gpu::tensor_by_index(pdof, num_dims, grid.num_indexes(), grid.gpu_indexes(),
            entry.gpu_consts[0].data(), entry.gpu_consts[1].data(), entry.gpu_consts[2].data(),
            entry.gpu_consts[3].data(), entry.gpu_consts[4].data(), entry.gpu_consts[5].data(),
            data);
      };

    // update the constant components
    for (auto &src : sources)
    {
      if (src.is_time_dependent() or not resources.owns(src.rec)) continue;

      // see the CPU version, only tensoring when there's time-independent component
      tensor_consts(src);
    }

    // update the constant components
    for (auto &bc : bcs)
    {
      if (not resources.owns(terms[bc.term_index].rec)) continue;

      if (bc.is_time_dependent())
        continue;

      // In addition to the tensoring, the boundary condition case
      // may require application of the chain operators

      if (terms[bc.term_index].is_chain_link()) { // if chain (not top link)
        // tensor into a temp, rechain and put the final result into swork
        tensor_consts(bc, gpu_t1[0].data());
        rechain(gpu::device{0}, bc, P{1}, swork.data() + bc.ilump * num_entries);
      } else
        tensor_consts(bc);
    }

    if (sources_have_time_dep or bcs_have_time_dep)
      rebuild_mass_matrices(grid);

    sources_gpu_grid_gen = grid.generation();
  }

  // NOTE: when adding the "boundary" and "edge" sources, the sign is flipped

  sweights.resize(0);

  // the non-separable in time sources are still processed on the CPU
  // if such entry is encountered, zero and store in t1
  bool s1_initialized = false;
  auto using_cpu_s1 = [&, this]() -> void {
    if (not s1_initialized) {
      cpu_s1.resize(num_entries);
      std::fill(cpu_s1.begin(), cpu_s1.end(), P{0});
      s1_initialized = true;
    }
  };

  // most entries delay updating y until the end but rechaining updates on-the-fly
  if constexpr (dmode == data_mode::replace or dmode == data_mode::scal_rep)
    compute->fill_zeros(num_entries, y);

  indexrange isrng = (group == group_id::all()) ? indexrange(sources)
                                                : source_groups[group()].source_range;

  for (int is : isrng) {
    auto const &src = sources[is];
    if (not resources.owns(src.rec)) continue;

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
        using_cpu_s1();
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          hier.template project_separable<data_mode::increment>
              (std::get<separable_func<P>>(src.func), grid, lmass, time, alpha, cpu_s1.data());
        else
          hier.template project_separable<data_mode::scal_inc>
              (std::get<separable_func<P>>(src.func), grid, lmass, time, alpha, cpu_s1.data());
        break;
      default:
        // unreachable here
        break;
    }
  }

  indexrange ibrng = (group == group_id::all()) ? indexrange(bcs)
                                                : source_groups[group()].bc_range;

  for (int ib : ibrng) {
    auto &bc = bcs[ib]; // non-const for the time-dependent case
    if (not resources.owns(terms[bc.term_index].rec)) continue;

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
              (bc.flux.func(), grid, lmass, time, 1, t2.data());
          gpu_t1[0] = t2;
          if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
            rechain(gpu::device{0}, bc, P{-1}, y);
          else
            rechain(gpu::device{0}, bc, -alpha, y);
        } else {
          using_cpu_s1();
          if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
            hier.template project_separable<data_mode::increment>
                (bc.flux.func(), grid, lmass, time, P{-1}, cpu_s1.data());
          else
            hier.template project_separable<data_mode::scal_inc>
                (bc.flux.func(), grid, lmass, time, -alpha, cpu_s1.data());
        }
        break;
      default:
        // unreachable here
        break;
    }
  }

  indexrange irng = (group == group_id::all()) ? indexrange(0, num_lumped)
                                               : source_groups[group()].lump_range;

  if (not gpu_swork.empty())
  {
    gpu_sweights = sweights;
    compute->gemv(num_entries, irng.size(), 1,
                  gpu_swork.data() + num_entries * irng.ibegin(),
                  gpu_sweights.data(), 1, y);
  }

  auto interp_source = [&](source_entry_interp<P> const &src)
        -> void {
      if (src.is_gpu()) {
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          interp(gpu::device{0}, grid, conns, moms.get_cached_interps(gpu::device{0}), time,
                 1, src, 1, y, kwork, gpu_it1[0], gpu_it2[0]);
        else
          interp(gpu::device{0}, grid, conns, moms.get_cached_interps(gpu::device{0}), time,
                 alpha, src, 1, y, kwork, gpu_it1[0], gpu_it2[0]);
      } else {
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          interp(gpu::device{0}, grid, conns, moms.get_cached_interps(), time,
                 1, src, 1, y, kwork, cpu_it1[0], gpu_it1[0], gpu_it2[0]);
        else
          interp(gpu::device{0}, grid, conns, moms.get_cached_interps(), time,
                 alpha, src, 1, y, kwork, cpu_it1[0], gpu_it1[0], gpu_it2[0]);
      }
    };

  // interpolation sources
  if (group == group_id::all()) {
    for (auto const &src : sources_md) {
      if (not src or not resources.owns(src.rec)) continue;

      interp_source(src);
    }
  } else {
    if (resources.owns(sources_md[group()].rec) and !!sources_md[group()])
    {
      interp_source(sources_md[group()]);
    }
  }

  if (s1_initialized) {
    gpu_t1[0] = cpu_s1;
    compute->axpy(num_entries, 1, gpu_t1[0].data(), y);
  }
}
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template void term_manager<double>::apply_sources<data_mode::replace>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::increment>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_inc>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_rep>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);

#ifdef ASGARD_USE_GPU
template void term_manager<double>::apply_sources_gpu<data_mode::replace>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources_gpu<data_mode::increment>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources_gpu<data_mode::scal_inc>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
template void term_manager<double>::apply_sources_gpu<data_mode::scal_rep>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, double, double[]);
#endif

#endif

#ifdef ASGARD_ENABLE_FLOAT
template void term_manager<float>::apply_sources<data_mode::replace>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::increment>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_inc>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_rep>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);

#ifdef ASGARD_USE_GPU
template void term_manager<float>::apply_sources_gpu<data_mode::replace>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources_gpu<data_mode::increment>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources_gpu<data_mode::scal_inc>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
template void term_manager<float>::apply_sources_gpu<data_mode::scal_rep>(
    group_id, sparse_grid const &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, float, float[]);
#endif

#endif

}
