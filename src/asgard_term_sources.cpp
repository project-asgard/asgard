
#include "asgard_term_manager.hpp"

#include "asgard_blas.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_tensors.hpp"
#endif

namespace asgard
{

// ib_dim is the interpolated boundary dimension
template<typename P, data_mode dmode, int ib_dim>
void merge_boundary_grids(sparse_grid const &grid, sparse_grid const &subgrid,
                          std::vector<int> const &map,
                          std::vector<P> const &con1d, std::vector<P> const &bnd,
                          int pdof, P alpha, P y[])
{
  int const num_dims = grid.num_dims();
  assert(0 <= ib_dim and ib_dim < num_dims);
  assert(num_dims <= max_num_dimensions);

  assert(not con1d.empty());
  assert(bnd.size() == static_cast<size_t>(subgrid.num_dof()));

#pragma omp parallel for
  for (int64_t i = 0; i < grid.num_indexes(); i++)
  {
    int const *idx = grid[i];
    int const isub = map[i];

    P *out = y + i * grid.block_size();

    P const *block1d  = con1d.data() + pdof * idx[ib_dim];
    P const *subblock = bnd.data() + subgrid.block_size() * isub;

    std::array<int, max_num_dimensions> v;
    std::fill_n(v.begin(), num_dims, 0);

    int const ib_init = (ib_dim == 0) ? 1 : 0;
    int const ib_post = (ib_dim == 0) ? 2 : ib_dim + 1;

    bool is_in = true;
    int c = 0;
    while (is_in or c > 0)
    {
      if (is_in)
      {
        int ib = v[ib_init];
        for (int d = ib_init + 1; d < ib_dim; d++) {
          ib *= pdof;
          ib += v[d];
        }
        if constexpr (ib_dim < 5)
          for (int d = ib_post; d < num_dims; d++) {
            ib *= pdof;
            ib += v[d];
          }

        P const b1 = block1d[v[ib_dim]];
        P const b2 = subblock[ib];

        if constexpr (dmode == data_mode::replace)
          *out++ = b1 * b2;
        else if constexpr (dmode == data_mode::scal_rep)
          *out++ = alpha * b1 * b2;
        else if constexpr (dmode == data_mode::increment)
          *out++ += b1 * b2;
        else if constexpr (dmode == data_mode::scal_inc)
          *out++ += alpha * b1 * b2;

        c = num_dims - 1;
        v[c]++;
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

template<typename P, data_mode dmode>
void merge_boundary_grids(sparse_grid const &grid, sparse_grid const &subgrid, int flux_dim,
                          std::vector<int> const &map,
                          std::vector<P> const &con1d, std::vector<P> const &bnd,
                          int pdof, P alpha, P y[])
{
  switch(flux_dim) {
      case 0: merge_boundary_grids<P, dmode, 0>
              (grid, subgrid, map, con1d, bnd, pdof, alpha, y);
              break;
      case 1: merge_boundary_grids<P, dmode, 1>
              (grid, subgrid, map, con1d, bnd, pdof, alpha, y);
              break;
      case 2: merge_boundary_grids<P, dmode, 2>
              (grid, subgrid, map, con1d, bnd, pdof, alpha, y);
              break;
      case 3: merge_boundary_grids<P, dmode, 3>
              (grid, subgrid, map, con1d, bnd, pdof, alpha, y);
              break;
      case 4: merge_boundary_grids<P, dmode, 4>
              (grid, subgrid, map, con1d, bnd, pdof, alpha, y);
              break;
      case 5: merge_boundary_grids<P, dmode, 5>
              (grid, subgrid, map, con1d, bnd, pdof, alpha, y);
              break;
      default:
        break;
    };
}


template<typename P>
template<data_mode dmode>
void term_manager<P>::apply_sources(group_id group, P time, P alpha, P y[])
{
  // make all sources/bc lumped, except the time-dependent ones
  // if lumped size is small, use the addition for-loop
  // move the lumped workspace (sources/weights) into the terms class

  tools::time_event perf_("sources apply");

  int64_t const block_size  = grid.block_size();
  int64_t const num_entries = grid.num_dof();

  // if a boundary entry is at a lower link of a chain, go back and apply the previous links
  // x are base files at the bottom of the chain, out is the output, w is workspace
  // all sizes must be num_entries, use t1 and t2 workspace vectors
  // when computing cached entries, beta is 0
  // when storing directly in y (e.g., for non-sep in time and interpolated BC), beta is 1
  auto rechain = [&, this](boundary_entry<P> &bc, P al, P x[], P beta, P out[], P w[]) -> void
    {
      // push the vectors through the term_md chain
      // intermediate chains are stored in w, final chain result is added to out
      int tid = bc.term_index - 1;
      while (terms[tid].is_chain_link()) {
        kron_term(terms[tid], 1, x, 0, w);
        std::swap(x, w);

        --tid;
      }
      // apply the top chain and put the result in the final place
      kron_term(terms[tid], al, x, beta, out);
    };

  // update the const-components of the sources, if the grid has updated
  if (grid.generation() != sources_grid_gen)
  {
    tools::time_event perf2_("sources grid update");
    swork.resize(num_lumped * num_entries);

    int const num_dims = grid.num_dims();
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
      if (src.is_time_non_sep() or not resources.owns(src.rec))
        continue;

      // the time-dependent case will construct both the 1D can mD vector for each t
      // the rest of the cases will have constant components in space and a time variable
      // this handles the space vector 1D -> mD tensoring
      tensor_consts(src);
    }

    // update the constant components
    for (auto &bc : bcs)
    {
      if (not bc.is_separable()
          or bc.is_time_non_sep()
          or not resources.owns(terms[bc.term_index].rec))
        continue;

      // In addition to the tensoring, the boundary condition case
      // may require application of the chain operators

      if (terms[bc.term_index].is_chain_link()) { // if chain (not top link)
        // tensor into a temp, rechain and put the final result into swork
        tensor_consts(bc, t1.data());
        rechain(bc, P{1}, t1.data(), P{0}, swork.data() + bc.ilump * num_entries, t2.data());
      } else
        tensor_consts(bc);
    }

    if (sources_have_time_dep or bcs_have_time_dep)
      rebuild_mass_matrices();

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

    switch (src.func.get_time_mode()) {
      case separable_func<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          sweights.push_back(P{1});
        else
          sweights.push_back(alpha);
        break;
      case separable_func<P>::time_mode::separable: {
          P t = src.func.time_at(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            sweights.push_back(alpha * t);
          else
            sweights.push_back(t);
        }
        break;
      case separable_func<P>::time_mode::non_separable:
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
        interp(grid, conn, moms.get_cached_interps(), time, 1, src, 1, y, kwork);
      else
        interp(grid, conn, moms.get_cached_interps(), time, alpha, src, 1, y, kwork);
    }
  } else {
    if (resources.owns(sources_md[group()].rec) and !!sources_md[group()])
    {
      if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
        interp(grid, conn, moms.get_cached_interps(), time, 1, sources_md[group()],
               1, y, kwork);
      else
        interp(grid, conn, moms.get_cached_interps(), time, alpha, sources_md[group()],
               1, y, kwork);
    }
  }

  indexrange ibrng = (group == group_id::all()) ? indexrange(bcs)
                                                : source_groups[group()].bc_range;

  for (int ib : ibrng) {
    auto &bc = bcs[ib]; // non-const for the time-dependent case
    // auto const &trm = terms[bc.term_index];
    if (not resources.owns(terms[bc.term_index].rec)) continue;

    if (not bc.is_separable()) {
      // this works like the time-dependent case, the assumption is that we cannot reuse
      // the vector that has been computed ... should probably fix that

      // 1. check-update the grid
      int const flux_dim = terms[bc.term_index].flux_dim; // direction of the flux
      sparse_grid &subgrid = ibc_grid[flux_dim];
      if (subgrid.generation() != grid.generation()) {
        grid.subgrid(flux_dim, basis.pdof, subgrid, ibc_map[flux_dim]);
        switch (flux_dim) {
          case 0: interp.template nodes<0>(ibc_grid[flux_dim], ibc_nodes[0]); break;
          case 1: interp.template nodes<1>(ibc_grid[flux_dim], ibc_nodes[1]); break;
          case 2: interp.template nodes<2>(ibc_grid[flux_dim], ibc_nodes[2]); break;
          case 3: interp.template nodes<3>(ibc_grid[flux_dim], ibc_nodes[3]); break;
          case 4: interp.template nodes<4>(ibc_grid[flux_dim], ibc_nodes[4]); break;
          case 5: interp.template nodes<5>(ibc_grid[flux_dim], ibc_nodes[5]); break;
          default: // unreachable
            break;
        }
      }

      // 2. set the interpolation work-spaces
      size_t const nwork = interp.it1.size(); // needed to restore the size

      interp.it1.resize(subgrid.num_dof());
      interp.it2.resize(interp.it1.size());

      // 3. call the interpolated function on the nodes
      std::visit([&](auto const &func) {
          if constexpr (std::is_same_v<std::decay_t<decltype(func)>, md_func<P>>) {
            func(time, ibc_nodes[flux_dim], interp.it1);
          }
        }, bc.flux.var_func());

      // 4. construct hierarchical basis and project back on the interpolation nodes
      block_cpu(basis.pdof, subgrid, conn, ibc_perm_low, interp.matrix_nodal2hier(),
                P{1}, interp.it1.data(), P{0}, interp.it2.data(), kwork);

      block_cpu(basis.pdof, subgrid, conn, ibc_perm_up, interp.matrix_hier2wav(),
                ibc_iwavscale[flux_dim], interp.it2.data(), P{0}, interp.it1.data(), kwork);

      if (terms[bc.term_index].is_chain_link())
      {
        merge_boundary_grids<P, data_mode::replace>
            (grid, ibc_grid[flux_dim], flux_dim, ibc_map[flux_dim], bc.consts[flux_dim],
             interp.it1, basis.pdof, 1, t1.data());

        interp.it1.resize(nwork);
        interp.it2.resize(nwork);

        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          rechain(bc, P{-1}, t1.data(), P{1}, y, t2.data());
        else
          rechain(bc, -alpha, t1.data(), P{1}, y, t2.data());
      }
      else
      {
        constexpr data_mode effective_mode = [&]() -> data_mode {
          if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
            return data_mode::increment;
          else
            return data_mode::scal_inc;
        }();

        merge_boundary_grids<P, effective_mode>
            (grid, ibc_grid[flux_dim], flux_dim, ibc_map[flux_dim], bc.consts[flux_dim],
             interp.it1, basis.pdof, -alpha, y);

        interp.it1.resize(nwork);
        interp.it2.resize(nwork);
      }

      continue;
    }

    switch (bc.flux.func().get_time_mode()) {
      case separable_func<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          sweights.push_back(-P{1});
        else
          sweights.push_back(-alpha);
        break;
      case separable_func<P>::time_mode::separable: {
          P t = bc.flux.func().time_at(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            sweights.push_back(-alpha * t);
          else
            sweights.push_back(-t);
        }
        break;
      case separable_func<P>::time_mode::non_separable:
        if (terms[bc.term_index].is_chain_link()) {
          hier.template project_separable<data_mode::replace>
              (bc.flux.func(), grid, lmass, time, 1, t1.data());
          if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
            rechain(bc, P{-1}, t1.data(), P{1}, y, t2.data());
          else
            rechain(bc, -alpha, t1.data(), P{1}, y, t2.data());
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
void term_manager<P>::apply_sources_gpu(group_id group, P time, P alpha, P y[])
{
  // make all sources/bc lumped, except the time-dependent ones
  // if lumped size is small, use the addition for-loop
  // move the lumped workspace (sources/weights) into the terms class

  tools::time_event perf_("sources gpu-apply");

  gpu::device const dev{0};
  compute->set_device(dev); // rework for multi-GPU

  int64_t const num_entries = grid.num_dof();

  // if a boundary entry is at a lower link of a chain, go back and apply the previous links
  // see the CPU case
  auto rechain = [&, this](gpu::device g, boundary_entry<P> &bc,
                           P al, P x[], P beta, P out[], P w[]) -> void
    {
      int tid = bc.term_index - 1;
      while (terms[tid].is_chain_link()) {
        // TODO: move this to the GPU with the rest of the sources/bc terms
        kron_term(g, terms[tid], 1, x, 0, w);
        std::swap(x, w);

        --tid;
      }
      // apply the top chain and put the result in the final place
      kron_term(g, terms[tid], al, x, beta, out);
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

        gpu::tensor_by_index(pdof, grid.num_dims(), grid.num_indexes(), grid.gpu_indexes(),
            entry.gpu_consts[0].data(), entry.gpu_consts[1].data(), entry.gpu_consts[2].data(),
            entry.gpu_consts[3].data(), entry.gpu_consts[4].data(), entry.gpu_consts[5].data(),
            data);
      };

    // update the constant components
    for (auto &src : sources)
    {
      if (src.is_time_non_sep() or not resources.owns(src.rec)) continue;

      // see the CPU version, only tensoring when there's time-independent component
      tensor_consts(src);
    }

    // update the constant components
    for (auto &bc : bcs)
    {
      if (not bc.is_separable()) continue;



      if (not bc.is_separable()
          or bc.is_time_non_sep()
          or not resources.owns(terms[bc.term_index].rec))
        continue;

      // In addition to the tensoring, the boundary condition case
      // may require application of the chain operators

      if (terms[bc.term_index].is_chain_link()) { // if chain (not top link)
        // tensor into a temp, rechain and put the final result into swork
        tensor_consts(bc, gpu_t1[dev()].data());
        rechain(dev, bc, P{1}, gpu_t1[dev()].data(),
                P{0}, gpu_swork.data() + bc.ilump * num_entries, gpu_t2[dev()].data());
      } else
        tensor_consts(bc);
    }

    if (sources_have_time_dep or bcs_have_time_dep)
      rebuild_mass_matrices();

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

    switch (src.func.get_time_mode()) {
      case separable_func<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          sweights.push_back(P{1});
        else
          sweights.push_back(alpha);
        break;
      case separable_func<P>::time_mode::separable: {
          P t = src.func.time_at(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            sweights.push_back(alpha * t);
          else
            sweights.push_back(t);
        }
        break;
      case separable_func<P>::time_mode::non_separable:
        using_cpu_s1();
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          hier.template project_separable<data_mode::increment>
              (src.func, grid, lmass, time, alpha, cpu_s1.data());
        else
          hier.template project_separable<data_mode::scal_inc>
              (src.func, grid, lmass, time, alpha, cpu_s1.data());
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

    if (not bc.is_separable()) {
      // this works like the time-dependent case, the assumption is that we cannot reuse
      // the vector that has been computed ... should probably fix that

      // 1. check-update the grid
      int const flux_dim = terms[bc.term_index].flux_dim; // direction of the flux
      sparse_grid &subgrid = ibc_grid[flux_dim];
      if (subgrid.generation() != grid.generation()) {
        grid.subgrid(flux_dim, basis.pdof, subgrid, ibc_map[flux_dim]);
        ibc_gpu_map[flux_dim] = ibc_map[flux_dim];
        switch (flux_dim) {
          case 0: interp.template nodes<0>(ibc_grid[0], ibc_nodes[0]); break;
          case 1: interp.template nodes<1>(ibc_grid[1], ibc_nodes[1]); break;
          case 2: interp.template nodes<2>(ibc_grid[2], ibc_nodes[2]); break;
          case 3: interp.template nodes<3>(ibc_grid[3], ibc_nodes[3]); break;
          case 4: interp.template nodes<4>(ibc_grid[4], ibc_nodes[4]); break;
          case 5: interp.template nodes<5>(ibc_grid[5], ibc_nodes[5]); break;
          default: // unreachable
            break;
        }
        ibc_gpu_nodes[flux_dim] = ibc_nodes[flux_dim].data_vector();
      }

      gpu::vector<P> &work1 = interp.gpu_it1[dev()];
      gpu::vector<P> &work2 = interp.gpu_it2[dev()];
      int64_t const ndof = subgrid.num_dof();
      assert(work1.size() >= ndof);
      assert(work2.size() >= ndof);
      gpu::wrap_array<P> w1(work1.data(), ndof);
      gpu::wrap_array<P> w2(work2.data(), ndof);

      P *p1 = w1.vec.data();
      P *p2 = w2.vec.data();

      // 3. call the interpolated function on the nodes
      std::visit([&](auto const &func) {
          if constexpr (std::is_same_v<std::decay_t<decltype(func)>, md_func<P>>) {
            interp.it1.resize(subgrid.num_dof());
            func(time, ibc_nodes[flux_dim], interp.it1);
            w1.vec = interp.it1;
            interp.it1.resize(grid.num_dof());
          } else if constexpr (std::is_same_v<std::decay_t<decltype(func)>, md_gpu_func<P>>) {
            func(subgrid.num_dof(), time, ibc_gpu_nodes[flux_dim].data(), p1);
          }
        }, bc.flux.var_func());

      // 4. construct hierarchical basis and project back on the interpolation nodes
      block_gpu(dev, basis.pdof, subgrid, conn, ibc_perm_low,
                interp.matrix_gpu_nodal2hier(dev), P{1}, p1, P{0}, p2, kwork,
                interp.matrix_nodal2hier());

      block_gpu(dev, basis.pdof, subgrid, conn, ibc_perm_up,
                interp.matrix_gpu_hier2wav(dev),
                ibc_iwavscale[flux_dim], p2, P{0}, p1, kwork,
                interp.matrix_hier2wav());

      if (terms[bc.term_index].is_chain_link())
      {
        gpu::merge_boundary_grids<P, data_mode::replace>
            (grid, ibc_grid[flux_dim], flux_dim, ibc_gpu_map[flux_dim], bc.gpu_consts[flux_dim],
             w1.vec, basis.pdof, 1, gpu_t1[dev()].data());

        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          rechain(dev, bc, P{-1}, gpu_t1[dev()].data(), P{1}, y, gpu_t2[dev()].data());
        else
          rechain(dev, bc, -alpha, gpu_t1[dev()].data(), P{1}, y, gpu_t2[dev()].data());
      }
      else
      {
        constexpr data_mode effective_mode = [&]() -> data_mode {
          if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
            return data_mode::increment;
          else
            return data_mode::scal_inc;
        }();

        gpu::merge_boundary_grids<P, effective_mode>
            (grid, ibc_grid[flux_dim], flux_dim, ibc_gpu_map[flux_dim], bc.gpu_consts[flux_dim],
             w1.vec, basis.pdof, -alpha, y);
      }

      continue;
    }

    switch (bc.flux.func().get_time_mode()) {
      case separable_func<P>::time_mode::constant:
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          sweights.push_back(-P{1});
        else
          sweights.push_back(-alpha);
        break;
      case separable_func<P>::time_mode::separable: {
          P t = bc.flux.func().time_at(time);
          if constexpr (dmode == data_mode::scal_inc or dmode == data_mode::scal_rep)
            sweights.push_back(-alpha * t);
          else
            sweights.push_back(-t);
        }
        break;
      case separable_func<P>::time_mode::non_separable:
        if (terms[bc.term_index].is_chain_link()) {
          hier.template project_separable<data_mode::replace>
              (bc.flux.func(), grid, lmass, time, 1, t2.data());
          gpu_t1[dev()] = t2;
          if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
            rechain(dev, bc, P{-1}, gpu_t1[dev()].data(), P{1}, y, gpu_t2[dev()].data());
          else
            rechain(dev, bc, -alpha, gpu_t1[dev()].data(), P{1}, y, gpu_t2[dev()].data());
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
          interp(dev, grid, conn, moms.get_cached_interps(dev), time,
                 1, src, 1, y, kwork);
        else
          interp(dev, grid, conn, moms.get_cached_interps(dev), time,
                 alpha, src, 1, y, kwork);
      } else {
        if constexpr (dmode == data_mode::increment or dmode == data_mode::replace)
          interp(dev, grid, conn, moms.get_cached_interps(), time,
                 1, src, 1, y, kwork);
        else
          interp(dev, grid, conn, moms.get_cached_interps(), time,
                 alpha, src, 1, y, kwork);
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
    gpu_t1[dev()] = cpu_s1;
    compute->axpy(num_entries, 1, gpu_t1[dev()].data(), y);
  }
}
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template void term_manager<double>::apply_sources<data_mode::replace>(
    group_id, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::increment>(
    group_id, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_inc>(
    group_id, double, double, double[]);
template void term_manager<double>::apply_sources<data_mode::scal_rep>(
    group_id, double, double, double[]);

#ifdef ASGARD_USE_GPU
template void term_manager<double>::apply_sources_gpu<data_mode::replace>(
    group_id, double, double, double[]);
template void term_manager<double>::apply_sources_gpu<data_mode::increment>(
    group_id, double, double, double[]);
template void term_manager<double>::apply_sources_gpu<data_mode::scal_inc>(
    group_id, double, double, double[]);
template void term_manager<double>::apply_sources_gpu<data_mode::scal_rep>(
    group_id, double, double, double[]);
#endif

#endif

#ifdef ASGARD_ENABLE_FLOAT
template void term_manager<float>::apply_sources<data_mode::replace>(
    group_id, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::increment>(
    group_id, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_inc>(
    group_id, float, float, float[]);
template void term_manager<float>::apply_sources<data_mode::scal_rep>(
    group_id, float, float, float[]);

#ifdef ASGARD_USE_GPU
template void term_manager<float>::apply_sources_gpu<data_mode::replace>(
    group_id, float, float, float[]);
template void term_manager<float>::apply_sources_gpu<data_mode::increment>(
    group_id, float, float, float[]);
template void term_manager<float>::apply_sources_gpu<data_mode::scal_inc>(
    group_id, float, float, float[]);
template void term_manager<float>::apply_sources_gpu<data_mode::scal_rep>(
    group_id, float, float, float[]);
#endif

#endif

}
