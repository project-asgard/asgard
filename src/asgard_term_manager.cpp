#include "asgard_term_manager.hpp"

#include "asgard_blas.hpp"

namespace asgard
{

template<typename P>
void term_manager<P>::mass_apply(
    sparse_grid const &grid, connection_patterns const &conns,
    P alpha, std::vector<P> const &x, P beta, std::vector<P> &y) const
{
  if (beta == 0) {
    y.resize(x.size());
  } else {
    expect(y.size() == x.size());
  }
  if (mass_term) {
    block_cpu(basis.pdof, grid, conns, mass_perm, mass_forward,
              alpha, x.data(), beta, y.data(), kwork);
  } else {
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 0; i < x.size(); i++)
      y[i] = alpha * x[i] + beta * y[i];
  }
}
template<typename P>
P term_manager<P>::normL2(
    sparse_grid const &grid, connection_patterns const &conns,
    std::vector<P> const &x) const
{
  if (mass_term) {
    mass_apply(grid, conns, 1, x, 0, t1);
    P nrm = 0;
    for (size_t i = 0; i < x.size(); i++)
      nrm += x[i] * t1[i];
    return std::sqrt(nrm);
  } else {
    P nrm = 0;
    for (size_t i = 0; i < x.size(); i++)
      nrm += x[i] * x[i];
    return std::sqrt(nrm);
  }
}

template<typename P>
template<typename vector_type_x, typename vector_type_y>
void term_manager<P>::apply_tmpl(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    P alpha, vector_type_x x, P beta, vector_type_y y) const
{
  bool constexpr using_vectors = std::is_same_v<vector_type_x, std::vector<P> const &>;

  if constexpr (using_vectors)
  {
    expect(x.size() == y.size());
    expect(x.size() == kwork.w1.size());
  }
  expect(all_groups <= gid and gid < static_cast<int>(term_groups.size()));

  auto kterm = [&grid, &conns, this](term_entry<P> const &tme, P al, P const in[], P be, P out[])
    -> void {
      if (tme.is_interpolatory()) {
        interp(tme.interplan, grid, conns, momset, 0, in, ifield, al, tme.tmd, be, out, kwork, it1, it2);
      } else {
        block_cpu(basis.pdof, grid, conns, tme.perm, tme.coeffs,
                  al, in, be, out, kwork);
      }
    };

  P b = beta; // on first iteration, overwrite y

  P const *px = [&]()
        -> P const * {
      if constexpr (using_vectors)
        return x.data();
      else
        return x;
      }();
  P *py = [&]()
        -> P * {
      if constexpr (using_vectors)
        return y.data();
      else
        return y;
      }();

  if (not ifield.empty()) // using interpolation and will need the field
    interp.wav2nodal(grid, px, ifield, kwork);

  auto const group = terms_group_range(gid);
  int icurrent = group.ibegin();
  while (icurrent < group.iend())
  {
    auto it = terms.begin() + icurrent;

    #ifdef ASGARD_USE_MPI
    if (not resources.owns(it->rec)) {
      icurrent += it->num_chain;
      continue;
    }
    #endif

    if (it->num_chain == 1) {
      kterm(*it, alpha, px, b, py);
      ++icurrent;
    } else {
      // dealing with a chain
      int const num_chain = it->num_chain;

      kterm(*(it + num_chain - 1), 1, px, 0, t1.data());
      for (int i = num_chain - 2; i > 0; --i) {
        kterm(*(it + i), 1, t1.data(), 0, t2.data());
        std::swap(t1, t2);
      }
      kterm(*it, alpha, t1.data(), b, py);

      icurrent += num_chain;
    }

    b = 1; // next iteration appends on y
  }

  if (not has_terms_) {
    int64_t const num = grid.num_indexes() * fm::ipow(basis.pdof, num_dims);
    if (beta == 0) {
      std::fill_n(py, num, 0);
    } else {
      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < num; i++)
        py[i] *= beta;
    }
  }
}

#ifdef ASGARD_USE_FLOPCOUNTER
template<typename P>
int64_t term_manager<P>::flop_count(
    int gid, sparse_grid const &grid, connection_patterns const &conns) const
{
  #ifdef ASGARD_USE_MPI
  if (not is_leader())
    return -1;
  #endif

  expect(-1 <= gid and gid < static_cast<int>(term_groups.size()));

  int const gidx = gid + 1;
  if (flop_info.size() <= static_cast<size_t>(gidx))
    flop_info.resize(gidx + 1);

  if (flop_info[gidx].grid_gen == grid.generation())
    return flop_info[gidx].flops;

  int64_t flops = 0;

  auto kterm = [&grid, &conns, &flops, this](term_entry<P> const &tme)
    -> void {
      if (not tme.tmd.is_interpolatory())
        flops += block_cpu(basis.pdof, grid, conns, tme.perm, kwork);
    };

  int icurrent   = (gid == -1) ? 0                              : term_groups[gid].begin();
  int const iend = (gid == -1) ? static_cast<int>(terms.size()) : term_groups[gid].end();
  while (icurrent < iend)
  {
    auto it = terms.begin() + icurrent;

    if (it->num_chain == 1) {
      kterm(*it);
      ++icurrent;
    } else {
      // dealing with a chain
      int const num_chain = it->num_chain;

      kterm(*(it + num_chain - 1));

      for (int i = num_chain - 2; i > 0; --i)
        kterm(*(it + i));

      kterm(*it);

      icurrent += num_chain;
    }
  }

  if (not has_terms_)
    flops += static_cast<int64_t>(kwork.w1.size());

  flop_info[gidx].grid_gen = grid.generation();
  flop_info[gidx].flops    = flops;

  return flops;
}
#endif

#ifdef ASGARD_USE_GPU
template<typename P>
void term_manager<P>::prapare_kron_workspace_gpu(int64_t num_entries)
{
  int const num_gpus = compute->num_gpus();

  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});
    // GPU 0 always uses gpu_t1[0] for scratch-space when collecting local data
    if ((not t1.empty() or g == 0) and gpu_t1[g].size() < num_entries)
      gpu_t1[g].resize(num_entries);
    if (not t2.empty() and gpu_t2[g].size() < num_entries)
      gpu_t2[g].resize(num_entries);
    kwork.gpu_w1[g].resize(num_entries);
    kwork.gpu_w2[g].resize(num_entries);

    if (interp) {
      cpu_it1[g].resize(num_entries);
      cpu_it2[g].resize(num_entries);
      gpu_it1[g].resize(num_entries);
      gpu_it2[g].resize(num_entries);
    }
  }
}

template<typename P>
template<typename vector_type_x, typename vector_type_y, compute_mode mode>
void term_manager<P>::apply_tmpl_gpu(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    P alpha, vector_type_x x, P beta, vector_type_y y) const
{
  bool constexpr using_cpu_vectors = std::is_same_v<vector_type_x, std::vector<P> const &>;
  bool constexpr using_gpu_vectors = std::is_same_v<vector_type_x, gpu::vector<P> const &>;

  bool constexpr using_vectors = using_cpu_vectors or using_gpu_vectors;

  // general idea
  // 1. If CPU -> mode data to GPU 0
  // 2. If GPU 0 (moved or provided there), distribute to all GPUs
  //    --- have multi-GPU scratch space, data for x and y on each device
  // 3. !!!! Fix the load of the coefficient matrices, currently doesn't respect the device
  // 4. Loop over all devices and perform kron only for the local terms
  //    --- respect the has-term per device
  // 5. Bring all data back to device 0 and add it up
  //    --- apply alpha on every device, do beta at the end
  //
  //  -- maybe have 2 versions, 1 GPU and multi-GPUs

  if constexpr (using_cpu_vectors)
    static_assert(mode == compute_mode::cpu, "std::vector requires compute_mode::cpu");
  if constexpr (using_gpu_vectors)
    static_assert(mode == compute_mode::gpu, "gpu::vector requires compute_mode::gpu");
  // no reasonable way to check if pointers are on the CPU or GPU, assume "mode" is set correctly

  if constexpr (using_vectors)
    expect(x.size() == y.size());

  int64_t const num_entries  = fm::ipow(basis.pdof, grid.num_dims()) * grid.num_indexes();

  expect(-1 <= gid and gid < static_cast<int>(term_groups.size()));

  auto kterm = [&grid, &conns, this]
               (gpu::device dev, term_entry<P> const &tme, P al, P const in[], P be, P out[])
    -> void {
      if (tme.is_interpolatory()) {
        interp(dev, tme.interplan, grid, conns, momset, 0, in, ifield, al, tme.tmd, be, out, kwork,
               cpu_it1[dev.id], cpu_it2[dev.id], gpu_it1[dev.id], gpu_it2[dev.id]);
      } else {
        block_gpu(dev, basis.pdof, grid, conns, tme.perm, tme.gpu_coeffs,
                  al, in, be, out, kwork, tme.coeffs);
      }
    };

  // if doing out-of-core, load data onto the device and sync across devices, device 0 is always the "root"
  if constexpr (mode == compute_mode::cpu) {
    // tools::time_event performance_("copy_from_host");
    compute->set_device(gpu::device{0});
    if constexpr (using_cpu_vectors) {
      gpu_x[0] = x;
      if (beta == 0) // allocate memory, no need to copy
        gpu_y[0].resize(num_entries);
      else
        gpu_y[0] = y;
    } else {
      gpu_x[0].resize(num_entries);
      gpu_x[0].copy_from_host(num_entries, x);
      gpu_y[0].resize(num_entries);
      gpu_y[0].copy_from_host(num_entries, y);
    }
  }

  int const num_gpus = compute->num_gpus();

  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});

    // effective x/y, either x or gpu_x[id]
    P const *xpntr = nullptr;
    P *ypntr = nullptr;

    if (g == 0) {
      if constexpr (mode == compute_mode::cpu) {
        xpntr = gpu_x[0].data();
        ypntr = gpu_y[0].data();
      } else {
        if constexpr (using_vectors) {
          xpntr = x.data();
          ypntr = y.data();
        } else {
          xpntr = x;
          ypntr = y;
        }
      }
    } else {
      gpu_x[g].resize(num_entries);
      gpu_y[g].resize(num_entries);
      if constexpr (mode == compute_mode::cpu) {
        gpu::mcopy(gpu::device{0}, gpu_x[0], gpu::device{g}, gpu_x[g]);
        gpu::mcopy(gpu::device{0}, gpu_y[0], gpu::device{g}, gpu_y[g]);
      } else {
        gpu::mcopy(gpu::device{0}, x, gpu::device{g}, gpu_x[g]);
        gpu::mcopy(gpu::device{0}, y, gpu::device{g}, gpu_y[g]);
      }
      xpntr = gpu_x[g].data();
      ypntr = gpu_y[g].data();
    }

    P b = (g == 0) ? beta : 0; // on first iteration, overwrite y

    if (not ifield.empty()) {
      interp.wav2nodal(gpu::device{0}, grid, xpntr, gpu_it1[0].data(), kwork);
      gpu_it1[0].copy_to_host(ifield);
    }

    bool term_found = false; // does this GPU have at least 1 term

    auto const group = terms_group_range(gid);

    int icurrent = group.ibegin();
    while (icurrent < group.iend())
    {
      auto it = terms.begin() + icurrent;

      // skip the terms associated with other MPI ranks or devices
      #ifdef ASGARD_USE_MPI
      if (not resources.owns(it->rec) or it->rec.device != g) {
        icurrent += it->num_chain;
        continue;
      }
      #else
      if (it->rec.device != g) {
        icurrent += it->num_chain;
        continue;
      }
      #endif

      if (it->num_chain == 1) {
        kterm(gpu::device{g}, *it, alpha, xpntr, b, ypntr);
      } else {
        // dealing with a chain
        int const num_chain = it->num_chain;

        kterm(gpu::device{g}, *(it + num_chain - 1), 1, xpntr, 0, gpu_t1[g].data());
        for (int i = num_chain - 2; i > 0; --i) {
          kterm(gpu::device{g}, *(it + i), 1, gpu_t1[g].data(), 0, gpu_t2[g].data());
          std::swap(gpu_t1[g], gpu_t2[g]);
        }
        kterm(gpu::device{g}, *it, alpha, gpu_t1[g].data(), b, ypntr);
      }

      icurrent += it->num_chain;

      term_found = true; // something got computed above
      b = 1; // next iteration appends on y
    }

    // handle the case when a GPU has no terms
    if (not term_found) {
      if (g == 0 and beta != 0) { // main GPU is expected to scale y
        compute->scal(num_entries, beta, ypntr);
      } else {
        // either scale by zero or no terms, so set to zero
        compute->fill_zeros(num_entries, ypntr);
      }
    }
    compute->device_synchronize();
  }

  // #ifdef ASGARD_GPU_MEMGREEDY
  // std::cout << " memory used: " << grid.used_xy_ram() << "MB\n";
  // #endif

  // collect the data across the GPUs
  for (int g = 1; g < num_gpus; g++) {
    gpu::mcopy(num_entries, gpu::device{g}, gpu_y[g].data(), gpu::device{0}, gpu_t1[0].data());
    if constexpr (mode == compute_mode::cpu) {
      compute->axpy(num_entries, gpu_t1[0].data(), gpu_y[0].data());
    } else {
      if constexpr (using_gpu_vectors)
        compute->axpy(num_entries, gpu_t1[0].data(), y.data());
      else
        compute->axpy(num_entries, gpu_t1[0].data(), y);
    }
  }

  if constexpr (mode == compute_mode::cpu) {// send back to the CPU
    gpu_y[0].copy_to_host(y);
  }
}
#endif

template<typename P>
void term_manager<P>::make_jacobi(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    std::vector<P> &y) const
{
  int const block_size      = fm::ipow(basis.pdof, grid.num_dims());
  int64_t const num_entries = block_size * grid.num_indexes();

  if (y.size() == 0)
    y.resize(num_entries);
  else {
    y.resize(num_entries);
    std::fill(y.begin(), y.end(), P{0});
  }

  kwork.w1.resize(num_entries);

  int icurrent   = (gid == -1) ? 0                              : term_groups[gid].begin();
  int const iend = (gid == -1) ? static_cast<int>(terms.size()) : term_groups[gid].end();
  while (icurrent < iend)
  {
    auto it = terms.begin() + icurrent;

    #ifdef ASGARD_USE_MPI
    if (not resources.owns(it->rec)) {
      icurrent += it->num_chain;
      continue;
    }
    #endif

    if (it->num_chain == 1) {
      kron_diag<data_mode::increment>(grid, conns, *it, block_size, y);

      icurrent++;
    } else {
      // dealing with a chain
      int const num_chain = it->num_chain;

      std::fill(kwork.w1.begin(), kwork.w1.end(), P{0});

      kron_diag<data_mode::increment>(grid, conns, *(it + num_chain - 1),
                                      block_size, kwork.w1);

      for (int i = num_chain - 2; i >= 0; --i) {
        kron_diag<data_mode::multiply>(grid, conns, *(it + i),
                                       block_size, kwork.w1);
      }
ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < num_entries; i++)
        y[i] += kwork.w1[i];

      icurrent += num_chain;
    }
  }
}

template<typename P>
template<data_mode mode>
void term_manager<P>::kron_diag(
    sparse_grid const &grid, connection_patterns const &conn,
    term_entry<P> const &tme, int const block_size, std::vector<P> &y) const
{
  static_assert(mode == data_mode::increment or mode == data_mode::multiply);

#pragma omp parallel
  {
    std::array<P const *, max_num_dimensions> amats;

#pragma omp for
    for (int i = 0; i < grid.num_indexes(); i++) {
      for (int d : iindexof(num_dims))
        if (tme.coeffs[d].empty())
          amats[d] = nullptr;
        else
          amats[d] = tme.coeffs[d][conn[tme.coeffs[d]].row_diag(grid[i][d])];

      for (int t : iindexof(block_size)) {
        P a = 1;
        int tt = i;
        for (int d = num_dims - 1; d >= 0; --d)
        {
          if (amats[d] != nullptr) {
            int const rc = tt % basis.pdof;
            a *= amats[d][rc * basis.pdof + rc];
          }
          tt /= basis.pdof;
        }
        if constexpr (mode == data_mode::increment)
          y[i * block_size + t] += a;
        else if constexpr (mode == data_mode::multiply)
          y[i * block_size + t] *= a;
      }
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct term_manager<double>;

template void term_manager<double>::kron_diag<data_mode::increment>(
    sparse_grid const &, connection_patterns const &,
    term_entry<double> const &, int const, std::vector<double> &) const;
template void term_manager<double>::kron_diag<data_mode::multiply>(
    sparse_grid const &, connection_patterns const &,
    term_entry<double> const &, int const, std::vector<double> &) const;

template void term_manager<double>::apply_tmpl<std::vector<double> const &, std::vector<double> &>(
    int, sparse_grid const &, connection_patterns const &, double,
    std::vector<double> const &, double, std::vector<double> &) const;
template void term_manager<double>::apply_tmpl<double const[], double[]>(
    int, sparse_grid const &, connection_patterns const &, double,
    double const[], double, double[]) const;

#ifdef ASGARD_USE_GPU
template void term_manager<double>::apply_tmpl_gpu<std::vector<double> const &, std::vector<double> &, compute_mode::cpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    double alpha, std::vector<double> const &x, double beta, std::vector<double> &y) const;
template void term_manager<double>::apply_tmpl_gpu<double const[], double[], compute_mode::cpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    double alpha, double const x[], double beta, double y[]) const;
template void term_manager<double>::apply_tmpl_gpu<gpu::vector<double> const &, gpu::vector<double> &, compute_mode::gpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    double alpha, gpu::vector<double> const &x, double beta, gpu::vector<double> &y) const;
template void term_manager<double>::apply_tmpl_gpu<double const[], double[], compute_mode::gpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    double alpha, double const x[], double beta, double y[]) const;
#endif

#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct term_manager<float>;

template void term_manager<float>::kron_diag<data_mode::increment>(
    sparse_grid const &, connection_patterns const &,
    term_entry<float> const &, int const, std::vector<float> &) const;
template void term_manager<float>::kron_diag<data_mode::multiply>(
    sparse_grid const &, connection_patterns const &,
    term_entry<float> const &, int const, std::vector<float> &) const;

template void term_manager<float>::apply_tmpl<std::vector<float> const &, std::vector<float> &>(
    int, sparse_grid const &, connection_patterns const &, float,
    std::vector<float> const &, float, std::vector<float> &) const;
template void term_manager<float>::apply_tmpl<float const[], float[]>(
    int, sparse_grid const &, connection_patterns const &, float,
    float const[], float, float[]) const;

#ifdef ASGARD_USE_GPU
template void term_manager<float>::apply_tmpl_gpu<std::vector<float> const &, std::vector<float> &, compute_mode::cpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    float alpha, std::vector<float> const &x, float beta, std::vector<float> &y) const;
template void term_manager<float>::apply_tmpl_gpu<float const[], float[], compute_mode::cpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    float alpha, float const x[], float beta, float y[]) const;
template void term_manager<float>::apply_tmpl_gpu<gpu::vector<float> const &, gpu::vector<float> &, compute_mode::gpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    float alpha, gpu::vector<float> const &x, float beta, gpu::vector<float> &y) const;
template void term_manager<float>::apply_tmpl_gpu<float const[], float[], compute_mode::gpu>(
    int gid, sparse_grid const &grid, connection_patterns const &conns,
    float alpha, float const x[], float beta, float y[]) const;
#endif

#endif

}

