#pragma once

#include "asgard_interpolation.hpp"

namespace asgard
{

#ifndef KRON_MODE_GLOBAL
/*!
 * \brief Holds a list of matrices used for time-stepping.
 *
 * There are multiple types of matrices based on the time-stepping and the
 * different terms being used. Matrices are grouped in one object so they can go
 * as a set and reduce the number of matrix making.
 */
template<typename precision>
struct kron_operators
{
  //! \brief Makes a list of uninitialized matrices
  kron_operators(connection_patterns const *conn = nullptr,
                 verbosity_level verb_in = verbosity_level::high)
      : verbosity(verb_in), conn_(conn)
  {
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    load_stream = nullptr;
#endif
  }
  //! \brief Frees the matrix list and any cache vectors
  ~kron_operators()
  {
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    if (load_stream != nullptr)
    {
      auto status = cudaStreamDestroy(load_stream);
      expect(status == cudaSuccess);
    }
#endif
  }

  template<resource rec = resource::host>
  void apply(imex_flag entry, precision alpha, precision const x[], precision beta, precision y[]) const
  {
    apply<rec>(entry, 0, alpha, x, beta, y);
  }
  //! \brief Apply the given matrix entry
  template<resource rec = resource::host>
  void apply(imex_flag entry, precision time, precision alpha, precision const x[], precision beta, precision y[]) const
  {
    ignore(time);
    matrices[static_cast<int>(entry)].template apply<rec>(alpha, x, beta, y);
  }
  int64_t flops(imex_flag entry) const
  {
    return matrices[static_cast<int>(entry)].flops();
  }

  //! \brief Make the matrix for the given entry
  void make(imex_flag entry, PDE<precision> const &pde,
            coefficient_matrices<precision> &cmats,
            adapt::distributed_grid<precision> const &grid)
  {
    if (not mem_stats)
      mem_stats = compute_mem_usage(pde, grid, entry, spcache);

    int const ientry = static_cast<int>(entry);
    if (not matrices[ientry])
      matrices[ientry] = make_local_kronmult_matrix(
          pde, cmats, *conn_, grid, mem_stats, entry, spcache, verbosity);

#ifdef ASGARD_USE_CUDA
    if (matrices[ientry].input_size() != xdev.size())
    {
      xdev = fk::vector<precision, mem_type::owner, resource::device>();
      xdev = fk::vector<precision, mem_type::owner, resource::device>(
          matrices[ientry].input_size());
    }
    if (matrices[ientry].output_size() != ydev.size())
    {
      ydev = fk::vector<precision, mem_type::owner, resource::device>();
      ydev = fk::vector<precision, mem_type::owner, resource::device>(
          matrices[ientry].output_size());
    }
    matrices[ientry].set_workspace(xdev, ydev);
#endif
#ifdef ASGARD_USE_GPU_MEM_LIMIT
    if (mem_stats.kron_call == memory_usage::multi_calls)
    {
      // doing multiple calls, prepare streams and workspaces
      if (load_stream == nullptr)
        cudaStreamCreate(&load_stream);
      if (worka.size() < static_cast<int>(mem_stats.work_size))
      {
        worka = fk::vector<int, mem_type::owner, resource::device>();
        workb = fk::vector<int, mem_type::owner, resource::device>();
        worka = fk::vector<int, mem_type::owner, resource::device>(
            mem_stats.work_size);
        workb = fk::vector<int, mem_type::owner, resource::device>(
            mem_stats.work_size);
        if (matrices[ientry].is_dense())
        {
          irowa = fk::vector<int, mem_type::owner, resource::device>();
          irowb = fk::vector<int, mem_type::owner, resource::device>();
          icola = fk::vector<int, mem_type::owner, resource::device>();
          icolb = fk::vector<int, mem_type::owner, resource::device>();
          irowa = fk::vector<int, mem_type::owner, resource::device>(
              mem_stats.row_work_size);
          irowb = fk::vector<int, mem_type::owner, resource::device>(
              mem_stats.row_work_size);
          icola = fk::vector<int, mem_type::owner, resource::device>(
              mem_stats.row_work_size);
          icolb = fk::vector<int, mem_type::owner, resource::device>(
              mem_stats.row_work_size);
        }
      }
    }
    matrices[ientry].set_workspace_ooc(worka, workb, load_stream);
    matrices[ientry].set_workspace_ooc_sparse(irowa, irowb, icola, icolb);
#endif
  }
  /*!
   * \brief Either makes the matrix or if it exists, just updates only the
   *        coefficients
   */
  void reset_coefficients(imex_flag entry, PDE<precision> const &pde,
                          coefficient_matrices<precision> &cmats,
                          adapt::distributed_grid<precision> const &grid)
  {
    int const ientry = static_cast<int>(entry);
    if (not matrices[ientry])
      make(entry, pde, cmats, grid);
    else
      update_kronmult_coefficients(pde, cmats, *conn_, entry, spcache,
                                   matrices[ientry]);
  }

  //! \brief Clear all matrices
  void clear()
  {
    for (auto &matrix : matrices)
      if (matrix)
        matrix = local_kronmult_matrix<precision>();
    mem_stats.reset();
  }

  //! \brief Returns the preconditioner.
  template<resource rec>
  auto const &get_diagonal_preconditioner() const
  {
    if (matrices[static_cast<int>(imex_flag::imex_implicit)])
      return matrices[static_cast<int>(imex_flag::imex_implicit)].template get_diagonal_preconditioner<rec>();
    else
      return matrices[static_cast<int>(imex_flag::unspecified)].template get_diagonal_preconditioner<rec>();
  }

  vector2d<precision> const &get_inodes() const
  {
    throw std::runtime_error("interpolation/inodes cannot be used with local kronmult");
    static vector2d<precision> inodes;
    return inodes;
  }

  //! adjusts the verbosity level
  verbosity_level verbosity = verbosity_level::high;

private:
  //! \brief Holds the matrices
  std::array<local_kronmult_matrix<precision>, num_imex_variants> matrices;
  //! \brief Holds the 1d connection patterns
  connection_patterns const *conn_ = nullptr;

  //! \brief Cache holding the memory stats, limits bounds etc.
  memory_usage mem_stats;

  //! \brief Cache holding sparse parameters to avoid recomputing data
  kron_sparse_cache spcache;

#ifdef ASGARD_USE_CUDA
  //! \brief Work buffers for the input and output
  mutable fk::vector<precision, mem_type::owner, resource::device> xdev, ydev;
#endif
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  mutable fk::vector<int, mem_type::owner, resource::device> worka;
  mutable fk::vector<int, mem_type::owner, resource::device> workb;
  mutable fk::vector<int, mem_type::owner, resource::device> irowa;
  mutable fk::vector<int, mem_type::owner, resource::device> irowb;
  mutable fk::vector<int, mem_type::owner, resource::device> icola;
  mutable fk::vector<int, mem_type::owner, resource::device> icolb;
  cudaStream_t load_stream;
#endif
};
#endif

#ifdef KRON_MODE_GLOBAL

template<typename precision>
struct kron_operators
{
  kron_operators() {}

  kron_operators(connection_patterns const *conn,
                 verbosity_level verb_in = verbosity_level::high)
      : verbosity(verb_in), pde_(nullptr), conn_(conn)
  {}

  template<resource rec = resource::host>
  void apply(imex_flag entry, precision alpha, precision const x[], precision beta, precision y[]) const
  {
    apply<rec>(entry, precision{0}, alpha, x, beta, y);
  }

  //! \brief Apply the given matrix entry
  template<resource rec = resource::host>
  void apply(imex_flag entry, precision time, precision alpha, precision const x[], precision beta, precision y[]) const
  {
    // prep stage for the operator application
    // apply the beta parameter, all operations are incremental
    if (beta == 0)
      kronmult::set_buffer_to_zero<rec>(kglobal.num_active(), y);
    else
      lib_dispatch::scal<resource::host>(kglobal.num_active(), beta, y, 1);

    // if any work will be done, copy x into the padded workspace
    if (kglobal.is_active(entry) or interp)
      std::copy_n(x, kglobal.num_active(), workspace.x.begin());

    kglobal.template apply<rec>(entry, alpha, y);

    if (interp)
    {
      // using the fact that workspace.x contains the padded values of x
      // and workspace.y is extra scratch space that can be used here
      interp.get_nodal_values(kglobal, domain_scale, workspace.x, workspace.y);
      if (pde_->interp_nox())
      {
        pde_->interp_nox()(time, workspace.y, finterp);
      }
      else // must be interp_x
      {
        check_make_inodes();
        pde_->interp_x()(time, inodes, workspace.y, finterp);
      }
      interp.compute_hierarchical_coeffs(kglobal, finterp);
      interp.get_projection_coeffs(kglobal, finterp, workspace.y);
      alpha /= domain_scale;
      for (int64_t i = 0; i < kglobal.num_active(); i++)
        y[i] += alpha * workspace.y[i];
    }
  }

  int64_t flops(imex_flag entry) const
  {
    return kglobal.flops(entry);
  }

  //! \brief Make the matrix for the given entry
  void make(imex_flag entry, PDE<precision> const &pde,
            coefficient_matrices<precision> &cmats,
            adapt::distributed_grid<precision> const &grid)
  {
    if (pde_ == nullptr and pde.has_interp())
    {
      pde.get_domain_bounds(dmin, dslope);
      domain_scale = precision{1};
      for (int d = 0; d < pde.num_dims(); d++)
      {
        dslope[d] -= dmin[d];
        domain_scale *= dslope[d];
      }
      domain_scale = precision{1} / std::sqrt(domain_scale);

      pde_   = &pde;
      interp = interpolation(pde_->num_dims(), conn_->get(connect_1d::hierarchy::volume), &workspace);
    }
    if (not kglobal)
    {
      kglobal = make_block_global_kron_matrix(
          pde, grid, conn_->get(connect_1d::hierarchy::volume),
          conn_->get(connect_1d::hierarchy::full), &workspace, verbosity);
      set_specific_mode(pde, cmats, *conn_, grid, entry, kglobal);
      if (interp)
      {
        finterp.resize(workspace.x.size());
        inodes.clear();
      }
    }
    else if (not kglobal.specific_is_set(entry))
      set_specific_mode(pde, cmats, *conn_, grid, entry, kglobal);
  }

  /*!
   * \brief Either makes the matrix or if it exists, just updates only the
   *        coefficients
   */
  void reset_coefficients(imex_flag entry, PDE<precision> const &pde,
                          coefficient_matrices<precision> &cmats,
                          adapt::distributed_grid<precision> const &grid)
  {
    if (not kglobal)
      make(entry, pde, cmats, grid);
    else
      set_specific_mode(pde, cmats, *conn_, grid, entry, kglobal);
  }

  //! \brief Clear all matrices
  void clear()
  {
    if (kglobal)
      kglobal = block_global_kron_matrix<precision>();
  }

  //! \brief Returns the preconditioner.
  template<resource rec>
  auto const &get_diagonal_preconditioner() const
  {
    return kglobal.template get_diagonal_preconditioner<rec>();
  }

  /*!
   * \brief Returns the inteprolation nodes, uses the padded grid.
   *
   * Used for computing the initial conditions and exact solution.
   */
  vector2d<precision> const &get_inodes() const
  {
    rassert(!!interp, "get_inodes() requires enabled interpolation and made operators");
    check_make_inodes();
    return inodes;
  }

  /*!
   * \brief Convert to nodal values at the inodes, uses the padded grid.
   *
   * Gives the nodal values of the solution at the inodes.
   */
  std::vector<precision> get_nodals(precision const x[]) const
  {
    if (not interp or not kglobal)
      throw std::runtime_error("get_nodals() requires enabled interpolation and made operators");
    std::copy_n(x, kglobal.num_active(), workspace.x.begin());
    interp.get_nodal_values(kglobal, domain_scale, workspace.x, finterp);
    return finterp;
  }
  /*!
   * \brief Convert to projection from nodal values, uses the padded grid.
   *
   * Puts the output inside the container_type
   * (works with std::vector and fk::vector),
   * the size will be set to include any padding.
   */
  template<typename container_type> // std::vector or fk::vector
  void get_project(precision const nodal[], container_type &proj) const
  {
    if (not interp or not kglobal)
      throw std::runtime_error("get_nodals() requires enabled interpolation and made operators");
    proj.resize(workspace.x.size());
    std::copy_n(nodal, kglobal.num_active(), workspace.x.begin());
    interp.compute_hierarchical_coeffs(kglobal, workspace.x);
    interp.get_projection_coeffs(kglobal.get_cells(), kglobal.get_dsort(),
                                 precision{1} / domain_scale,
                                 workspace.x.data(), proj.data());
    proj.resize(kglobal.num_active());
  }

  //! adjusts the verbosity level
  verbosity_level verbosity = verbosity_level::high;

private:
  void check_make_inodes() const
  {
    if (inodes.empty())
    {
      inodes = interp.get_nodes(kglobal.get_cells());
      for (int64_t i = 0; i < inodes.num_strips(); i++)
      {
        for (int d = 0; d < pde_->num_dims(); d++)
          inodes[i][d] = dmin[d] + inodes[i][d] * dslope[d];
      }
    }
  }

  PDE<precision> const *pde_ = nullptr;
  precision domain_scale;
  std::array<precision, max_num_dimensions> dmin, dslope;
  connection_patterns const *conn_ = nullptr;

  block_global_kron_matrix<precision> kglobal;

  interpolation<precision> interp;
  mutable vector2d<precision> inodes;

  mutable kronmult::block_global_workspace<precision> workspace;
  mutable std::vector<precision> finterp; // scratch used only for interpolation
};

#endif

} // namespace asgard
