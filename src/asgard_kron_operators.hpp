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
  kron_operators()
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

  //! \brief Apply the given matrix entry
  template<resource rec = resource::host>
  void apply(imex_flag entry, precision alpha, precision const x[], precision beta, precision y[]) const
  {
    matrices[static_cast<int>(entry)].template apply<rec>(alpha, x, beta, y);
  }
  int64_t flops(imex_flag entry) const
  {
    return matrices[static_cast<int>(entry)].flops();
  }

  //! \brief Make the matrix for the given entry
  void make(imex_flag entry, PDE<precision> const &pde,
            adapt::distributed_grid<precision> const &grid, options const &opts)
  {
    if (not mem_stats)
      mem_stats = compute_mem_usage(pde, grid, opts, entry, spcache);

    int const ientry = static_cast<int>(entry);
    if (not matrices[ientry])
      matrices[ientry] = make_local_kronmult_matrix(
          pde, grid, opts, mem_stats, entry, spcache);

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
                          adapt::distributed_grid<precision> const &grid,
                          options const &opts)
  {
    int const ientry = static_cast<int>(entry);
    if (not matrices[ientry])
      make(entry, pde, grid, opts);
    else
      update_kronmult_coefficients(pde, opts, entry, spcache,
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

private:
  //! \brief Holds the matrices
  std::array<local_kronmult_matrix<precision>, num_imex_variants> matrices;

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

#ifdef KRON_MODE_GLOBAL_BLOCK

template<typename precision>
struct kron_operators
{
  //! \brief Apply the given matrix entry
  template<resource rec = resource::host>
  void apply(imex_flag entry, precision alpha, precision const x[], precision beta, precision y[]) const
  {
    kglobal.template apply<rec>(entry, alpha, x, beta, y);
  }

  int64_t flops(imex_flag entry) const
  {
    return kglobal.flops(entry);
  }

  //! \brief Make the matrix for the given entry
  void make(imex_flag entry, PDE<precision> const &pde,
            adapt::distributed_grid<precision> const &grid, options const &opts)
  {
    if (not kglobal)
    {
      kglobal = make_block_global_kron_matrix(pde, grid, opts, &workspace);
      set_specific_mode(pde, grid, opts, entry, kglobal);
    }
    else if (not kglobal.specific_is_set(entry))
      set_specific_mode(pde, grid, opts, entry, kglobal);
  }

  /*!
   * \brief Either makes the matrix or if it exists, just updates only the
   *        coefficients
   */
  void reset_coefficients(imex_flag entry, PDE<precision> const &pde,
                          adapt::distributed_grid<precision> const &grid,
                          options const &opts)
  {
    if (not kglobal)
      make(entry, pde, grid, opts);
    else
      set_specific_mode(pde, grid, opts, entry, kglobal);
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

private:
  block_global_kron_matrix<precision> kglobal;

  interpolation<precision> interp;

  kronmult::block_global_workspace<precision> workspace;
};

#endif

}
