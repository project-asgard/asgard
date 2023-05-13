#include "asgard_kronmult_matrix.hpp"
#include "batch.hpp"
#include "lib_dispatch.hpp"
#include "tools.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif

#include <cstdlib>
#include <limits.h>
#include <mutex>
#include <vector>

namespace asgard
{

template<typename precision>
kronmult_matrix<precision>
make_kronmult_dense(PDE<precision> const &pde, adapt::distributed_grid<precision> const &discretization,
                    options const &program_options, imex_flag const imex)
{
  // convert pde to kronmult dense matrix
  auto const &grid         = discretization.get_subgrid(get_rank());
  int const num_dimensions = pde.num_dims;
  int const kron_size      = pde.get_dimensions()[0].get_degree();
  int const num_terms      = pde.num_terms;
  int const num_rows       = grid.row_stop - grid.row_start + 1;

  int64_t lda = kron_size * fm::two_raised_to((program_options.do_adapt_levels) ? program_options.max_level : pde.max_level);

  int64_t osize = 0;
  std::vector<int64_t> dim_term_offset(num_terms * pde.num_dims + 1);
  for (int t = 0; t < num_terms; t++)
  {
    for (int d = 0; d < num_dimensions; d++)
    {
      dim_term_offset[t * num_dimensions + d] = osize;
      osize += pde.get_coefficients(t, d).size();
    }
  }

  fk::vector<int, mem_type::owner, resource::host> iA(num_rows * num_rows * num_terms * num_dimensions);
  fk::vector<precision, mem_type::owner, resource::host> vA(osize);

#ifdef ASGARD_USE_CUDA
  std::cout << "  kronmult dense matrix allocation (MB): "
            << get_MB<int>(iA.size()) + get_MB<precision>(vA.size()) << "\n";
#endif
  // will print the command to use for performance testing
  //std::cout << "./asgard_kronmult_benchmark " << num_dimensions << " " << kron_size
  //          << " " << num_rows << " " << num_terms
  //          << " " << osize / (kron_size * kron_size) << "\n";

  // load the matrices into a contiguous data-structure
  // keep the column major format
  // but stack on the columns so the new leading dimensions is kron_size
  precision *pA = vA.data();
  for (int t = 0; t < num_terms; t++)
  {
    for (int d = 0; d < num_dimensions; d++)
    {
      if (!program_options.use_imex_stepping ||
          (program_options.use_imex_stepping &&
           pde.get_terms()[t][d].flag == imex))
      {
        auto const &ops = pde.get_coefficients(t, d); // this is an fk::matrix
        int const num_ops = ops.nrows() / kron_size;

        // the matrices of the kron products are organized into blocks
        // of a large matrix, the matrix is square with size num-ops by kron-size
        // rearrange in a sequential way (by columns) to avoid the lda
        for (int ocol = 0; ocol < num_ops; ocol++)
          for (int orow = 0; orow < num_ops; orow++)
            for (int i=0; i<kron_size; i++)
              pA = std::copy_n(ops.data() + kron_size * orow + lda * (kron_size * ocol + i), kron_size, pA);
      }
      else
      {
        pA = std::fill_n(pA, pde.get_coefficients(t, d).size(), 0);
      }
    }
  }

  // compute the indexes for the matrices for the kron-products
  int const *const flattened_table = discretization.get_table().get_active_table().data();
  std::vector<int> oprow(num_dimensions);
  std::vector<int> opcol(num_dimensions);
  auto ia = iA.begin();
  for (int64_t row=grid.row_start; row < grid.row_stop+1; row++)
  {
    int const *const row_coords = flattened_table + 2 * num_dimensions * row;
    for(int i=0; i<num_dimensions; i++)
      oprow[i] = (row_coords[i] == 0) ? 0 : ((1 << (row_coords[i] - 1)) + row_coords[i + num_dimensions]);

    for (int64_t col=grid.col_start; col < grid.col_stop+1; col++)
    {
      int const *const col_coords = flattened_table + 2 * num_dimensions * col;
      for(int i=0; i<num_dimensions; i++)
        opcol[i] = (col_coords[i] == 0) ? 0 : ((1 << (col_coords[i] - 1)) + col_coords[i + num_dimensions]);

      for (int t = 0; t < num_terms; t++)
      {
        for (int d = 0; d < num_dimensions; d++)
        {
          int64_t const num_ops = pde.get_coefficients(t, d).nrows() / kron_size;
          *ia++ = dim_term_offset[t * num_dimensions + d] + (oprow[d] + opcol[d] * num_ops) * kron_size * kron_size;
        }
      }
    }
  }

#ifdef ASGARD_USE_CUDA
  // if using CUDA, copy the matrices onto the GPU
  return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows, num_terms, iA.clone_onto_device(), vA.clone_onto_device());
#else
  // if using the CPU, move the vectors into the matrix structure
  return kronmult_matrix<precision>(num_dimensions, kron_size, num_rows, num_terms, std::move(iA), std::move(vA));
#endif
}

template kronmult_matrix<float>
make_kronmult_dense<float>
(PDE<float> const&, adapt::distributed_grid<float> const&, options const&, imex_flag const);
template kronmult_matrix<double>
make_kronmult_dense<double>
(PDE<double> const&, adapt::distributed_grid<double> const&, options const&, imex_flag const);

}
