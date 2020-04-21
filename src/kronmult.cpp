#include "kronmult.hpp"

// TODO tidy these includes
#include "kronmult1_pbatched.hpp"
#include "kronmult2_pbatched.hpp"
#include "kronmult3_pbatched.hpp"
#include "kronmult4_pbatched.hpp"
#include "kronmult5_pbatched.hpp"
#include "kronmult6_pbatched.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif

#include <mutex>
#include <vector>

#include "connectivity.hpp"
#include "lib_dispatch.hpp"
#include "timer.hpp"
#include <limits.h>

namespace kronmult
{
// helper - calculate element coordinates -> operator matrix indices
inline void
get_indices(fk::vector<int> const &coords, int indices[], int const degree)
{
  assert(degree > 0);

  int const indices_size = coords.size() / 2;
  for (int i = 0; i < indices_size; ++i)
  {
    indices[i] = get_1d_index(coords(i), coords(i + indices_size)) * degree;
  }
}

// helper - call the right kronmult
// --------------------------------------------
// note  the input memory referenced by x_ptrs will be over-written
// --------------------------------------------

template<typename P>
void call_kronmult(int const n, P const operators[], P *x_ptrs[],
                   P *output_ptrs[], P *work_ptrs[], int const num_krons,
                   int const num_dims)
{
#ifdef USE_GPU
  {
    int constexpr warpsize    = 32;
    int constexpr nwarps      = 8;
    int constexpr num_threads = nwarps * warpsize;
    switch (num_dims)
    {
    case 1:
      kronmult1_pbatched<P><<<num_krons, num_threads>>>(
          n, operators, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 2:
      kronmult2_pbatched<P><<<num_krons, num_threads>>>(
          n, operators, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 3:
      kronmult3_pbatched<P><<<num_krons, num_threads>>>(
          n, operators, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 4:
      kronmult4_pbatched<P><<<num_krons, num_threads>>>(
          n, operators, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 5:
      kronmult5_pbatched<P><<<num_krons, num_threads>>>(
          n, operators, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 6:
      kronmult6_pbatched<P><<<num_krons, num_threads>>>(
          n, operators, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    default:
      assert(false);
    };

    // -------------------------------------------
    // note important to wait for kernel to finish
    // -------------------------------------------
    cudaError_t const istat = cudaDeviceSynchronize();
    assert(istat == cudaSuccess);
  }
#else

  {
    switch (num_dims)
    {
    case 1:
      kronmult1_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 2:
      kronmult2_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 3:
      kronmult3_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 4:
      kronmult4_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 5:
      kronmult5_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    case 6:
      kronmult6_pbatched<P>(n, operators, x_ptrs, output_ptrs, work_ptrs,
                            num_krons);
      break;
    default:
      assert(false);
    };
  }
#endif
}

// helper - copy between device matrices
template<typename P>
static void copy_matrix_on_device(P const *const source, int const source_lda,
                                  P *const dest, int const dest_lda,
                                  int const nrows, int const ncols)
{
#ifdef ASGARD_USE_CUDA
  auto const success =
      cudaMemcpy2D(dest, dest_lda * sizeof(P), source, source_lda * sizeof(P),
                   nrows * sizeof(P), ncols, cudaMemcpyDeviceToDevice);
  assert(success == 0);
#else
  for (int i = 0; i < nrows; ++i)
  {
    for (int j = 0; j < ncols; ++j)
    {
      dest[i + j * dest_lda] = source[i + j * source_lda];
    }
  }
#endif
}
template<typename P>
fk::vector<P, mem_type::owner, resource::device>
execute(PDE<P> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid,
        fk::vector<P, mem_type::owner, resource::device> const &x)
{
  static std::once_flag print_flag;

  auto const num_elements = my_subgrid.size();

  // FIXME code relies on uniform degree across dimensions
  auto const degree     = pde.get_dimensions()[0].get_degree();
  auto const deg_to_dim = static_cast<int>(std::pow(degree, pde.num_dims));

  // size of input/working space for kronmults - need 2
  auto const workspace_size = num_elements * deg_to_dim * pde.num_terms;
  // size of small deg*deg matrices for kronmults
  auto const operator_size =
      num_elements * std::pow(degree, 2) * pde.num_dims * pde.num_terms;
  // size of output
  auto const output_size = my_subgrid.nrows() * deg_to_dim;

  // for now, our vectors are indexed with a 32 bit int
  assert(workspace_size < INT_MAX);
  assert(operator_size < INT_MAX);
  assert(output_size < INT_MAX);

  std::call_once(print_flag, [&pde, workspace_size, operator_size,
                              output_size] {
    // FIXME assumes (with everything else) that coefficients are equally sized
    auto const coefficients_size_MB =
        get_MB<P>(static_cast<int64_t>(pde.get_coefficients(0, 0).size()) *
                  pde.num_terms * pde.num_dims);
    node_out() << "kron workspace size..." << '\n';
    node_out() << "coefficient size (existing allocation)..."
               << coefficients_size_MB << '\n';
    node_out() << "workspace allocation (MB): " << get_MB<P>(workspace_size * 2)
               << '\n';
    node_out() << "operator staging allocation (MB): "
               << get_MB<P>(operator_size) << '\n';

    node_out() << "output allocation (MB): " << get_MB<P>(output_size) << '\n';
  });

  // FIXME all of below will be default init'd to 0
  fk::vector<P, mem_type::owner, resource::device> operators(operator_size);
  fk::vector<P, mem_type::owner, resource::device> element_x(workspace_size);
  fk::vector<P, mem_type::owner, resource::device> element_work(workspace_size);
  fk::vector<P, mem_type::owner, resource::device> output(output_size);

  auto const total_kronmults = num_elements * pde.num_terms;

  std::vector<P *> input_ptrs(total_kronmults);
  std::vector<P *> work_ptrs(total_kronmults);
  std::vector<P *> output_ptrs(total_kronmults);

// loop over assigned elements, staging inputs and operators
#pragma omp parallel for
  for (auto i = my_subgrid.row_start; i <= my_subgrid.row_stop; ++i)
  {
    // calculate and store operator row indices for this element
    static int constexpr max_dims = 6;
    assert(pde.num_dims < max_dims);
    int operator_row[max_dims];
    fk::vector<int> const &row_coords = elem_table.get_coords(i);
    assert(row_coords.size() == pde.num_dims * 2);
    get_indices(row_coords, operator_row, degree);

    for (auto j = my_subgrid.col_start; j <= my_subgrid.col_stop; ++j)
    {
      // calculate and store operator col indices for this element
      int operator_col[max_dims];
      fk::vector<int> const &col_coords = elem_table.get_coords(j);
      assert(col_coords.size() == pde.num_dims * 2);
      get_indices(col_coords, operator_col, degree);

      // also prepare x_window all terms will use
      auto const x_start = my_subgrid.to_local_col(j) * deg_to_dim;
      fk::vector<P, mem_type::const_view, resource::device> x_window(
          x, x_start, x_start + deg_to_dim - 1);

      for (auto t = 0; t < pde.num_terms; ++t)
      {
        // get preallocated vector positions for this kronmult
        auto const num_kron =
            (i - my_subgrid.row_start) * my_subgrid.ncols() * pde.num_terms +
            (j - my_subgrid.col_start) * pde.num_terms + t;

        auto const operator_start = num_kron * degree * degree * pde.num_dims;

        // stage inputs FIXME may eventually have to remove views
        // auto const x_start = element_x.data(num_kron * deg_to_dim);
        // int n              = deg_to_dim;
        // int one            = 1; // sigh

        // copy_matrix_on_device(x.data(my_subgrid.to_local_col(j) *
        // deg_to_dim),
        //                      n, x_start, n, n, 1);
        // lib_dispatch::copy(&n, x.data(my_subgrid.to_local_col(j) *
        // deg_to_dim),
        //                 &one, x_start, &one);

        fk::vector<P, mem_type::view, resource::device> my_x(
            element_x, num_kron * deg_to_dim, (num_kron + 1) * deg_to_dim - 1);
        my_x                 = x_window;
        input_ptrs[num_kron] = my_x.data();

        // input_ptrs[num_kron] = x_start;
        // stage work/output
        work_ptrs[num_kron] = element_work.data(num_kron * deg_to_dim);
        output_ptrs[num_kron] =
            output.data(my_subgrid.to_local_row(i) * deg_to_dim);

        // stage operators FIXME may eventually have to remove views
        for (auto d = 0; d < pde.num_dims; ++d)
        {
          // auto const &coeff = pde.get_coefficients(t, d);
          // copy_matrix_on_device(
          //    coeff.data(operator_row[d], operator_col[d]), coeff.stride(),
          //    operators.data(degree * degree * d), degree, degree, degree);
          fk::matrix<P, mem_type::const_view, resource::device> const
              coefficient_window(pde.get_coefficients(t, d), operator_row[d],
                                 operator_row[d] + degree - 1, operator_col[d],
                                 operator_col[d] + degree - 1);
          fk::matrix<P, mem_type::view, resource::device> my_A(
              operators, degree, degree, operator_start + degree * degree * d);

          my_A = coefficient_window;
        }
      }
    }
  }

  timer::record.start("kronmult");
  call_kronmult(degree, operators.data(), input_ptrs.data(), output_ptrs.data(),
                work_ptrs.data(), total_kronmults, pde.num_dims);

  timer::record.stop("kronmult");
  return output;
}

template fk::vector<float, mem_type::owner, resource::device>
execute(PDE<float> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid,
        fk::vector<float, mem_type::owner, resource::device> const &x);

template fk::vector<double, mem_type::owner, resource::device>
execute(PDE<double> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid,
        fk::vector<double, mem_type::owner, resource::device> const &x);

} // namespace kronmult
