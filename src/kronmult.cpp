#include "kronmult.hpp"
#include "kronmult_cuda.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif

#include <cstdlib>
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

template<typename P>
fk::vector<P, mem_type::owner, resource::host>
execute(PDE<P> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid,
        fk::vector<P, mem_type::owner, resource::host> const &x)
{
  static std::once_flag print_flag;

  auto const num_elements = my_subgrid.size();

  // FIXME code relies on uniform degree across dimensions
  auto const degree     = pde.get_dimensions()[0].get_degree();
  auto const deg_to_dim = static_cast<int>(std::pow(degree, pde.num_dims));

  // size of input/working space for kronmults - need 2
  auto const workspace_size = num_elements * deg_to_dim * pde.num_terms;

  // size of output
  auto const output_size = my_subgrid.nrows() * deg_to_dim;

  // for now, our vectors are indexed with a 32 bit int
  assert(workspace_size < INT_MAX);
  assert(output_size < INT_MAX);

  std::call_once(print_flag, [&pde, workspace_size, output_size] {
    // FIXME assumes (with everything else) that coefficients are equally sized
    auto const coefficients_size_MB =
        get_MB<P>(static_cast<int64_t>(pde.get_coefficients(0, 0).size()) *
                  pde.num_terms * pde.num_dims);
    node_out() << "kron workspace size..." << '\n';
    node_out() << "coefficient size (MB, existing allocation): "
               << coefficients_size_MB << '\n';
    node_out() << "workspace allocation (MB): " << get_MB<P>(workspace_size * 2)
               << '\n';
    node_out() << "output allocation (MB): " << get_MB<P>(output_size) << '\n';
  });

  P *element_x;
  P *element_work;
  fk::allocate_device(element_x, workspace_size);
  fk::allocate_device(element_work, workspace_size);

  // build x workspace with the desired pattern on CPU
 // auto const stage_x =
 //     static_cast<P *>(std::malloc(workspace_size * sizeof(P)));
  // stage x vector in writable regions for each element
  //#pragma omp parallel for
  
  fk::vector<P, mem_type::owner, resource::device> const x_d(x.clone_onto_device());
  for (auto i = 0; i < my_subgrid.nrows() * pde.num_terms; ++i)
  {
     fk::copy_on_device(element_x + i * x_d.size(), x_d.data(), x_d.size());
  }
  // now, single copy to GPU
  //fk::copy_to_device(element_x, stage_x, workspace_size);
  //std::free(stage_x);

  fk::vector<P, mem_type::owner, resource::device> output(output_size);

  // loop over assigned elements, staging inputs and operators
  auto const total_kronmults = num_elements * pde.num_terms;

  std::vector<P *> input_ptrs(total_kronmults);
  std::vector<P *> work_ptrs(total_kronmults);
  std::vector<P *> output_ptrs(total_kronmults);
  std::vector<P *> operator_ptrs(total_kronmults * pde.num_dims);

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

      // auto const x_start = element_x.data(my_subgrid.to_local_row(i) *
      // pde.num_terms * x.size()
      // + my_subgrid.to_local_col(j) * deg_to_dim);

      auto const x_start =
          element_x + (my_subgrid.to_local_row(i) * pde.num_terms * x.size() +
                       my_subgrid.to_local_col(j) * deg_to_dim);
      for (auto t = 0; t < pde.num_terms; ++t)
      {
        // get preallocated vector positions for this kronmult
        auto const num_kron =
            my_subgrid.to_local_row(i) * my_subgrid.ncols() * pde.num_terms +
            my_subgrid.to_local_col(j) * pde.num_terms + t;

        // point to inputs
        input_ptrs[num_kron] = x_start + t * x.size();
        // point to work/output
        work_ptrs[num_kron] =
            // element_work.data(num_kron * deg_to_dim);
            element_work + num_kron * deg_to_dim;
        output_ptrs[num_kron] =
            output.data(my_subgrid.to_local_row(i) * deg_to_dim);
        // output + (my_subgrid.to_local_row(i) * deg_to_dim);
        // point to operators
        auto const operator_start = num_kron * pde.num_dims;
        for (auto d = 0; d < pde.num_dims; ++d)
        {
          auto const &coeff = pde.get_coefficients(t, d);
          operator_ptrs[operator_start + d] =
              coeff.data(operator_row[d], operator_col[d]);
        }
      }
    }
  }

  // FIXME assume all operators same size
  auto const lda = pde.get_coefficients(0, 0)
                       .stride(); // leading dimension of coefficient matrices
  timer::record.start("kronmult");
  call_kronmult(degree, input_ptrs.data(), output_ptrs.data(), work_ptrs.data(),
                operator_ptrs.data(), lda, total_kronmults, pde.num_dims);
  timer::record.stop("kronmult");

  fk::delete_device(element_x);
  fk::delete_device(element_work);

  return output.clone_onto_host();
}

template fk::vector<float, mem_type::owner, resource::host>
execute(PDE<float> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid,
        fk::vector<float, mem_type::owner, resource::host> const &x);

template fk::vector<double, mem_type::owner, resource::host>
execute(PDE<double> const &pde, element_table const &elem_table,
        element_subgrid const &my_subgrid,
        fk::vector<double, mem_type::owner, resource::host> const &x);

} // namespace kronmult
