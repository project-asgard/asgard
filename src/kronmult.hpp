
#pragma once

#include "distribution.hpp"
#include "elements.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "adapt.hpp"

#include "./device/asgard_kronmult.hpp"

// this interface between the low level kernels in src/device
// and the higher level data-structures

namespace asgard
{

template<typename P, mem_type mem, resource resrc>
using vector = fk::vector<P, mem, resrc>;

namespace kronmult{
// execute one subgrid by breaking into smaller subgrids to
// fit workspace limit MB
template<typename P>
fk::vector<P, mem_type::owner, resource::host>
execute(PDE<P> const &pde, elements::table const &elem_table,
        options const &program_options, element_subgrid const &my_subgrid,
        fk::vector<P, mem_type::owner, resource::host> const &x,
        imex_flag const imex = imex_flag::unspecified);

} // namespace kronmult

/*!
 * \brief Contains persistent data for a kronmult operation.
 *
 * Holds the data for one batch of kronmult operations
 * so multiple calls can be made without reloading data
 * onto the device or recomputing pointers.
 * Especially useful for iterative operations, such as GMRES.
 *
 * \tparam precision is double or float
 * \tparam data_mode is resource::host (CPU) or resource::device (GPU)
 *
 * This is the dense implementation, assuming that the discretization
 * uses a dense matrix with number of columns equal to the number of
 * operator terms times the number of rows.
 * Each row/column entry corresponds to a Kronecker product.
 */
template<typename precision>
class kronmult_matrix
{
public:
  kronmult_matrix()
    : num_dimensions_(0), kron_size_(0), num_rows_(0),
      num_columns_(0), num_terms_(0), input_size_(0), flops_(0)
  {}
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_columns,
                  fk::vector<int, mem_type::const_view, resource::host> const &index_A,
                  fk::vector<precision, mem_type::const_view, resource::host> const &values_A)
    : num_dimensions_(num_dimensions), kron_size_(kron_size), num_rows_(num_rows),
      num_columns_(num_columns), num_terms_(num_columns / num_rows), input_size_(1),
      iA(index_A.size()), vA(values_A.size())
  {
    for(int d=0; d<num_dimensions_; d++)
      input_size_ *= kron_size_;

    if constexpr (data_mode == resource::host){
        iA = index_A;
        vA = values_A;
    }else{
        iA = index_A.clone_onto_device();
        vA = index_A.clone_onto_device();
    }

    compute_flops();
  }
  template<resource input_mode>
  kronmult_matrix(int num_dimensions, int kron_size, int num_rows, int num_columns,
                  fk::vector<int, mem_type::owner, input_mode> &&index_A,
                  fk::vector<precision, mem_type::owner, input_mode> &&values_A)
    : num_dimensions_(num_dimensions), kron_size_(kron_size), num_rows_(num_rows),
      num_columns_(num_columns), num_terms_(num_columns / num_rows), input_size_(1),
      iA(std::move(index_A)), vA(std::move(values_A))
  {
#ifdef ASGARD_USE_CUDA
    static_assert(input_mode == resource::device, "the GPU is enabled, so r-value inputs must have resource::device");
#else
    static_assert(input_mode == resource::host, "the GPU is disabled, so r-value inputs must have resource::host");
#endif
    for(int d=0; d<num_dimensions_; d++)
      input_size_ *= kron_size_;

    compute_flops();
  }

  //! \brief Computes y = alpha * kronmult_matrix * x + beta * y
  void apply(precision alpha, precision const x[], precision beta, precision y[]) const
  {
#ifdef ASGARD_USE_CUDA
#else
    kronmult::cpu_dense(num_dimensions_, kron_size_, num_rows_, num_terms_, iA.data(), vA.data(), alpha, x, beta, y);
#endif
  }

  int input_size() const
  {
    return input_size_;
  }

  operator bool () const
  {
      return (num_dimensions_ > 0);
  }

  int64_t flops() const
  {
    return flops_;
  }

protected:
  void compute_flops()
  {
    flops_ = kron_size_;
    for(int i=0; i<num_dimensions_; i++)
      flops_ *= kron_size_;
    flops_ *= 2 * num_dimensions_ * num_rows_ * num_rows_ * num_terms_;
  }

private:
  int num_dimensions_;
  int kron_size_; // i.e., n - size of the matrices
  int num_rows_;
  int num_columns_;
  int num_terms_;
  int input_size_;
  int64_t flops_;

#ifdef ASGARD_USE_CUDA
  static constexpr resource data_mode = resource::device;
#else
  static constexpr resource data_mode = resource::host;
#endif
  // index of the matrices for each kronmult product
  fk::vector<int, mem_type::owner, data_mode> iA;
  // list of the operators
  fk::vector<precision, mem_type::owner, data_mode> vA;
};

template<typename P>
kronmult_matrix<P>
make_kronmult_dense(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
                    options const &program_options,
                    imex_flag const imex = imex_flag::unspecified);

} // namespace asgard
