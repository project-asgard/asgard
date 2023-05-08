
#pragma once

#include "distribution.hpp"
#include "elements.hpp"
#include "pde.hpp"
#include "tensors.hpp"

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
template<typename precision, resource data_mode>
class kronmult_matrix
{
public:
  kronmult_matrix(int num_dimensions_, int kron_size_, int num_rows_, int num_columns_,
                  fk::vector<int, mem_type::const_view, resource::host> const &index_A,
                  fk::vector<precision, mem_type::const_view, resource::host> const &values_A)
    : num_dimensions(num_dimensions_), kron_size(kron_size_), num_rows(num_rows_),
      num_columns(num_columns_), num_terms(num_columns / num_rows),
      iA(index_A.size()), vA(values_A.size())
  {
    if constexpr (data_mode == resource::host){
        iA = index_A;
        vA = values_A;
    }else{
        iA = index_A.clone_onto_device();
        vA = index_A.clone_onto_device();
    }
  }
  kronmult_matrix(int num_dimensions_, int kron_size_, int num_rows_, int num_columns_,
                  fk::vector<int, mem_type::owner, resource::host> &&index_A,
                  fk::vector<precision, mem_type::owner, resource::host> &&values_A,
                  std::enable_if_t<data_mode == resource::host>* = nullptr)
    : num_dimensions(num_dimensions_), kron_size(kron_size_), num_rows(num_rows_),
      num_columns(num_columns_), num_terms(num_columns / num_rows),
      iA(std::move(index_A)), vA(std::move(values_A))
  {}

  //! \brief Computes y = alpha * kronmult_matrix + beta * y
  void apply(precision alpha, precision const x[], precision beta, precision y[]){
    kronmult::cpu_dense(num_dimensions, kron_size, num_rows, 1, iA.data(), vA.data(), alpha, x, beta, y);
  }

private:
  int num_dimensions;
  int kron_size; // i.e., n - size of the matrices
  int num_rows;
  int num_columns;
  int num_terms;

  // index of the matrices for each kronmult product
  fk::vector<int, mem_type::owner, data_mode> iA;
  // list of the operators
  fk::vector<precision, mem_type::owner, data_mode> vA;
};

} // namespace asgard
