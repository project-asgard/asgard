/*
Comments assume the user is reading the lines of this file in order.

Problem relevant to functions in this file:

given a vector "x" of length "x_size" and list of matrices of arbitrary
dimension in "matrix": { m0, m1, ... , m_last }, calculate ( m0 kron m1 kron ...
kron m_end ) * x

*/
#include "batch.hpp"
#include "fast_math.hpp"
#include "tensors.hpp"

#include <array>
#include <iostream>
#include <numeric>

template<typename P, resource resrc>
class batch_chain
{
public:
  /* allocates batches */
  batch_chain(std::vector<fk::matrix<P, mem_type::view, resrc>> const &matrix,
              fk::vector<P, mem_type::view, resrc> const &x,
              std::array<fk::vector<P, mem_type::view, resrc>, 2> &workspace,
              fk::vector<P, mem_type::view, resrc> &final_output);

  /* assigns data to allocated batch chains */
  void initialize_batch_chain();

  void execute_batch_chain();

private:
  fk::vector<P, mem_type::view, resrc> const &get_input_workspace();

  fk::vector<P, mem_type::view, resrc> const &get_output_workspace();

  void swap_workspaces();

  std::vector<fk::matrix<P, mem_type::view, resrc>> const &matrix;

  fk::vector<P, mem_type::view, resrc> const &x;

  std::array<fk::vector<P, mem_type::view, resrc>, 2> workspace;

  fk::vector<P, mem_type::view, resrc> &final_output;

  int in;

  int out;

  std::vector<batch<P, resrc>> left;

  std::vector<batch<P, resrc>> right;

  std::vector<batch<P, resrc>> product;
};

extern template class batch_chain<double, resource::device>;
extern template class batch_chain<double, resource::host>;
extern template class batch_chain<float, resource::device>;
extern template class batch_chain<float, resource::host>;

/* Calculates necessary workspace length for the Kron algorithm. See .cpp file
 * for more details */
template<typename P, resource resrc>
int calculate_workspace_len(
    std::vector<fk::matrix<P, mem_type::view, resrc>> const &matrix,
    int const x_size);

/* external explicit instantiations */
extern template int calculate_workspace_len(
    std::vector<fk::matrix<double, mem_type::view, resource::device>> const
        &matrix,
    int const x_size);

extern template int calculate_workspace_len(
    std::vector<fk::matrix<double, mem_type::view, resource::host>> const
        &matrix,
    int const x_size);

extern template int calculate_workspace_len(
    std::vector<fk::matrix<float, mem_type::view, resource::device>> const
        &matrix,
    int const x_size);

extern template int calculate_workspace_len(
    std::vector<fk::matrix<float, mem_type::view, resource::host>> const
        &matrix,
    int const x_size);
