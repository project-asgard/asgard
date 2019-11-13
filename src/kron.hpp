/*
Comments assume the user is reading the lines of this file in order.

Problem relevant to functions in this file:

given a vector "x" of length "x_size" and list of matrices of arbitrary dimension
in "matrix": { m0, m1, ... , m_last }, calculate ( m0 kron m1 kron ... kron m_end ) * x

*/
#include "batch.hpp"
#include "fast_math.hpp"
#include "tensors.hpp"

#include <array>
#include <iostream>
#include <numeric>

/*
What it is:

Stores 3 equal length batches that contain information for a set of matrix multiplications.

How it is used:

Each multiplication is of the form: left[ i ] * right[ i ] = product[ i ]
each "left[ i ]" is a matrix formed from a subset of the elements of an input fk::vector
each "right[ i ]" is the same matrix
each "product[ i ]" is a subset of the elements of an output fk::vector
*/
template<typename P, resource resrc>
class batch_set
{
public:

  batch_set( int const left_num_entries, int const left_num_rows, int const left_num_cols,
             int const left_stride, bool const left_do_trans,

             int const right_num_entries, int const right_num_rows, int const right_num_cols,
             int const right_stride, bool const right_do_trans,

             int const product_num_entries,
             int const product_num_rows,
             int const product_num_cols,
             int const product_stride,
             bool const product_do_trans )
    :
    left( left_num_entries, left_num_rows, left_num_cols, left_stride, left_do_trans ),
    right( right_num_entries, right_num_rows, right_num_cols, right_stride, right_do_trans ),
    product( product_num_entries, product_num_rows, product_num_cols, product_stride, 
             product_do_trans )
  { return; }

  batch_set(batch<P, resrc> &&bs)
      : left(std::move(bs.left)), right(std::move(bs.right)),
        product(std::move(bs.product))
  { return; }

  batch<P, resrc> left;
  batch<P, resrc> right;
  batch<P, resrc> product;
};

/*
What is is:
Stores many batch sets where the output of each one is the input of the next one. The fk::vectors
in "workspace" alternate between input/output for each successive batch_set. The initial
input is in the vector "x". The final output will be one of the fk::vectors in "workspace".
*/
template<typename P, resource resrc>
class batch_job
{
public:
  batch_job(P const alpha, P const beta, int const workspace_len,
            fk::vector<P, mem_type::view, resrc> const &x, int const y_size);

  batch_job(batch_job<P, resrc> &&job) = default;

  void add_batch_set(batch_set<P, resrc> const &&bs);

  void swap_workspaces();

  fk::vector<P, mem_type::owner, resrc> const &get_input_workspace();

  fk::vector<P, mem_type::owner, resrc> const &get_output_workspace();

  std::vector<batch_set<P, resrc>> batches;

  /* alpha and beta arguments for BLAS gemm call */
  P const alpha;
  P const beta;

  /* length of output vector */
  int const y_size;

  /* alternating input and output of each stage */
  std::array<fk::vector<P, mem_type::owner, resrc>, 2> workspace;

private:
  /* "in" and "out" are state variables that indicate the index of the input and output vector,
     respectively, within "workspace", that the next batch_set will use */
  /* "in" alternates between 0 and 1 starting at 0. */
  int in;
  /* "out" alternates between 0 and 1 starting at 1. */
  int out;
};

/* Given a list of matrices in "matrix" and a vector "x", this function creates a "batch_job"
   object and populates it with "batch_set" objects. The result will be sent to
   "execute_batch_job". The dataflow delineated in the batch_job implements the problem */
template<typename P, resource resrc>
batch_job<P, resrc>
kron_batch(std::vector<fk::matrix<P, mem_type::view, resrc>> const &matrix,
           fk::vector<P, mem_type::view, resrc> const &x);

/* extern explicit instantiations */
extern template batch_job<float, resource::device>
kron_batch(std::vector<fk::matrix<float, mem_type::view, resource::device>> const &matrix,
           fk::vector<float, mem_type::view, resource::device> const &x);

extern template batch_job<float, resource::host>
kron_batch(std::vector<fk::matrix<float, mem_type::view, resource::host>> const &matrix,
           fk::vector<float, mem_type::view, resource::host> const &x);

extern template batch_job<double, resource::device>
kron_batch(std::vector<fk::matrix<double, mem_type::view, resource::device>> const &matrix,
           fk::vector<double, mem_type::view, resource::device> const &x);

extern template batch_job<double, resource::host>
kron_batch(std::vector<fk::matrix<double, mem_type::view, resource::host>> const &matrix,
           fk::vector<double, mem_type::view, resource::host> const &x);

/* explicit instantiations */
extern template class batch_job<double, resource::device >;
extern template class batch_job<double, resource::host>;
extern template class batch_job<float, resource::device >;
extern template class batch_job<float, resource::host>;

/* Given a batch_job class that describes the dataflow of the problem described at top of file,
   this function carries out the math */
template<typename P, resource resrc>
fk::vector<P, mem_type::owner, resrc>
execute_batch_job(batch_job<P, resrc> &job);

extern template
fk::vector< double, mem_type::owner, resource::device >
execute_batch_job< double, resource::device >
(batch_job<double, resource::device > &job);

extern template fk::vector< float, mem_type::owner, resource::device >
execute_batch_job< float, resource::device >
(batch_job<float, resource::device > &job);

extern template fk::vector< double, mem_type::owner, resource::host>
execute_batch_job< double, resource::host>
(batch_job<double, resource::host> &job);

extern template fk::vector< float, mem_type::owner, resource::host>
execute_batch_job< float, resource::host>
(batch_job<float, resource::host> &job);

/* Calculates necessary workspace length for the Kron algorithm. See .cpp file for more details */
template<typename P, resource resrc>
int calculate_workspace_len(
    std::vector<fk::matrix<P, mem_type::view, resrc>> const &matrix,
    int const x_size);

/* external explicit instantiations */
extern template int calculate_workspace_len(
    std::vector<fk::matrix<double, mem_type::view, resource::device >> const &matrix,
    int const x_size);

extern template int calculate_workspace_len(
    std::vector<fk::matrix<double, mem_type::view, resource::host >> const &matrix,
    int const x_size);

extern template int calculate_workspace_len(
    std::vector<fk::matrix<float, mem_type::view, resource::device >> const &matrix,
    int const x_size);

extern template int calculate_workspace_len(
    std::vector<fk::matrix<float, mem_type::view, resource::host >> const &matrix,
    int const x_size);
