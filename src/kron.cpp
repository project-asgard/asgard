/*
Comments assume the user is reading the lines of this file in order.

Problem relevant to functions in this file:

given a vector "x" of length "x_size" and list of matrices of arbitrary
dimension in "matrix": { m0, m1, ... , m_last }, calculate ( m0 kron m1 kron ...
kron m_end ) * x

*/
#include "kron.hpp"

/*
For a list of "n" matrices in "matrix" parameter, the Kron algorithm in this
file will go through "n" rounds of batched matrix multiplications. The output of
each round becomes the input of the next round. Two workspaces are thus
sufficient, if each of them is as large as the largest output from any round.
This function calculates the largest output of any round.

For a Kronecker product m0 * m1 * m2 * x = m3:
matrix ~ ( rows, columns )
m0 ~ (a, b)
m1 ~ ( c, d )
m2 ~ ( e, f )
m3 ~ ( g, h )
x ~ ( b*d*f*h, 1 )
m3 ~ ( a*c*e*g, 1 )

Algorithm completes in 3 rounds.
Initial space needed: size of "x" vector: b*d*f*h
Round 0 output size: b*d*f*g
Round 1 output size: b*d*e*g
Round 2 output size: b*c*e*g
Round 3 ourput size: a*c*e*g

Max space needed is maximum of the above sizes.

Improvement idea: Each stage alternates between 2 workspaces,
so the first workspace needs to be the maximum of round 0 and 2, while the other
needs to be the max of round 1 and 3. As it stands now, each workspace is the
size of the max of all rounds.
*/
template<typename P, resource resrc>
int calculate_workspace_len(
    std::vector<fk::matrix<P, mem_type::view, resrc>> const &matrix,
    int const x_size)
{
  int greatest = x_size;
  int r_prod   = 1;
  int c_prod   = 1;

  typename std::vector<
      fk::matrix<P, mem_type::view, resrc>>::const_reverse_iterator iter;

  for (iter = matrix.rbegin(); iter != matrix.rend(); ++iter)
  {
    c_prod *= iter->ncols();
    assert(c_prod > 0);

    r_prod *= iter->nrows();

    int const size = x_size / c_prod * r_prod;
    if (size > greatest)
      greatest = size;
  }

  return greatest;
}

/* explicit instantions of function above */
template int calculate_workspace_len<double, resource::device>(
    std::vector<fk::matrix<double, mem_type::view, resource::device>> const
        &matrix,
    int const x_size);

template int calculate_workspace_len<double, resource::host>(
    std::vector<fk::matrix<double, mem_type::view, resource::host>> const
        &matrix,
    int const x_size);

template int calculate_workspace_len<float, resource::device>(
    std::vector<fk::matrix<float, mem_type::view, resource::device>> const
        &matrix,
    int const x_size);

template int calculate_workspace_len<float, resource::host>(
    std::vector<fk::matrix<float, mem_type::view, resource::host>> const
        &matrix,
    int const x_size);

/* delineate the dataflow for the algorithm that implements the particular
   problem described in the comment at the top of the file. The algorithm
   proceeds in stages, with a stage for each matrix in the list. The input for
   each stage is the output of the previous. The initial input is the vector
   "x". The output of each stage is a vector. The output of the final stage is
   the solution to the problem. */
template<typename P, resource resrc>
batch_chain<P, resrc>::batch_chain(
    std::vector<fk::matrix<P, mem_type::view, resrc>> const &matrix,
    fk::vector<P, mem_type::view, resrc> const &x,
    std::array<fk::vector<P, mem_type::view, resrc>, 2> &workspace,
    fk::vector<P, mem_type::view, resrc> &final_output)
    : matrix(matrix), x(x),
      workspace(
          workspace), // I assume this makes a copy of the views - low overhead
      final_output(final_output), in(0), out(1)
{
  /* validation */
  assert(matrix.size() > 0);

  /* ensure "x" is correct size - should be the product of all the matrices
     respective number of columns */
  assert(x.size() == std::accumulate(matrix.begin(), matrix.end(), 1,
                                     [](int const i, auto const &m) {
                                       return i * m.ncols();
                                     }));

  /* ensure the workspaces are big enough for the problem */
  int const workspace_len = calculate_workspace_len(matrix, x.size());
  assert(workspace[0].size() >= workspace_len);
  assert(workspace[1].size() >= workspace_len);

  /*
    The algorithm iterates over each matrix in "matrix" in reverse order,
    creates a batch_set object for each matrix, and assigns input and output for
    each multiplication in that batch_set. The first and last iterations are
    handled differently than the others, so the loop is unrolled accordingly.
  */
  {
    /* only one large matrix multiply is needed for this first iteration. We
       want to break up "x" into consecutive subvectors of length iter->ncols().
       Each of these subvectors will be transformed into a vector of length
       iter->nrows() by multiplying it by the last matrix in the list. These
       results are concatenated to produce the output of the first stage. The
       easiest way to implement the above process is with a large matrix
       multiply: */

    /* define the sizes of the operands */
    int const rows = x.size() / matrix.back().ncols();

    left.emplace_back(1, matrix.back().nrows(), matrix.back().ncols(),
                      matrix.back().stride(), false);
    right.emplace_back(1, matrix.back().ncols(), rows, matrix.back().ncols(),
                       false);
    product.emplace_back(1, matrix.back().nrows(), rows, matrix.back().nrows(),
                         false);

    /* assign the data to the batches */
    fk::matrix<P, mem_type::view, resrc> input(x, matrix.back().ncols(), rows,
                                               0);

    /* If there is only 1 iteration, write immediately to final_output */
    fk::vector<P, mem_type::view, resrc> destination(get_output_workspace());
    fk::matrix<P, mem_type::view, resrc> output(
        (matrix.size() > 1 ? destination : final_output), matrix.back().nrows(),
        rows, 0);

    right.back().assign_entry(input, 0);
    left.back().assign_entry(
        fk::matrix<P, mem_type::view, resrc>(matrix.back()), 0);
    product.back().assign_entry(output, 0);

    /* output and input space will switch roles for the next iteration */
    swap_workspaces();
  }

  /* The remaining iterations are a general case of the first one */

  /* distance in between first element of output subvectors above */
  int stride = matrix.back().nrows();
  /* initial value for the size of the input vector for the stage to follow:
     each iter->ncols()
     length consecutive subvector was previously replaced by one of length
     iter->nrows() */
  int v_size = x.size() / matrix.back().ncols() * matrix.back().nrows();

  /* second to last iteration must be unrolled */
  for (int i = matrix.size() - 2; i > 0; --i)
  {
    /* total number of input matrices encoded in linear input memory */
    int const n_gemms = v_size / stride / matrix[i].ncols();

    /*
      left operand's data comes from a linear input array. The transposes ensure
      that the input can be interpreted in column major format and that the
      output will be implicitly stored that way.
    */
    left.emplace_back(n_gemms, stride, matrix[i].ncols(), stride, false);
    right.emplace_back(n_gemms, matrix[i].nrows(), matrix[i].ncols(),
                       matrix[i].stride(), true);
    product.emplace_back(n_gemms, stride, matrix[i].nrows(), stride, false);

    /* assign actual data to the batches */
    for (int j = 0; j < n_gemms; ++j)
    {
      fk::matrix<P, mem_type::view, resrc> input(
          get_input_workspace(), stride, matrix[i].ncols(),
          j * stride * matrix[i].ncols());

      fk::matrix<P, mem_type::view, resrc> output(
          get_output_workspace(), stride, matrix[i].nrows(),
          j * stride * matrix[i].nrows());

      left.back().assign_entry(input, j);
      right.back().assign_entry(fk::matrix<P, mem_type::view, resrc>(matrix[i]),
                                j);
      product.back().assign_entry(output, j);
    }

    /* output and input space will switch roles for the next iteration */
    swap_workspaces();

    /* update variables for next iteration */
    v_size = n_gemms * stride * matrix[i].nrows();
    stride *= matrix[i].nrows();
  }

  /* final loop iteration - output goes into output space instead of workspace,
     iterator already points to correct element because of loop above */
  if (matrix.size() > 1)
  {
    int const n_gemms = v_size / stride / matrix.front().ncols();

    left.emplace_back(n_gemms, stride, matrix.front().ncols(), stride, false);
    right.emplace_back(n_gemms, matrix.front().nrows(), matrix.front().ncols(),
                       matrix.front().stride(), true);
    product.emplace_back(n_gemms, stride, matrix.front().nrows(), stride,
                         false);

    for (int j = 0; j < n_gemms; ++j)
    {
      fk::matrix<P, mem_type::view, resrc> input(
          get_input_workspace(), stride, matrix.front().ncols(),
          j * stride * matrix.front().ncols());

      fk::matrix<P, mem_type::view, resrc> output(
          final_output, stride, matrix.front().nrows(),
          j * stride * matrix.front().nrows());

      left.back().assign_entry(input, j);
      right.back().assign_entry(
          fk::matrix<P, mem_type::view, resrc>(matrix.front()), j);
      product.back().assign_entry(output, j);
    }
  }

  return;
}

template<typename P, resource resrc>
fk::vector<P, mem_type::view, resrc> const &
batch_chain<P, resrc>::get_input_workspace()
{
  return workspace[in];
}

template<typename P, resource resrc>
fk::vector<P, mem_type::view, resrc> const &
batch_chain<P, resrc>::get_output_workspace()
{
  return workspace[out];
}

template<typename P, resource resrc>
void batch_chain<P, resrc>::swap_workspaces()
{
  int const tmp = in;
  in            = out;
  out           = tmp;

  return;
}

template<typename P, resource resrc>
void batch_chain<P, resrc>::execute_batch_chain()
{
  assert(left.size() == right.size());
  assert(right.size() == product.size());

  for (int i = 0; i < static_cast<int>(left.size()); ++i)
  {
    batched_gemm(left[i], right[i], product[i], (P)1, (P)0);
  }

  return;
}

template class batch_chain<double, resource::device>;
template class batch_chain<double, resource::host>;
template class batch_chain<float, resource::device>;
template class batch_chain<float, resource::host>;
