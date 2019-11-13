/*
Comments assume the user is reading the lines of this file in order.

Problem relevant to functions in this file:

given a vector "x" of length "x_size" and list of matrices of arbitrary dimension
in "matrix": { m0, m1, ... , m_last }, calculate ( m0 kron m1 kron ... kron m_end ) * x

*/
#include "kron.hpp"


/*
For a list of "n" matrices in "matrix" parameter, the Kron algorithm in this file will
go through "n" rounds of batched matrix multiplications. The output of each round becomes the
input of the next round. Two workspaces are thus sufficient, if each of them is as large as the
largest output from any round. This function calculates the largest output of any round.

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
so the first workspace needs to be the maximum of round 0 and 2, while the other needs to be
the max of round 1 and 3. As it stands now, each workspace is the size of the max of all rounds.
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
    assert( c_prod > 0 );

    r_prod *= iter->nrows();

    int const size = x_size / c_prod * r_prod;
    if (size > greatest)
      greatest = size;
  }

  return greatest;
}

/* explicit instantions of function above */
template int calculate_workspace_len< double, resource::device >(
    std::vector<fk::matrix<double, mem_type::view, resource::device >> const &matrix,
    int const x_size);

template int calculate_workspace_len< double, resource::host >(
    std::vector<fk::matrix<double, mem_type::view, resource::host >> const &matrix,
    int const x_size);

template int calculate_workspace_len< float, resource::device >(
    std::vector<fk::matrix<float, mem_type::view, resource::device >> const &matrix,
    int const x_size);

template int calculate_workspace_len< float, resource::host >(
    std::vector<fk::matrix<float, mem_type::view, resource::host >> const &matrix,
    int const x_size);

/* class batch_job member functions */
template< typename P, resource resrc >
batch_job< P, resrc >::batch_job(P const alpha, P const beta, int const workspace_len,
            fk::vector<P, mem_type::view, resrc> const &x, int const y_size)
      : alpha(alpha), beta(beta), y_size(y_size),
        workspace({fk::vector<P, mem_type::owner, resrc>(workspace_len),
                   fk::vector<P, mem_type::owner, resrc>(workspace_len)}), in( 0 ), out( 1 )
{
  /* put the data in "x" into the first input workspace. The workspace may be longer than
     the input vector. If so, the first x.size() elements of the workspace should equal x */
  assert( workspace[ 0 ].size() >= x.size() );
  fk::vector<P, mem_type::view, resrc> init_w0(workspace[0], 0, x.size() - 1);
  init_w0 = x;
}

/* For each matrix in the "matrix" list in the problem description, a batch_set object is created,
   and the input/output must be flipped so the next batch_set's input will be the previous ones
   output */
template< typename P, resource resrc >
void batch_job< P, resrc >::add_batch_set(batch_set<P, resrc> const &&bs)
{
  batches.emplace_back(bs);

  return;
}

template< typename P, resource resrc >
void batch_job< P, resrc >::swap_workspaces()
{

  int const tmp = in;
  in            = out;
  out           = tmp;

  return;
}

template< typename P, resource resrc >
fk::vector<P, mem_type::owner, resrc> const &
batch_job< P, resrc>::get_input_workspace()
{
  return workspace[in];
}

template< typename P, resource resrc >
fk::vector<P, mem_type::owner, resrc> const &
batch_job< P, resrc >::get_output_workspace()
{
  return workspace[out];
}

/* explicit instantiations of batch_job class */
template class batch_job<double, resource::device >;
template class batch_job<double, resource::host>;
template class batch_job<float, resource::device >;
template class batch_job<float, resource::host>;

/* Carry out the math according to the dataflow encapsulated in the batch_job object */
template<typename P, resource resrc>
fk::vector<P, mem_type::owner, resrc> execute_batch_job(batch_job<P, resrc> &job)
{
  for (auto const &bs : job.batches)
  {
    batched_gemm(bs.left, bs.right, bs.product, job.alpha, job.beta);
  }

  fk::vector<P, mem_type::view, resrc> v(job.get_input_workspace(), 0,
                                         job.y_size - 1);

  fk::vector<P, mem_type::owner, resrc> r(v);

  return r;
}

/* execute_batch_job explicit instantiations */
template fk::vector< double, mem_type::owner, resource::device >
execute_batch_job< double, resource::device >
(batch_job<double, resource::device > &job);

template fk::vector< float, mem_type::owner, resource::device >
execute_batch_job< float, resource::device >
(batch_job<float, resource::device > &job);

template fk::vector< double, mem_type::owner, resource::host>
execute_batch_job< double, resource::host>
(batch_job<double, resource::host> &job);

template fk::vector< float, mem_type::owner, resource::host>
execute_batch_job< float, resource::host>
(batch_job<float, resource::host> &job);

/* delineate the dataflow for the algorithm that implements the particular problem described in
   the comment at the top of the file. The algorithm proceeds in stages, with a stage for each
   matrix in the list. The input for each stage is the output of the previous. The initial input
   is the vector "x". The output of each stage is a vector. The output of the final stage is
   the solution to the problem. */
template<typename P, resource resrc>
batch_job<P, resrc>
kron_batch(std::vector<fk::matrix<P, mem_type::view, resrc>> const &matrix,
           fk::vector<P, mem_type::view, resrc> const &x)
{
  assert(matrix.size() > 0);

  /* ensure "x" is correct size - should be the product of all the matrices respective number
     of columns */
  assert(x.size() == std::accumulate(matrix.begin(), matrix.end(), 1,
                                     [](int const i, auto const &m) {
                                       return i * m.ncols();
                                     }));

  /* determine correct workspace length */
  int const workspace_len = calculate_workspace_len(matrix, x.size());

  /*
    final output vector's size will be the product of all the matrices respective number of rows
  */
  int const y_size =
      std::accumulate(matrix.begin(), matrix.end(), 1,
                      [](int const i, auto const &m) { return i * m.nrows(); });

  batch_job<P, resrc> job(1, 0, workspace_len, x, y_size);

  /*
    The algorithm iterates over each matrix in "matrix" in reverse order, creates a batch_set 
    object for each matrix, and assigns input and output for each multiplication in that
    batch_set. The first iteration is handled differently than the ones that follow, so the loop
    is unrolled one iteration.
  */
  typename std::vector<
      fk::matrix<P, mem_type::view, resrc>>::const_reverse_iterator iter =
      matrix.rbegin();

  {
    /* only one large matrix multiply is needed for this first iteration. We want to break up
       "x" into consecutive subvectors of length iter->ncols(). Each of these subvectors will
       be transformed into a vector of length iter->nrows() by multiplying it by the last
       matrix in the list. These results are concatenated to produce the output of the first
       stage. The easiest way to implement the above process is with a large matrix multiply: */

    /* define the sizes of the operands */
    int const rows = x.size() / iter->ncols();

    /* Create uninitialized batch operands */
    batch_set< P, resrc > bs( 1, iter->nrows(), iter->ncols(), iter->stride(), false,
                              1, iter->ncols(), rows, iter->ncols(), false,
                              1, iter->nrows(), rows, iter->nrows(), false );

    /* assign the data to the batches */
    fk::matrix<P, mem_type::view, resrc> input(job.get_input_workspace(),
                                               iter->ncols(), rows, 0);

    fk::matrix<P, mem_type::view, resrc> output(job.get_output_workspace(),
                                                iter->nrows(), rows, 0);

    bs.right.assign_entry(input, 0);
    bs.left.assign_entry(fk::matrix<P, mem_type::view, resrc>((*iter)), 0);
    bs.product.assign_entry(output, 0);

    /* add the batch_set to the batch job */
    job.add_batch_set(std::move(bs));

    /* output and input space will switch roles for the next iteration */
    job.swap_workspaces();
  }
  /* The remaining iterations are a general case of the first one */

  /* distance in between first element of output subvectors above */
  int stride = iter->nrows();
  /* initial value for the size of the input vector for the stage to follow: each iter->ncols()
     length consecutive subvector was previously replaced by one of length iter->nrows() */
  int v_size = x.size() / iter->ncols() * iter->nrows();
  ++iter;

  for (; iter != matrix.rend(); ++iter)
  {
    /* the input matrix contains "read_stride" elements. Since they are consecutive in memory,
       this number also represents the stride, or distance in between the first elements of
       successive matrices in the linear memory bank */
    int const read_stride  = stride * iter->ncols();
    /* output matrix, follows same logic of input matrix */
    int const write_stride = stride * iter->nrows();

    /* total number of input matrices encoded in linear input memory */
    int const n_gemms      = v_size / read_stride;

    /*
      left operand's data comes from a linear input array. The transposes ensure that the input
      can be interpreted in column major format and that the output will be implicitly stored 
      that way.
    */
    batch_set< P, resrc > bs( n_gemms, stride, iter->ncols(), stride, false,
                              n_gemms, iter->nrows(), iter->ncols(), iter->stride(), true,
                              n_gemms, stride, iter->nrows(), stride, false );

    /* assign actual data to the batches */
    for (int j = 0; j < n_gemms; ++j)
    {
      fk::matrix<P, mem_type::view, resrc> input(
          job.get_input_workspace(), stride, iter->ncols(), j * read_stride);

      fk::matrix<P, mem_type::view, resrc> output(
          job.get_output_workspace(), stride, iter->nrows(), j * write_stride);

      bs.left.assign_entry(input, j);
      bs.right.assign_entry(fk::matrix<P, mem_type::view, resrc>((*iter)), j);
      bs.product.assign_entry(output, j);
    }

    /* add the batch_set to the batch_job */
    job.add_batch_set(std::move(bs));

    /* output and input space will switch roles for the next iteration */
    job.swap_workspaces();

    /* update variables for next iteration */
    v_size = n_gemms * write_stride;
    stride = write_stride;
  }

  return job;
}

/* explicit instantiations of kron_batch() */
template batch_job<float, resource::device>
kron_batch(std::vector<fk::matrix<float, mem_type::view, resource::device>> const &matrix,
           fk::vector<float, mem_type::view, resource::device> const &x);

template batch_job<float, resource::host>
kron_batch(std::vector<fk::matrix<float, mem_type::view, resource::host>> const &matrix,
           fk::vector<float, mem_type::view, resource::host> const &x);

template batch_job<double, resource::device>
kron_batch(std::vector<fk::matrix<double, mem_type::view, resource::device>> const &matrix,
           fk::vector<double, mem_type::view, resource::device> const &x);

template batch_job<double, resource::host>
kron_batch(std::vector<fk::matrix<double, mem_type::view, resource::host>> const &matrix,
           fk::vector<double, mem_type::view, resource::host> const &x);
