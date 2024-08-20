#include "asgard_batch.hpp"

#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif

namespace asgard
{
// utilized as the primary data structure for other functions
// within this component.
template<typename P, resource resrc>
batch<P, resrc>::batch(int const num_entries, int const nrows, int const ncols,
                       int const stride, bool const do_trans)
    : num_entries_(num_entries), nrows_(nrows), ncols_(ncols), stride_(stride),
      do_trans_(do_trans), batch_{new P *[num_entries]()}
{
  expect(num_entries > 0);
  expect(nrows > 0);
  expect(ncols > 0);
  expect(stride > 0);
}

template<typename P, resource resrc>
batch<P, resrc>::batch(batch<P, resrc> const &other)
    : num_entries_(other.num_entries()), nrows_(other.nrows()),
      ncols_(other.ncols()), stride_(other.get_stride()),
      do_trans_(other.get_trans()), batch_{new P *[other.num_entries()]()}
{
  std::memcpy(batch_, other.batch_, other.num_entries() * sizeof(P *));
}

template<typename P, resource resrc>
batch<P, resrc> &batch<P, resrc>::operator=(batch<P, resrc> const &other)
{
  if (&other == this)
  {
    return *this;
  }
  expect(num_entries() == other.num_entries());
  expect(nrows() == other.nrows());
  expect(ncols() == other.ncols());
  expect(get_stride() == other.get_stride());
  expect(get_trans() == other.get_trans());
  std::memcpy(batch_, other.batch_, other.num_entries() * sizeof(P *));
  return *this;
}

template<typename P, resource resrc>
batch<P, resrc>::batch(batch<P, resrc> &&other)
    : num_entries_(other.num_entries()), nrows_(other.nrows()),
      ncols_(other.ncols()), stride_(other.get_stride()),
      do_trans_(other.get_trans()), batch_{other.batch_}
{
  other.batch_ = nullptr;
}

template<typename P, resource resrc>
batch<P, resrc> &batch<P, resrc>::operator=(batch<P, resrc> &&other)
{
  if (&other == this)
  {
    return *this;
  }

  expect(num_entries() == other.num_entries());
  expect(nrows() == other.nrows());
  expect(ncols() == other.ncols());
  expect(get_stride() == other.get_stride());

  expect(get_trans() == other.get_trans());
  batch_       = other.batch_;
  other.batch_ = nullptr;
  return *this;
}

template<typename P, resource resrc>
batch<P, resrc>::~batch()
{
  delete[] batch_;
}

template<typename P, resource resrc>
bool batch<P, resrc>::operator==(batch<P, resrc> const &other) const
{
  if (nrows() != other.nrows())
  {
    return false;
  }
  if (ncols() != other.ncols())
  {
    return false;
  }
  if (get_stride() != other.get_stride())
  {
    return false;
  }
  if (num_entries() != other.num_entries())
  {
    return false;
  }
  if (get_trans() != other.get_trans())
  {
    return false;
  }

  for (int i = 0; i < num_entries(); ++i)
  {
    if (batch_[i] != other.batch_[i])
    {
      return false;
    }
  }

  return true;
}

template<typename P, resource resrc>
P *batch<P, resrc>::operator()(int const position) const
{
  expect(position >= 0);
  expect(position < num_entries());
  return batch_[position];
}

// assign the provided view's data pointer
// at the index indicated by position argument
// cannot overwrite previous assignment
template<typename P, resource resrc>
template<mem_type mem>
void batch<P, resrc>::assign_entry(fk::matrix<P, mem, resrc> const &a,
                                   int const position)
{
  // make sure this matrix is the
  // same dimensions as others in batch
  expect(a.nrows() == nrows());
  expect(a.ncols() == ncols());

  // if this is a batch of vectors,
  // we won't check the single column
  // matrix view a's stride
  if (get_stride() != 1)
  {
    expect(a.stride() == get_stride());
  }

  // ensure position is valid
  expect(position >= 0);
  expect(position < num_entries());

  // just aliasing, probably bad
  batch_[position] = const_cast<P *>(a.data());
}

// clear one assignment
// returns true if there was a previous assignment,
// false if nothing was assigned
template<typename P, resource resrc>
bool batch<P, resrc>::clear_entry(int const position)
{
  P *temp          = batch_[position];
  batch_[position] = nullptr;
  return temp;
}

// get a pointer to the batch's
// pointers for batched blas call
// for performance, may have to
// provide a direct access to P**
// from batch_, ~~but avoid for now~~ yep :-(
template<typename P, resource resrc>
P **batch<P, resrc>::get_list() const
{
  return batch_;
}

// verify that every allocated pointer
// has been assigned to
template<typename P, resource resrc>
bool batch<P, resrc>::is_filled() const
{
  for (P *const ptr : (*this))
  {
    if (!ptr)
    {
      return false;
    }
  }
  return true;
}

// clear assignments
template<typename P, resource resrc>
batch<P, resrc> &batch<P, resrc>::clear_all()
{
  for (P *&ptr : (*this))
  {
    ptr = nullptr;
  }
  return *this;
}

// execute a batched gemm given a, b, c batch lists
// and other blas information
// if we store info in the batch about where it is
// resident, this could be an abstraction point
// for calling cpu/gpu blas etc.
template<typename P, resource resrc>
void batched_gemm(batch<P, resrc> const &a, batch<P, resrc> const &b,
                  batch<P, resrc> const &c, P const alpha, P const beta)
{
  // check cardinality of sets
  expect(a.num_entries() == b.num_entries());
  expect(b.num_entries() == c.num_entries());

  // not allowed by blas interface
  // can be removed if we decide
  // we need to consider the transpose
  // of C later
  expect(!c.get_trans());

  // check dimensions for gemm
  //
  // rows_a/b and cols_a/b are the
  // number of rows/cols of a/b
  // after the optional transpose
  int const rows_a = a.get_trans() ? a.ncols() : a.nrows();
  int const cols_a = a.get_trans() ? a.nrows() : a.ncols();
  int const rows_b = b.get_trans() ? b.ncols() : b.nrows();
  int const cols_b = b.get_trans() ? b.nrows() : b.ncols();

  expect(cols_a == rows_b);
  expect(c.nrows() == rows_a);
  expect(c.ncols() == cols_b);

  // setup blas args
  int m = rows_a;
  int n = cols_b; // technically these should be of op(A) or (B), but our
                  // dims are the same when transposing
  int k              = cols_a;
  int lda            = a.get_stride();
  int ldb            = b.get_stride();
  int ldc            = c.get_stride();
  char const trans_a = a.get_trans() ? 't' : 'n';
  char const trans_b = b.get_trans() ? 't' : 'n';

  P alpha_ = alpha;
  P beta_  = beta;

  int num_batch = a.num_entries();

  lib_dispatch::batched_gemm<resrc>(a.get_list(), lda, trans_a, b.get_list(),
                                    ldb, trans_b, c.get_list(), ldc, m, n, k,
                                    alpha_, beta_, num_batch);
}

// Problem solved by batch_chain class:

// perform one or many (A1 kron A2 ... kron An) * x via constructing pointer
// lists to small gemms and invoking BLAS

// Realspace Transform

// For a list of "n" matrices in "matrix" parameter, this constructor
// will prepare "n" rounds of batched matrix multiplications. The output of
// each round becomes the input of the next round. Two workspaces are thus
// sufficient, if each of them is as large as the largest output from any round.
// This function calculates the largest output of any round.

// For a Kronecker product m0 * m1 * m2 * x = m3:
// matrix ~ ( rows, columns )
// m0 ~ (a, b)
// m1 ~ ( c, d )
// m2 ~ ( e, f )
// m3 ~ ( g, h )
// x ~ ( b*d*f*h, 1 )
// m3 ~ ( a*c*e*g, 1 )

// Algorithm completes in 3 rounds.
// Initial space needed: size of "x" vector: b*d*f*h
// Round 0 output size: b*d*f*g
// Round 1 output size: b*d*e*g
// Round 2 output size: b*c*e*g
// Round 3 ourput size: a*c*e*g

template<typename P, resource resrc, chain_method method>
template<chain_method, typename>
batch_chain<P, resrc, method>::batch_chain(
    std::vector<fk::matrix<P, mem_type::const_view, resrc>> const &matrices,
    fk::vector<P, mem_type::const_view, resrc> const &x,
    std::array<fk::vector<P, mem_type::view, resrc>, 2> &workspace,
    fk::vector<P, mem_type::view, resrc> &final_output)
{
  /* validation */
  expect(matrices.size() > 0);
  expect(workspace.size() == 2);

  /* ensure "x" is correct size - should be the product of all the matrices
     respective number of columns */
  expect(x.size() == std::accumulate(matrices.begin(), matrices.end(), 1,
                                     [](int const i, auto const &m) {
                                       return i * m.ncols();
                                     }));

  /* these are used to index "workspace" - the input/output role alternates
     between workspace[ 0 ] and workspace[ 1 ] in this algorithm */
  int in  = 0;
  int out = 1;

  /* ensure the workspaces are big enough for the problem */
  int const workspace_len = calculate_workspace_length(matrices, x.size());
  expect(workspace[0].size() >= workspace_len);
  expect(workspace[1].size() >= workspace_len);

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
    int const rows = x.size() / matrices.back().ncols();

    left_.emplace_back(1, matrices.back().nrows(), matrices.back().ncols(),
                       matrices.back().stride(), false);
    right_.emplace_back(1, matrices.back().ncols(), rows,
                        matrices.back().ncols(), false);
    product_.emplace_back(1, matrices.back().nrows(), rows,
                          matrices.back().nrows(), false);

    /* assign the data to the batches */
    fk::matrix<P, mem_type::const_view, resrc> input(x, matrices.back().ncols(),
                                                     rows, 0);

    /* If there is only 1 iteration, write immediately to final_output */
    fk::vector<P, mem_type::view, resrc> destination(workspace[out]);
    fk::matrix<P, mem_type::view, resrc> output(
        (matrices.size() > 1 ? destination : final_output),
        matrices.back().nrows(), rows, 0);

    right_.back().assign_entry(input, 0);
    left_.back().assign_entry(matrices.back(), 0);
    product_.back().assign_entry(output, 0);

    /* output and input space will switch roles for the next iteration */
    std::swap(in, out);
  }

  /* The remaining iterations are a general case of the first one */

  /* distance in between first element of output subvectors above */
  int stride = matrices.back().nrows();
  /* initial value for the size of the input vector for the stage to follow:
     each iter->ncols()
     length consecutive subvector was previously replaced by one of length
     iter->nrows() */
  int v_size = x.size() / matrices.back().ncols() * matrices.back().nrows();

  /* second to last iteration must be unrolled */
  for (int i = matrices.size() - 2; i > 0; --i)
  {
    /* total number of input matrices encoded in linear input memory */
    int const n_gemms = v_size / stride / matrices[i].ncols();

    /*
      left operand's data comes from a linear input array. The transposes ensure
      that the input can be interpreted in column major format and that the
      output will be implicitly stored that way.
    */
    left_.emplace_back(n_gemms, stride, matrices[i].ncols(), stride, false);
    right_.emplace_back(n_gemms, matrices[i].nrows(), matrices[i].ncols(),
                        matrices[i].stride(), true);
    product_.emplace_back(n_gemms, stride, matrices[i].nrows(), stride, false);

    /* assign actual data to the batches */
    for (int j = 0; j < n_gemms; ++j)
    {
      fk::matrix<P, mem_type::view, resrc> input(
          workspace[in], stride, matrices[i].ncols(),
          j * stride * matrices[i].ncols());

      fk::matrix<P, mem_type::view, resrc> output(
          workspace[out], stride, matrices[i].nrows(),
          j * stride * matrices[i].nrows());

      left_.back().assign_entry(input, j);
      right_.back().assign_entry(matrices[i], j);
      product_.back().assign_entry(output, j);
    }

    /* output and input space will switch roles for the next iteration */
    std::swap(in, out);

    /* update variables for next iteration */
    v_size = n_gemms * stride * matrices[i].nrows();
    stride *= matrices[i].nrows();
  }

  /* final loop iteration - output goes into output space instead of workspace
   */
  if (matrices.size() > 1)
  {
    int const n_gemms = v_size / stride / matrices.front().ncols();

    left_.emplace_back(n_gemms, stride, matrices.front().ncols(), stride,
                       false);
    right_.emplace_back(n_gemms, matrices.front().nrows(),
                        matrices.front().ncols(), matrices.front().stride(),
                        true);
    product_.emplace_back(n_gemms, stride, matrices.front().nrows(), stride,
                          false);

    for (int j = 0; j < n_gemms; ++j)
    {
      fk::matrix<P, mem_type::view, resrc> input(
          workspace[in], stride, matrices.front().ncols(),
          j * stride * matrices.front().ncols());

      fk::matrix<P, mem_type::view, resrc> output(
          final_output, stride, matrices.front().nrows(),
          j * stride * matrices.front().nrows());

      left_.back().assign_entry(input, j);
      right_.back().assign_entry(matrices.front(), j);
      product_.back().assign_entry(output, j);
    }
  }
}

template<typename P, resource resrc, chain_method method>
void batch_chain<P, resrc, method>::execute() const
{
  expect(left_.size() == right_.size());
  expect(right_.size() == product_.size());

  for (int i = 0; i < static_cast<int>(left_.size()); ++i)
  {
    batched_gemm(left_[i], right_[i], product_[i], (P)1, (P)0);
  }

  return;
}

template<typename P>
inline fk::vector<int>
linear_coords_to_indices(PDE<P> const &pde, int const degree,
                         fk::vector<int> const &coords)
{
  fk::vector<int> indices(coords.size());
  for (int d = 0; d < pde.num_dims(); ++d)
  {
    indices(d) = coords(d) * (degree + 1);
  }
  return indices;
}

#ifdef ASGARD_ENABLE_DOUBLE
#ifdef ASGARD_USE_CUDA
template class batch<double>;
#endif
template class batch<double, resource::host>;

template void batch<double, resource::host>::assign_entry(
    fk::matrix<double, mem_type::owner, resource::host> const &a,
    int const position);
template void batch<double, resource::host>::assign_entry(
    fk::matrix<double, mem_type::view, resource::host> const &a,
    int const position);
template void batch<double, resource::host>::assign_entry(
    fk::matrix<double, mem_type::const_view, resource::host> const &a,
    int const position);
#ifdef ASGARD_USE_CUDA
template void batch<double, resource::device>::assign_entry(
    fk::matrix<double, mem_type::owner, resource::device> const &a,
    int const position);
template void batch<double, resource::device>::assign_entry(
    fk::matrix<double, mem_type::view, resource::device> const &a,
    int const position);
template void batch<double, resource::device>::assign_entry(
    fk::matrix<double, mem_type::const_view, resource::device> const &a,
    int const position);

template void batched_gemm(batch<double> const &a, batch<double> const &b,
                           batch<double> const &c, double const alpha,
                           double const beta);
#endif

template void batched_gemm(batch<double, resource::host> const &a,
                           batch<double, resource::host> const &b,
                           batch<double, resource::host> const &c,
                           double const alpha, double const beta);

template class batch_chain<double, resource::host, chain_method::realspace>;
#ifdef ASGARD_USE_CUDA
template class batch_chain<double, resource::device, chain_method::realspace>;
template class batch_chain<double, resource::device, chain_method::advance>;
#endif

template batch_chain<double, resource::host, chain_method::realspace>::
    batch_chain(
        std::vector<fk::matrix<double, mem_type::const_view,
                               resource::host>> const &matrices,
        fk::vector<double, mem_type::const_view, resource::host> const &x,
        std::array<fk::vector<double, mem_type::view, resource::host>, 2>
            &workspace,
        fk::vector<double, mem_type::view, resource::host> &final_output);
#ifdef ASGARD_USE_CUDA
template batch_chain<double, resource::device, chain_method::realspace>::
    batch_chain(
        std::vector<fk::matrix<double, mem_type::const_view,
                               resource::device>> const &matrices,
        fk::vector<double, mem_type::const_view, resource::device> const &x,
        std::array<fk::vector<double, mem_type::view, resource::device>, 2>
            &workspace,
        fk::vector<double, mem_type::view, resource::device> &final_output);
#endif
#endif

#ifdef ASGARD_ENABLE_FLOAT
#ifdef ASGARD_USE_CUDA
template class batch<float>;
#endif
template class batch<float, resource::host>;

template void batch<float, resource::host>::assign_entry(
    fk::matrix<float, mem_type::owner, resource::host> const &a,
    int const position);
template void batch<float, resource::host>::assign_entry(
    fk::matrix<float, mem_type::view, resource::host> const &a,
    int const position);
template void batch<float, resource::host>::assign_entry(
    fk::matrix<float, mem_type::const_view, resource::host> const &a,
    int const position);
#ifdef ASGARD_USE_CUDA
template void batch<float, resource::device>::assign_entry(
    fk::matrix<float, mem_type::owner, resource::device> const &a,
    int const position);
template void batch<float, resource::device>::assign_entry(
    fk::matrix<float, mem_type::view, resource::device> const &a,
    int const position);
template void batch<float, resource::device>::assign_entry(
    fk::matrix<float, mem_type::const_view, resource::device> const &a,
    int const position);

template void batched_gemm(batch<float> const &a, batch<float> const &b,
                           batch<float> const &c, float const alpha,
                           float const beta);
#endif
template void batched_gemm(batch<float, resource::host> const &a,
                           batch<float, resource::host> const &b,
                           batch<float, resource::host> const &c,
                           float const alpha, float const beta);

#ifdef ASGARD_USE_CUDA
template class batch_chain<float, resource::device, chain_method::realspace>;
template class batch_chain<float, resource::device, chain_method::advance>;
#endif
template class batch_chain<float, resource::host, chain_method::realspace>;

template batch_chain<float, resource::host, chain_method::realspace>::
    batch_chain(
        std::vector<fk::matrix<float, mem_type::const_view,
                               resource::host>> const &matrices,
        fk::vector<float, mem_type::const_view, resource::host> const &x,
        std::array<fk::vector<float, mem_type::view, resource::host>, 2>
            &workspace,
        fk::vector<float, mem_type::view, resource::host> &final_output);
#ifdef ASGARD_USE_CUDA
template batch_chain<float, resource::device, chain_method::realspace>::
    batch_chain(
        std::vector<fk::matrix<float, mem_type::const_view,
                               resource::device>> const &matrices,
        fk::vector<float, mem_type::const_view, resource::device> const &x,
        std::array<fk::vector<float, mem_type::view, resource::device>, 2>
            &workspace,
        fk::vector<float, mem_type::view, resource::device> &final_output);
#endif
#endif

} // namespace asgard
