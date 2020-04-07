#include "batch.hpp"
#include "build_info.hpp"
#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif
#include "chunk.hpp"
#include "connectivity.hpp"
#include "lib_dispatch.hpp"
#include "tensors.hpp"
#include <limits.h>

// utilized as the primary data structure for other functions
// within this component.
template<typename P, resource resrc>
batch<P, resrc>::batch(int const capacity, int const nrows, int const ncols,
                       int const stride, bool const do_trans)
    : capacity_(capacity), num_entries_(capacity), nrows_(nrows), ncols_(ncols),
      stride_(stride), do_trans_(do_trans), batch_{new P *[capacity]()}
{
  assert(capacity > 0);
  assert(nrows > 0);
  assert(ncols > 0);
  assert(stride > 0);

  // FIXME
  /*for (P *&ptr : (*this))
  {
    ptr = nullptr;
  }*/
}

template<typename P, resource resrc>
batch<P, resrc>::batch(batch<P, resrc> const &other)
    : capacity_(other.get_capacity()), num_entries_(other.num_entries()),
      nrows_(other.nrows()), ncols_(other.ncols()), stride_(other.get_stride()),
      do_trans_(other.get_trans()), batch_{new P *[other.get_capacity()]()}
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
  assert(get_capacity() == other.get_capacity());
  set_num_entries(other.num_entries());
  assert(nrows() == other.nrows());
  assert(ncols() == other.ncols());
  assert(get_stride() == other.get_stride());
  assert(get_trans() == other.get_trans());
  std::memcpy(batch_, other.batch_, other.num_entries() * sizeof(P *));
  return *this;
}

template<typename P, resource resrc>
batch<P, resrc>::batch(batch<P, resrc> &&other)
    : capacity_(other.get_capacity()), num_entries_(other.num_entries()),
      nrows_(other.nrows()), ncols_(other.ncols()), stride_(other.get_stride()),
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
  assert(get_capacity() == other.get_capacity());
  set_num_entries(other.num_entries());

  assert(nrows() == other.nrows());
  assert(ncols() == other.ncols());
  assert(get_stride() == other.get_stride());

  assert(get_trans() == other.get_trans());
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
  assert(position >= 0);
  assert(position < num_entries());
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
  assert(a.nrows() == nrows());
  assert(a.ncols() == ncols());

  // if this is a batch of vectors,
  // we won't check the single column
  // matrix view a's stride
  if (get_stride() != 1)
  {
    assert(a.stride() == get_stride());
  }

  // ensure position is valid
  assert(position >= 0);
  assert(position < num_entries());

  // FIXME
  // ensure nothing already assigned
  // assert(!batch_[position]);

  batch_[position] = a.data();
}

template<typename P, resource resrc>
void batch<P, resrc>::assign_raw(P *const a, int const position)
{
  assert(position >= 0);
  assert(position < num_entries());
  // ensure nothing already assigned
  assert(!batch_[position]);
  batch_[position] = a;
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
// resrcident, this could be an abstraction point
// for calling cpu/gpu blas etc.
template<typename P, resource resrc>
void batched_gemm(batch<P, resrc> const &a, batch<P, resrc> const &b,
                  batch<P, resrc> const &c, P const alpha, P const beta)
{
  // check cardinality of sets
  assert(a.num_entries() == b.num_entries());
  assert(b.num_entries() == c.num_entries());

  // not allowed by blas interface
  // can be removed if we decide
  // we need to consider the transpose
  // of C later
  assert(!c.get_trans());

  // check dimensions for gemm
  //
  // rows_a/b and cols_a/b are the
  // number of rows/cols of a/b
  // after the optional transpose
  int const rows_a = a.get_trans() ? a.ncols() : a.nrows();
  int const cols_a = a.get_trans() ? a.nrows() : a.ncols();
  int const rows_b = b.get_trans() ? b.ncols() : b.nrows();
  int const cols_b = b.get_trans() ? b.nrows() : b.ncols();

  assert(cols_a == rows_b);
  assert(c.nrows() == rows_a);
  assert(c.ncols() == cols_b);

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

  lib_dispatch::batched_gemm(a.get_list(), &lda, &trans_a, b.get_list(), &ldb,
                             &trans_b, c.get_list(), &ldc, &m, &n, &k, &alpha_,
                             &beta_, &num_batch, resrc);
}

// execute a batched gemv given a, b, c batch lists
// and other blas information
template<typename P, resource resrc>
void batched_gemv(batch<P, resrc> const &a, batch<P, resrc> const &b,
                  batch<P, resrc> const &c, P const alpha, P const beta)
{
  // check cardinality of sets
  assert(a.num_entries() == b.num_entries());
  assert(b.num_entries() == c.num_entries());
  int const num_entries = a.num_entries();

  // our gemv will be set up for a column vector,
  // so b cannot be transposed.
  //
  // we can remove either or both of these if
  // we want to support more flexible operations
  assert(!b.get_trans() && !c.get_trans());

  // check dimensions for gemv
  assert((a.get_trans() ? a.nrows() : a.ncols()) == b.nrows());
  assert(b.ncols() == 1);
  assert(c.ncols() == 1);

  // setup blas args
  int m   = a.nrows();
  int n   = a.ncols();
  int lda = a.get_stride();

  char const transpose_a = a.get_trans() ? 't' : 'n';
  P alpha_               = alpha;
  P beta_                = beta;

  int num_batch = num_entries;

  lib_dispatch::batched_gemv(a.get_list(), &lda, &transpose_a, b.get_list(),
                             c.get_list(), &m, &n, &alpha_, &beta_, &num_batch,
                             resrc);
}

// helpers for calculating 1d indices for elements

// performant (no-alloc) version
inline void linearize(fk::vector<int> const &coords, int output[])
{
  int const output_size = coords.size() / 2;
  for (int i = 0; i < output_size; ++i)
  {
    output[i] = get_1d_index(coords(i), coords(i + output_size));
  }
}

// "safe" version
inline fk::vector<int> linearize(fk::vector<int> const &coords)
{
  fk::vector<int> linear(coords.size() / 2);
  for (int i = 0; i < linear.size(); ++i)
  {
    linear(i) = get_1d_index(coords(i), coords(i + linear.size()));
  }
  return linear;
}

// helpers for converting linear coordinates into operator matrix indices

// performant (no-alloc) version
template<typename P>
inline void
linear_coords_to_indices(PDE<P> const &pde, int const degree, int coords[])
{
  for (int d = 0; d < pde.num_dims; ++d)
  {
    coords[d] = coords[d] * degree;
  }
}

// "safe" version
template<typename P>
inline fk::vector<int>
linear_coords_to_indices(PDE<P> const &pde, int const degree,
                         fk::vector<int> const &coords)
{
  fk::vector<int> indices(coords.size());
  for (int d = 0; d < pde.num_dims; ++d)
  {
    indices(d) = coords(d) * degree;
  }
  return indices;
}

// FIXME this needs trimming
/*

Problem relevant to batch_chain class:

given a vector "x" of length "x_size" and list of matrices of arbitrary
dimension in "matrix": { m0, m1, ... , m_last }, calculate ( m0 kron m1 kron ...
kron m_end ) * x

*/

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

/* batch chain delineates the dataflow for the algorithm that implements the
   particular problem described in the comment at the top of the file. The
   algorithm proceeds in stages, with a stage for each matrix in the list. The
   input for each stage is the output of the previous. The initial input is the
   vector "x". The output of each stage is a vector. The output of the final
   stage is the solution to the problem. */

template<typename P, resource resrc, chain_method method>
template<chain_method, typename>
batch_chain<P, resrc, method>::batch_chain(
    std::vector<fk::matrix<P, mem_type::const_view, resrc>> const &matrices,
    fk::vector<P, mem_type::const_view, resrc> const &x,
    std::array<fk::vector<P, mem_type::view, resrc>, 2> &workspace,
    fk::vector<P, mem_type::view, resrc> &final_output)
{
  /* validation */
  assert(matrices.size() > 0);
  assert(workspace.size() == 2);

  /* ensure "x" is correct size - should be the product of all the matrices
     respective number of columns */
  assert(x.size() == std::accumulate(matrices.begin(), matrices.end(), 1,
                                     [](int const i, auto const &m) {
                                       return i * m.ncols();
                                     }));

  /* these are used to index "workspace" - the input/output role alternates
     between workspace[ 0 ] and workspace[ 1 ] in this algorithm */
  int in  = 0;
  int out = 1;

  /* ensure the workspaces are big enough for the problem */
  int const workspace_len = calculate_workspace_length(matrices, x.size());
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
template<chain_method, typename>
batch_chain<P, resrc, method>::batch_chain(PDE<P> const &pde,
                                           element_table const &elem_table,
                                           device_workspace<P> const &workspace,
                                           element_subgrid const &subgrid,
                                           element_chunk const &chunk)
{
  // 1 -- allocate batches

  int const num_elems = num_elements_in_chunk(chunk);

  // FIXME code relies on uniform degree across dimensions
  int const degree = pde.get_dimensions()[0].get_degree();

  // add the first (lowest dimension) batch
  bool const do_trans         = false;
  int const num_gemms         = pde.num_terms * num_elems;
  matrix_size_set const sizes = compute_dimensions(degree, pde.num_dims, 0);

  // get stride of first coefficient matrix in 0th term set.
  // note all the coefficient matrices for each term have the
  // same dimensions
  int const stride = pde.get_coefficients(0, 0).stride();

  left_.emplace_back(std::move(
      batch<P>(num_gemms, sizes.rows_a, sizes.cols_a, stride, do_trans)));
  right_.emplace_back(std::move(
      batch<P>(num_gemms, sizes.rows_b, sizes.cols_b, sizes.rows_b, do_trans)));
  product_.emplace_back(std::move(
      batch<P>(num_gemms, sizes.rows_a, sizes.cols_b, sizes.rows_a, false)));

  // remaining batches
  for (int i = 1; i < pde.num_dims; ++i)
  {
    int const num_gemms =
        compute_batch_size(degree, pde.num_dims, i) * pde.num_terms * num_elems;
    matrix_size_set const sizes = compute_dimensions(degree, pde.num_dims, i);
    bool const trans_a          = false;
    bool const trans_b          = true;

    int const stride = pde.get_coefficients(0, i).stride();

    left_.emplace_back(std::move(batch<P>(num_gemms, sizes.rows_a, sizes.cols_a,
                                          sizes.rows_a, trans_a)));
    right_.emplace_back(std::move(
        batch<P>(num_gemms, sizes.rows_b, sizes.cols_b, stride, trans_b)));
    product_.emplace_back(std::move(
        batch<P>(num_gemms, sizes.rows_a, sizes.rows_b, sizes.rows_a, false)));
  }

  // 2 -- populate

  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));

  auto const x_size = (subgrid.col_stop - subgrid.col_start + 1) *
                      static_cast<int64_t>(elem_size);
  assert(workspace.batch_input.size() >= x_size);

  int const elements_in_chunk = num_elements_in_chunk(chunk);

  // this can be smaller w/ atomic batched gemm e.g. ed's modified magma
  assert(workspace.reduction_space.size() >=
         (elem_size * elements_in_chunk * pde.num_terms));

  // intermediate workspaces for kron product.
  int const num_workspaces = std::min(pde.num_dims - 1, 2);
  assert(workspace.batch_intermediate.size() ==
         workspace.reduction_space.size() * num_workspaces);

  int const max_connected       = max_connected_in_chunk(chunk);
  int const max_items_to_reduce = pde.num_terms * max_connected;
  assert(workspace.get_unit_vector().size() >= max_items_to_reduce);

  // here we map integers 0->chunk_size-1 to row_0->row_last in the chunk
  // we could iterate over the chunk (which is a map rows->connected)
  // but openmp requires traditional loop, not foreachs
  std::vector<int> const index_to_key = [&chunk]() {
    std::vector<int> builder;
    for (auto const &[i, connected] : chunk)
    {
      builder.push_back(i);
      ignore(connected);
    }
    return builder;
  }();

// loop over elements
#ifdef ASGARD_USE_OPENMP
  int const threads = omp_get_num_procs();
#pragma omp parallel for num_threads(threads)
#endif
  for (int chunk_num = 0; chunk_num < static_cast<int>(chunk.size());
       ++chunk_num)
  {
    // allocate on thread stack
    static int constexpr max_dims       = 6;
    static int constexpr max_workspaces = 2;
    int operator_row[max_dims];
    int operator_col[max_dims];
    P *workspace_ptrs[max_workspaces];
    P *operator_ptrs[max_dims];

    // i: row we are addressing in element grid
    int const i = index_to_key[chunk_num];
    // connected: start/stop grid elements for this row
    auto const connected = chunk.at(i);

    // first, get linearized indices for this element
    //
    // calculate from the level/cell indices for each
    // dimension
    fk::vector<int> const &coords = elem_table.get_coords(i);
    assert(coords.size() == pde.num_dims * 2);

    linearize(coords, operator_row);

    // calculate the row portion of the
    // operator position used for this
    // element's gemm calls
    linear_coords_to_indices(pde, degree, operator_row);

    // calculate number of elements in previous rows
    // for later indexing in term (k) loop
    int const prev_row_elems = [i = i, &chunk] {
      if (i == chunk.begin()->first)
      {
        return 0;
      }
      int prev_elems = 0;
      for (int r = chunk.begin()->first; r < i; ++r)
      {
        prev_elems += chunk.at(r).stop - chunk.at(r).start + 1;
      }
      return prev_elems;
    }();

    // loop over connected elements. for now, we assume
    // full connectivity
    for (int j = connected.start; j <= connected.stop; ++j)
    {
      // get linearized indices for this connected element
      fk::vector<int> const &coords = elem_table.get_coords(j);
      assert(coords.size() == pde.num_dims * 2);

      linearize(coords, operator_col);

      // calculate the col portion of the
      // operator position used for this
      // element's gemm calls
      linear_coords_to_indices(pde, degree, operator_col);

      for (int k = 0; k < pde.num_terms; ++k)
      {
        // term major y-space layout, followed by connected items, finally work
        // items.
        int const total_prev_elems = prev_row_elems + j - connected.start;
        int const kron_index       = k + total_prev_elems * pde.num_terms;

        // y space, where kron outputs are written
        int const y_index = elem_size * kron_index;
        P *const y_ptr    = workspace.reduction_space.data() + y_index;

        // work space, intermediate kron data
        int const work_index =
            elem_size * kron_index * std::min(pde.num_dims - 1, 2);

        if (num_workspaces > 0)
          workspace_ptrs[0] = workspace.batch_intermediate.data() + work_index;
        if (num_workspaces == 2)
          workspace_ptrs[1] =
              workspace.batch_intermediate.data() + work_index + elem_size;

        // index into operator matrices
        for (int d = pde.num_dims - 1; d >= 0; --d)
        {
          operator_ptrs[(pde.num_dims - 1) - d] =
              pde.get_coefficients(k, d).data() + operator_row[d] +
              operator_col[d] * pde.get_coefficients(k, d).stride();
        }

        // determine the index for the input vector
        int const x_index = subgrid.to_local_col(j) * elem_size;

        // x vector input to kronmult
        P *const x_ptr = workspace.batch_input.data() + x_index;

        kronmult_to_batch_sets(operator_ptrs, x_ptr, y_ptr, workspace_ptrs,
                               kron_index, pde);
      }
    }
  }
}

template<typename P, resource resrc, chain_method method>
void batch_chain<P, resrc, method>::execute() const
{
  assert(left_.size() == right_.size());
  assert(right_.size() == product_.size());

  for (int i = 0; i < static_cast<int>(left_.size()); ++i)
  {
    batched_gemm(left_[i], right_[i], product_[i], (P)1, (P)0);
  }

  return;
}

// function to transform a single kronecker product * vector into a
// series of batched gemm calls, where the kronecker product is
// tensor encoded in the view vector A. x is the input vector; y is the output
// vector. work is a vector of vectors (max size 2), each of which is the same
// size as y. these store intermediate kron products - lower dimensional outputs
// are higher dimensional inputs.
//
//
// on entry the batches argument contains empty (pre-allocated) pointer lists
// that this function will populate to perform the above operation

// unsafe version; uses raw pointers rather than views as view construction
// destruction incurred significant runtime expense (alloc sync in parallel
// loop)

template<typename P, resource resrc, chain_method method>
template<chain_method, typename>
void batch_chain<P, resrc, method>::kronmult_to_batch_sets(
    P *const *const A, P *const x, P *const y, P *const *const work,
    int const batch_offset, PDE<P> const &pde)
{
  // FIXME when we allow varying degree by dimension, all
  // this code will have to change...
  int const degree = pde.get_dimensions()[0].get_degree();

  // batch offset describes the ordinal position of the
  // connected element we are on - should be non-negative
  assert(batch_offset >= 0);

  // first, enqueue gemms for the lowest dimension
  left_[0].assign_raw(A[0], batch_offset);
  right_[0].assign_raw(x, batch_offset);

  // in a single dimensional PDE, we have to write lowest-dimension output
  // directly into the output vector
  if (pde.num_dims == 1)
  {
    product_[0].assign_raw(y, batch_offset);
    return;
  }

  // otherwise, we write into a work vector to serve as input for next-highest
  // dimension
  product_[0].assign_raw(work[0], batch_offset);

  // loop over intermediate dimensions, enqueueing gemms
  for (int dimension = 1; dimension < pde.num_dims - 1; ++dimension)
  {
    // determine a and b matrix sizes at this dimension for all gemms
    matrix_size_set const sizes =
        compute_dimensions(degree, pde.num_dims, dimension);
    // determine how many gemms we will enqueue for this dimension
    int const num_gemms = compute_batch_size(degree, pde.num_dims, dimension);
    int const offset    = sizes.rows_a * sizes.cols_a;
    assert((offset * num_gemms) ==
           static_cast<int>(std::pow(degree, pde.num_dims)));

    // loop over gemms for this dimension and enqueue
    for (int gemm = 0; gemm < num_gemms; ++gemm)
    {
      // the modulus here is to alternate input/output workspaces per dimension

      P *const x_ptr = work[(dimension - 1) % 2] + offset * gemm;
      left_[dimension].assign_raw(x_ptr, batch_offset * num_gemms + gemm);

      right_[dimension].assign_raw(A[dimension],
                                   batch_offset * num_gemms + gemm);

      P *const work_ptr = work[dimension % 2] + offset * gemm;
      right_[dimension].assign_raw(work_ptr, batch_offset * num_gemms + gemm);
    }
  }

  // enqueue gemms for the highest dimension
  P *const x_ptr = work[pde.num_dims % 2];
  left_[pde.num_dims - 1].assign_raw(x_ptr, batch_offset);
  right_[pde.num_dims - 1].assign_raw(A[pde.num_dims - 1], batch_offset);
  product_[pde.num_dims - 1].assign_raw(y, batch_offset);
}

// function to allocate and build implicit system.
// given a problem instance (pde/elem table)
// does not utilize batching, here because it shares underlying structure and
// routines with explicit time advance
template<typename P>
void build_system_matrix(PDE<P> const &pde, element_table const &elem_table,
                         element_chunk const &chunk, fk::matrix<P> &A)
{
  // assume uniform degree for now
  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));
  int const A_size    = elem_size * elem_table.size();

  assert(A.ncols() == A_size && A.nrows() == A_size);

  using key_type = std::pair<int, int>;
  using val_type = fk::matrix<P, mem_type::owner, resource::host>;
  std::map<key_type, val_type> coef_cache;

  assert(A.ncols() == A_size && A.nrows() == A_size);
  // Copy coefficients to host for subsequent use
  for (int k = 0; k < pde.num_terms; ++k)
  {
    for (int d = 0; d < pde.num_dims; d++)
    {
      coef_cache.insert(std::pair<key_type, val_type>(
          key_type(k, d), pde.get_coefficients(k, d).clone_onto_host()));
    }
  }

  // loop over elements
  // FIXME eventually want to do this in parallel
  for (auto const &[i, connected] : chunk)
  {
    // first, get linearized indices for this element
    //
    // calculate from the level/cell indices for each
    // dimension
    fk::vector<int> const coords = elem_table.get_coords(i);
    assert(coords.size() == pde.num_dims * 2);
    fk::vector<int> const elem_indices = linearize(coords);

    int const global_row = i * elem_size;

    // calculate the row portion of the
    // operator position used for this
    // element's gemm calls
    fk::vector<int> const operator_row =
        linear_coords_to_indices(pde, degree, elem_indices);

    // loop over connected elements. for now, we assume
    // full connectivity
    for (int j = connected.start; j <= connected.stop; ++j)
    {
      // get linearized indices for this connected element
      fk::vector<int> const coords_nD = elem_table.get_coords(j);
      assert(coords_nD.size() == pde.num_dims * 2);
      fk::vector<int> const connected_indices = linearize(coords_nD);

      // calculate the col portion of the
      // operator position used for this
      // element's gemm calls
      fk::vector<int> const operator_col =
          linear_coords_to_indices(pde, degree, connected_indices);

      for (int k = 0; k < pde.num_terms; ++k)
      {
        std::vector<fk::matrix<P>> kron_vals;
        fk::matrix<P> kron0(1, 1);
        kron0(0, 0) = 1.0;
        kron_vals.push_back(kron0);
        for (int d = 0; d < pde.num_dims; d++)
        {
          fk::matrix<P, mem_type::view> op_view = fk::matrix<P, mem_type::view>(
              coef_cache[key_type(k, d)], operator_row(d),
              operator_row(d) + degree - 1, operator_col(d),
              operator_col(d) + degree - 1);
          fk::matrix<P> k_new = kron_vals[d].kron(op_view);
          kron_vals.push_back(k_new);
        }

        // calculate the position of this element in the
        // global system matrix
        int const global_col = j * elem_size;
        auto const &k_tmp    = kron_vals.back();

        fk::matrix<P, mem_type::view> A_view(
            A, global_row, global_row + k_tmp.nrows() - 1, global_col,
            global_col + k_tmp.ncols() - 1);

        A_view = A_view + k_tmp;
      }
    }
  }
}

template class batch<float>;
template class batch<double>;
template class batch<float, resource::host>;
template class batch<double, resource::host>;

template void batched_gemm(batch<float> const &a, batch<float> const &b,
                           batch<float> const &c, float const alpha,
                           float const beta);
template void batched_gemm(batch<double> const &a, batch<double> const &b,
                           batch<double> const &c, double const alpha,
                           double const beta);

template void batched_gemm(batch<float, resource::host> const &a,
                           batch<float, resource::host> const &b,
                           batch<float, resource::host> const &c,
                           float const alpha, float const beta);
template void batched_gemm(batch<double, resource::host> const &a,
                           batch<double, resource::host> const &b,
                           batch<double, resource::host> const &c,
                           double const alpha, double const beta);

template void batched_gemv(batch<float> const &a, batch<float> const &b,
                           batch<float> const &c, float const alpha,
                           float const beta);
template void batched_gemv(batch<double> const &a, batch<double> const &b,
                           batch<double> const &c, double const alpha,
                           double const beta);

template void batched_gemv(batch<float, resource::host> const &a,
                           batch<float, resource::host> const &b,
                           batch<float, resource::host> const &c,
                           float const alpha, float const beta);
template void batched_gemv(batch<double, resource::host> const &a,
                           batch<double, resource::host> const &b,
                           batch<double, resource::host> const &c,
                           double const alpha, double const beta);

template void
build_system_matrix(PDE<double> const &pde, element_table const &elem_table,
                    element_chunk const &chunk, fk::matrix<double> &A);
template void
build_system_matrix(PDE<float> const &pde, element_table const &elem_table,
                    element_chunk const &chunk, fk::matrix<float> &A);

template class batch_chain<double, resource::device, chain_method::realspace>;
template class batch_chain<double, resource::host, chain_method::realspace>;
template class batch_chain<float, resource::device, chain_method::realspace>;
template class batch_chain<float, resource::host, chain_method::realspace>;

template class batch_chain<double, resource::device, chain_method::advance>;
template class batch_chain<float, resource::device, chain_method::advance>;

template batch_chain<float, resource::host, chain_method::realspace>::
    batch_chain(
        std::vector<fk::matrix<float, mem_type::const_view,
                               resource::host>> const &matrices,
        fk::vector<float, mem_type::const_view, resource::host> const &x,
        std::array<fk::vector<float, mem_type::view, resource::host>, 2>
            &workspace,
        fk::vector<float, mem_type::view, resource::host> &final_output);

template batch_chain<double, resource::host, chain_method::realspace>::
    batch_chain(
        std::vector<fk::matrix<double, mem_type::const_view,
                               resource::host>> const &matrices,
        fk::vector<double, mem_type::const_view, resource::host> const &x,
        std::array<fk::vector<double, mem_type::view, resource::host>, 2>
            &workspace,
        fk::vector<double, mem_type::view, resource::host> &final_output);

template batch_chain<float, resource::device, chain_method::realspace>::
    batch_chain(
        std::vector<fk::matrix<float, mem_type::const_view,
                               resource::device>> const &matrices,
        fk::vector<float, mem_type::const_view, resource::device> const &x,
        std::array<fk::vector<float, mem_type::view, resource::device>, 2>
            &workspace,
        fk::vector<float, mem_type::view, resource::device> &final_output);

template batch_chain<double, resource::device, chain_method::realspace>::
    batch_chain(
        std::vector<fk::matrix<double, mem_type::const_view,
                               resource::device>> const &matrices,
        fk::vector<double, mem_type::const_view, resource::device> const &x,
        std::array<fk::vector<double, mem_type::view, resource::device>, 2>
            &workspace,
        fk::vector<double, mem_type::view, resource::device> &final_output);

template batch_chain<float, resource::device, chain_method::advance>::
    batch_chain(PDE<float> const &pde, element_table const &elem_table,
                device_workspace<float> const &workspace,
                element_subgrid const &subgrid, element_chunk const &chunk);

template batch_chain<double, resource::device, chain_method::advance>::
    batch_chain(PDE<double> const &pde, element_table const &elem_table,
                device_workspace<double> const &workspace,
                element_subgrid const &subgrid, element_chunk const &chunk);

template void batch_chain<float, resource::device, chain_method::advance>::
    kronmult_to_batch_sets(float *const *const A, float *const x,
                           float *const y, float *const *const work,
                           int const batch_offset, PDE<float> const &pde);

template void batch_chain<double, resource::device, chain_method::advance>::
    kronmult_to_batch_sets(double *const *const A, double *const x,
                           double *const y, double *const *const work,
                           int const batch_offset, PDE<double> const &pde);
