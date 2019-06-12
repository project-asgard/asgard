#include "batch.hpp"
#include "connectivity.hpp"
#include "lib_dispatch.hpp"
#include "tensors.hpp" // for views

// object to store lists of operands for batched gemm/gemv.
// utilized as the primary data structure for other functions
// within this component.
template<typename P>
batch<P>::batch(int const num_entries, int const nrows, int const ncols,
                int const stride, bool const do_trans)
    : num_entries_(num_entries), nrows_(nrows), ncols_(ncols), stride_(stride),
      do_trans_(do_trans), batch_{new P *[num_entries]()}
{
  assert(num_entries > 0);
  assert(nrows > 0);
  assert(ncols > 0);
  assert(stride > 0);
  for (P *&ptr : (*this))
  {
    ptr = nullptr;
  }
}

template<typename P>
batch<P>::batch(batch<P> const &other)
    : num_entries_(other.num_entries()), nrows_(other.nrows()),
      ncols_(other.ncols()), stride_(other.get_stride()),
      do_trans_(other.get_trans()), batch_{new P *[other.num_entries()]()}
{
  std::memcpy(batch_, other.batch_, other.num_entries() * sizeof(P *));
}

template<typename P>
batch<P> &batch<P>::operator=(batch<P> const &other)
{
  if (&other == this)
  {
    return *this;
  }
  assert(num_entries() == other.num_entries());
  assert(nrows() == other.nrows());
  assert(ncols() == other.ncols());
  assert(get_stride() == other.get_stride());
  assert(get_trans() == other.get_trans());
  std::memcpy(batch_, other.batch_, other.num_entries() * sizeof(P *));
  return *this;
}

template<typename P>
batch<P>::batch(batch<P> &&other)
    : num_entries_(other.num_entries()), nrows_(other.nrows()),
      ncols_(other.ncols()), stride_(other.get_stride()),
      do_trans_(other.get_trans()), batch_{other.batch_}
{
  other.batch_ = nullptr;
}

template<typename P>
batch<P> &batch<P>::operator=(batch<P> &&other)
{
  if (&other == this)
  {
    return *this;
  }
  assert(num_entries() == other.num_entries());
  assert(nrows() == other.nrows());
  assert(ncols() == other.ncols());
  assert(get_stride() == other.get_stride());

  assert(get_trans() == other.get_trans());
  batch_       = other.batch_;
  other.batch_ = nullptr;
  return *this;
}

template<typename P>
batch<P>::~batch()
{
  delete[] batch_;
}

template<typename P>
bool batch<P>::operator==(batch<P> other) const
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

template<typename P>
P *batch<P>::operator()(int const position) const
{
  assert(position >= 0);
  assert(position < num_entries());
  return batch_[position];
}

// assign the provided view's data pointer
// at the index indicated by position argument
// cannot overwrite previous assignment
template<typename P>
void batch<P>::assign_entry(fk::matrix<P, mem_type::view> const a,
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

  // ensure nothing already assigned
  assert(!batch_[position]);

  batch_[position] = a.data();
}

// clear one assignment
// returns true if there was a previous assignment,
// false if nothing was assigned
template<typename P>
bool batch<P>::clear_entry(int const position)
{
  P *temp          = batch_[position];
  batch_[position] = nullptr;
  return temp;
}

// get a pointer to the batch's
// pointers for batched blas call
// for performance, may have to
// provide a direct access to P**
// from batch_, but avoid for now
template<typename P>
P *const *batch<P>::get_list() const
{
  return batch_;
}

// verify that every allocated pointer
// has been assigned to
template<typename P>
bool batch<P>::is_filled() const
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
template<typename P>
batch<P> &batch<P>::clear_all()
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
template<typename P>
void batched_gemm(batch<P> const &a, batch<P> const &b, batch<P> const &c,
                  P const alpha, P const beta)
{
  // check cardinality of sets
  assert(a.num_entries() == b.num_entries());
  assert(b.num_entries() == c.num_entries());
  int const num_entries = a.num_entries();

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
  int n = cols_b; // technically these should be of op(A) or (B), but our dims
                  // are the same when transposing
  int k                  = cols_a;
  int lda                = a.get_stride();
  int ldb                = b.get_stride();
  int ldc                = c.get_stride();
  char const transpose_a = a.get_trans() ? 't' : 'n';
  char const transpose_b = b.get_trans() ? 't' : 'n';
  P alpha_               = alpha;
  P beta_                = beta;

  for (int i = 0; i < num_entries; ++i)
  {
    if (a(i) && b(i) && c(i))
      lib_dispatch::gemm(&transpose_a, &transpose_b, &m, &n, &k, &alpha_, a(i),
                         &lda, b(i), &ldb, &beta_, c(i), &ldc);
  }
}

// execute a batched gemv given a, b, c batch lists
// and other blas information
template<typename P>
void batched_gemv(batch<P> const &a, batch<P> const &b, batch<P> const &c,
                  P const alpha, P const beta)
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
  int const rows_a = a.get_trans() ? a.ncols() : a.nrows();
  int const cols_a = a.get_trans() ? a.nrows() : a.ncols();
  int const rows_b = b.nrows();

  assert(cols_a == rows_b);
  assert(b.ncols() == 1);
  assert(c.ncols() == 1);

  // setup blas args
  int m                  = rows_a;
  int n                  = cols_a;
  int lda                = a.get_stride();
  int stride_b           = b.get_stride();
  int stride_c           = c.get_stride();
  char const transpose_a = a.get_trans() ? 't' : 'n';
  P alpha_               = alpha;
  P beta_                = beta;

  for (int i = 0; i < num_entries; ++i)
  {
    if (a(i) && b(i) && c(i))
      lib_dispatch::gemv(&transpose_a, &m, &n, &alpha_, a(i), &lda, b(i),
                         &stride_b, &beta_, c(i), &stride_c);
  }
}

// --- batch allocation code --- /

// static helper - compute how many gemms a single call to
// kronmult_to_batch_sets adds for at a given PDE dimension
//
// num_dims is the total number of dimensions for the PDE
static int
compute_batch_size(int const degree, int const num_dims, int const dimension)
{
  assert(dimension >= 0);
  assert(dimension < num_dims);
  assert(num_dims > 0);
  assert(degree > 0);

  if (dimension == 0 || dimension == num_dims - 1)
  {
    return 1;
  }

  return std::pow(degree, (num_dims - dimension - 1));
}

// static helper - compute the dimensions (nrows/ncols) for gemm at a given
// dimension
static matrix_size_set
compute_dimensions(int const degree, int const num_dims, int const dimension)
{
  assert(dimension >= 0);
  assert(dimension < num_dims);
  assert(num_dims > 0);
  assert(degree > 0);

  if (dimension == 0)
  {
    return matrix_size_set(degree, degree, degree,
                           static_cast<int>(std::pow(degree, num_dims - 1)));
  }
  return matrix_size_set(static_cast<int>(std::pow(degree, dimension)), degree,
                         degree, degree);
}

// create empty batches w/ correct dims/cardinality for a pde
template<typename P>
std::vector<batch_operands_set<P>>
allocate_batches(PDE<P> const &pde, int const num_elems)
{
  std::vector<batch_operands_set<P>> batches;

  // FIXME when we allow varying degree by dimension, all
  // this code will have to change...
  int const degree = pde.get_dimensions()[0].get_degree();

  // add the first (lowest dimension) batch
  bool const do_trans         = false;
  int const num_gemms         = pde.num_terms * num_elems;
  matrix_size_set const sizes = compute_dimensions(degree, pde.num_dims, 0);

  // get stride of first coefficient matrix in 0th term set.
  // note all the coefficient matrices for each term have the
  // same dimensions
  int const stride = pde.get_coefficients(0, 0).stride();
  batches.emplace_back(std::vector<batch<P>>{
      batch<P>(num_gemms, sizes.rows_a, sizes.cols_a, stride, do_trans),
      batch<P>(num_gemms, sizes.rows_b, sizes.cols_b, sizes.rows_b, do_trans),
      batch<P>(num_gemms, sizes.rows_a, sizes.cols_b, sizes.rows_a, false)});

  // remaining batches
  for (int i = 1; i < pde.num_dims; ++i)
  {
    int const num_gemms =
        compute_batch_size(degree, pde.num_dims, i) * pde.num_terms * num_elems;
    matrix_size_set const sizes = compute_dimensions(degree, pde.num_dims, i);
    bool const trans_a          = false;
    bool const trans_b          = true;

    int const stride = pde.get_coefficients(0, i).stride();
    batches.emplace_back(std::vector<batch<P>>{
        batch<P>(num_gemms, sizes.rows_a, sizes.cols_a, sizes.rows_a, trans_a),
        batch<P>(num_gemms, sizes.rows_b, sizes.cols_b, stride, trans_b),
        batch<P>(num_gemms, sizes.rows_a, sizes.rows_b, sizes.rows_a, false)});
  }
  return batches;
}

// --- kronmult batching code --- //

// helper for lowest level of kronmult
// enqueue the gemms to perform A*x
// for the 0th dimension of the PDE.
//
// A is a tensor view into the 0th dimension
// operator matrix; x is the portion of the vector
// for this element; y is the output vector.
//
// batch offset is the 0-indexed ordinal numbering
// of the connected element these gemms address
template<typename P>
static void
kron_base(fk::matrix<P, mem_type::view> const A,
          fk::vector<P, mem_type::view> x, fk::vector<P, mem_type::view> y,
          batch_operands_set<P> &batches, int const batch_offset,
          int const degree, int const num_dims)
{
  batches[0].assign_entry(A, batch_offset);
  matrix_size_set const sizes = compute_dimensions(degree, num_dims, 0);
  fk::matrix<P, mem_type::view> x_view(x, sizes.rows_b, sizes.cols_b);
  batches[1].assign_entry(x_view, batch_offset);
  fk::matrix<P, mem_type::view> y_view(y, sizes.rows_a, sizes.cols_b);
  batches[2].assign_entry(y_view, batch_offset);
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
//
// batches is a vector of dimensions+1 batch sets:
//
//   dimension-many batch sets of 3 batch lists - a,b,c
//   to perform the kronmult
//
//   1 batch set to perform the reduction of connected element
//   contributions to each work item.
template<typename P>
void kronmult_to_batch_sets(
    std::vector<fk::matrix<P, mem_type::view>> const A,
    fk::vector<P, mem_type::view> x, fk::vector<P, mem_type::view> y,
    std::vector<fk::vector<P, mem_type::view>> const work,
    std::vector<batch_operands_set<P>> &batches, int const batch_offset,
    PDE<P> const &pde)
{
  // FIXME when we allow varying degree by dimension, all
  // this code will have to change...
  int const degree = pde.get_dimensions()[0].get_degree();

  // check vector sizes
  int const result_size = std::pow(degree, pde.num_dims);
  assert(x.size() == result_size);
  assert(y.size() == result_size);

  // check workspace sizes
  assert(static_cast<int>(work.size()) == std::min(pde.num_dims - 1, 2));
  for (fk::vector<P, mem_type::view> const &vector : work)
  {
    assert(vector.size() == result_size);
  }

  // check matrix sizes
  for (fk::matrix<P, mem_type::view> const &matrix : A)
  {
    assert(matrix.nrows() == degree);
    assert(matrix.ncols() == degree);
  }

  // we need an operand set for each dimension on entry
  assert(static_cast<int>(batches.size()) == pde.num_dims);

  // batch offset describes the ordinal position of the
  // connected element we are on - should be non-negative
  assert(batch_offset >= 0);

  // first, enqueue gemms for the lowest dimension
  if (pde.num_dims == 1)
  {
    // in a single dimensional PDE, we have to write lowest-dimension output
    // directly into the output vector
    kron_base(A[0], x, y, batches[0], batch_offset, degree, pde.num_dims);
    return;
  }

  // otherwise, we write into a work vector to serve as input for next-highest
  // dimension
  kron_base(A[0], x, work[0], batches[0], batch_offset, degree, pde.num_dims);

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
      fk::matrix<P, mem_type::view> x_view(
          work[(dimension - 1) % 2], sizes.rows_a, sizes.cols_a, offset * gemm);
      batches[dimension][0].assign_entry(x_view,
                                         batch_offset * num_gemms + gemm);
      batches[dimension][1].assign_entry(A[dimension],
                                         batch_offset * num_gemms + gemm);
      fk::matrix<P, mem_type::view> work_view(work[dimension % 2], sizes.rows_a,
                                              sizes.cols_a, offset * gemm);
      batches[dimension][2].assign_entry(work_view,
                                         batch_offset * num_gemms + gemm);
    }
  }

  // enqueue gemms for the highest dimension
  matrix_size_set const sizes =
      compute_dimensions(degree, pde.num_dims, pde.num_dims - 1);

  fk::matrix<P, mem_type::view> x_view(work[pde.num_dims % 2], sizes.rows_a,
                                       sizes.cols_a);
  batches[pde.num_dims - 1][0].assign_entry(x_view, batch_offset);
  batches[pde.num_dims - 1][1].assign_entry(A[pde.num_dims - 1], batch_offset);
  fk::matrix<P, mem_type::view> y_view(y, sizes.rows_a, sizes.cols_a);
  batches[pde.num_dims - 1][2].assign_entry(y_view, batch_offset);
}

// helper for calculating 1d indices for elements
static fk::vector<int> linearize(fk::vector<int> const &coords)
{
  fk::vector<int> elem_indices(coords.size() / 2);
  for (int i = 0; i < elem_indices.size(); ++i)
  {
    elem_indices(i) = get_1d_index(coords(i), coords(i + elem_indices.size()));
  }
  return elem_indices;
}

// function to allocate and build batch lists.
// given a problem instance (pde/elem table) and
// memory allocations (x, y, work), enqueue the
// batch gemms/reduction gemv to perform A*x
template<typename P>
std::vector<batch_operands_set<P>>
build_batches(PDE<P> const &pde, element_table const &elem_table,
              explicit_system<P> const &system, int const connected_start,
              int const elements_per_batch)
{
  // assume uniform degree for now
  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));
  int const x_size    = elem_size * elem_table.size();
  assert(system.batch_input.size() == x_size);

  // check our batch partitioning arguments
  assert(connected_start >= 0);
  assert(connected_start < elem_table.size());
  assert(elements_per_batch >= -1);
  assert(elements_per_batch <= elem_table.size());
  int const elements_in_batch =
      elements_per_batch < 0 ? elem_table.size() : elements_per_batch;

  // this can be smaller w/ atomic batched gemm e.g. ed's modified magma

  assert(system.reduction_space.size() >=
         (x_size * elements_in_batch * pde.num_terms));

  // intermediate workspaces for kron product.
  int const num_workspaces = std::min(pde.num_dims - 1, 2);
  assert(system.batch_intermediate.size() ==
         system.reduction_space.size() * num_workspaces);

  int const items_to_reduce = pde.num_terms * elements_in_batch;
  assert(system.get_unit_vector().size() >= items_to_reduce);

  std::vector<batch_operands_set<P>> batches =
      allocate_batches<P>(pde, elem_table.size() * elements_in_batch);

  batch_operands_set<P> reduction_batch = {
      batch<P>(elem_table.size(), elem_size, items_to_reduce, elem_size, false),
      batch<P>(elem_table.size(), items_to_reduce, 1, 1, false),
      batch<P>(elem_table.size(), elem_size, 1, 1, false)};

  // loop over elements
  // FIXME eventually want to do this in parallel
  for (int i = 0; i < elem_table.size(); ++i)
  {
    // first, get linearized indices for this element
    //
    // calculate from the level/cell indices for each
    // dimension
    fk::vector<int> const coords = elem_table.get_coords(i);
    assert(coords.size() == pde.num_dims * 2);
    fk::vector<int> elem_indices = linearize(coords);

    // calculate the row portion of the
    // operator position used for this
    // element's gemm calls
    fk::vector<int> operator_row = [&] {
      fk::vector<int> op_row(pde.num_dims);
      for (int d = 0; d < pde.num_dims; ++d)
      {
        // FIXME here we would have to use each dimension's
        // degree when calculating the index if we want different
        // degree in each dim
        op_row(d) = elem_indices(d) * degree;
      }
      return op_row;
    }();

    // calculate reduction inputs

    // calculate the position of this element in the
    // global system matrix
    // this is where the item reduces to in the output vector
    int const global_row = i * elem_size;

    int const workspace_size = elem_size * items_to_reduce;
    int const reduce_index   = workspace_size * i;
    fk::matrix<P, mem_type::view> reduction_space(
        system.reduction_space, elem_size, items_to_reduce, reduce_index);
    // assign_entry into reduction batch
    reduction_batch[0].assign_entry(reduction_space, i);
    reduction_batch[1].assign_entry(
        fk::matrix<P, mem_type::view>(system.get_unit_vector(), items_to_reduce,
                                      1, 0),
        i);
    reduction_batch[2].assign_entry(
        fk::matrix<P, mem_type::view>(system.batch_output, elem_size, 1,
                                      global_row),
        i);

    // loop over connected elements. for now, we assume
    // full connectivity
    int const connected_stop =
        std::min(connected_start + elements_in_batch, elem_table.size());
    for (int j = connected_start; j < connected_stop; ++j)
    {
      // get linearized indices for this connected element
      fk::vector<int> coords = elem_table.get_coords(j);
      assert(coords.size() == pde.num_dims * 2);
      fk::vector<int> connected_indices = linearize(coords);

      // calculate the col portion of the
      // operator position used for this
      // element's gemm calls
      fk::vector<int> operator_col = [&] {
        fk::vector<int> op_col(pde.num_dims);
        for (int d = 0; d < pde.num_dims; ++d)
        {
          // FIXME here we would have to use each dimension's
          // degree when calculating the index if we want different
          // degree in each dim
          op_col(d) = connected_indices(d) * degree;
        }
        return op_col;
      }();

      for (int k = 0; k < pde.num_terms; ++k)
      {
        // term major y-space layout, followed by connected items, finally work
        // items note that this index assumes uniform connected items (full
        // connectivity) if connectivity is instead computed for each element, i
        // * elem_table.size() * terms must be replaced by sum of lower indexed
        // connected items (scan) * terms

        int const kron_index = k + (j - connected_start) * pde.num_terms +
                               i * elements_in_batch * pde.num_terms;

        // y space, where kron outputs are written
        int const y_index = elem_size * kron_index;
        fk::vector<P, mem_type::view> const y_view(
            system.reduction_space, y_index, y_index + elem_size - 1);

        // work space, intermediate kron data
        int const work_index =
            elem_size * kron_index * std::min(pde.num_dims - 1, 2);
        std::vector<fk::vector<P, mem_type::view>> work_views(
            num_workspaces,
            fk::vector<P, mem_type::view>(system.batch_intermediate, work_index,
                                          work_index + elem_size - 1));
        if (num_workspaces == 2)
        {
          work_views[1] = fk::vector<P, mem_type::view>(
              system.batch_intermediate, work_index + elem_size,
              work_index + elem_size * 2 - 1);
        }

        // operator views, windows into operator matrix
        std::vector<fk::matrix<P, mem_type::view>> operator_views;
        for (int d = pde.num_dims - 1; d >= 0; --d)
        {
          operator_views.push_back(fk::matrix<P, mem_type::view>(
              pde.get_coefficients(k, d), operator_row(d),
              operator_row(d) + degree - 1, operator_col(d),
              operator_col(d) + degree - 1));
        }

        // calculate the position of this element in the
        // global system matrix
        int const global_col = j * elem_size;

        // x vector input to kronmult
        fk::vector<P, mem_type::view> const x_view(
            system.batch_input, global_col, global_col + elem_size - 1);

        kronmult_to_batch_sets(operator_views, x_view, y_view, work_views,
                               batches, kron_index, pde);
      }
    }
  }

  // emplace the reduction batch
  batches.push_back(std::move(reduction_batch));
  return batches;
}

// determine how many connected elements will be assigned to each work_set,
// given the workspace limit argument. because our partioning is done in terms
// of connected elements, we may not be able to exactly meet the limit.
//
// e.g. if a single connected element requires 2MB to compute, and the provided
// limit is 1MB, the work_sets will consume twice the provided limit
//
// -1 passed in workspace_MB (default arg) is used to indicate no workspace
// limiting.
//
// we may in future implement partitioning across work elements. this will give
// us finer discritization to meet provided limits. however, we haven't done
// that yet, because we may use work element partitioning to distribute work
// across nodes.
template<typename P>
int get_elements_per_set(PDE<P> const &pde, element_table const &elem_table,
                         int const workspace_MB)
{
  assert(workspace_MB >= -1);
  auto const get_MB = [](auto const num_elems) {
    double const bytes     = num_elems * sizeof(P);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

  // are we splitting the batches?
  bool const do_chunk      = workspace_MB > 0;
  int const degree         = pde.get_dimensions()[0].get_degree();
  int const elem_size      = static_cast<int>(std::pow(degree, pde.num_dims));
  int const num_workspaces = std::min(pde.num_dims - 1, 2);

  // calc size of reduction space for a single work item
  double const elem_reduction_space_MB =
      get_MB(pde.num_terms * elem_table.size() * elem_size);
  // calc size of intermediate space for a single work item
  double const elem_intermediate_space_MB =
      get_MB(static_cast<double>(num_workspaces) * pde.num_terms *
             elem_table.size() * elem_size);
  double const elem_MB = elem_reduction_space_MB + elem_intermediate_space_MB;
  // number of elements that will be batched and calculated together
  double const elem_limit =
      do_chunk ? workspace_MB / elem_MB : elem_table.size();
  return std::min(static_cast<int>(std::ceil(elem_limit)), elem_table.size());
}

template<typename P>
explicit_system<P>::explicit_system(PDE<P> const &pde,
                                    element_table const &table,
                                    int const limit_MB)
{
  int const degree        = pde.get_dimensions()[0].get_degree();
  int const elem_size     = static_cast<int>(std::pow(degree, pde.num_dims));
  int const elems_per_set = get_elements_per_set(pde, table, limit_MB);

  // FIXME note that if problem size/limit are misconfigured for a machine,
  // bad alloc can be thrown here
  batch_input.resize(elem_size * table.size());
  batch_output.resize(batch_input.size());
  reduction_space.resize(batch_input.size() * elems_per_set * pde.num_terms);

  // intermediate workspaces for kron product.
  int const num_workspaces = std::min(pde.num_dims - 1, 2);
  batch_intermediate.resize(reduction_space.size() * num_workspaces);
  unit_vector_.resize(pde.num_terms * elems_per_set);
  std::fill(unit_vector_.begin(), unit_vector_.end(), 1.0);

  // time advance workspace
  x_orig.resize(batch_input.size());
  scaled_source.resize(batch_input.size());
  result_1.resize(batch_input.size());
  result_2.resize(batch_input.size());
  result_3.resize(batch_input.size());
}

template<typename P>
fk::vector<P> const &explicit_system<P>::get_unit_vector() const
{
  return unit_vector_;
}

template<typename P>
work_set<P>
build_work_set(PDE<P> const &pde, element_table const &elem_table,
               explicit_system<P> const &system, int const workspace_MB)
{
  work_set<P> work_split;

  int const elem_limit = get_elements_per_set(pde, elem_table, workspace_MB);

  // number of element sets that will share workspace
  int const num_work_sets =
      workspace_MB > 0
          ? std::ceil(static_cast<double>(elem_table.size()) / elem_limit)
          : 1;

  for (int i = 0; i < num_work_sets; ++i)
  {
    work_split.emplace_back(build_batches(
        pde, elem_table, system, i * elem_limit,
        std::min(elem_limit, elem_table.size() - i * elem_limit)));
  }

  return work_split;
}

template class batch<float>;
template class batch<double>;

template class explicit_system<float>;
template class explicit_system<double>;

template void batched_gemm(batch<float> const &a, batch<float> const &b,
                           batch<float> const &c, float const alpha,
                           float const beta);

template void batched_gemm(batch<double> const &a, batch<double> const &b,
                           batch<double> const &c, double const alpha,
                           double const beta);

template void batched_gemv(batch<float> const &a, batch<float> const &b,
                           batch<float> const &c, float const alpha,
                           float const beta);
template void batched_gemv(batch<double> const &a, batch<double> const &b,
                           batch<double> const &c, double const alpha,
                           double const beta);

template std::vector<batch_operands_set<float>>
allocate_batches(PDE<float> const &pde, int const num_elems);
template std::vector<batch_operands_set<double>>
allocate_batches(PDE<double> const &pde, int const num_elems);

template void kronmult_to_batch_sets(
    std::vector<fk::matrix<float, mem_type::view>> const A,
    fk::vector<float, mem_type::view> x, fk::vector<float, mem_type::view> y,
    std::vector<fk::vector<float, mem_type::view>> const work,
    std::vector<batch_operands_set<float>> &batches, int const batch_offset,
    PDE<float> const &pde);

template void kronmult_to_batch_sets(
    std::vector<fk::matrix<double, mem_type::view>> const A,
    fk::vector<double, mem_type::view> x, fk::vector<double, mem_type::view> y,
    std::vector<fk::vector<double, mem_type::view>> const work,
    std::vector<batch_operands_set<double>> &batches, int const batch_offset,
    PDE<double> const &pde);

template std::vector<batch_operands_set<float>>
build_batches(PDE<float> const &pde, element_table const &elem_table,
              explicit_system<float> const &system, int const connected_start,
              int const elements_per_batch);
template std::vector<batch_operands_set<double>>
build_batches(PDE<double> const &pde, element_table const &elem_table,
              explicit_system<double> const &system, int const connected_start,
              int const elements_per_batch);

template work_set<float>
build_work_set(PDE<float> const &pde, element_table const &elem_table,
               explicit_system<float> const &system, int const workspace_MB);

template work_set<double>
build_work_set(PDE<double> const &pde, element_table const &elem_table,
               explicit_system<double> const &system, int const workspace_MB);
