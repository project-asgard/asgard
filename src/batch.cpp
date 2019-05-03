#include "batch.hpp"
#include "connectivity.hpp"
#include "tensors.hpp" // for views/blas

template<typename P>
batch_list<P>::batch_list(int const num_batch, int const nrows, int const ncols,
                          int const stride, bool const do_trans)
    : num_batch(num_batch), nrows(nrows), ncols(ncols), stride(stride),
      do_trans(do_trans), batch_list_{new P *[num_batch]()}
{
  assert(num_batch > 0);
  assert(nrows > 0);
  assert(ncols > 0);
  assert(stride >= nrows);

  for (P *&ptr : (*this))
  {
    ptr = nullptr;
  }
}

template<typename P>
batch_list<P>::batch_list(batch_list<P> const &other)
    : num_batch(other.num_batch), nrows(other.nrows), ncols(other.ncols),
      stride(other.stride),
      do_trans(other.do_trans), batch_list_{new P *[other.num_batch]()}
{
  std::memcpy(batch_list_, other.batch_list_, other.num_batch * sizeof(P *));
}

template<typename P>
batch_list<P> &batch_list<P>::operator=(batch_list<P> const &other)
{
  if (&other == this)
  {
    return *this;
  }
  assert(num_batch == other.num_batch);
  assert(nrows == other.nrows);
  assert(ncols == other.ncols);
  assert(stride == other.stride);
  assert(do_trans == other.do_trans);
  std::memcpy(batch_list_, other.batch_list_, other.num_batch * sizeof(P *));
  return *this;
}

template<typename P>
batch_list<P>::batch_list(batch_list<P> &&other)
    : num_batch(other.num_batch), nrows(other.nrows), ncols(other.ncols),
      stride(other.stride),
      do_trans(other.do_trans), batch_list_{other.batch_list_}
{
  other.batch_list_ = nullptr;
}

template<typename P>
batch_list<P> &batch_list<P>::operator=(batch_list<P> &&other)
{
  if (&other == this)
  {
    return *this;
  }
  assert(num_batch == other.num_batch);
  assert(nrows == other.nrows);
  assert(ncols == other.ncols);
  assert(stride == other.stride);

  assert(do_trans == other.do_trans);
  batch_list_       = other.batch_list_;
  other.batch_list_ = nullptr;
  return *this;
}

template<typename P>
batch_list<P>::~batch_list()
{
  delete[] batch_list_;
}

template<typename P>
bool batch_list<P>::operator==(batch_list<P> other) const
{
  if (nrows != other.nrows)
  {
    return false;
  }
  if (ncols != other.ncols)
  {
    return false;
  }
  if (stride != other.stride)
  {
    return false;
  }
  if (num_batch != other.num_batch)
  {
    return false;
  }
  if (do_trans != other.do_trans)
  {
    return false;
  }

  for (int i = 0; i < num_batch; ++i)
  {
    if (batch_list_[i] != other.batch_list_[i])
    {
      return false;
    }
  }

  return true;
}

template<typename P>
P *batch_list<P>::operator()(int const position) const
{
  assert(position >= 0);
  assert(position < num_batch);
  return batch_list_[position];
}

// insert the provided view's data pointer
// at the index indicated by position argument
// cannot overwrite previous assignment
template<typename P>
void batch_list<P>::insert(fk::matrix<P, mem_type::view> const a,
                           int const position)
{
  // make sure this matrix is the
  // same dimensions as others in batch
  assert(a.nrows() == nrows);
  assert(a.ncols() == ncols);

  if (a.stride() != stride)
  {
    std::cout << a.stride() << std::endl;

    std::cout << stride << std::endl;
  }
  assert(a.stride() == stride);

  // ensure position is valid
  assert(position >= 0);
  assert(position < num_batch);

  // ensure nothing already assigned
  assert(!batch_list_[position]);

  batch_list_[position] = a.data();
}

// clear one assignment
// returns true if there was a previous assignment,
// false if nothing was assigned
template<typename P>
bool batch_list<P>::clear(int const position)
{
  P *temp               = batch_list_[position];
  batch_list_[position] = nullptr;
  return temp;
}

// get a pointer to the batch_list's
// pointers for batched blas call
// for performance, may have to
// provide a direct access to P**
// from batch_list_, but avoid for now
template<typename P>
P **batch_list<P>::get_list() const
{
  P **const list_copy = new P *[num_batch]();
  std::memcpy(list_copy, batch_list_, num_batch * sizeof(P *));
  return list_copy;
}

// verify that every allocated pointer
// has been assigned to
template<typename P>
bool batch_list<P>::is_filled() const
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
batch_list<P> &batch_list<P>::clear_all()
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
void batched_gemm(batch_list<P> const a, batch_list<P> const b,
                  batch_list<P> const c, P const alpha, P const beta)
{
  // check data validity
  assert(a.is_filled() && b.is_filled() && c.is_filled());

  // check cardinality of sets
  assert(a.num_batch == b.num_batch);
  assert(b.num_batch == c.num_batch);
  int const num_batch = a.num_batch;

  assert(!c.do_trans);

  // check dimensions for gemm
  int const rows_a = a.do_trans ? a.ncols : a.nrows;
  int const cols_a = a.do_trans ? a.nrows : a.ncols;
  int const rows_b = b.do_trans ? b.ncols : b.nrows;
  int const cols_b = b.do_trans ? b.nrows : b.ncols;
  assert(cols_a == rows_b);
  assert(c.nrows == rows_a);
  assert(c.ncols == cols_b);

  // setup blas args
  int m = rows_a;
  int n = cols_b; // technically these should be of op(A) or (B), but our dims
                  // are the same when transposing
  int k                  = cols_a;
  int lda                = a.stride;
  int ldb                = b.stride;
  int ldc                = c.stride;
  char const transpose_a = a.do_trans ? 't' : 'n';
  char const transpose_b = b.do_trans ? 't' : 'n';
  P alpha_               = alpha;
  P beta_                = beta;

  if constexpr (std::is_same<P, double>::value)
  {
    for (int i = 0; i < num_batch; ++i)
    {
      fk::dgemm_(&transpose_a, &transpose_b, &m, &n, &k, &alpha_, a(i), &lda,
                 b(i), &ldb, &beta_, c(i), &ldc);
    }
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    for (int i = 0; i < num_batch; ++i)
    {
      fk::sgemm_(&transpose_a, &transpose_b, &m, &n, &k, &alpha_, a(i), &lda,
                 b(i), &ldb, &beta_, c(i), &ldc);
    }
  }
}

// --- batch allocation code --- /

// static helper - compute how many gemms a single call to batch_for_kronmult
// adds for a given dimension
static int
compute_batch_size(int const degree, int const num_dims, int const dimension)
{
  assert(dimension >= 0);
  assert(dimension < num_dims);
  assert(num_dims > 0);
  assert(degree > 0);

  if (dimension == 0)
  {
    return 1;
  }

  // intermediate levels
  if (dimension < num_dims - 1)
  {
    return std::pow(
        degree, (num_dims - dimension - 1)); /// std::pow(degree, dimension);
  }

  // highest level; single gemm
  return 1;
}

// static helper - compute the dimensions for gemm at a given dimension
static gemm_dims
compute_dimensions(int const degree, int const num_dims, int const dimension)
{
  assert(dimension >= 0);
  assert(dimension < num_dims);
  assert(num_dims > 0);
  assert(degree > 0);

  if (dimension == 0)
  {
    return gemm_dims(degree, degree, degree,
                     static_cast<int>(std::pow(degree, num_dims - 1)));
  }
  return gemm_dims(static_cast<int>(std::pow(degree, dimension)), degree,
                   degree, degree);
}

// create empty batches w/ correct dims/cardinality
template<typename P>
std::vector<batch_set<P>>
allocate_batches(PDE<P> const &pde, int const num_elems)
{
  std::vector<batch_set<P>> batches;

  // FIXME when we allow varying degree by dimension, all
  // this code will have to change...
  int const degree = pde.get_dimensions()[0].get_degree();

  // add the first (lowest dimension) batch
  bool const do_trans   = false;
  int const num_gemms   = pde.num_terms * num_elems;
  gemm_dims const sizes = compute_dimensions(degree, pde.num_dims, 0);

  // get stride of first coefficient matrix in 0th term set.
  // note all the coefficient matrices for each term have the
  // same dimensions
  int const stride = pde.get_terms()[0][0].get_coefficients().stride();
  batches.emplace_back(std::vector<batch_list<P>>{
      batch_list<P>(num_gemms, sizes.rows_a, sizes.cols_a, stride, do_trans),
      batch_list<P>(num_gemms, sizes.rows_b, sizes.cols_b, sizes.rows_b,
                    do_trans),
      batch_list<P>(num_gemms, sizes.rows_a, sizes.cols_b, sizes.rows_a,
                    false)});

  // remaining batches
  for (int i = 1; i < pde.num_dims; ++i)
  {
    int const num_gemms =
        compute_batch_size(degree, pde.num_dims, i) * pde.num_terms * num_elems;
    gemm_dims const sizes = compute_dimensions(degree, pde.num_dims, i);
    bool const trans_a    = false;
    bool const trans_b    = true;

    int const stride = pde.get_terms()[0][i].get_coefficients().stride();
    batches.emplace_back(std::vector<batch_list<P>>{
        batch_list<P>(num_gemms, sizes.rows_a, sizes.cols_a, sizes.rows_a,
                      trans_a),
        batch_list<P>(num_gemms, sizes.rows_b, sizes.cols_b, stride, trans_b),
        batch_list<P>(num_gemms, sizes.rows_a, sizes.rows_b, sizes.rows_a,
                      false)});
  }
  return batches;
}

// --- kronmult batching code --- //

// helper for lowest level of kronmult
template<typename P>
static void
kron_base(fk::matrix<P, mem_type::view> const A,
          fk::vector<P, mem_type::view> x, fk::vector<P, mem_type::view> y,
          batch_set<P> &batch_lists, int const batch_offset, int const degree,
          int const num_dims)
{
  batch_lists[0].insert(A, batch_offset);
  gemm_dims const sizes = compute_dimensions(degree, num_dims, 0);
  fk::matrix<P, mem_type::view> x_view(x, sizes.rows_b, sizes.cols_b);
  batch_lists[1].insert(x_view, batch_offset);
  fk::matrix<P, mem_type::view> y_view(y, sizes.rows_a, sizes.cols_b);
  batch_lists[2].insert(y_view, batch_offset);
}

template<typename P>
void batch_for_kronmult(std::vector<fk::matrix<P, mem_type::view>> const A,
                        fk::vector<P, mem_type::view> x,
                        fk::vector<P, mem_type::view> y,
                        std::vector<fk::vector<P, mem_type::view>> const work,
                        std::vector<batch_set<P>> &batch_lists,
                        int const batch_offset, PDE<P> const &pde)
{
  // FIXME when we allow varying degree by dimension, all
  // this code will have to change...
  int const degree = pde.get_dimensions()[0].get_degree();

  // check vector sizes
  int const result_size = std::pow(degree, pde.num_dims);
  assert(x.size() == result_size);
  assert(y.size() == result_size);

  // check workspace sizes
  assert(static_cast<int>(work.size()) == pde.num_dims - 1);
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

  assert(static_cast<int>(batch_lists.size()) == pde.num_dims);

  assert(batch_offset >= 0);

  if (pde.num_dims == 1)
  {
    kron_base(A[0], x, y, batch_lists[0], batch_offset, degree, pde.num_dims);
    return;
  }

  kron_base(A[0], x, work[0], batch_lists[0], batch_offset, degree,
            pde.num_dims);

  for (int dimension = 1; dimension < pde.num_dims - 1; ++dimension)
  {
    gemm_dims const sizes = compute_dimensions(degree, pde.num_dims, dimension);
    int const num_gemms   = compute_batch_size(degree, pde.num_dims, dimension);
    int const offset      = sizes.rows_a * sizes.cols_a;
    assert((offset * num_gemms) ==
           static_cast<int>(std::pow(degree, pde.num_dims)));

    for (int gemm = 0; gemm < num_gemms; ++gemm)
    {
      fk::matrix<P, mem_type::view> x_view(work[dimension - 1], sizes.rows_a,
                                           sizes.cols_a, offset * gemm);
      batch_lists[dimension][0].insert(x_view, batch_offset * num_gemms + gemm);
      batch_lists[dimension][1].insert(A[dimension],
                                       batch_offset * num_gemms + gemm);
      fk::matrix<P, mem_type::view> work_view(work[dimension], sizes.rows_a,
                                              sizes.cols_a, offset * gemm);
      batch_lists[dimension][2].insert(work_view,
                                       batch_offset * num_gemms + gemm);
    }
  }

  gemm_dims const sizes =
      compute_dimensions(degree, pde.num_dims, pde.num_dims - 1);

  fk::matrix<P, mem_type::view> x_view(work[pde.num_dims - 2], sizes.rows_a,
                                       sizes.cols_a);
  batch_lists[pde.num_dims - 1][0].insert(x_view, batch_offset);
  batch_lists[pde.num_dims - 1][1].insert(A[pde.num_dims - 1], batch_offset);
  fk::matrix<P, mem_type::view> y_view(y, sizes.rows_a, sizes.cols_a);
  batch_lists[pde.num_dims - 1][2].insert(y_view, batch_offset);
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

template<typename P>
std::vector<batch_set<P>>
build_batches(PDE<P> const &pde, element_table const &elem_table,
              fk::vector<P> const &x, fk::vector<P> const &y,
              fk::vector<P> const &work)
{
  // assume uniform degree for now
  int const degree       = pde.get_dimensions()[0].get_degree();
  int const element_size = static_cast<int>(std::pow(degree, pde.num_dims));
  int const x_size       = static_cast<int>(std::pow(degree, pde.num_dims));
  assert(x.size() == x_size);
  // this can be smaller w/ atomic batched gemm e.g. ed's modified magma
  assert(y.size() ==
         x_size * elem_table.size() * elem_table.size() * pde.num_terms);

  // intermediate workspaces for kron product. can be reduced to 3/5ths
  // this size for the 6 dimensional case if we alternate input/output spaces
  // per dimension.
  assert(work.size() == y.size() * (pde.num_dims - 1));

  std::vector<batch_set<P>> batches =
      allocate_batches<P>(pde, elem_table.size());

  // loop over elements
  // FIXME eventually want to do this in parallel
  for (int i = 0; i < elem_table.size(); ++i)
  {
    // first, get linearized indices for this element
    //
    // calculate from the level/cell indices for each
    // dimension
    fk::vector<int> coords = elem_table.get_coords(i);
    assert(coords.size() == pde.num_dims * 2);
    fk::vector<int> elem_indices = linearize(coords);

    // calculate the position of this element in the
    // global system matrix
    // int const global_row = i * element_size;
    // this is where the item reduces into; TODO
    // once I start on reduction

    // calculate the row portion of the
    // operator position used for this
    // element's gemm calls
    fk::vector<int> operator_row(pde.num_dims);
    for (int j = 0; j < pde.num_dims; ++j)
    {
      // FIXME here we would have to use each dimension's
      // degree when calculating the index if we want different
      // degree in each dim
      operator_row(j) = elem_indices(j) * degree;
    }
    // loop over connected elements. for now, we assume
    // full connectivity
    for (int j = 0; j < elem_table.size(); ++j)
    {
      // get linearized indices for this connected element
      fk::vector<int> coords = elem_table.get_coords(j);
      assert(coords.size() == pde.num_dims * 2);
      fk::vector<int> connected_indices = linearize(coords);

      // calculate the position of this element in the
      // global system matrix
      int const global_col = j * element_size;

      // calculate the row portion of the
      // operator position used for this
      // element's gemm calls
      fk::vector<int> operator_col(pde.num_dims);

      for (int k = 0; k < pde.num_dims; ++k)
      {
        // FIXME here we would have to use each dimension's
        // degree when calculating the index if we want different
        // degree in each dim
        operator_col(k) = connected_indices(k) * degree;
      }

      auto terms = pde.get_terms();
      for (int k = 0; k < pde.num_terms; ++k)
      {
        auto term = terms[k];

        // term major y-space layout, followed by connected items, finally work
        // items note that this index assumes uniform connected items (full
        // connectivity) if connectivity is instead computed for each element, i
        // * elem_table.size() must be replaced by sum of lower indexed
        // connected items (scan)
        int const kron_index = k + j * pde.num_terms + i * elem_table.size();
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * (pde.num_dims - 1);

        fk::vector<P, mem_type::view> y_view(y, y_index, y_index + x_size - 1);

        std::vector<fk::vector<P, mem_type::view>> work_views(
            pde.num_dims - 1, fk::vector<P, mem_type::view>(
                                  work, work_index, work_index + x_size - 1));
        for (int d = 1; d < pde.num_dims; ++d)
        {
          work_views[i] = fk::vector<P, mem_type::view>(
              work, work_index + x_size * d, work_index + (x_size + 1) * d - 1);
        }

        std::vector<fk::matrix<P, mem_type::view>> operator_views;
        for (int d = 0; d < pde.num_dims; ++d)
        {
          // FIXME need to figure out how to avoid the coefficient copy
          operator_views.emplace_back(fk::matrix<P, mem_type::view>(
              term[d].get_coefficients(), operator_row(d), operator_col(d),
              degree, degree));
        }
        fk::vector<P, mem_type::view> x_view(x, global_col,
                                             global_col + x_size - 1);
        batch_for_kronmult(operator_views, x_view, y_view, work_views, batches,
                           kron_index, pde);
      }
    }
  }
  return batches;
}

template class batch_list<float>;
template class batch_list<double>;

template void batched_gemm(batch_list<float> const a, batch_list<float> const b,
                           batch_list<float> const c, float const alpha,
                           float const beta);

template void
batched_gemm(batch_list<double> const a, batch_list<double> const b,
             batch_list<double> const c, double const alpha, double const beta);
template std::vector<batch_set<float>>
allocate_batches(PDE<float> const &pde, int const num_elems);
template std::vector<batch_set<double>>
allocate_batches(PDE<double> const &pde, int const num_elems);

template void
batch_for_kronmult(std::vector<fk::matrix<float, mem_type::view>> const A,
                   fk::vector<float, mem_type::view> x,
                   fk::vector<float, mem_type::view> y,
                   std::vector<fk::vector<float, mem_type::view>> const work,
                   std::vector<batch_set<float>> &batch_lists,
                   int const batch_offset, PDE<float> const &pde);

template void
batch_for_kronmult(std::vector<fk::matrix<double, mem_type::view>> const A,
                   fk::vector<double, mem_type::view> x,
                   fk::vector<double, mem_type::view> y,
                   std::vector<fk::vector<double, mem_type::view>> const work,
                   std::vector<batch_set<double>> &batch_lists,
                   int const batch_offset, PDE<double> const &pde);

template std::vector<batch_set<float>>
build_batches(PDE<float> const &pde, element_table const &elem_table,
              fk::vector<float> const &x, fk::vector<float> const &y,
              fk::vector<float> const &work);
template std::vector<batch_set<double>>
build_batches(PDE<double> const &pde, element_table const &elem_table,
              fk::vector<double> const &x, fk::vector<double> const &y,
              fk::vector<double> const &work);
