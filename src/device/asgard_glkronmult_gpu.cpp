
#include "asgard_kronmult.hpp"

namespace asgard::kronmult
{
#ifdef KRON_MODE_GLOBAL
#ifdef ASGARD_USE_CUDA

template<typename precision>
void split_pattern(std::vector<int> const &pntr, std::vector<int> const &indx,
                   std::vector<int> const &diag, std::vector<precision> const &vals,
                   permutes::matrix_fill mode,
                   gpu::vector<int> &gpntr, gpu::vector<int> &gindx,
                   gpu::vector<int> &gvals)
{
  if (mode == permutes::matrix_fill::both)
  {
    if (gpntr.empty())
      gpntr = pntr;
    if (gindx.empty())
      gindx = indx;
    gvals = vals;
  }

  // copy the lower/upper part of the pattern into the vectors prefixed with "a"
  std::vector<int> apntr(pntr.size());
  std::vector<int> aindx;
  aindx.reserve(indx.size());
  std::vector<precision> avals;
  avals.reserve(avals.size());

  for (size_t r = 0; r < apntr.size(); r++)
  {
    apntr[r] = static_cast<int>(aindx.size());
    if (mode == permutes::matrix_fill::lower)
      for (int j = pntr[r]; r < diag[r]; r++)
      {
        aindx.push_back(indx[j]);
        avals.push_back(vals[j]);
      }
    else
      for (int j = diag[r]; r < pntr[r + 1]; r++)
      {
        aindx.push_back(indx[j]);
        avals.push_back(vals[j]);
      }
  }
  apntr.back() = static_cast<int>(aindx.size());

  // send the result to the gpu
  if (gpntr.empty())
    gpntr = apntr;
  if (gindx.empty())
    gindx = aindx;
  gvals = avals;
}

template<typename precision>
global_gpu_operations::
global_gpu_operations(gpu::sparse_handle const &hndl, int num_dimensions,
                      std::vector<permutes> const &perms,
                      std::vector<std::vector<int>> const &gpntr,
                      std::vector<std::vector<int>> const &gindx,
                      std::vector<std::vector<precision>> const &gvals,
                      std::vector<int> const &terms,
                      precision const *x, precision *y,
                      precision *work1, precision *work2)
    // each matrix has 3 potential variants (both, lower, upper)
  : hndl(hndl_), gpntr_(gpntr.size()), gindx_(gindx.size()), gvals_(gvals.size())
{
  int64_t const num_rows = static_cast<int64_t>(gpntr.front().size() - 1);

  mats_.reserve(terms.size() * num_dimensions); // max possible number of matrices

  for (int t : terms)
  {
    // terms can have different effective dimension, since some of them are identity
    permutes const &perm = perms[t];
    int const dims       = perm.num_dimensions();
    if (dims == 0) // all terms here are identity, redundant term, why is it here?
      continue;

    for (size_t i = 0; i < perm.fill.size(); i++)
    {
      precision *w1 = work1;
      precision *w2 = work2;

      precision ybeta = (i == 0) ? 0 : 1; // first one zeros out y, then we increment

      for (int d = 0; d < dims; d++)
      {
        int dir                    = perm.direction[i][d];
        permutes::matrix_fill mode = perm.fill[i][d];
        int const mode_id          = (mode == permutes::matrix_fill::both) ? 0 : ((mode == permutes::matrix_fill::lower) ? 1 : 2);

        // pattern and value ids for the matrices
        int const pid = 3 * dir + mode_id;
        int const vid = 3 * (t * num_dimensions + dir) + mode_id;

        if (gpntr_[pid].empty())
          gpntr_[pid] = gpntr[pid];
        if (gindx_[pid].empty())
          gindx_[pid] = gindx[pid];
        if (gvals_[vid].empty())
          gvals_[vid] = gvals[pid];

        mats_.push_back(gpu::sparse_matrix(num_rows, num_rows, gindx_[pid].size(),
                                           gpntr_[pid].data(), gindx_[pid].data(), gvals_[vid].data()));

        if (d == 0 and d == dims - 1) // only one matrix
          mats_.back().set_vectors(num_rows, precision{1}, x, ybeta, y);
        else if (d == 0)
          mats_.back().set_vectors(num_rows, precision{1}, x, precision{0}, w2);
        else if (d == dims - 1)
          mats_.back().set_vectors(num_rows, precision{1}, w1, ybeta, y);
        else
          mats_.back().set_vectors(num_rows, precision{1}, w1, precision{0}, w2);

        std::swap(w1, w2);
      }
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct global_gpu_operations<double>;

template global_gpu_operations<double>::global_gpu_operations(gpu::sparse_handle const &hndl, int num_dimensions,
                      std::vector<permutes> const &perms,
                      std::vector<std::vector<int>> const &gpntr,
                      std::vector<std::vector<int>> const &gindx,
                      std::vector<std::vector<precision>> const &gvals,
                      std::vector<int> const &terms,
                      precision const *x, precision *y,
                      precision *work1, precision *work2);
#endif
#ifdef ASGARD_ENABLE_FLOAT
template struct global_gpu_operations<float>;
#endif

#endif
#endif
} // namespace asgard::kronmult
