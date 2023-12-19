
#include "asgard_kronmult.hpp"

namespace asgard::kronmult
{
#ifdef KRON_MODE_GLOBAL
#ifdef ASGARD_USE_CUDA

template<typename precision>
global_gpu_operations<precision>
::global_gpu_operations(gpu::sparse_handle const &hndl, int num_dimensions,
                        std::vector<permutes> const &perms,
                        std::vector<std::vector<int>> const &gpntr,
                        std::vector<std::vector<int>> const &gindx,
                        std::vector<std::vector<precision>> const &gvals,
                        std::vector<int> const &terms,
                        precision *x, precision *y,
                        precision *work1, precision *work2)
    // each matrix has 3 potential variants (both, lower, upper)
  : hndl_(hndl), gpntr_(gpntr.size()), gindx_(gindx.size()), gvals_(gvals.size()),
    buffer_(nullptr)
{
  int64_t const num_rows = static_cast<int64_t>(gpntr.front().size() - 1);

  size_t num_matrices = 0;
  for (int t : terms)
    num_matrices += perms[t].num_dimensions() * perms[t].fill.size();
  mats_.reserve(num_matrices); // max possible number of matrices

  // the first matrix will zero out the y
  // follow on operations will switch to increment after
  precision ybeta = 0;

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

      for (int d = 0; d < dims; d++)
      {
        int dir                    = perm.direction[i][d];
        permutes::matrix_fill mode = perm.fill[i][d];
        int const mode_id          = (mode == permutes::matrix_fill::both) ? 0 :
                                     ((mode == permutes::matrix_fill::lower) ? 1 : 2);

        std::cout << " permute, term = " << t << " - dir = " << dir << "  mode = " << perm.fill_name(i, d) << "\n";

        // pattern and value ids for the matrices
        int const pid = 3 * dir + mode_id;
        int const vid = 3 * t * num_dimensions + pid;

        if (gpntr_[pid].empty())
          gpntr_[pid] = gpntr[pid];
        if (gindx_[pid].empty())
          gindx_[pid] = gindx[pid];
        if (gvals_[vid].empty())
          gvals_[vid] = gvals[vid];

        mats_.emplace_back(num_rows, num_rows, gindx_[pid].size(),
                                          gpntr_[pid].data(), gindx_[pid].data(),
                                          gvals_[vid].data());

        if (d == 0 and d == dims - 1) // only one matrix
        {
          mats_.back().set_vectors(num_rows, precision{1}, x, ybeta, y);
          ybeta = precision{1};
        }
        else if (d == 0)
          mats_.back().set_vectors(num_rows, precision{1}, x, precision{0}, w2);
        else if (d == dims - 1)
        {
          mats_.back().set_vectors(num_rows, precision{1}, w1, ybeta, y);
          ybeta = precision{1};
        }
        else
          mats_.back().set_vectors(num_rows, precision{1}, w1, precision{0}, w2);

        std::swap(w1, w2);
      }
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct global_gpu_operations<double>;
#endif
#ifdef ASGARD_ENABLE_FLOAT
template struct global_gpu_operations<float>;
#endif

#endif
#endif
} // namespace asgard::kronmult
