#pragma once
#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"

// -----------------------------------------------------------------------------
// permutations
// this components's purpose is to provide utilities used in
// construction of the element table
// -----------------------------------------------------------------------------

/*!
 * \defgroup permuts Permutations and index selection
 *
 * Collection of algorithms for generating multi-indexes
 * with different order and selection criteria.
 */

namespace asgard::permutations
{
/*!
 * \ingroup permuts
 * \brief Generates the list of multi-indexes that satisfy a general selection
 * criteria.
 *
 * A helper method to select_indexex(), it generates a list of vectors,
 * where the first index is in entries 0 to `num_dims-1`
 * and all follow on indexes are appended one after the other;
 * the order of the indexes is lexicographical ascending starting with the first
 * dimension.
 */
inline std::vector<int> generate_lower_index_set(
    size_t num_dims, std::function<bool(std::array<int, max_num_dimensions> const &index)> inside)
{
  size_t c   = 0;
  bool is_in = true;
  std::array<int, max_num_dimensions> root;
  std::fill_n(root.begin(), num_dims, 0);
  std::vector<int> indexes;
  while (is_in || (c < num_dims))
  {
    if (is_in)
    {
      indexes.insert(indexes.end(), root.begin(), root.begin() + num_dims);
      c = 0;
      root[c]++;
    }
    else
    {
      std::fill(root.begin(), root.begin() + c + 1, 0);
      c++;
      if (c == num_dims)
        break;
      root[c]++;
    }
    is_in = inside(root);
  }
  return indexes;
}
/*!
 * \ingroup permuts
 * \brief Generates the list of multi-indexes that satisfy a general selection
 * criteria.
 *
 * Identical to generate_lower_index_set(), but returns the list sorted in
 * levels. The first vector in the result will always be the zeroth vector, the
 * second vector will contain the selected set of indexes from among the indexes
 * that have one entry equal to 1, the L-th vector contains the selected indexes
 * with sum of the entries equal to L.
 */
inline std::vector<std::vector<int>> generate_lower_index_level_sets(
    size_t num_dims, std::function<bool(std::array<int, max_num_dimensions> const &index)> inside)
{
  size_t c = 0, level = 0;
  bool is_in = true;
  std::array<int, max_num_dimensions> root;
  std::fill_n(root.begin(), num_dims, 0);
  std::vector<std::vector<int>> indexes(1);
  while (is_in || (c < num_dims))
  {
    if (is_in)
    {
      if (level == indexes.size())
      {
        indexes.emplace_back(root.begin(), root.begin() + num_dims);
      }
      else
      {
        indexes[level].insert(indexes[level].end(), root.begin(), root.begin() + num_dims);
      }
      c = 0;
      root[c]++;
      level++;
    }
    else
    {
      std::fill(root.begin(), root.begin() + c + 1, 0);
      c++;
      if (c == num_dims)
        break;
      root[c]++;
      level = root[0];
      for (size_t i = 1; i < num_dims; i++)
        level += root[i];
    }
    is_in = inside(root);
  }
  return indexes;
}
/*!
 * \ingroup permuts
 * \brief Converts the output of generate_lower_index_set() into
 * fk::matrix<int>.
 *
 * \param num_dims must be the same as in the call to generate_lower_index_set()
 * \param indexes must be the output of generate_lower_index_set()
 *
 * \returns an fk::matrix<int> containing all the indexes.
 */
inline fk::matrix<int>
transp_data(int const num_dims, std::vector<int> const &indexes)
{
  int const nrows = indexes.size() / num_dims;
  fk::matrix<int> result(nrows, num_dims);
  for (int i = 0; i < nrows; i++)
  {
    for (int j = 0; j < num_dims; j++)
    {
      result(i, j) = indexes[i * num_dims + j];
    }
  }
  return result;
}
/*!
 * \ingroup permuts
 * \brief Converts the output of generate_lower_index_level_sets() into
 * fk::matrix<int>.
 *
 * See transp_data().
 *
 * \b levels_in_reverse will switch the order of levels, so the zeroth-vector
 * will be last.
 */
inline fk::matrix<int> transp_data(int const num_dims, bool levels_in_reverse,
                                   std::vector<std::vector<int>> const &indexes)
{
  int nrows = 0;
  for (auto const &ivec : indexes)
    nrows += ivec.size() / num_dims;       // count all indexes
  fk::matrix<int> result(nrows, num_dims); // allocate the matrix

  if (levels_in_reverse)
  {
    int count = nrows; // count the current set of rows
    for (auto const &ivec : indexes)
    {
      nrows = ivec.size() / num_dims;
      count -= nrows;
      for (int i = 0; i < nrows; i++)
      {
        for (int j = 0; j < num_dims; j++)
        {
          result(i + count, j) = ivec[i * num_dims + j];
        }
      }
    }
  }
  else
  {
    int count = 0; // count the current set of rows
    for (auto const &ivec : indexes)
    {
      nrows = ivec.size() / num_dims;
      for (int i = 0; i < nrows; i++)
      {
        for (int j = 0; j < num_dims; j++)
        {
          result(i + count, j) = ivec[i * num_dims + j];
        }
      }
      count += nrows;
    }
  }
  return result;
}
/*!
 * \ingroup permuts
 * \brief Generates a set of multi-indexes and returns them into a matrix with
 * the appropriate sorting order.
 *
 * \param num_dims is the number of dimensions or the size of the multi-index
 * tuple, must be at least 1 \param order_by_n indicates whether the indexes
 * should be sorted by the level, if order_by_n is true, and the sum of entries
 * of i is less than the sum entries of j, then i will be on a row before j
 * \param levels_in_reverse indicates whether the zeroth vector should be first
 * (false) or last (true) in the ordering, used only when order_by_n is true.
 * \param inside is a callable method that returns true whenever the provided
 * index satisfies the criteria
 * - the input to \b inside will always be a vector of size num_dims
 * - the criteria given by inside must describe a lower completer set
 *   a set \b A is lower complete whenever \f$ j \leq i \f$ and \f$ i \in A \f$
 * guarantees \f$ j \in A \f$ (an alternative definition is that a lower
 * complete set is the union of dense/full sets)
 *
 * \returns fk::matrix<int> containing the selected multi-indexes
 */
inline fk::matrix<int>
select_indexex(
    int const num_dims, bool order_by_n, bool levels_in_reverse,
    std::function<bool(std::array<int, max_num_dimensions> const &index)> inside)
{
  if (order_by_n)
  {
    return transp_data(num_dims, levels_in_reverse,
                       generate_lower_index_level_sets(num_dims, inside));
  }
  else
  {
    return transp_data(num_dims, generate_lower_index_set(num_dims, inside));
  }
}

/*!
 * \ingroup permuts
 * \brief Select multi-indexes with sum of the entries below a limit.
 *
 * \param num_dims is dimensions of the multi-indexes
 * \param limit is the cut-off criteria, selected indexes will have
 *        vector one norm (or sum of entries) less or equal to this limit
 * \param order_by_n indicates the order of the indexes, if set to true the
 * indexes will be grouped by matching the sum of indexes
 *
 * \returns matrix with num_dims columns where each row is a multi-index
 */
fk::matrix<int>
get_lequal(int const num_dims, int const limit, bool const order_by_n);
/*!
 * \ingroup permuts
 * \brief Select similar to get_lequal() but also applies limits in different
 * directions.
 *
 * Uses the same selection as permutations::get_lequal() but imposes additional
 * constraint in each direction.
 * Every index entry will be no greater than the corresponding entry in \b
 * levels, naturally, expecting levels.size() == num_dims and levels[j] >= 0.
 */
fk::matrix<int> get_lequal_multi(fk::vector<int> const &levels,
                                 int const num_dims, int const limit,
                                 bool const increasing_sum_order);

/*!
 * \ingroup permuts
 * \brief Splits the dimensions in two groups, uses sparse grid within the group, but full grid between the groups.
 *
 * The two groups are 0 ... num_first_group -1 and num_first_group ... num_dims
 * - 1. Sparse grid selection is used within each group, but the two groups are
 * tensored.
 */
fk::matrix<int> get_mix_leqmax_multi(fk::vector<int> const &levels,
                                     int const num_dims,
                                     fk::vector<int> const &mixed_max,
                                     int const num_first_group,
                                     bool const increasing_sum_order);

/*!
 * \ingroup permuts
 * \brief Generates multi-indexes for a full/dense grid.
 *
 * \param num_dims is dimensions of the multi-indexes
 * \param limit is the largest index entry
 * \param last_index_decreasing indicates whether to sort the indexes
 *        in ascending (false) or descending order
 *
 * \returns matrix with num_dims columns where each row is a multi-index
 */
fk::matrix<int>
get_max(int const num_dims, int const limit, bool const last_index_decreasing);

/*!
 * \ingroup permuts
 * \brief Generates non-uniform (anisotropic) multi-indexes for a full/dense
 * grid.
 *
 * \param levels indicates the largest index in each direction,
 *        expecting levels.size() == num_dims and levels[j] >= 0
 * \param num_dims is dimensions of the multi-indexes
 * \param last_index_decreasing indicates whether to sort the indexes
 *        in ascending (false) or descending order
 *
 * \returns matrix with num_dims columns where each row is a multi-index
 */
fk::matrix<int> get_max_multi(fk::vector<int> const &levels, int const num_dims,
                              bool const last_index_decreasing);

using list_set = std::vector<fk::vector<int>>;

// Index counter

int count_leq_max_indices(list_set const &lists, int const num_dims,
                          int const max_sum, int const max_val);

// Index finder

fk::matrix<int> get_leq_max_indices(list_set const &lists, int const num_dims,
                                    int const max_sum, int const max_val);

} // namespace asgard::permutations
