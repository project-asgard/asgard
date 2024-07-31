#include "asgard_indexset.hpp"

namespace asgard
{
template<typename data_container>
indexset make_index_set(organize2d<int, data_container> const &indexes)
{
  int64_t num_indexes    = indexes.num_strips();
  int64_t num_dimensions = indexes.stride();

  // compute a map that gives the sorted order of the indexes
  std::vector<int64_t> map(num_indexes);
  std::iota(map.begin(), map.end(), 0);

  std::sort(map.begin(), map.end(),
            [&](int64_t a, int64_t b) -> bool {
              for (int64_t d = 0; d < num_dimensions; d++)
              {
                if (indexes[a][d] < indexes[b][d])
                  return true;
                if (indexes[a][d] > indexes[b][d])
                  return false;
              }
              return false; // equal should be false, as in < operator
            });

  // in the sorted order, it's easy to find repeated entries
  int64_t repeated_indexes = 0;
  for (int64_t i = 0; i < num_indexes - 1; i++)
  {
    bool is_repeated = [&]() -> bool {
      for (int64_t d = 0; d < num_dimensions; d++)
        if (indexes[map[i]][d] != indexes[map[i + 1]][d])
          return false;
      return true;
    }();
    if (is_repeated)
    {
      map[i] = -1;
      repeated_indexes += 1;
    }
  }

  // map the unique entries into a vector and load into an indexset
  std::vector<int> sorted_indexes;
  sorted_indexes.reserve(num_dimensions * (num_indexes - repeated_indexes));
  for (int64_t i = 0; i < num_indexes; i++)
  {
    if (map[i] != -1)
      sorted_indexes.insert(sorted_indexes.end(), indexes[map[i]],
                            indexes[map[i]] + num_dimensions);
  }

  return indexset(num_dimensions, std::move(sorted_indexes));
}

template indexset
make_index_set(organize2d<int, std::vector<int>> const &indexes);
template indexset
make_index_set(organize2d<int, int *> const &indexes);
template indexset
make_index_set(organize2d<int, int const *> const &indexes);

dimension_sort::dimension_sort(indexset const &iset) : iorder_(iset.num_dimensions())
{
  int num_dimensions = iset.num_dimensions();
  int num_indexes    = iset.num_indexes();

  iorder_ = std::vector<std::vector<int>>(num_dimensions, std::vector<int>(num_indexes));

#pragma omp parallel for
  for (int d = 0; d < num_dimensions - 1; d++)
  {
    // for each dimension, use the map to group points together
    std::iota(iorder_[d].begin(), iorder_[d].end(), 0);
    std::sort(iorder_[d].begin(), iorder_[d].end(), [&](int a, int b) -> bool {
      const int *idxa = iset[a];
      const int *idxb = iset[b];
      for (int j = 0; j < num_dimensions; j++)
      {
        if (j != d)
        {
          if (idxa[j] < idxb[j])
            return true;
          if (idxa[j] > idxb[j])
            return false;
        }
      }
      // lexigographical order, dimension d is the fastest moving one
      if (idxa[d] < idxb[d])
        return true;
      if (idxa[d] > idxb[d])
        return false;
      return false;
    });
  }
  // no sort needed for the last dimenison
  std::iota(iorder_[num_dimensions - 1].begin(), iorder_[num_dimensions - 1].end(), 0);

  // check if multi-indexes match in all but one dimension
  auto match_outside_dim = [&](int d, int const *a, int const *b) -> bool {
    for (int j = 0; j < d; j++)
      if (a[j] != b[j])
        return false;
    for (int j = d + 1; j < num_dimensions; j++)
      if (a[j] != b[j])
        return false;
    return true;
  };

  // offsets of sorted "vectors" in each dimension
  pntr_ = std::vector<std::vector<int>>(num_dimensions);

  // split the map into vectors of identical coefficients in all but 1d
  if (num_dimensions == 1)
  {
    pntr_[0].push_back(0);
    pntr_[0].push_back(num_indexes);
  }
  else
  {
#pragma omp parallel for
    for (int d = 0; d < num_dimensions; d++)
    {
      int const *c_index = iset[iorder_[d][0]];
      pntr_[d].push_back(0);
      for (int i = 1; i < num_indexes; i++)
      {
        if (not match_outside_dim(d, c_index, iset[iorder_[d][i]]))
        {
          pntr_[d].push_back(i);
          c_index = iset[iorder_[d][i]];
        }
      }
      pntr_[d].push_back(num_indexes);
    }
  }
}

dimension_sort::dimension_sort(vector2d<int> const &list) : iorder_(list.stride())
{
  int num_dimensions = list.stride();
  int num_indexes    = list.num_strips();

  iorder_ = std::vector<std::vector<int>>(num_dimensions, std::vector<int>(num_indexes));

#pragma omp parallel for
  for (int d = 0; d < num_dimensions; d++)
  {
    // for each dimension, use the map to group points together
    std::iota(iorder_[d].begin(), iorder_[d].end(), 0);
    std::sort(iorder_[d].begin(), iorder_[d].end(), [&](int a, int b) -> bool {
      const int *idxa = list[a];
      const int *idxb = list[b];
      for (int j = 0; j < num_dimensions; j++)
      {
        if (j != d)
        {
          if (idxa[j] < idxb[j])
            return true;
          if (idxa[j] > idxb[j])
            return false;
        }
      }
      // lexigographical order, dimension d is the fastest moving one
      if (idxa[d] < idxb[d])
        return true;
      if (idxa[d] > idxb[d])
        return false;
      return false;
    });
  }

  // check if multi-indexes match in all but one dimension
  auto match_outside_dim = [&](int d, int const *a, int const *b) -> bool {
    for (int j = 0; j < d; j++)
      if (a[j] != b[j])
        return false;
    for (int j = d + 1; j < num_dimensions; j++)
      if (a[j] != b[j])
        return false;
    return true;
  };

  // offsets of sorted "sparse vectors" in each dimension
  pntr_ = std::vector<std::vector<int>>(num_dimensions);

  // split the map into vectors of identical coefficients in all but 1d
  if (num_dimensions == 1)
  {
    pntr_[0].push_back(0);
    pntr_[0].push_back(num_indexes);
  }
  else
  {
#pragma omp parallel for
    for (int d = 0; d < num_dimensions; d++)
    {
      int const *c_index = list[iorder_[d][0]];
      pntr_[d].push_back(0);
      for (int i = 1; i < num_indexes; i++)
      {
        if (not match_outside_dim(d, c_index, list[iorder_[d][i]]))
        {
          pntr_[d].push_back(i);
          c_index = list[iorder_[d][i]];
        }
      }
      pntr_[d].push_back(num_indexes);
    }
  }
}

/*!
 * \brief Make a callback for all 1d ancestors of the set
 *
 * Given a set of indexes and 1d ancestry, constructs all the ancestors
 * (on the fly, not explicitly) and makes a callback for each one.
 * The ancestors work in 1d only, this does not consider the cross terms.
 * - the method can be called recursively to process all ancestors in all
 *   directions
 * - the method work for cases like edge neighbors, where we process one
 *   "parent" only
 *
 * The scratch-space must have size iset.num_dimensions() and will be filled
 * with one ancestor at a time, but the user should not touch the scratch
 * directly.
 * The callback should be a callable that accepts std::vector<int> const&
 *
 * Usage:
 * \code
 *   std::vector<int> scratch(iset.num_dimensions());
 *   parse_ancestry_1d(iset, connect_1d(max_level), scratch,
 *                     [](std::vector<int> const &ancestor)
 *                       -> void {
 *                         if (iset.missing(ancestor))
 *                           std::cout << " iset is not complete \n";
 *                       });
 * \endcode
 *
 */
template<typename callback_lambda>
void parse_ancestry_1d(indexset const &iset, connect_1d const &ancestry,
                       std::vector<int> &scratch, callback_lambda callback)
{
  int const num_dimensions = iset.num_dimensions();
  expect(scratch.size() == static_cast<size_t>(num_dimensions));

  for (int i = 0; i < iset.num_indexes(); i++)
  {
    // construct all parents, even considering the edges
    std::copy_n(iset[i], num_dimensions, scratch.begin());
    // check the parents in each direction
    for (int d = 0; d < num_dimensions; d++)
    {
      int const row = scratch[d];
      for (int j = ancestry.row_begin(row); j < ancestry.row_diag(row); j++)
      {
        scratch[d] = ancestry[j];
        callback(scratch);
      }
      scratch[d] = row;
    }
  }
}

indexset compute_ancestry_completion(indexset const &iset,
                                     connect_1d const &hierarchy)
{
  int const num_dimensions = iset.num_dimensions();

  // store all missing ancestors here
  vector2d<int> missing_ancestors(num_dimensions, 0);

  // workspace for the algorithms
  std::vector<int> scratch(num_dimensions);

  // do just one pass, considering the indexes in the iset only
  // after this, missing_ancestors will hold those from iset
  // we need to recurs only on the missing_ancestors from now on
  parse_ancestry_1d(iset, hierarchy, scratch,
                    [&](std::vector<int> const &ancestor)
                        -> void {
                      if (iset.missing(ancestor))
                        missing_ancestors.append(ancestor);
                    });

  bool ancestry_complete = missing_ancestors.empty();

  indexset pad_indexes = make_index_set(missing_ancestors);

  // the assumption here it that the padded set will be smaller
  // then the iset, so we do one loop over the large set and then we work
  // only with the smaller pad_indexes
  while (not ancestry_complete)
  {
    // all new found indexes are already in pad_indexes
    // missing_ancestors holds the ones from this iteration only
    missing_ancestors.clear();

    parse_ancestry_1d(pad_indexes, hierarchy, scratch,
                      [&](std::vector<int> const &ancestor)
                          -> void {
                        if (iset.missing(ancestor) and
                            pad_indexes.missing(ancestor))
                          missing_ancestors.append(ancestor);
                      });

    // check if every ancestor is already in either iset or pad_indexes
    ancestry_complete = missing_ancestors.empty();

    if (not ancestry_complete)
    {
      // add the new indexes into the pad_indexes (could be improved)
      missing_ancestors.append(pad_indexes[0], pad_indexes.num_indexes());
      pad_indexes = make_index_set(missing_ancestors);
    }
  }

  return pad_indexes;
}

/*!
 * \brief Helper method, fills the indexes with the polynomial degree of freedom
 *
 * The cells are the current set of cells to process,
 * pdof is the number of polynomial terms,
 * e.g., 2 for linear and 3 for quadratic.
 * tsize is the size of the tensor within a cell,
 * i.e., tsize = pdof to power num_dimensions
 */
template<typename itype>
void complete_poly_order(span2d<itype> const &cells, int64_t pdof,
                         int64_t tsize, span2d<int> indexes)
{
  int num_dimensions = cells.stride();
  int64_t num_cells  = cells.num_strips();

#pragma omp parallel for
  for (int64_t i = 0; i < num_cells; i++)
  {
    int const *cell = cells[i];

    for (int64_t ipoly = 0; ipoly < tsize; ipoly++)
    {
      int64_t t = ipoly;
      int *idx  = indexes[i * tsize + ipoly];

      for (int d = num_dimensions - 1; d >= 0; d--)
      {
        idx[d] = cell[d] * pdof + static_cast<int>(t % pdof);
        t /= pdof;
      }
    }
  }
}

vector2d<int> complete_poly_order(vector2d<int> const &cells, int degree)
{
  int const num_dimensions = cells.stride();

  int64_t const num_cells = cells.num_strips();

  int64_t const pdof = degree + 1;

  int64_t const tsize = fm::ipow(pdof, num_dimensions);

  vector2d<int> indexes(num_dimensions, tsize * num_cells);

  complete_poly_order(
      span2d(num_dimensions, num_cells, cells[0]), pdof, tsize,
      span2d(num_dimensions, tsize * num_cells, indexes[0]));

  return indexes;
}

vector2d<int> complete_poly_order(vector2d<int> const &cells,
                                  indexset const &padded, int degree)
{
  expect(padded.num_indexes() == 0 or padded.num_dimensions() == cells.stride());

  int num_dimensions = cells.stride();

  int64_t const num_cells  = cells.num_strips();
  int64_t const num_padded = padded.num_indexes();

  int64_t const pdof = degree + 1;

  int64_t const tsize = fm::ipow(pdof, num_dimensions);

  vector2d<int> indexes(num_dimensions, tsize * (num_cells + num_padded));

  complete_poly_order(
      span2d(num_dimensions, num_cells, cells[0]), pdof, tsize,
      span2d(num_dimensions, tsize * num_cells, indexes[0]));

  if (num_padded > 0)
    complete_poly_order(
        span2d(num_dimensions, num_padded, padded[0]), pdof, tsize,
        span2d(num_dimensions, tsize * num_padded, indexes[tsize * num_cells]));

  return indexes;
}

} // namespace asgard
