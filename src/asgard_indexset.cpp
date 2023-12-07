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
            [&](int64_t a, int64_t b)-> bool{
              for(int64_t d=0; d<num_dimensions; d++)
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
  for(int64_t i=0; i<num_indexes - 1; i++)
  {
    bool is_repeated = [&]()
        ->bool {
            for(int64_t d=0; d<num_dimensions; d++)
              if (indexes[map[i]][d] != indexes[map[i+1]][d])
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
  for(int64_t i=0; i<num_indexes; i++)
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


dimension_sort::dimension_sort(indexset const &iset) : map_(iset.num_dimensions())
{
  int num_dimensions = iset.num_dimensions();
  int num_indexes    = iset.num_indexes();
  map_ = std::vector<std::vector<int>>(num_dimensions, std::vector<int>(num_indexes));

  #pragma omp parallel for
  for(int d=0; d<num_dimensions -1; d++) {
      // for each dimension, use the map to group points together
      std::iota(map_[d].begin(), map_[d].end(), 0);
      std::sort(map_[d].begin(), map_[d].end(), [&](int a, int b)->bool{
        const int * idxa = iset[a];
        const int * idxb = iset[b];
        for(int j=0; j<num_dimensions; j++) {
          if (j != d){
            if (idxa[j] < idxb[j]) return true;
            if (idxa[j] > idxb[j]) return false;
          }
        }
        // lexigographical order, dimension d is the fastest moving one
        if (idxa[d] < idxb[d]) return true;
        if (idxa[d] > idxb[d]) return false;
        return false;
      });
  }
  // no sort needed for the last dimenison
  std::iota(map_[num_dimensions-1].begin(), map_[num_dimensions-1].end(), 0);

  // check if multi-indexes match in all but one dimension
  auto match_outside_dim = [&](int d, int const*a, int const *b) -> bool {
      for(int j=0; j<d; j++)
        if (a[j] != b[j]) return false;
      for(int j=d+1; j<num_dimensions; j++)
        if (a[j] != b[j]) return false;
      return true;
    };

  // offsets of sorted "vectors" in each dimension
  pntr_ = std::vector<std::vector<int>>(num_dimensions);

  // split the map into vectors of identical coefficients in all but 1d
  if (num_dimensions == 1) {
    pntr_[0].push_back(0);
    pntr_[0].push_back(num_indexes);
  } else {
     #pragma omp parallel for
     for(int d=0; d<num_dimensions; d++)
     {
       int const *c_index = iset[map_[d][0]];
       pntr_[d].push_back(0);
       for(int i=1; i<num_indexes; i++) {
         if (not match_outside_dim(d, c_index, iset[map_[d][i]]))
         {
           pntr_[d].push_back(i);
           c_index = iset[map_[d][i]];
         }
       }
       pntr_[d].push_back(num_indexes);
    }
  }
}

dimension_sort::dimension_sort(vector2d<int> const &list) : map_(list.stride())
{
  int num_dimensions = list.stride();
  int num_indexes    = list.num_strips();
  map_ = std::vector<std::vector<int>>(num_dimensions, std::vector<int>(num_indexes));

  #pragma omp parallel for
  for(int d=0; d<num_dimensions; d++) {
      // for each dimension, use the map to group points together
      std::iota(map_[d].begin(), map_[d].end(), 0);
      std::sort(map_[d].begin(), map_[d].end(), [&](int a, int b)->bool{
        const int * idxa = list[a];
        const int * idxb = list[b];
        for(int j=0; j<num_dimensions; j++) {
          if (j != d){
            if (idxa[j] < idxb[j]) return true;
            if (idxa[j] > idxb[j]) return false;
          }
        }
        // lexigographical order, dimension d is the fastest moving one
        if (idxa[d] < idxb[d]) return true;
        if (idxa[d] > idxb[d]) return false;
        return false;
      });
  }

  // check if multi-indexes match in all but one dimension
  auto match_outside_dim = [&](int d, int const*a, int const *b) -> bool {
      for(int j=0; j<d; j++)
        if (a[j] != b[j]) return false;
      for(int j=d+1; j<num_dimensions; j++)
        if (a[j] != b[j]) return false;
      return true;
    };

  // offsets of sorted "sparse vectors" in each dimension
  pntr_ = std::vector<std::vector<int>>(num_dimensions);

  // split the map into vectors of identical coefficients in all but 1d
  if (num_dimensions == 1) {
    pntr_[0].push_back(0);
    pntr_[0].push_back(num_indexes);
  } else {
     #pragma omp parallel for
     for(int d=0; d<num_dimensions; d++)
     {
       int const *c_index = list[map_[d][0]];
       pntr_[d].push_back(0);
       for(int i=1; i<num_indexes; i++) {
         if (not match_outside_dim(d, c_index, list[map_[d][i]]))
         {
           pntr_[d].push_back(i);
           c_index = list[map_[d][i]];
         }
       }
       pntr_[d].push_back(num_indexes);
    }
  }
}

indexset compute_ancestry_completion(indexset const &iset,
                                     connect_1d const &pattern1d)
{
  int const num_dimensions = iset.num_dimensions();

  // store all missing ancestors here
  vector2d<int> missing_ancestors(num_dimensions, 0);

  // do just one pass, considering the indexes in the iset only
  std::vector<int> ancestor(num_dimensions);
  for(int i = 0; i < iset.num_indexes(); i++)
  {
    // construct all parents, even considering the edges
    std::copy_n(iset[i], num_dimensions, ancestor.begin());
    // check the parents in each direction
    for(int d = 0; d < num_dimensions; d++)
    {
      int const row = ancestor[d];
      for(int j = pattern1d.row_begin(row); j < pattern1d.row_diag(row); j++)
      {
        ancestor[d] = pattern1d[j];
        if (iset.missing(ancestor))
          missing_ancestors.append(ancestor);
      }
      ancestor[d] = row;
    }
  }

  bool ancestry_complete = missing_ancestors.empty();

  indexset pad_indexes = make_index_set(missing_ancestors);

  // the assumption here it that the padded set will be smaller
  // then the iset, so we do one loop over the large set and then we work
  // only with the smaller pad_indexes
  while(not ancestry_complete)
  {
    // all new found indexes are already in pad_indexes
    // missing_ancestors holds the ones from this iteration only
    missing_ancestors.clear();

    for(int i = 0; i < pad_indexes.num_indexes(); i++)
    {
      // construct all parents, even considering the edges
      std::copy_n(pad_indexes[i], num_dimensions, ancestor.begin());
      // check the parents in each direction
      for(int d = 0; d < num_dimensions; d++)
      {
        int const row = ancestor[d];
        for(int j = pattern1d.row_begin(row); j < pattern1d.row_diag(row); j++)
        {
          ancestor[d] = pattern1d[j];
          if (iset.missing(ancestor) and pad_indexes.missing(ancestor))
            missing_ancestors.append(ancestor);
        }
        ancestor[d] = row;
      }
    }

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

vector2d<int> complete_poly_order(vector2d<int> const &active_cells,
                                  indexset const &pad_cells, int porder)
{
  expect(active_cells.stride() == pad_cells.num_dimensions());
  int num_dimensions = pad_cells.num_dimensions();

  int64_t num_active_cells = active_cells.num_strips();

  int64_t pterms = porder + 1;

  int64_t tsize = pterms;
  for(int64_t d=1; d<num_dimensions; d++)
    tsize *= pterms;

  int64_t total_cells = num_active_cells + pad_cells.num_indexes();

  vector2d<int> indexes(num_dimensions, tsize * total_cells);

  // expand with the polynomial indexes in two stages
  // first work with the active_indexes, then with the padded ones
#pragma omp parallel for
  for(int64_t i = 0; i < total_cells; i++)
  {
    int const *cell = (i < num_active_cells) ? active_cells[i] : pad_cells[i - num_active_cells];

    for(int64_t ipoly = 0; ipoly < tsize; ipoly++)
    {
      int64_t t = ipoly;
      int *idx  = indexes[i*tsize + ipoly];

      for(int d = num_dimensions-1; d >= 0; d--)
      {
        idx[d] = cell[d] * pterms + static_cast<int>(t % pterms);
        t /= pterms;
      }
    }
  }

  return indexes;
}

}
