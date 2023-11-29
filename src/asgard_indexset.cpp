#include "asgard_indexset.hpp"

namespace asgard
{

indexset make_index_set(int num_dimensions, std::vector<int> const &indexes){
  expect(indexes.size() % num_dimensions == 0);
  int num_indexes = static_cast<int>(indexes.size() / num_dimensions);
  std::vector<int> map(num_indexes);
  std::iota(map.begin(), map.end(), 0);

  std::sort(map.begin(), map.end(),
            [&](int a, int b)-> bool{
              for(int j=0; j<num_dimensions; j++)
              {
                if (indexes[a*num_dimensions + j] < indexes[b*num_dimensions + j])
                  return true;
                if (indexes[a*num_dimensions + j] > indexes[b*num_dimensions + j])
                  return false;
              }
              return true; // equal, should never happen
            });

  int repeated_indexes = 0;
  for(int i=0; i<num_indexes - 1; i++)
  {
    bool is_repeated = [&]()
        ->bool {
            for(int j=0; j<num_dimensions; j++)
              if (indexes[map[i] * num_dimensions + j] != indexes[map[i+1] * num_dimensions + j])
                return false;
            return true;
        }();
    if (is_repeated)
    {
      map[i] = -1;
      repeated_indexes += 1;
    }
  }

  std::vector<int> sorted_indexes;
  sorted_indexes.reserve(num_dimensions * (num_indexes - repeated_indexes));
  for(int i=0; i<num_indexes; i++)
  {
    if (map[i] != -1)
      sorted_indexes.insert(sorted_indexes.end(), &indexes[map[i] * num_dimensions], &indexes[map[i] * num_dimensions] + num_dimensions);
  }

  return indexset(num_dimensions, std::move(sorted_indexes));
}

indexset reindex_map::remap(std::vector<int> const &indexes)
{
  expect(indexes.size() % num_dimensions_ == 0);
  int num_indexes = static_cast<int>(indexes.size() / num_dimensions_);
  map_.resize(num_indexes);
  std::iota(map_.begin(), map_.end(), 0);

  std::sort(map_.begin(), map_.end(),
            [&](int a, int b)-> bool{
              for(int j=0; j<num_dimensions_; j++)
              {
                if (indexes[a*num_dimensions_ + j] < indexes[b*num_dimensions_ + j])
                  return true;
                if (indexes[a*num_dimensions_ + j] > indexes[b*num_dimensions_ + j])
                  return false;
              }
              return true; // equal, should never happen
            });

  std::vector<int> sorded_indexes(num_dimensions_ * num_indexes);
  #pragma omp parallel for
  for(int i=0; i<num_indexes; i++)
  {
    std::copy_n(&indexes[map_[i] * num_dimensions_], num_dimensions_, &sorded_indexes[i * num_dimensions_]);
  }

  return indexset(num_dimensions_, std::move(sorded_indexes));
}

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
        const int * idxa = iset.index(a);
        const int * idxb = iset.index(b);
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
       int const *c_index = iset.index(map_[d][0]);
       pntr_[d].push_back(0);
       for(int i=1; i<num_indexes; i++) {
         if (not match_outside_dim(d, c_index, iset.index(map_[d][i])))
         {
           pntr_[d].push_back(i);
           c_index = iset.index(map_[d][i]);
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
  // do just one pass, considering the indexes in the iset only
  std::vector<int> missing_ancestors;
  for(int i=0; i<iset.num_indexes(); i++)
  {
    // construct all parents, even considering the edges
    std::vector<int> ancetor(iset.index(i), iset.index(i) + num_dimensions);
    // check the parents in each direction
    for(int d=0; d<num_dimensions; d++)
    {
      int const row = ancetor[d];
      for(int j=pattern1d.row_begin(row); j<pattern1d.row_diag(row); j++)
      {
        ancetor[d] = pattern1d[j];
        if (iset.find(ancetor) == -1)
          missing_ancestors.insert(missing_ancestors.end(), ancetor.begin(), ancetor.end());
      }
      ancetor[d] = row;
    }
  }

  indexset pad_indexes;
  if (not missing_ancestors.empty())
  {
    pad_indexes = make_index_set(num_dimensions, missing_ancestors);

    int last_added = pad_indexes.num_indexes();

    // the assumption here it that the padded set will be smaller
    // then the iset, so we do one loop over the large set and then we work
    // only with the smaller completion
    while(last_added > 0)
    {
      missing_ancestors.clear();

      for(int i=0; i<pad_indexes.num_indexes(); i++)
      {
        // construct all parents, even considering the edges
        std::vector<int> ancetor(pad_indexes.index(i), pad_indexes.index(i) + num_dimensions);
        // check the parents in each direction
        for(int d=0; d<num_dimensions; d++)
        {
          int const row = ancetor[d];
          for(int j=pattern1d.row_begin(row); j<pattern1d.row_diag(row); j++)
          {
            ancetor[d] = pattern1d[j];
            if (iset.find(ancetor) == -1 and pad_indexes.find(ancetor) == -1)
              missing_ancestors.insert(missing_ancestors.end(), ancetor.begin(), ancetor.end());
          }
          ancetor[d] = row;
        }
      }

      if (missing_ancestors.empty())
      {
        last_added = 0;
      }
      else
      {
        missing_ancestors.insert(missing_ancestors.end(), pad_indexes.index(0), pad_indexes.index(0) + pad_indexes.size());
        pad_indexes = make_index_set(num_dimensions, missing_ancestors);
        last_added  = pad_indexes.num_indexes();
      }
    }
  }
  return pad_indexes;
}

}
