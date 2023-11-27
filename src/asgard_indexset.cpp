#include "asgard_indexset.hpp"

indexset reindex_map::remap(std::vector<int> const &indexes)
{
  expect(indexes.size() % num_dimensions_ == 0);
  int num_indexes_ = static_cast<int>(indexes.size() / num_dimensions_);
  map_.resize(num_indexes_);
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

  std::vector<int> sorded_indexes(num_dimensions_ * num_indexes_);
  #pragma omp parallel for
  for(int i=0; i<num_indexes_; i++)
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
