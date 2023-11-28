#include "tests_general.hpp"

#include "asgard_indexset.hpp"

using namespace asgard;

void verify_1d(dimension_sort const &dsort, indexset const &iset, int dimension, int i,
               std::vector<int> const &offsets, std::vector<int> const &index1d)
{
  REQUIRE(dsort.vec_end(dimension, i) - dsort.vec_begin(dimension, i)
          == static_cast<int>(offsets.size()));
  REQUIRE(offsets.size() == index1d.size()); // if this is off, the test is wrong
  auto ioff = offsets.begin();
  auto idx1 = index1d.begin();
  for(int j=dsort.vec_begin(dimension, i); j < dsort.vec_end(dimension, i); j++)
  {
    REQUIRE(dsort.map(dimension, j) == *ioff++);
    REQUIRE(dsort(iset, dimension, j) == *idx1++);
  }
}

TEST_CASE("indexset sort", "[sort]")
{
  // indexes (0, 0), (0, 1), (1, 0), (1, 1), (2, 0)
  std::vector<int> sorted   = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0};
  std::vector<int> unsorted = {1, 1, 1, 0, 2, 0, 0, 0, 0, 1};

  reindex_map rmap(2);

  indexset iset = rmap.remap(unsorted);
  // check the dimensions and number of indexes
  REQUIRE(iset.num_dimensions() == 2);
  REQUIRE(iset.num_indexes() == 5);

  REQUIRE(iset.find({0, 1}) ==  1);
  REQUIRE(iset.find({1, 1}) ==  3);
  REQUIRE(iset.find({0, 2}) == -1);

  // check the sorted order
  for(int i=0; i<iset.num_indexes(); i++)
    for(int j=0; j<iset.num_dimensions(); j++)
      REQUIRE(iset.index(i)[j] == sorted[2 * i + j]);

  dimension_sort dsort(iset);
  // number of 1d vectors
  REQUIRE(dsort.num_vecs(0) == 2);
  REQUIRE(dsort.num_vecs(1) == 3);

  // first vector uses sorted indexes {0, 2, 4} and the 1d index goes {0, 1, 2}
  // the second index should always go 0, 1, 2 ...
  //    unless the grid is large and adaptivity was used
  verify_1d(dsort, iset, 0, 0, {0, 2, 4}, {0, 1, 2});
  verify_1d(dsort, iset, 0, 1, {1, 3}, {0, 1});
  verify_1d(dsort, iset, 1, 0, {0, 1}, {0, 1});
  verify_1d(dsort, iset, 1, 1, {2, 3}, {0, 1});
  verify_1d(dsort, iset, 1, 2, {4,}, {0,});
}
