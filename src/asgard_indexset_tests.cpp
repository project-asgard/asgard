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

  indexset iset = make_index_set(2, unsorted);
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

TEST_CASE("connectivity expand", "[connectivity]")
{
  connect_1d cells(3, connect_1d::level_edge_skip);
  REQUIRE(cells.num_cells() == 8);
  REQUIRE(cells.num_connections() == 50);

  std::vector<int> gold_num_connect = {8, 8, 7, 7, 5, 5, 5, 5};
  for(int row=0; row<cells.num_cells(); row++)
    REQUIRE(gold_num_connect[row] == cells.row_end(row) - cells.row_begin(row));

  std::vector<int> gold_connect_row4 = {0, 1, 2, 3, 4};
  for(int col=cells.row_begin(4); col<cells.row_end(4); col++)
    REQUIRE(gold_connect_row4[col - cells.row_begin(4)] == cells[col]);

  // expand the cells by adding the degrees of freedom for quadratic basis
  // i.e., each entry in the sparse matrix is replaced with a 3x3 block
  connect_1d expanded(cells, 3);
  REQUIRE(expanded.num_cells() == 3 * 8);
  // there are fewer connection since we removed the self-connection
  REQUIRE(expanded.num_connections() == (50 - 8) * 3 * 3);

  // compare the connectivity to the 12-th element
  std::vector<int> gold_connect_row12 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for(int col=expanded.row_begin(12); col<expanded.row_end(12); col++)
    REQUIRE(gold_connect_row12[col - expanded.row_begin(12)] == expanded[col]);

  // connectivity for 12 should be the same as 13
  for(int col=expanded.row_begin(13); col<expanded.row_end(13); col++)
    REQUIRE(gold_connect_row12[col - expanded.row_begin(13)] == expanded[col]);
}
