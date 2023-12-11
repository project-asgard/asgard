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

TEST_CASE("data manipulation in 2d", "[order2d]")
{
  // some assertions to make sure I didn't mess things up
  static_assert(std::is_move_constructible<vector2d<int>>::value);
  static_assert(std::is_move_constructible<span2d<int>>::value);
  static_assert(std::is_copy_constructible<vector2d<int>>::value);
  static_assert(std::is_copy_constructible<span2d<int>>::value);

  vector2d<int> data(2, 3);
  REQUIRE(data.stride()     == 2);
  REQUIRE(data.num_strips() == 3);
  REQUIRE(data[1][0]        == 0);

  for(int i=0; i<3; i++)
    data[0][i] = i;
  REQUIRE(data[1][0] == 2);

  std::vector<int> raw_data = {3, 4, 5, 2, 3, 4};
  span2d<int> spdata(3, 2, raw_data.data());
  REQUIRE(spdata[0][0] == 3);
  REQUIRE(spdata[0][2] == 5);
  REQUIRE(spdata[1][0] == 2);
  REQUIRE(spdata[1][2] == 4);
}

TEST_CASE("indexset sort", "[sort]")
{
  // indexes (0, 0), (0, 1), (1, 0), (1, 1), (2, 0)
  std::vector<int> sorted   = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0};
  std::vector<int> unsorted = {1, 1, 1, 0, 2, 0, 0, 0, 0, 1};

  indexset iset = make_index_set(vector2d<int>(2, unsorted));
  // check the dimensions and number of indexes
  REQUIRE(iset.num_dimensions() == 2);
  REQUIRE(iset.num_indexes() == 5);

  REQUIRE(iset.find({0, 1}) ==  1);
  REQUIRE(iset.find({1, 1}) ==  3);
  REQUIRE(not iset.missing({1, 1}));
  REQUIRE(iset.find({0, 2}) == -1);
  REQUIRE(iset.missing({0, 2}));

  // check the sorted order
  for(int i=0; i<iset.num_indexes(); i++)
    for(int j=0; j<iset.num_dimensions(); j++)
      REQUIRE(iset[i][j] == sorted[2 * i + j]);

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
  REQUIRE(cells.num_rows() == 8);
  REQUIRE(cells.num_connections() == 50);

  std::vector<int> gold_num_connect = {8, 8, 7, 7, 5, 5, 5, 5};
  for(int row=0; row<cells.num_rows(); row++)
    REQUIRE(gold_num_connect[row] == cells.row_end(row) - cells.row_begin(row));

  std::vector<int> gold_connect_row4 = {0, 1, 2, 3, 4};
  for(int col=cells.row_begin(4); col<cells.row_end(4); col++)
    REQUIRE(gold_connect_row4[col - cells.row_begin(4)] == cells[col]);

  //connect_1d(cells, 0).dump(); // uncomment to double-check (non-automated)

  // expand the cells by adding the degrees of freedom for quadratic basis
  // i.e., each entry in the sparse matrix is replaced with a 3x3 block
  int const porder = 2;
  connect_1d expanded(cells, porder);
  REQUIRE(expanded.num_rows() == (porder+1) * 8);
  // there are fewer connection since we removed the self-connection
  REQUIRE(expanded.num_connections() == 50 * (porder+1) * (porder+1));

  // compare the connectivity to the 12-th element
  std::vector<int> gold_connect_row12 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  for(int col=expanded.row_begin(12); col<expanded.row_end(12); col++)
    REQUIRE(gold_connect_row12[col - expanded.row_begin(12)] == expanded[col]);

  // connectivity for 12 should be the same as 13
  for(int col=expanded.row_begin(13); col<expanded.row_end(13); col++)
    REQUIRE(gold_connect_row12[col - expanded.row_begin(13)] == expanded[col]);
}

TEST_CASE("testing edge connections", "[edge connect]")
{
  connect_1d cells(1, connect_1d::level_edge_only);
  // cells on level 0 and 1 only connect the themselves
  REQUIRE(cells.num_rows() == 2);
  REQUIRE(cells.num_connections() == 2);
  REQUIRE((cells[0] == 0 and cells[1] == 1));

  cells = connect_1d(4, connect_1d::level_edge_only);
  REQUIRE(cells.num_rows() == 16);
  REQUIRE(cells.num_connections() == 42);

  std::vector<int> gold_num_connect = {1, 1, 2, 2};
  while(gold_num_connect.size() < 16)
    gold_num_connect.push_back(3);
  for(int i=0; i<16; i++)
    REQUIRE(cells.row_end(i) - cells.row_begin(i) == gold_num_connect[i]);

  // check the first two rows only
  std::vector<int> gold_connect = {0, 1, 2, 3, 2, 3, 4, 5, 7, 4, 5, 6, 5, 6, 7, 4, 6, 7};
  for(int j=0; j<static_cast<int>(gold_connect.size()); j++)
    REQUIRE(cells[j] == gold_connect[j]);
}

TEST_CASE("testing completion", "[ancestry completion]")
{
  connect_1d conn(3, connect_1d::level_edge_skip);

  indexset incomplete(2, {1, 1});

  indexset completion = compute_ancestry_completion(incomplete, conn);
  REQUIRE(completion.num_dimensions() == incomplete.num_dimensions());
  REQUIRE(completion.num_indexes()    == 3);

  std::vector<int> gold_complete = {0, 0, 0, 1, 1, 0};
  for(int i=0; i<3; i++)
    for(int d=0; d<2; d++)
      REQUIRE(completion[i][d] == gold_complete[2*i + d]);

  incomplete = indexset(2, {0, 0, 0, 1, 0, 2, 0, 5}); // missing (0, 3)

  completion = compute_ancestry_completion(incomplete, conn);
  REQUIRE(completion.num_indexes() == 1);
  REQUIRE(completion[0][0] == 0);
  REQUIRE(completion[0][1] == 3);
}
