#include "tests_general.hpp"

#include "asgard_indexset.hpp"

using namespace asgard;

void verify_1d(dimension_sort const &dsort, indexset const &iset, int dimension, int i,
               std::vector<int> const &offsets, std::vector<int> const &index1d)
{
  REQUIRE(dsort.vec_end(dimension, i) - dsort.vec_begin(dimension, i) == static_cast<int>(offsets.size()));
  REQUIRE(offsets.size() == index1d.size()); // if this is off, the test is wrong
  auto ioff = offsets.begin();
  auto idx1 = index1d.begin();
  for (int j = dsort.vec_begin(dimension, i); j < dsort.vec_end(dimension, i); j++)
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
  REQUIRE(data.stride() == 2);
  REQUIRE(data.num_strips() == 3);
  REQUIRE(data[1][0] == 0);

  for (int i = 0; i < 3; i++)
    data[0][i] = i;
  REQUIRE(data[1][0] == 2);

  std::vector<int> raw_data = {3, 4, 5, 2, 3, 4};
  span2d<int> spdata(3, 2, raw_data.data());
  REQUIRE(spdata[0][0] == 3);
  REQUIRE(spdata[0][2] == 5);
  REQUIRE(spdata[1][0] == 2);
  REQUIRE(spdata[1][2] == 4);
}

TEST_CASE("indexset union", "[union]")
{
  std::vector<int> data1 = {0, 0, 0, 1, 0, 2, 1, 1, 1, 4};
  std::vector<int> data2 = {0, 0, 0, 3, 1, 1, 1, 5};
  std::vector<int> ref   = {0, 0, 0, 1, 0, 2, 0, 3, 1, 1, 1, 4, 1, 5};
  indexset iset1(2, std::vector<int>(data1));
  indexset iset2(2, std::vector<int>(data2));

  indexset u = iset1;
  u += iset2;

  REQUIRE(u.num_indexes() == 7);
  for (size_t i = 0; i < ref.size(); i++)
    REQUIRE(u[0][i] == ref[i]);

  u = iset2;
  REQUIRE(u.num_indexes() == iset2.num_indexes());
  for (int i = 0; i < iset2.num_indexes(); i++)
    for (int d = 0; d < iset2.num_dimensions(); d++)
      REQUIRE(u[i][d] == iset2[i][d]);

  u += iset1;
  REQUIRE(u.num_indexes() == 7);
  for (size_t i = 0; i < ref.size(); i++)
    REQUIRE(u[0][i] == ref[i]);
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

  REQUIRE(iset.find({0, 1}) == 1);
  REQUIRE(iset.find({1, 1}) == 3);
  REQUIRE(not iset.missing({1, 1}));
  REQUIRE(iset.find({0, 2}) == -1);
  REQUIRE(iset.missing({0, 2}));

  // check the sorted order
  for (int i = 0; i < iset.num_indexes(); i++)
    for (int j = 0; j < iset.num_dimensions(); j++)
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
  verify_1d(dsort, iset, 1, 2, std::vector<int>(1, 4), std::vector<int>(1, 0));
}

TEST_CASE("connectivity full and expanded to dof", "[connectivity]")
{
  connect_1d cells(3, connect_1d::hierarchy::full);
  REQUIRE(cells.num_rows() == 8);
  REQUIRE(cells.num_connections() == 60);

  std::vector<int> gold_num_connect = {8, 8, 8, 8, 7, 7, 7, 7};
  for (int row = 0; row < cells.num_rows(); row++)
    REQUIRE(gold_num_connect[row] == cells.row_end(row) - cells.row_begin(row));

  std::vector<int> gold_connect_row4 = {0, 1, 2, 3, 4, 5, 7};
  for (int col = cells.row_begin(4); col < cells.row_end(4); col++)
    REQUIRE(gold_connect_row4[col - cells.row_begin(4)] == cells[col]);

  //connect_1d(cells, 0).print(std::cerr); // uncomment for manual check

  // expand the cells by adding the degrees of freedom for quadratic basis
  // i.e., each entry in the sparse matrix is replaced with a 3x3 block
  int const degree = 2;
  connect_1d expanded(cells, degree);
  REQUIRE(expanded.num_rows() == (degree + 1) * 8);
  // there are fewer connection since we removed the self-connection
  REQUIRE(expanded.num_connections() == 60 * (degree + 1) * (degree + 1));

  // compare the connectivity to the 12-th element (first in cell 4)
  REQUIRE(expanded.row_end(12) - expanded.row_begin(12) == 21);
  for (int col = 0; col < 18; col++)
    REQUIRE(col == expanded[expanded.row_begin(12) + col]);
  for (int col = 18; col < 21; col++)
    REQUIRE(col + 3 == expanded[expanded.row_begin(12) + col]);

  // connectivity for 12 should be the same as 13
  REQUIRE(expanded.row_end(13) - expanded.row_begin(13) == 21);
  for (int col = 0; col < 18; col++)
    REQUIRE(col == expanded[expanded.row_begin(13) + col]);
  for (int col = 18; col < 21; col++)
    REQUIRE(col + 3 == expanded[expanded.row_begin(13) + col]);

  cells = connect_1d(4, connect_1d::hierarchy::full);
  cells.print();
}

TEST_CASE("testing volume connections", "[connectivity]")
{
  connect_1d cells(1, connect_1d::hierarchy::volume);
  // cells on level 0 and 1 only connect the themselves
  REQUIRE(cells.num_rows() == 2);
  REQUIRE(cells.num_connections() == 4);
  REQUIRE((cells[0] == 0 and cells[3] == 1));
  REQUIRE((cells[1] == 1 and cells[2] == 0));

  cells = connect_1d(4, connect_1d::hierarchy::volume);
  REQUIRE(cells.num_rows() == 16);
  REQUIRE(cells.num_connections() == 114);

  std::vector<int> gold_connect = {0, 1, 2, 4, 8};
  REQUIRE(cells.row_end(8) - cells.row_begin(8) ==
          static_cast<int>(gold_connect.size()));
  for (int j = cells.row_begin(8); j < cells.row_end(8); j++)
    REQUIRE(cells[j] == gold_connect[j - cells.row_begin(8)]);

  gold_connect = std::vector<int>{0, 1, 2, 5, 10, 11};
  REQUIRE(cells.row_end(5) - cells.row_begin(5) ==
          static_cast<int>(gold_connect.size()));
  for (int j = cells.row_begin(5); j < cells.row_end(5); j++)
    REQUIRE(cells[j] == gold_connect[j - cells.row_begin(5)]);

  gold_connect = std::vector<int>{0, 1, 3, 6, 13};
  REQUIRE(cells.row_end(13) - cells.row_begin(13) ==
          static_cast<int>(gold_connect.size()));
  for (int j = cells.row_begin(13); j < cells.row_end(13); j++)
    REQUIRE(cells[j] == gold_connect[j - cells.row_begin(13)]);
}

TEST_CASE("testing column transform connections", "[transformation]")
{
  SECTION("level 1, edge connect")
  {
    connect_1d cells(1, connect_1d::hierarchy::full);
    connect_1d col_cells(cells, connect_1d::col_extend_hierarchy);
    // cells on level 0 and 1 only connect the themselves
    REQUIRE(col_cells.num_rows() == 2);
    REQUIRE(col_cells.num_connections() == 4);
    REQUIRE((col_cells[0] == 0 and col_cells[3] == 1));
    REQUIRE((col_cells[1] == 1 and col_cells[2] == 0));
  }
  SECTION("level 3, edge connect")
  {
    connect_1d cells(3, connect_1d::hierarchy::full);
    connect_1d col_cells(cells, connect_1d::col_extend_hierarchy);
    // cells on level 0 and 1 only connect the themselves
    REQUIRE(col_cells.num_rows() == 14);
    for (int r = 4; r < 8; r++)
    {
      int r1 = col_cells.num_rows() - (8 - r) * 2;
      int r2 = r1 + 1;
      REQUIRE(cells.row_end(r) - cells.row_begin(r) == col_cells.row_end(r1) - col_cells.row_begin(r1));
      REQUIRE(cells.row_end(r) - cells.row_begin(r) == col_cells.row_end(r2) - col_cells.row_begin(r2));
      int j1 = col_cells.row_begin(r1);
      int j2 = col_cells.row_begin(r2);
      for (int j = cells.row_begin(r); j < cells.row_end(r); j++)
      {
        REQUIRE(cells[j] == col_cells[j1]);
        REQUIRE(cells[j] == col_cells[j2]);
        ++j1;
        ++j2;
      }
    }
  }
}
