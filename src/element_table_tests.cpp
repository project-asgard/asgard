#include "element_table.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"

TEST_CASE("element table constructor/accessors/size", "[element_table]")
{
  int const levels = 1;
  int const dims   = 1;
  Options o        = make_options({"-l", std::to_string(levels)});
  element_table t(o, dims);
  fk::vector<int> element_0 = {0, 0};
  fk::vector<int> element_1 = {1, 0};
  REQUIRE(t.get_index(element_0) == 0);
  REQUIRE(t.get_index(element_1) == 1);
  REQUIRE(t.get_coords(0) == element_0);
  REQUIRE(t.get_coords(1) == element_1);

  int const levels_2 = 3;
  int const dims_2   = 2;
  Options o_2        = make_options({"-l", std::to_string(levels_2)});
  element_table t_2(o_2, dims_2);
  fk::vector<int> element_17 = {0, 3, 0, 1};
  REQUIRE(t_2.get_index(element_17) == 17);
  REQUIRE(t_2.get_coords(17) == element_17);

  int const levels_3 = 4;
  int const dims_3   = 3;
  // test full grid
  Options o_3 = make_options({"-l", std::to_string(levels_3), "-f"});
  element_table t_3(o_3, dims_3);
  fk::vector<int> element_4000 = {4, 4, 4, 0, 4, 6};
  REQUIRE(t_3.get_index(element_4000) == 4000);
  REQUIRE(t_3.get_coords(4000) == element_4000);

  SECTION("element table size", "[element_table]")
  {
    REQUIRE(t.size() == 2);
    REQUIRE(t_2.size() == 20);
    REQUIRE(t_3.size() == 4096);
  }
}

TEST_CASE("Static helper - cell builder", "[element_table]")
{
  element_table t(make_options({}), 1);

  SECTION("cell index set builder")
  {
    std::vector<fk::vector<int>> levels_set = {{1}, {1, 2}, {2, 1}, {2, 3}};

    std::vector<fk::matrix<int>> gold_set = {
        {{0}},
        {{0, 0}, {0, 1}},
        {{0, 0}, {1, 0}},
        {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}}};

    for (size_t i = 0; i < gold_set.size(); ++i)
    {
      REQUIRE(t.get_cell_index_set(levels_set[i]) == gold_set[i]);
    }
  }
}
