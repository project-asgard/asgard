#include "connectivity.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <string>
#include <vector>

TEST_CASE("element table constructor/accessors/size", "[connectivity]")
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

  SECTION("element table size", "[connectivity]")
  {
    REQUIRE(t.size() == 2);
    REQUIRE(t_2.size() == 20);
    REQUIRE(t_3.size() == 4096);
  }
}

// TEMPORARY TESTS FOR STATIC HELPERS
//
// these aren't part of the API and can be removed after class development
// if we want to make these static functions private

TEST_CASE("Lev/cell indexing functions", "[connectivity]")
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

TEST_CASE("Permutations enumerators", "[connectivity]")
{
  element_table t(make_options({}), 1);
  SECTION("permutations eq enumeration")
  {
    std::vector<int> const golds{1, 1, 1001, 4598126};
    std::vector<int> const dims{1, 1, 5, 5};
    std::vector<int> const ns{1, 10, 10, 100};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.count_eq_permutations(dims[i], ns[i]) == golds[i]);
    }
  }
  SECTION("permutations leq enumeration")
  {
    std::vector<int> const golds{2, 11, 3003, 96560646};
    std::vector<int> const dims{1, 1, 5, 5};
    std::vector<int> const ns{1, 10, 10, 100};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.count_leq_permutations(dims[i], ns[i]) == golds[i]);
    }
  }
  SECTION("permutations max enumeration")
  {
    std::vector<int> const golds{2, 11, 161051};
    std::vector<int> const dims{1, 1, 5};
    std::vector<int> const ns{1, 10, 10};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.count_max_permutations(dims[i], ns[i]) == golds[i]);
    }
  }
}

TEST_CASE("Permutations builders", "[connectivity]")
{
  element_table t(make_options({}), 1);

  std::vector<int> const dims{5, 2, 2, 5};
  std::vector<int> const ns{0, 1, 1, 2};
  std::vector<bool> const ord_by_ns{false, false, true, false};

  SECTION("permutations eq")
  {
    // clang-format off
    std::vector<fk::matrix<int>> const golds{
        {{0, 0, 0, 0, 0}},
        eye<int>(2),
        {{0, 1}, {1, 0}},
        {{2, 0, 0, 0, 0},
         {1, 1, 0, 0, 0},
         {0, 2, 0, 0, 0},
         {1, 0, 1, 0, 0},
         {0, 1, 1, 0, 0},
         {0, 0, 2, 0, 0},
         {1, 0, 0, 1, 0},
         {0, 1, 0, 1, 0},
         {0, 0, 1, 1, 0},
         {0, 0, 0, 2, 0},
         {1, 0, 0, 0, 1},
         {0, 1, 0, 0, 1},
         {0, 0, 1, 0, 1},
         {0, 0, 0, 1, 1},
         {0, 0, 0, 0, 2}}}; // clang-format on

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.get_eq_permutations(dims[i], ns[i], ord_by_ns[i]) == golds[i]);
    }
  }

  SECTION("permutations leq")
  {
    // clang-format off
    std::vector<fk::matrix<int>> const golds{
        {{0, 0, 0, 0, 0}},
        {{0, 0}, {1, 0}, {0, 1}},
        {{0, 0}, {1, 0}, {0, 1}},
        {{0, 0, 0, 0, 0},
	 {1, 0, 0, 0, 0},
	 {2, 0, 0, 0, 0},
	 {0, 1, 0, 0, 0},
         {1, 1, 0, 0, 0},
	 {0, 2, 0, 0, 0},
	 {0, 0, 1, 0, 0},
	 {1, 0, 1, 0, 0},
         {0, 1, 1, 0, 0},
	 {0, 0, 2, 0, 0},
	 {0, 0, 0, 1, 0},
	 {1, 0, 0, 1, 0},
         {0, 1, 0, 1, 0},
	 {0, 0, 1, 1, 0},
	 {0, 0, 0, 2, 0},
	 {0, 0, 0, 0, 1},
         {1, 0, 0, 0, 1},
	 {0, 1, 0, 0, 1},
	 {0, 0, 1, 0, 1},
	 {0, 0, 0, 1, 1},
         {0, 0, 0, 0, 2}}}; // clang-format on

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.get_leq_permutations(dims[i], ns[i], ord_by_ns[i]) == golds[i]);
    }
  }

  SECTION("permutations max")
  {
    std::vector<int> const dims{5, 2, 2, 2};
    std::vector<int> const ns{0, 1, 1, 3};
    std::vector<bool> const ord_by_ns{false, false, true, true};

    // clang-format off
    std::vector<fk::matrix<int>> const golds{
        {{0, 0, 0, 0, 0}},
        {{0, 0},
	 {1, 0},
	 {0, 1},
	 {1, 1}},
	{{1, 1},
	 {0, 1},
	 {1, 0},
	 {0, 0}},
        {{3, 3},
         {2, 3},
         {1, 3},
         {0, 3},
         {3, 2},
         {2, 2},
         {1, 2},
         {0, 2},
         {3, 1},
         {2, 1},
         {1, 1},
         {0, 1},
         {3, 0},
         {2, 0},
         {1, 0},
         {0, 0}}}; // clang-format on

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.get_max_permutations(dims[i], ns[i], ord_by_ns[i]) == golds[i]);
    }
  }
}

TEST_CASE("one-dimensional indexing", "[connectivity]")
{
  SECTION("simple test for indexing function")
  {
    // test some vals calc'ed by hand :)
    std::vector<int> const golds{0, 0, 524388};
    std::vector<int> const levels{0, 0, 20};
    std::vector<int> const cells{0, 500, 100};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(get_1d_index(levels[i], cells[i]) == golds[i]);
    }
  }
}
