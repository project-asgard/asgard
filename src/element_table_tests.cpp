#include "element_table.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <vector>

// TEMPORARY TESTS FOR STATIC HELPERS
//
// these aren't part of the API and will be removed after class development

TEST_CASE("Indexing functions", "[element_table]")
{
  element_table t(0, 0, 0);
  SECTION("one dimensional indexing function")
  {
    // test some vals calc'ed by hand :)
    std::vector<int> const golds{1, 1, 524389};
    std::vector<int> const levels{0, 0, 20};
    std::vector<int> const cells{0, 500, 100};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.get_1d_index(levels[i], cells[i]) == golds[i]);
    }
  }
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
      REQUIRE(t.get_index_set(levels_set[i]) == gold_set[i]);
    }
  }
}

TEST_CASE("Permutations enumerators", "[element_table]")
{
  element_table t(0, 0, 0);
  SECTION("permutations eq enumeration")
  {
    std::vector<int> const golds{1, 1, 1001, 4598126};
    std::vector<int> const dims{1, 1, 5, 5};
    std::vector<int> const ns{1, 10, 10, 100};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.permutations_eq_count(dims[i], ns[i]) == golds[i]);
    }
  }
  SECTION("permutations leq enumeration")
  {
    std::vector<int> const golds{2, 11, 3003, 96560646};
    std::vector<int> const dims{1, 1, 5, 5};
    std::vector<int> const ns{1, 10, 10, 100};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.permutations_leq_count(dims[i], ns[i]) == golds[i]);
    }
  }
  SECTION("permutations max enumeration")
  {
    std::vector<int> const golds{2, 11, 161051};
    std::vector<int> const dims{1, 1, 5};
    std::vector<int> const ns{1, 10, 10};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(t.permutations_max_count(dims[i], ns[i]) == golds[i]);
    }
  }
}

TEST_CASE("Permutations builders", "[element_table]")
{
  element_table t(0, 0, 0);

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
      REQUIRE(t.permutations_eq(dims[i], ns[i], ord_by_ns[i]) == golds[i]);
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
      REQUIRE(t.permutations_leq(dims[i], ns[i], ord_by_ns[i]) == golds[i]);
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
      REQUIRE(t.permutations_max(dims[i], ns[i], ord_by_ns[i]) == golds[i]);
    }
  }
}
