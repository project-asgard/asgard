#include "permutations.hpp"
#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <vector>

TEST_CASE("Permutations enumerators", "[permutations]")
{
  SECTION("permutations eq enumeration")
  {
    std::vector<int> const golds{1, 1, 1001, 4598126};
    std::vector<int> const dims{1, 1, 5, 5};
    std::vector<int> const ns{1, 10, 10, 100};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(count_eq_permutations(dims[i], ns[i]) == golds[i]);
    }
  }
  SECTION("permutations leq enumeration")
  {
    std::vector<int> const golds{2, 11, 3003, 96560646};
    std::vector<int> const dims{1, 1, 5, 5};
    std::vector<int> const ns{1, 10, 10, 100};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(count_leq_permutations(dims[i], ns[i]) == golds[i]);
    }
  }
  SECTION("permutations max enumeration")
  {
    std::vector<int> const golds{2, 11, 161051};
    std::vector<int> const dims{1, 1, 5};
    std::vector<int> const ns{1, 10, 10};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(count_max_permutations(dims[i], ns[i]) == golds[i]);
    }
  }
}

TEST_CASE("Permutations builders", "[permutations]")
{
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
      REQUIRE(get_eq_permutations(dims[i], ns[i], ord_by_ns[i]) == golds[i]);
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
      REQUIRE(get_leq_permutations(dims[i], ns[i], ord_by_ns[i]) == golds[i]);
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
      REQUIRE(get_max_permutations(dims[i], ns[i], ord_by_ns[i]) == golds[i]);
    }
  }
}
