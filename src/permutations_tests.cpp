#include "matlab_utilities.hpp"
#include "permutations.hpp"
#include "tests_general.hpp"
#include <vector>

TEST_CASE("Permutations builders", "[permutations]")
{
  std::string zero = "0";
  std::string one  = "1";
  std::vector<int> const dims{1, 2, 4, 6};
  std::vector<int> const ns{1, 4, 6, 8};
  std::vector<bool> const ord_by_ns{false, true, false, true};

  SECTION("permutations eq") {}

  SECTION("permutations leq")
  {
    std::string out_base = "../testing/generated-inputs/perm_leq_";
    for (size_t i = 0; i < dims.size(); ++i)
    {
      std::string file_base = out_base + std::to_string(dims[i]) + "_" +
                              std::to_string(ns[i]) + "_" +
                              (ord_by_ns[i] ? one : zero);
      std::string file_path  = file_base + ".dat";
      std::string count_path = file_base + "_count.dat";
      fk::matrix<int> gold   = readMatrixFromTxtFile(file_path);
      int count_gold = static_cast<int>(readScalarFromTxtFile(count_path));
      REQUIRE(get_leq_permutations(dims[i], ns[i], ord_by_ns[i]) == gold);
      REQUIRE(count_leq_permutations(dims[i], ns[i]) == count_gold);
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
  SECTION("index leq max build")
  {
    list_set lists{{0, 1}, {0, 3}, {0, 1}, {2, 5}};
    int const max_sum = 5;
    int const max_val = 3;
    //clang-format off
    fk::matrix<int> const gold = {
        {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {1, 0, 1, 0}};
    //clang-format on
    REQUIRE(get_leq_max_indices(lists, lists.size(), max_sum, max_val) == gold);
  }
}
