#include "connectivity.hpp"

#include "element_table.hpp"
#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <numeric>
#include <string>
#include <vector>

TEST_CASE("one-dimensional connectivity", "[connectivity]")
{
  SECTION("simple test for indexing function")
  {
    std::string const base_path =
        "../testing/generated-inputs/connectivity/get_1d_";

    std::vector<int> const levels{0, 0, 5};
    std::vector<int> const cells{0, 1, 9};

    for (int i = 0; i < static_cast<int>(levels.size()); ++i)
    {
      std::string const file_path = base_path + std::to_string(levels[i]) +
                                    "_" + std::to_string(cells[i]) + ".dat";
      int const gold = static_cast<int>(read_scalar_from_txt_file(file_path));
      // indexing function - adjust MATLAB indexing by -1
      REQUIRE(get_1d_index(levels[i], cells[i]) == gold - 1);
    }
  }

  SECTION("one-dimensional connectivity function")
  {
    std::string const base_path =
        "../testing/generated-inputs/connectivity/connect_1_";
    std::vector<int> const levels{2, 3, 8};
    for (int const level : levels)
    {
      std::string const file_path = base_path + std::to_string(level) + ".dat";
      fk::matrix<int> const gold =
          fk::matrix<int>(read_matrix_from_txt_file(file_path));
      REQUIRE(make_1d_connectivity(level) == gold);
    }
  }
}

void test_connectivity(int const dims, int const levels,
                       std::string const &gold_path,
                       bool const full_grid = false)
{
  std::string const grid_str = full_grid ? "-f" : "";
  options const o = make_options({"-l", std::to_string(levels), grid_str});
  element_table const t(o, dims);
  int const max_levels        = full_grid ? dims * levels : levels;
  list_set const connectivity = make_connectivity(t, dims, max_levels, levels);

  list_set gold;
  for (int i = 0; i < static_cast<int>(connectivity.size()); ++i)
  {
    std::string const file_path = gold_path + std::to_string(i + 1) + ".dat";
    fk::vector<int> element =
        fk::vector<int>(read_vector_from_txt_file(file_path));
    // adjust matlab indexing
    std::transform(element.begin(), element.end(), element.begin(),
                   [](int &elem) { return elem - 1; });
    gold.push_back(element);
  }

  REQUIRE(connectivity == gold);
}

TEST_CASE("n-dimensional connectivity", "[connectivity]")
{
  SECTION("3 dimensions, level 4, sparse grid")
  {
    int const levels = 4;
    int const dims   = 3;
    std::string const gold_path =
        "../testing/generated-inputs/connectivity/connect_n_3_4_SG_";
    test_connectivity(dims, levels, gold_path);
  }

  SECTION("2 dimensions, level 3, full grid")
  {
    int const levels = 3;
    int const dims   = 2;
    std::string const gold_path =
        "../testing/generated-inputs/connectivity/connect_n_2_3_FG_";
    bool const full_grid = true;
    test_connectivity(dims, levels, gold_path, full_grid);
  }
}
