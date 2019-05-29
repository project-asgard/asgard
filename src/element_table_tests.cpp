#include "element_table.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <string>

TEST_CASE("element table constructor/accessors/size", "[element_table]")
{
  std::string out_base =
      "../testing/generated-inputs/element_table/element_table";

  int const levels = 1;
  int const dims   = 1;
  options o        = make_options({"-l", std::to_string(levels)});
  element_table t(o, dims);

  std::string test_base = out_base + "_1_1_SG_";
  for (auto i = 0; i < t.size(); ++i)
  {
    std::string file_path = test_base + std::to_string(i + 1) + ".dat";
    fk::vector<int> gold =
        fk::vector<int>(read_vector_from_txt_file(file_path));
    REQUIRE(t.get_coords(i) == gold);
    REQUIRE(t.get_index(gold) == i);
  }

  int const levels_2 = 3;
  int const dims_2   = 2;
  options o_2        = make_options({"-l", std::to_string(levels_2)});
  element_table t_2(o_2, dims_2);
  test_base = out_base + "_2_3_SG_";
  for (auto i = 0; i < t_2.size(); ++i)
  {
    std::string file_path = test_base + std::to_string(i + 1) + ".dat";
    fk::vector<int> gold =
        fk::vector<int>(read_vector_from_txt_file(file_path));
    REQUIRE(t_2.get_coords(i) == gold);
    REQUIRE(t_2.get_index(gold) == i);
  }

  int const levels_3 = 4;
  int const dims_3   = 3;
  // test full grid
  options o_3 = make_options({"-l", std::to_string(levels_3), "-f"});
  element_table t_3(o_3, dims_3);
  test_base = out_base + "_3_4_FG_";
  for (auto i = 0; i < t_3.size(); ++i)
  {
    std::string file_path = test_base + std::to_string(i + 1) + ".dat";
    fk::vector<int> gold =
        fk::vector<int>(read_vector_from_txt_file(file_path));
    REQUIRE(t_3.get_coords(i) == gold);
    REQUIRE(t_3.get_index(gold) == i);
  }

  SECTION("element table size", "[element_table]")
  {
    REQUIRE(t.size() == 2);
    REQUIRE(t_2.size() == 20);
    REQUIRE(t_3.size() == 4096);
  }
}

TEST_CASE("Static helper - cell builder", "[element_table]")
{
  element_table t(
      make_options({"-l", std::to_string(3), "-d", std::to_string(2)}), 1);

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
