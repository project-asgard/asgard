#include "element_table.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <regex>
#include <string>

void test_element_table(int const levels, int const dims,
                        std::string const gold_filename,
                        bool const full_grid = false)
{
  std::string const grid_str = full_grid ? "-f" : "";
  options const o = make_options({"-l", std::to_string(levels), grid_str});

  element_table const t(o, levels, dims);
  fk::vector<int> const dev_table(t.get_device_table().clone_onto_host());
  auto const gold = fk::matrix<int>(read_matrix_from_txt_file(gold_filename));
  for (int i = 0; i < static_cast<int>(t.size()); ++i)
  {
    fk::vector<int> const gold_coords =
        gold.extract_submatrix(i, 0, 1, gold.ncols());
    REQUIRE(t.get_coords(i) == gold_coords);
    REQUIRE(t.get_index(gold_coords) == i);
    fk::vector<int, mem_type::const_view> const dev_coords(
        dev_table, i * dims * 2, (i + 1) * dims * 2 - 1);
    REQUIRE(dev_coords == gold_coords);
  }
}

void test_element_table(PDE_opts const pde_choice,
                        fk::vector<int> const &levels,
                        std::string const &gold_filename,
                        bool const full_grid = false)
{
  parser const cli_mock(pde_choice, levels, full_grid);
  options const opts(cli_mock);
  auto const pde = make_PDE<double>(cli_mock);
  element_table const elem_table(opts, *pde);

   auto const gold_table =
          fk::matrix<int>(read_matrix_from_txt_file(gold_filename));

   auto const gold_ids = fk::vector<double>(read_vector_from_txt_file(std::regex_replace(gold_filename, std::regex("table_"), "ids_")));

  for (int i = 0; i < static_cast<int>(elem_table.size()); ++i)
  {

  }
}

TEST_CASE("element table constructor/accessors/size", "[element_table]")
{
  static std::string const gold_base =
      "../testing/generated-inputs/element_table/element_table";
  SECTION("1D element table")
  {
    int const levels            = 2;
    int const dims              = 1;
    std::string const gold_path = gold_base + "_1_2_SG.dat";
    test_element_table(levels, dims, gold_path);
  }

  SECTION("2D element table")
  {
    int const levels            = 3;
    int const dims              = 2;
    std::string const gold_path = gold_base + "_2_3_SG.dat";
    test_element_table(levels, dims, gold_path);
  }

  SECTION("3D element table, full grid")
  {
    int const levels            = 4;
    int const dims              = 3;
    std::string const gold_path = gold_base + "_3_4_FG.dat";
    bool const full_grid        = true;
    test_element_table(levels, dims, gold_path, full_grid);
  }
}

TEST_CASE("Static helper - cell builder", "[element_table]")
{
  int const levels = 3;
  int const degree = 2;
  element_table const t(make_options({"-l", std::to_string(levels), "-d",
                                      std::to_string(degree)}),
                        levels, 1);

  SECTION("cell index set builder")
  {
    std::vector<fk::vector<int>> const levels_set = {
        {1}, {1, 2}, {2, 1}, {2, 3}};

    std::vector<fk::matrix<int>> const gold_set = {
        {{0}},
        {{0, 0}, {0, 1}},
        {{0, 0}, {1, 0}},
        {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}}};

    for (int i = 0; i < static_cast<int>(gold_set.size()); ++i)
    {
      REQUIRE(t.get_cell_index_set(levels_set[i]) == gold_set[i]);
    }
  }
}

TEST_CASE("element table constructors/accessors/size", "[element_table]")
{
  std::vector<fk::vector<int>> const test_levels{{9}, {5, 2}, {3, 2, 3}};
  std::vector<PDE_opts> const test_pdes{
      PDE_opts::continuity_1, PDE_opts::continuity_2, PDE_opts::continuity_3};
  std::string const gold_base =
      "../testing/generated-inputs/element_table/table_";

  SECTION("test table construction")
  {
    assert(test_levels.size() == test_pdes.size());

    for (auto i = 0; i < static_cast<int>(test_levels.size()); ++i)
    {
      auto const levels = test_levels[i];
      auto const choice = test_pdes[i];

      auto const full_gold_str = gold_base +
                                 std::to_string(test_levels[i].size() / 2 + 1) +
                                 "d_FG.dat";
      test_element_table(choice, levels, full_gold_str, true);
      
      auto const sparse_gold_str =
          gold_base + std::to_string(test_levels[i].size() / 2 + 1) +
          "d_SG.dat";

      test_element_table(choice, levels, sparse_gold_str, false);
    }
  }
}
