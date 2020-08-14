#include "element_table.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <regex>
#include <string>

void test_element_table(PDE_opts const pde_choice,
                        fk::vector<int> const &levels,
                        std::string const &gold_filename,
                        int64_t const max_level, bool const full_grid = false)
{
  parser const cli_mock(pde_choice, levels, full_grid, max_level);
  options const opts(cli_mock);
  auto const pde = make_PDE<double>(cli_mock);
  element_table const elem_table(opts, *pde);
  std::cout << gold_filename << '\n';
  auto const gold_table =
      fk::matrix<int>(read_matrix_from_txt_file(gold_filename));

  auto const gold_ids = fk::vector<double>(read_vector_from_txt_file(
      std::regex_replace(gold_filename, std::regex("table_"), "ids_")));

  // test size
  REQUIRE(elem_table.size() == gold_table.nrows());
  REQUIRE(elem_table.size() == gold_ids.size());

  for (int64_t i = 0; i < elem_table.size(); ++i)
  {
    // test ordering
    auto const gold_id = static_cast<int64_t>(gold_ids(i));
    auto const test_id = elem_table.get_element_id(i);

    REQUIRE(gold_id == test_id);

    // test id to coord mapping
    auto const &test_coords = elem_table.get_coords(test_id);
    fk::vector<int> const gold_coords =
        gold_table.extract_submatrix(i, 0, 1, pde->num_dims * 2);
    fk::vector<int> const mapped_coords = map_to_coords(test_id, opts, *pde);
    REQUIRE(mapped_coords == test_coords);
    REQUIRE(gold_coords == test_coords);

    // test mapping back to id
    auto const mapped_id = map_to_index(mapped_coords, opts, *pde);
    REQUIRE(mapped_id == gold_id);

    // FIXME test flattened table
  }
}

TEST_CASE("element table constructors/accessors/size/multid mapping",
          "[element_table]")
{
  std::vector<fk::vector<int>> const test_levels{{7}, {5, 2}, {3, 2, 3}};
  int const max_level = 7;
  std::vector<PDE_opts> const test_pdes{
      PDE_opts::continuity_1, PDE_opts::continuity_2, PDE_opts::continuity_3};
  std::string const gold_base =
      "../testing/generated-inputs/element_table/table_";

  SECTION("test table construction/mapping")
  {
    assert(test_levels.size() == test_pdes.size());

    for (auto i = 0; i < static_cast<int>(test_levels.size()); ++i)
    {
      auto const levels = test_levels[i];
      auto const choice = test_pdes[i];

      auto const full_gold_str =
          gold_base + std::to_string(test_levels[i].size()) + "d_FG.dat";
      auto const use_full_grid = true;
      //     test_element_table(choice, levels, full_gold_str, max_level,
      // use_full_grid);

      auto const sparse_gold_str =
          gold_base + std::to_string(test_levels[i].size()) + "d_SG.dat";
      test_element_table(choice, levels, sparse_gold_str, max_level);
    }
  }
}

TEST_CASE("1d mapping functions", "[element_table]")
{
  std::vector<fk::vector<int>> const pairs = {{0, 0}, {1, 0}, {2, 1}, {12, 5},
                                              {7, 0}, {4, 6}, {9, 3}, {30, 20}};
  std::string const gold_base =
      "../testing/generated-inputs/element_table/1d_index_";
  for (auto const &pair : pairs)
  {
    assert(pair.size() == 2);
    auto const id   = get_1d_index(pair(0), pair(1));
    auto const gold = static_cast<int64_t>(
        read_scalar_from_txt_file(gold_base + std::to_string(pair(0)) + "_" +
                                  std::to_string(pair(1)) + ".dat"));
    REQUIRE(id + 1 == gold);

    // map back to pair
    auto const [lev, cell] = get_level_cell(id);
    REQUIRE(lev == pair(0));
    REQUIRE(cell == pair(1));
  }
}

TEST_CASE("static helper - cell builder", "[element_table]")
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
