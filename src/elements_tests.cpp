#include "elements.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <regex>
#include <string>
#include <unordered_set>

void test_element_table(PDE_opts const pde_choice,
                        fk::vector<int> const &levels,
                        std::string const &gold_filename,
                        int64_t const max_level, bool const full_grid = false)
{
  auto const degree = parser::NO_USER_VALUE;
  auto const cfl    = parser::DEFAULT_CFL;
  parser const cli_mock(pde_choice, levels, degree, cfl, full_grid, max_level);
  options const opts(cli_mock);
  auto const pde = make_PDE<double>(cli_mock);
  elements::table const elem_table(opts, *pde);

  auto const gold_table =
      fk::matrix<int>(read_matrix_from_txt_file(gold_filename));
  auto const gold_ids = fk::vector<double>(read_vector_from_txt_file(
      std::regex_replace(gold_filename, std::regex("table_"), "ids_")));

  // test size
  REQUIRE(elem_table.size() == gold_table.nrows());
  REQUIRE(elem_table.size() == gold_ids.size());

  auto const flat_table = elem_table.get_active_table().clone_onto_host();

  for (int64_t i = 0; i < elem_table.size(); ++i)
  {
    // test ordering
    auto const gold_id = static_cast<int64_t>(gold_ids(i));
    auto const test_id = elem_table.get_element_id(i);

    REQUIRE(gold_id == test_id);

    // test id to coord mapping
    auto const &test_coords = elem_table.get_coords(i);
    fk::vector<int> const gold_coords =
        gold_table.extract_submatrix(i, 0, 1, pde->num_dims * 2);
    fk::vector<int> const mapped_coords =
        elements::map_to_coords(test_id, opts.max_level, pde->num_dims);
    REQUIRE(mapped_coords == test_coords);
    REQUIRE(gold_coords == test_coords);

    // test mapping back to id
    auto const mapped_id =
        elements::map_to_id(mapped_coords, opts.max_level, pde->num_dims);
    REQUIRE(mapped_id == gold_id);

    auto const coord_size         = pde->num_dims * 2;
    auto const element_flat_index = static_cast<int64_t>(coord_size) * i;
    fk::vector<int, mem_type::const_view> const flat_coords(
        flat_table, element_flat_index, element_flat_index + coord_size - 1);
    REQUIRE(flat_coords == gold_coords);
  }
}

void test_child_discovery(PDE_opts const pde_choice,
                          fk::vector<int> const &levels,
                          std::string const &gold_filename,
                          int64_t const max_level, bool const full_grid = false)
{
  auto const degree = parser::NO_USER_VALUE;
  auto const cfl    = parser::DEFAULT_CFL;
  parser const cli_mock(pde_choice, levels, degree, cfl, full_grid, max_level);
  options const opts(cli_mock);
  auto const pde = make_PDE<double>(cli_mock);
  elements::table const elem_table(opts, *pde);

  auto const gold_child_vect =
      fk::vector<int>(read_vector_from_txt_file(gold_filename));
  std::list<int64_t> gold_child_ids(gold_child_vect.begin(),
                                    gold_child_vect.end());

  auto const child_ids = [&elem_table, &opts]() {
    std::list<int64_t> child_ids;
    for (int64_t i = 0; i < elem_table.size(); ++i)
    {
      child_ids.splice(child_ids.end(), elem_table.get_child_elements(i, opts));
    }
    return child_ids;
  }();

  REQUIRE(child_ids == gold_child_ids);
}

void test_element_addition(PDE_opts const pde_choice,
                           fk::vector<int> const &levels,
                           int64_t const max_level,
                           bool const full_grid = false)
{
  auto const degree = parser::NO_USER_VALUE;
  auto const cfl    = parser::DEFAULT_CFL;
  parser const cli_mock(pde_choice, levels, degree, cfl, full_grid, max_level);
  options const opts(cli_mock);
  auto const pde = make_PDE<double>(cli_mock);
  elements::table elem_table(opts, *pde);

  // store existing ids
  auto const active_ids = [&elem_table]() {
    std::vector<int64_t> active_ids;
    active_ids.reserve(elem_table.size());
    for (int64_t i = 0; i < elem_table.size(); ++i)
    {
      active_ids.push_back(elem_table.get_element_id(i));
    }
    return active_ids;
  }();

  // store existing flat table
  auto const active_table = elem_table.get_active_table();

  // store ids to refine
  auto const id_list = [&elem_table, &opts]() {
    auto const n = 5;
    assert(elem_table.size() > n);

    std::list<int64_t> ids_to_add;
    int64_t counter = 0;
    for (int64_t i = 0; i < elem_table.size(); i += n)
    {
      ids_to_add.splice(ids_to_add.end(),
                        elem_table.get_child_elements(i, opts));
      // and sometimes refine a contiguous element
      if ((counter++ % 2) == 0 && (i + 1) < elem_table.size())
      {
        ids_to_add.splice(ids_to_add.end(),
                          elem_table.get_child_elements(i + 1, opts));
      }
    }
    return ids_to_add;
  }();

  std::unordered_set<int64_t> unique_add(id_list.begin(), id_list.end());
  std::unordered_set<int64_t> active_set(active_ids.begin(), active_ids.end());
  assert(active_set.size() == active_ids.size());
  int64_t should_add = 0;
  for (auto const child_id : unique_add)
  {
    if (active_set.count(child_id) == 0)
    {
      should_add++;
    }
  }

  auto const old_size = elem_table.size();
  std::vector<int64_t> ids_to_add(id_list.begin(), id_list.end());
  auto const num_added = elem_table.add_elements(ids_to_add, opts.max_level);

  REQUIRE(num_added == should_add);
  REQUIRE(elem_table.size() == old_size + should_add);
}

void test_element_deletion(PDE_opts const pde_choice,
                           fk::vector<int> const &levels,
                           int64_t const max_level,
                           bool const full_grid = false)
{
  auto const degree = parser::NO_USER_VALUE;
  auto const cfl    = parser::DEFAULT_CFL;
  parser const cli_mock(pde_choice, levels, degree, cfl, full_grid, max_level);
  options const opts(cli_mock);
  auto const pde = make_PDE<double>(cli_mock);
  elements::table elem_table(opts, *pde);

  // delete every nth element,
  auto const n        = 5;
  auto const old_size = elem_table.size();
  assert(old_size > n);

  // store deleted indices
  std::vector<int64_t> indices_to_delete;

  // store deleted ids
  std::unordered_set<int64_t> deleted_ids;

  indices_to_delete.reserve(elem_table.size() / n);
  int64_t counter = 0;
  for (int64_t i = 0; i < elem_table.size(); i += n)
  {
    indices_to_delete.push_back(i);
    deleted_ids.insert(elem_table.get_element_id(i));
    // and sometimes delete a contiguous element
    if ((counter++ % 2) == 0 && (i + 1) < elem_table.size())
    {
      indices_to_delete.push_back(i + 1);
      deleted_ids.insert(elem_table.get_element_id(i + 1));
    }
  }

  std::unordered_set<int64_t> const to_delete(indices_to_delete.begin(),
                                              indices_to_delete.end());
  auto const gold_flat_table = [&elem_table, &to_delete]() {
    auto table = fk::vector<int>();
    // auto gold_ids = std::vector<int64_t>();
    for (int64_t i = 0; i < elem_table.size(); ++i)
    {
      if (to_delete.count(i) == 1)
      {
        continue;
      }
      table.concat(elem_table.get_coords(i));
      // gold_ids.push_back(elem_table.get_element_id(i));
    }
    return table;
  }();

  elem_table.remove_elements(indices_to_delete);

  // check that table shrank appropriately
  REQUIRE(elem_table.size() ==
          old_size - static_cast<int>(indices_to_delete.size()));
  auto const flat_table = elem_table.get_active_table().clone_onto_host();
  auto const coord_size = pde->num_dims * 2;
  REQUIRE(flat_table.size() / coord_size == elem_table.size());

  // check that deleted ids are not present in table
  for (int64_t i = 0; i < elem_table.size(); ++i)
  {
    REQUIRE(deleted_ids.count(elem_table.get_element_id(i)) == 0);
  }

  // check that retained coords were shifted left
  REQUIRE(indices_to_delete.size() > 0);
  REQUIRE(gold_flat_table == flat_table);

  // check that index to coord mapping still works,
  // also checks that ids are present in correct position
  for (int64_t i = 0; i < elem_table.size(); ++i)
  {
    REQUIRE(fk::vector<int, mem_type::const_view>(flat_table, i * coord_size,
                                                  (i + 1) * coord_size - 1) ==
            elem_table.get_coords(i));
  }
}

TEST_CASE("element table object", "[element_table]")
{
  std::vector<fk::vector<int>> const test_levels{{7}, {5, 2}, {3, 2, 3}};
  auto const max_level = 7;
  std::vector<PDE_opts> const test_pdes{
      PDE_opts::continuity_1, PDE_opts::continuity_2, PDE_opts::continuity_3};

  std::string const gold_base =
      "../testing/generated-inputs/element_table/table_";
  std::string const child_gold_base =
      "../testing/generated-inputs/element_table/child_ids_";

  SECTION("test table construction/mapping")
  {
    REQUIRE(test_levels.size() == test_pdes.size());

    for (auto i = 0; i < static_cast<int>(test_levels.size()); ++i)
    {
      auto const levels = test_levels[i];
      auto const choice = test_pdes[i];

      auto const full_gold_str =
          gold_base + std::to_string(test_levels[i].size()) + "d_FG.dat";
      auto const use_full_grid = true;
      test_element_table(choice, levels, full_gold_str, max_level,
                         use_full_grid);

      auto const sparse_gold_str =
          gold_base + std::to_string(test_levels[i].size()) + "d_SG.dat";
      test_element_table(choice, levels, sparse_gold_str, max_level);
    }
  }
  SECTION("adaptivity: child id discovery, element addition, "
          "element deletion")
  {
    assert(test_levels.size() == test_pdes.size());

    for (auto i = 0; i < static_cast<int>(test_levels.size()); ++i)
    {
      auto const levels = test_levels[i];
      auto const choice = test_pdes[i];

      auto const full_gold_str =
          child_gold_base + std::to_string(test_levels[i].size()) + "d_FG.dat";
      auto const use_full_grid = true;
      test_child_discovery(choice, levels, full_gold_str, max_level,
                           use_full_grid);
      test_element_addition(choice, levels, max_level, use_full_grid);
      test_element_deletion(choice, levels, max_level, use_full_grid);

      auto const sparse_gold_str =
          child_gold_base + std::to_string(test_levels[i].size()) + "d_SG.dat";
      test_child_discovery(choice, levels, sparse_gold_str, max_level);
      test_element_addition(choice, levels, max_level);
      test_element_deletion(choice, levels, max_level);
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
    REQUIRE(pair.size() == 2);
    auto const id   = elements::get_1d_index(pair(0), pair(1));
    auto const gold = static_cast<int64_t>(
        read_scalar_from_txt_file(gold_base + std::to_string(pair(0)) + "_" +
                                  std::to_string(pair(1)) + ".dat"));
    REQUIRE(id + 1 == gold);

    // map back to pair
    auto const [lev, cell] = elements::get_level_cell(id);
    REQUIRE(lev == pair(0));
    REQUIRE(cell == pair(1));
  }
}

TEST_CASE("static helper - cell builder", "[element_table]")
{
  auto const levels = 3;
  auto const degree = 2;

  auto const pde = make_PDE<double>(PDE_opts::continuity_1, levels, degree);
  elements::table const t(make_options({"-l", std::to_string(levels), "-d",
                                        std::to_string(degree)}),
                          *pde);

  SECTION("cell index set builder")
  {
    std::vector<fk::vector<int>> const levels_set = {
        {1}, {1, 2}, {2, 1}, {2, 3}};

    std::vector<fk::matrix<int>> const gold_set = {
        {{0}},
        {{0, 0}, {0, 1}},
        {{0, 0}, {1, 0}},
        {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}}};

    for (auto i = 0; i < static_cast<int>(gold_set.size()); ++i)
    {
      REQUIRE(t.get_cell_index_set(levels_set[i]) == gold_set[i]);
    }
  }
}
