
#include "grouping.hpp"
#include "tests_general.hpp"

// check for complete, non-overlapping element assignment
auto const validity_check = [](std::vector<element_group> const &groups,
                               element_table const &table) {
  enum element_status
  {
    unassigned,
    assigned
  };

  fk::matrix<element_status> coverage(table.size(), table.size());

  // non-overlapping check
  for (element_group const &group : groups)
  {
    for (const auto &[row, cols] : group)
    {
      for (int col = cols.first; col <= cols.second; ++col)
      {
        REQUIRE(coverage(row, col) == element_status::unassigned);
        coverage(row, col) = element_status::assigned;
      }
    }
  }

  // complete check
  for (int i = 0; i < coverage.nrows(); ++i)
  {
    for (int j = 0; j < coverage.ncols(); ++j)
    {
      REQUIRE(coverage(i, j) == element_status::assigned);
    }
  }
};

// check that a given task vector occupies between 50% and 101% of the limiit
auto const size_check = [](std::vector<element_group> const &groups,
                           PDE<double> const &pde, int const limit_MB,
                           bool const large_problem) {
  rank_workspace const work(pde, groups);
  double lower_bound    = static_cast<double>(limit_MB * 0.5);
  double upper_bound    = static_cast<double>(limit_MB * 1.01);
  double workspace_size = work.size_MB();
  if (large_problem)
  {
    REQUIRE(workspace_size > lower_bound);
  }
  REQUIRE(workspace_size < upper_bound);
};

TEST_CASE("group convenience functions", "[grouping]")
{
  SECTION("elements in group - empty")
  {
    element_group g;
    assert(num_elements_in_group(g) == 0);
  }
  SECTION("elements in group - single row")
  {
    element_group g;
    g[2] = std::make_pair(0, 4);
    assert(num_elements_in_group(g) == 5);
  }

  SECTION("elements in group - multiple rows")
  {
    element_group g;
    g[3] = std::make_pair(1, 2);
    g[4] = std::make_pair(5, 10);
    assert(num_elements_in_group(g) == 8);
  }

  SECTION("max connected in group - empty")
  {
    element_group g;
    assert(max_connected_in_group(g) == 0);
  }
  SECTION("max connected in group - single row")
  {
    element_group g;
    g[2] = std::make_pair(0, 4);
    assert(max_connected_in_group(g) == 5);
  }
  SECTION("max connected in group - multiple rows")
  {
    element_group g;
    g[3] = std::make_pair(1, 2);
    g[4] = std::make_pair(5, 10);
    assert(max_connected_in_group(g) == 6);
  }

  SECTION("columns in group - single row")
  {
    element_group g;
    g[2] = std::make_pair(0, 4);
    assert(columns_in_group(g) == std::make_pair(0, 4));
  }

  SECTION("columns in group - multiple rows")
  {
    element_group g;
    g[3] = std::make_pair(1, 2);
    g[4] = std::make_pair(5, 10);
    assert(columns_in_group(g) == std::make_pair(1, 10));
  }

  SECTION("rows in group - single row")
  {
    element_group g;
    g[2] = std::make_pair(0, 4);
    assert(rows_in_group(g) == std::make_pair(2, 2));
  }
  SECTION("rows in group - multiple rows")
  {
    element_group g;
    g[3] = std::make_pair(1, 2);
    g[4] = std::make_pair(5, 10);
    assert(rows_in_group(g) == std::make_pair(3, 4));
  }
}

TEST_CASE("element grouping, continuity 2", "[grouping]")
{
  SECTION("1 rank, deg 5, level 6, 1-1000 MB")
  {
    int const degree = 5;
    int const level  = 6;
    int const ranks  = 1;

    auto const pde = make_PDE<double>(PDE_opts::continuity_2, level, degree);
    bool const large_problem = false;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);
    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);

      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }

  SECTION("2 ranks, deg 5, level 6, 1-1000 MB")
  {
    int const degree = 5;
    int const level  = 6;
    int const ranks  = 2;

    auto const pde = make_PDE<double>(PDE_opts::continuity_2, level, degree);
    bool const large_problem = false;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);
    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);

      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }
}

TEST_CASE("element grouping, continuity 3", "[grouping]")
{
  SECTION("1 rank, deg 5, level 6, 1-1000 MB")
  {
    int const degree = 5;
    int const level  = 6;
    int const ranks  = 1;

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);

    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);

    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);
      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }

  SECTION("2 ranks, deg 5, level 6, 1-1000 MB")
  {
    int const degree = 5;
    int const level  = 6;
    int const ranks  = 2;

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);
    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);
    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);

      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }

  SECTION("3 ranks, deg 5, level 6, 1-1000 MB")
  {
    int const degree = 5;
    int const level  = 6;
    int const ranks  = 3;

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);
    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);
    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);

      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }

  SECTION("1 rank, deg 4, level 6, 1-1000 MB")
  {
    int const degree = 4;
    int const level  = 6;
    int const ranks  = 1;

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);

    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);

    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);

      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }

  SECTION("2 ranks, deg 4, level 6, 1-1000 MB")
  {
    int const degree = 4;
    int const level  = 6;
    int const ranks  = 2;

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);
    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);
    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);

      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }

  SECTION("3 ranks, deg 4, level 6, 1-1000 MB")
  {
    int const degree = 4;
    int const level  = 6;
    int const ranks  = 3;

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);
    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);
    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);

      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }
}

TEST_CASE("element grouping, continuity 6", "[grouping]")
{
  SECTION("1 rank, deg 3, level 4, 10-10000 MB")
  {
    int const degree = 3;
    int const level  = 4;
    int const ranks  = 1;

    auto const pde = make_PDE<double>(PDE_opts::continuity_6, level, degree);

    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);

    for (int limit_MB = 1; limit_MB <= 10000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);
      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }

  SECTION("2 ranks, deg 3, level 4, 10-10000 MB")
  {
    int const degree = 3;
    int const level  = 4;
    int const ranks  = 2;

    auto const pde = make_PDE<double>(PDE_opts::continuity_6, level, degree);

    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);

    for (int limit_MB = 10; limit_MB <= 10000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);
      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }

  SECTION("11 ranks, deg 4, level 4, 10-10000 MB")
  {
    int const degree = 4;
    int const level  = 4;
    int const ranks  = 11;

    auto const pde = make_PDE<double>(PDE_opts::continuity_6, level, degree);

    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);

    for (int limit_MB = 10; limit_MB <= 10000; limit_MB *= 10)
    {
      int const num_groups = get_num_groups(table, *pde, ranks, limit_MB);
      auto const groups    = assign_elements(table, num_groups);
      assert(groups.size() % ranks == 0);
      assert(static_cast<int>(groups.size()) == num_groups);
      validity_check(groups, table);
      size_check(groups, *pde, limit_MB, large_problem);
    }
  }
}
