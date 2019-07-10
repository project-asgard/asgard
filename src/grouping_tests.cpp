
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

// FIXME we should eventually put this in the pde class
auto const element_segment_size = [](auto const &pde) {
  int const degree = pde.get_dimensions()[0].get_degree();
  return static_cast<int>(std::pow(degree, pde.num_dims));
};

auto const test_copy_in = [](PDE<double> const &pde, element_group const &group,
                             rank_workspace<double> const &rank_space,
                             host_workspace<double> const &host_space) {
  int const elem_size  = element_segment_size(pde);
  auto const x_range   = columns_in_group(group);
  auto const num_elems = (x_range.second - x_range.first + 1) * elem_size;

  for (int i = 0; i < num_elems; ++i)
  {
    REQUIRE(rank_space.batch_input(i) ==
            host_space.x(i + x_range.first * elem_size));
  }
};

auto const test_copy_out = [](PDE<double> const &pde,
                              element_group const &group,
                              rank_workspace<double> const &rank_space,
                              host_workspace<double> const &host_space,
                              fk::vector<double> const &fx_prior) {
  int const elem_size  = element_segment_size(pde);
  auto const y_range   = rows_in_group(group);
  auto const num_elems = (y_range.second - y_range.first + 1) * elem_size;

  for (int i = 0; i < num_elems; ++i)
  {
    int const fx_index = i + y_range.first * elem_size;
    REQUIRE(std::abs(host_space.fx(fx_index) - fx_prior(fx_index) -
                     rank_space.batch_output(i)) <
            std::numeric_limits<double>::epsilon() * num_elems);
  }
};

TEST_CASE("group data management functions", "[grouping]")
{
  SECTION("copy in deg 2/lev 4, continuity 1")
  {
    int const degree = 2;
    int const level  = 4;

    auto pde = make_PDE<double>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    host_workspace<double> host_space(*pde, elem_table);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(host_space.x.begin(), host_space.x.end(), gen);

    int const ranks    = 2;
    int const limit_MB = 1;
    auto const groups  = assign_elements(
        elem_table, get_num_groups(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, groups);

    for (auto const &group : groups)
    {
      // copy in inputs
      copy_group_inputs(*pde, rank_space, host_space, group);
      test_copy_in(*pde, group, rank_space, host_space);
    }
  }

  SECTION("copy in deg 4/lev 5, continuity 3")
  {
    int const degree = 4;
    int const level  = 5;

    auto pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    host_workspace<double> host_space(*pde, elem_table);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(host_space.x.begin(), host_space.x.end(), gen);

    int const ranks    = 3;
    int const limit_MB = 10;
    auto const groups  = assign_elements(
        elem_table, get_num_groups(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, groups);

    for (auto const &group : groups)
    {
      // copy in inputs
      copy_group_inputs(*pde, rank_space, host_space, group);
      test_copy_in(*pde, group, rank_space, host_space);
    }
  }

  SECTION("copy in deg 4/lev 2, continuity 6")
  {
    int const degree = 4;
    int const level  = 2;

    auto pde = make_PDE<double>(PDE_opts::continuity_6, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    host_workspace<double> host_space(*pde, elem_table);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(host_space.x.begin(), host_space.x.end(), gen);

    int const ranks    = 7;
    int const limit_MB = 100;
    auto const groups  = assign_elements(
        elem_table, get_num_groups(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, groups);

    for (auto const &group : groups)
    {
      // copy in inputs
      copy_group_inputs(*pde, rank_space, host_space, group);
      test_copy_in(*pde, group, rank_space, host_space);
    }
  }

  SECTION("copy out deg 2/lev 4, continuity 1")
  {
    int const degree = 2;
    int const level  = 4;

    auto pde = make_PDE<double>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    host_workspace<double> host_space(*pde, elem_table);

    int const ranks    = 2;
    int const limit_MB = 1;
    auto const groups  = assign_elements(
        elem_table, get_num_groups(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, groups);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.batch_output.begin(),
                  rank_space.batch_output.end(), gen);
    std::generate(host_space.fx.begin(), host_space.fx.end(), gen);

    for (auto const &group : groups)
    {
      fk::vector<double> fx_orig(host_space.fx);
      // copy out inputs
      copy_group_outputs(*pde, rank_space, host_space, group);
      test_copy_out(*pde, group, rank_space, host_space, fx_orig);
    }
  }

  SECTION("copy out deg 4/lev 5, continuity 3")
  {
    int const degree = 4;
    int const level  = 5;

    auto pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    host_workspace<double> host_space(*pde, elem_table);

    int const ranks    = 3;
    int const limit_MB = 10;
    auto const groups  = assign_elements(
        elem_table, get_num_groups(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, groups);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.batch_output.begin(),
                  rank_space.batch_output.end(), gen);
    std::generate(host_space.fx.begin(), host_space.fx.end(), gen);

    for (auto const &group : groups)
    {
      fk::vector<double> fx_orig(host_space.fx);
      // copy out inputs
      copy_group_outputs(*pde, rank_space, host_space, group);
      test_copy_out(*pde, group, rank_space, host_space, fx_orig);
    }
  }

  SECTION("copy out deg 4/lev 2, continuity 6")
  {
    int const degree = 4;
    int const level  = 2;

    auto pde = make_PDE<double>(PDE_opts::continuity_6, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    host_workspace<double> host_space(*pde, elem_table);

    int const ranks    = 7;
    int const limit_MB = 100;
    auto const groups  = assign_elements(
        elem_table, get_num_groups(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, groups);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.batch_output.begin(),
                  rank_space.batch_output.end(), gen);
    std::generate(host_space.fx.begin(), host_space.fx.end(), gen);

    for (auto const &group : groups)
    {
      fk::vector<double> fx_orig(host_space.fx);
      // copy out inputs
      copy_group_outputs(*pde, rank_space, host_space, group);
      test_copy_out(*pde, group, rank_space, host_space, fx_orig);
    }
  }
}

auto const test_reduction = [](PDE<double> const &pde,
                               element_group const &group,
                               rank_workspace<double> const &rank_space) {
  int const elem_size = element_segment_size(pde);
  auto const x_range  = columns_in_group(group);

  fk::vector<double> total_sum(rank_space.batch_output.size());
  for (auto const &[row, cols] : group)
  {
    int const prev_row_elems = [i = row, &group] {
      if (i == group.begin()->first)
      {
        return 0;
      }
      int prev_elems = 0;
      for (int r = group.begin()->first; r < i; ++r)
      {
        prev_elems += group.at(r).second - group.at(r).first + 1;
      }
      return prev_elems;
    }();
    int const reduction_offset = prev_row_elems * pde.num_terms * elem_size;
    fk::matrix<double, mem_type::view> const reduction_matrix(
        rank_space.reduction_space, elem_size,
        (cols.second - cols.first + 1) * pde.num_terms, reduction_offset);

    fk::vector<double> sum(reduction_matrix.nrows());
    for (int i = 0; i < reduction_matrix.nrows(); ++i)
    {
      for (int j = 0; j < reduction_matrix.ncols(); ++j)
        sum(i) += reduction_matrix(i, j);
    }
    int const row_this_task = row - group.begin()->first;
    fk::vector<double, mem_type::view> partial_sum(
        total_sum, row_this_task * elem_size,
        (row_this_task + 1) * elem_size - 1);

    partial_sum = partial_sum + sum;
  }

  fk::vector<double> const diff = rank_space.batch_output - total_sum;
  auto abs_compare              = [](double const a, double const b) {
    return (std::abs(a) < std::abs(b));
  };
  double const result =
      std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
  int const num_cols = (x_range.second - x_range.first + 1) * pde.num_terms;
  // tol = epsilon * possible number of additions for an element
  double const tol = std::numeric_limits<double>::epsilon() * num_cols * 3;
  REQUIRE(result <= tol);
};

TEST_CASE("group reduction function", "[grouping]")
{
  SECTION("reduction deg 2/lev 4, continuity 1")
  {
    int const degree = 2;
    int const level  = 4;

    auto pde = make_PDE<double>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    host_workspace<double> host_space(*pde, elem_table);

    int const ranks    = 2;
    int const limit_MB = 1;
    auto const groups  = assign_elements(
        elem_table, get_num_groups(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, groups);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.reduction_space.begin(),
                  rank_space.reduction_space.end(), gen);

    for (auto const &group : groups)
    {
      // reduce and test
      reduce_group(*pde, rank_space, group);
      test_reduction(*pde, group, rank_space);
    }
  }

  SECTION("reduction deg 5/lev 6, continuity 3")
  {
    int const degree = 5;
    int const level  = 6;

    auto pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    host_workspace<double> host_space(*pde, elem_table);

    int const ranks    = 4;
    int const limit_MB = 11;
    auto const groups  = assign_elements(
        elem_table, get_num_groups(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, groups);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.reduction_space.begin(),
                  rank_space.reduction_space.end(), gen);

    for (auto const &group : groups)
    {
      // reduce and test
      reduce_group(*pde, rank_space, group);
      test_reduction(*pde, group, rank_space);
    }
  }

  SECTION("reduction deg 3/lev 2, continuity 6")
  {
    int const degree = 3;
    int const level  = 2;

    auto pde = make_PDE<double>(PDE_opts::continuity_6, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    host_workspace<double> host_space(*pde, elem_table);

    int const ranks    = 7;
    int const limit_MB = 100;
    auto const groups  = assign_elements(
        elem_table, get_num_groups(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, groups);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.reduction_space.begin(),
                  rank_space.reduction_space.end(), gen);

    for (auto const &group : groups)
    {
      // reduce and test
      reduce_group(*pde, rank_space, group);
      test_reduction(*pde, group, rank_space);
    }
  }
}
