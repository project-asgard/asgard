
#include "tasking.hpp"
#include "tests_general.hpp"

// check for complete, non-overlapping element assignment
auto const validity_check = [](std::vector<task> const &tasks,
                               element_table const &table) {
  enum element_status
  {
    unassigned,
    assigned
  };

  fk::matrix<element_status> coverage(table.size(), table.size());

  // non-overlapping check
  for (task const &task : tasks)
  {
    if (task.elem_end > task.elem_start)
    {
      for (int i = task.elem_start + 1; i < task.elem_end; ++i)
      {
        for (int j = 0; j < table.size(); ++j)
        {
          REQUIRE(coverage(i, j) == element_status::unassigned);
          coverage(i, j) = element_status::assigned;
        }
      }

      for (int j = task.conn_start; j < table.size(); ++j)
      {
        REQUIRE(coverage(task.elem_start, j) == element_status::unassigned);
        coverage(task.elem_start, j) = element_status::assigned;
      }
      for (int j = 0; j <= task.conn_end; ++j)
      {
        REQUIRE(coverage(task.elem_end, j) == element_status::unassigned);
        coverage(task.elem_end, j) = element_status::assigned;
      }
    }
    else
    {
      for (int j = task.conn_start; j <= task.conn_end; ++j)
      {
        REQUIRE(coverage(task.elem_start, j) == element_status::unassigned);
        coverage(task.elem_start, j) = element_status::assigned;
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

// check that a given task vector occupies between 50% and 100% of the limit
// only expected to pass when problem size > limit_MB * num_ranks

auto const size_check = [](std::vector<task> const &tasks,
                           PDE<double> const &pde, element_table const &table,
                           int const limit_MB, bool const large_problem) {
  task_workspace const work(pde, table, tasks);
  double lower_bound    = static_cast<double>(limit_MB * 0.5);
  double upper_bound    = static_cast<double>(limit_MB * 1.0);
  double workspace_size = work.size_MB();
  if (large_problem)
  {
    REQUIRE(workspace_size > lower_bound);
  }
  REQUIRE(workspace_size < upper_bound);
};

TEST_CASE("tasking list generation, continuity 2", "[tasking]")
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
      int const num_tasks = get_num_tasks(table, *pde, ranks, limit_MB);
      auto const tasks    = assign_elements_to_tasks(table, num_tasks);

      assert(static_cast<int>(tasks.size()) == num_tasks);
      validity_check(tasks, table);
      size_check(tasks, *pde, table, limit_MB, large_problem);
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
      int const num_tasks = get_num_tasks(table, *pde, ranks, limit_MB);
      auto const tasks    = assign_elements_to_tasks(table, num_tasks);

      assert(static_cast<int>(tasks.size()) == num_tasks);
      validity_check(tasks, table);
      size_check(tasks, *pde, table, limit_MB, large_problem);
    }
  }
}

TEST_CASE("tasking list generation, continuity 3", "[tasking]")
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
      int const num_tasks = get_num_tasks(table, *pde, ranks, limit_MB);
      auto const tasks    = assign_elements_to_tasks(table, num_tasks);

      assert(static_cast<int>(tasks.size()) == num_tasks);
      validity_check(tasks, table);
      size_check(tasks, *pde, table, limit_MB, large_problem);
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
      int const num_tasks = get_num_tasks(table, *pde, ranks, limit_MB);
      auto const tasks    = assign_elements_to_tasks(table, num_tasks);

      assert(static_cast<int>(tasks.size()) == num_tasks);
      validity_check(tasks, table);
      size_check(tasks, *pde, table, limit_MB, large_problem);
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
      int const num_tasks = get_num_tasks(table, *pde, ranks, limit_MB);
      auto const tasks    = assign_elements_to_tasks(table, num_tasks);

      assert(static_cast<int>(tasks.size()) == num_tasks);
      validity_check(tasks, table);
      size_check(tasks, *pde, table, limit_MB, large_problem);
    }
  }

  SECTION("1 rank, deg 4, level 7, 1-1000 MB")
  {
    int const degree = 4;
    int const level  = 7;
    int const ranks  = 1;

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);

    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);

    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_tasks = get_num_tasks(table, *pde, ranks, limit_MB);
      auto const tasks    = assign_elements_to_tasks(table, num_tasks);

      assert(static_cast<int>(tasks.size()) == num_tasks);
      validity_check(tasks, table);
      size_check(tasks, *pde, table, limit_MB, large_problem);
    }
  }

  SECTION("2 ranks, deg 4, level 7, 1-1000 MB")
  {
    int const degree = 4;
    int const level  = 7;
    int const ranks  = 2;

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);
    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);
    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_tasks = get_num_tasks(table, *pde, ranks, limit_MB);
      auto const tasks    = assign_elements_to_tasks(table, num_tasks);

      assert(static_cast<int>(tasks.size()) == num_tasks);
      validity_check(tasks, table);
      size_check(tasks, *pde, table, limit_MB, large_problem);
    }
  }

  SECTION("3 ranks, deg 4, level 7, 1-1000 MB")
  {
    int const degree = 4;
    int const level  = 7;
    int const ranks  = 3;

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);
    bool const large_problem = true;
    options const o          = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});
    element_table const table(o, pde->num_dims);
    for (int limit_MB = 1; limit_MB <= 1000; limit_MB *= 10)
    {
      int const num_tasks = get_num_tasks(table, *pde, ranks, limit_MB);
      auto const tasks    = assign_elements_to_tasks(table, num_tasks);

      assert(static_cast<int>(tasks.size()) == num_tasks);
      validity_check(tasks, table);
      size_check(tasks, *pde, table, limit_MB, large_problem);
    }
  }
}
