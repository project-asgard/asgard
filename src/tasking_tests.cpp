
#include "tasking.hpp"
#include "tests_general.hpp"

// check for complete, non-overlapping element assignment
auto const validity_check = [](std::vector<task> const &tasks,
                               int const num_elems) {
  enum element_status
  {
    unassigned,
    assigned
  };
  fk::matrix<element_status> coverage(num_elems, num_elems);
  for (task const &task : tasks)
  {
    for (int i = task.elem_start; i <= task.elem_end; ++i)
    {
      for (int j = task.conn_start; j <= task.conn_end; ++j)
      {
        // non-overlapping check
        REQUIRE(coverage(i, j) == element_status::unassigned);
        coverage(i, j) = element_status::assigned;
      }
    }
  }
  for (element_status const &status : coverage)
  {
    // complete check
    REQUIRE(status == element_status::assigned);
  }
};

TEST_CASE("tasking list generation", "[tasking]")
{
  SECTION("1 rank, 1 MB")
  {
    int const degree = 2;
    int const level  = 4;

    int const ranks = 1;
    int const MB    = 1;
    auto pde        = make_PDE<double>(PDE_opts::continuity_2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, pde->num_dims);

    int const num_tasks = get_num_tasks(table, *pde, ranks, MB);
    auto const tasks    = assign_elements_to_tasks(table, num_tasks);
    assert(static_cast<int>(tasks.size()) == num_tasks);
    validity_check(tasks, table.size());
  }
}
