
#include "tasking.hpp"
#include "tests_general.hpp"

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
    assert(tasks.size() == num_tasks);
    REQUIRE(true);
  }
}
