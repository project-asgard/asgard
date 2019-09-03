#include "distribution.hpp"
#include "tests_general.hpp"

struct distribution_test_init
{
  distribution_test_init()
  {
    auto const [rank, total_ranks] = initialize_distribution();
    my_rank                        = rank;
    num_ranks                      = total_ranks;
  }
  ~distribution_test_init() { finalize_distribution(); }
  int get_my_rank() const { return my_rank; }
  int get_num_ranks() const { return num_ranks; }

private:
  int my_rank;
  int num_ranks;
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEST_CASE("subgrid struct", "[distribution]")
{
  int const row_start = 0;
  int const row_stop  = 4;
  int const col_start = 1;
  int const col_stop  = 3;
  element_subgrid const e(row_start, row_stop, col_start, col_stop);

  SECTION("construct/copy construct")
  {
    REQUIRE(e.row_start == row_start);
    REQUIRE(e.row_stop == row_stop);
    REQUIRE(e.col_start == col_start);
    REQUIRE(e.col_stop == col_stop);

    element_subgrid const e2(e);
    REQUIRE(e == e2);

    REQUIRE(e2.row_start == row_start);
    REQUIRE(e2.row_stop == row_stop);
    REQUIRE(e2.col_start == col_start);
    REQUIRE(e2.col_stop == col_stop);
  }

  SECTION("dimensions functions")
  {
    REQUIRE(e.nrows() == 5);
    REQUIRE(e.ncols() == 3);
    REQUIRE(e.size() == 15);
  }

  SECTION("translation functions")
  {
    REQUIRE(e.to_global_row(3) == 3);
    REQUIRE(e.to_global_col(2) == 3);
    REQUIRE(e.to_local_row(0) == 0);
    REQUIRE(e.to_local_col(1) == 0);

    int const row_start = 2;
    int const row_stop  = 5;
    int const col_start = 0;
    int const col_stop  = 4;
    element_subgrid const e2(row_start, row_stop, col_start, col_stop);

    REQUIRE(e2.to_global_row(3) == 5);
    REQUIRE(e2.to_global_col(0) == 0);
    REQUIRE(e2.to_local_row(3) == 1);
    REQUIRE(e2.to_local_col(2) == 2);
  }
}

auto const check_coverage = [](element_table const &table,
                               distribution_plan const &to_test) {
  enum element_status
  {
    unassigned,
    assigned
  };

  fk::matrix<element_status> coverage(table.size(), table.size());

  for (auto const &[rank, grid] : to_test)
  {
    ignore(rank);
    for (int row = grid.row_start; row <= grid.row_stop; ++row)
    {
      for (int col = grid.col_start; col <= grid.col_stop; ++col)
      {
        if (coverage(row, col) != element_status::unassigned)
        {
          return false;
        }
        coverage(row, col) = element_status::assigned;
      }
    }
  }
  for (auto const &elem : coverage)
  {
    if (elem != element_status::assigned)
    {
      return false;
    }
  }
  return true;
};

auto const check_even_sizing = [](element_table const &table,
                                  distribution_plan const &to_test) {
  auto const size = to_test.at(0).size();
  for (auto const &[rank, grid] : to_test)
  {
    ignore(rank);
    REQUIRE(std::abs(size - grid.size()) <
            table.size() *
                2); // at most, a subgrid's size should differ from
                    // another's (in the case of uneven division of elements
                    // by number of ranks) by one row and one column
  }
};

auto const check_rowmaj_layout = [](distribution_plan const &to_test,
                                    int const num_cols) {
  for (auto const &[rank, grid] : to_test)
  {
    int const my_col = rank % num_cols;
    if (my_col != 0)
    {
      REQUIRE(grid.col_start == to_test.at(rank - 1).col_stop + 1);
    }
  }
};

TEST_CASE("rank subgrid function", "[distribution]")
{
  SECTION("1 rank, whole problem")
  {
    int const degree   = 4;
    int const level    = 4;
    int const num_dims = 2;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks = 1;
    int const my_rank   = 0;

    element_subgrid const e(get_subgrid(num_ranks, my_rank, table));

    REQUIRE(e.row_start == 0);
    REQUIRE(e.row_stop == table.size() - 1);
    REQUIRE(e.col_start == 0);
    REQUIRE(e.col_stop == table.size() - 1);
  }

  SECTION("2 ranks")
  {
    int const degree   = 6;
    int const level    = 5;
    int const num_dims = 2;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks  = 2;
    int const first_rank = 0;

    // 2 ranks should divide by single vertical split
    element_subgrid const e(get_subgrid(num_ranks, first_rank, table));

    REQUIRE(e.row_start == 0);
    REQUIRE(e.row_stop == table.size() - 1);
    REQUIRE(e.col_start == 0);
    REQUIRE(e.col_stop == table.size() / 2 - 1);

    int const second_rank = 1;
    element_subgrid const e2(get_subgrid(num_ranks, second_rank, table));

    REQUIRE(e2.row_start == 0);
    REQUIRE(e2.row_stop == table.size() - 1);
    REQUIRE(e2.col_start == table.size() / 2);
    REQUIRE(e2.col_stop == table.size() - 1);
  }

  SECTION("4 ranks - even/square")
  {
    int const degree   = 10;
    int const level    = 7;
    int const num_dims = 3;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks  = 4;
    int const first_rank = 0;

    // testing the sizing and arrangement (row major) of subgrids
    element_subgrid const e(get_subgrid(num_ranks, first_rank, table));

    REQUIRE(e.row_start == 0);
    REQUIRE(e.row_stop == table.size() / 2 - 1);
    REQUIRE(e.col_start == 0);
    REQUIRE(e.col_stop == table.size() / 2 - 1);

    int const second_rank = 1;
    element_subgrid const e2(get_subgrid(num_ranks, second_rank, table));

    REQUIRE(e2.row_start == 0);
    REQUIRE(e2.row_stop == table.size() / 2 - 1);
    REQUIRE(e2.col_start == table.size() / 2);
    REQUIRE(e2.col_stop == table.size() - 1);

    int const third_rank = 2;
    element_subgrid const e3(get_subgrid(num_ranks, third_rank, table));

    REQUIRE(e3.row_start == table.size() / 2);
    REQUIRE(e3.row_stop == table.size() - 1);
    REQUIRE(e3.col_start == 0);
    REQUIRE(e3.col_stop == table.size() / 2 - 1);

    int const fourth_rank = 3;
    element_subgrid const e4(get_subgrid(num_ranks, fourth_rank, table));

    REQUIRE(e4.row_start == table.size() / 2);
    REQUIRE(e4.row_stop == table.size() - 1);
    REQUIRE(e4.col_start == table.size() / 2);
    REQUIRE(e4.col_stop == table.size() - 1);
  }

  SECTION("9 ranks - odd/square")
  {
    int const degree   = 4;
    int const level    = 5;
    int const num_dims = 6;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks = 9;

    distribution_plan plan;
    for (int i = 0; i < num_ranks; ++i)
    {
      element_subgrid const e(get_subgrid(num_ranks, i, table));
      REQUIRE(std::abs(e.nrows() - e.ncols()) <
              2); // square number of ranks should produce square subgrids
                  // left over elements are greedily assigned, maximum
                  // difference between row/col number should be one

      plan.emplace(i, e);
    }

    check_coverage(table, plan);    // and the subgrids should cover the table
    check_even_sizing(table, plan); // relatively similar sizing
    int const grid_cols = 3;
    check_rowmaj_layout(plan, grid_cols); // arranged row-major
  }
}

TEST_CASE("distribution plan function", "[distribution]")
{
  SECTION("1 rank")
  {
    int const degree   = 3;
    int const level    = 2;
    int const num_dims = 2;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks = 1;

    auto const plan = get_plan(num_ranks, table);
    check_coverage(table, plan);
  }

  SECTION("2 ranks - also, test 3rd rank ignored")
  {
    int const degree   = 8;
    int const level    = 5;
    int const num_dims = 2;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks = 2;
    auto const plan     = get_plan(num_ranks, table);
    check_coverage(table, plan);
    check_even_sizing(table, plan);
    int const ncols = 2;
    check_rowmaj_layout(plan, ncols);

    int const num_ranks_extra = 3;
    auto const plan_extra     = get_plan(num_ranks_extra, table);
    check_coverage(table, plan);

    assert(plan.size() == plan_extra.size());
    for (auto const &[rank, grid] : plan)
    {
      assert(grid == plan_extra.at(rank));
    }
  }

  SECTION("20 ranks")
  {
    int const degree   = 5;
    int const level    = 5;
    int const num_dims = 6;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks = 20;
    auto const plan     = get_plan(num_ranks, table);
    check_coverage(table, plan);
    check_even_sizing(table, plan);
    int const ncols = 5;
    check_rowmaj_layout(plan, ncols);
  }
}

TEST_CASE("reduction mapping function", "[distribution]")
{
  SECTION("1 rank")
  {
    int const degree   = 3;
    int const level    = 2;
    int const num_dims = 2;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks = 1;
    int const my_rank   = 0;
    auto const reduction_partners =
        get_reduction_partners(get_plan(num_ranks, table), my_rank);
    REQUIRE(reduction_partners == fk::vector<int>{0});
  }

  SECTION("2 ranks")
  {
    int const degree   = 8;
    int const level    = 5;
    int const num_dims = 2;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks = 2;
    auto const plan     = get_plan(num_ranks, table);

    int const first_rank            = 0;
    auto const reduction_partners_0 = get_reduction_partners(plan, first_rank);
    REQUIRE(reduction_partners_0 == fk::vector<int>{0, 1});

    int const second_rank           = 1;
    auto const reduction_partners_1 = get_reduction_partners(plan, second_rank);
    REQUIRE(reduction_partners_1 == fk::vector<int>{0, 1});
  }

  SECTION("9 ranks")
  {
    int const degree   = 5;
    int const level    = 5;
    int const num_dims = 6;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks = 9;
    auto const plan     = get_plan(num_ranks, table);

    for (int i = 0; i < 3; ++i)
    {
      auto const reduction_partners = get_reduction_partners(plan, i);
      REQUIRE(reduction_partners == fk::vector<int>{0, 1, 2});
    }
    for (int i = 3; i < 6; ++i)
    {
      auto const reduction_partners = get_reduction_partners(plan, i);
      REQUIRE(reduction_partners == fk::vector<int>{3, 4, 5});
    }
    for (int i = 6; i < 9; ++i)
    {
      auto const reduction_partners = get_reduction_partners(plan, i);
      REQUIRE(reduction_partners == fk::vector<int>{6, 7, 8});
    }
  }

  SECTION("20 ranks")
  {
    int const degree   = 10;
    int const level    = 5;
    int const num_dims = 3;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);

    int const num_ranks = 20;
    auto const plan     = get_plan(num_ranks, table);

    for (int i = 0; i < 5; ++i)
    {
      auto const reduction_partners = get_reduction_partners(plan, i);
      REQUIRE(reduction_partners == fk::vector<int>{0, 1, 2, 3, 4});
    }
    for (int i = 5; i < 10; ++i)
    {
      auto const reduction_partners = get_reduction_partners(plan, i);
      REQUIRE(reduction_partners == fk::vector<int>{5, 6, 7, 8, 9});
    }
    for (int i = 10; i < 15; ++i)
    {
      auto const reduction_partners = get_reduction_partners(plan, i);
      REQUIRE(reduction_partners == fk::vector<int>{10, 11, 12, 13, 14});
    }
    for (int i = 15; i < 20; ++i)
    {
      auto const reduction_partners = get_reduction_partners(plan, i);
      REQUIRE(reduction_partners == fk::vector<int>{15, 16, 17, 18, 19});
    }
  }
}

TEMPLATE_TEST_CASE("allreduce across row of subgrids", "[distribution]", float,
                   double)
{
  SECTION("1 rank")
  {
    auto const num_ranks = 1;
    auto const my_rank   = 0;
    int const degree     = 4;
    int const level      = 2;
    int const num_dims   = 2;

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, num_dims);
    auto const plan = get_plan(num_ranks, table);

    fk::vector<TestType> const gold{0, 1, 2, 3, 4, 5};
    fk::vector<TestType> const x(gold);
    fk::vector<TestType> fx(gold.size());
    reduce_results(x, fx, plan, my_rank);
    REQUIRE(fx == gold);
  }

  SECTION("multiple ranks")
  {
#ifdef ASGARD_USE_MPI

    int const my_rank   = distrib_test_info.get_my_rank();
    int const num_ranks = distrib_test_info.get_num_ranks();
    if (my_rank < num_ranks)
    {
      int const degree   = 5;
      int const level    = 4;
      int const num_dims = 3;

      options const o = make_options(
          {"-l", std::to_string(level), "-d", std::to_string(degree)});

      element_table const table(o, num_dims);
      auto const plan       = get_plan(num_ranks, table);
      int const vector_size = 10;

      std::vector<fk::vector<TestType>> rank_outputs;
      for (int i = 0; i < static_cast<int>(plan.size()); ++i)
      {
        fk::vector<TestType> rank_output(vector_size);
        std::iota(rank_output.begin(), rank_output.end(), i * vector_size);
        rank_outputs.push_back(rank_output);
      }
      int const my_row = my_rank / get_num_subgrid_cols(num_ranks);
      fk::vector<TestType> gold(vector_size);
      for (int i = 0; i < static_cast<int>(rank_outputs.size()); ++i)
      {
        if (i / get_num_subgrid_cols(num_ranks) == my_row)
        {
          gold = gold + rank_outputs[i];
        }
      }

      auto const &x =
          rank_outputs[std::min(my_rank, static_cast<int>(plan.size()) - 1)];
      fk::vector<TestType> fx(gold.size());
      reduce_results(x, fx, plan, my_rank);

      REQUIRE(fx == gold);
    }

#else
    REQUIRE(true);
#endif
  }
}
