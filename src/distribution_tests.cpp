#include "build_info.hpp"
#include "distribution.hpp"
#include "tests_general.hpp"

struct distribution_test_init
{
  distribution_test_init()
  {
#ifdef ASGARD_USE_MPI
    auto const [rank, total_ranks] = initialize_distribution();
    my_rank                        = rank;
    num_ranks                      = total_ranks;
#else
    my_rank   = 0;
    num_ranks = 1;
#endif
  }
  ~distribution_test_init()
  {
#ifdef ASGARD_USE_MPI
    finalize_distribution();
#endif
  }

  int get_my_rank() const { return my_rank; }
  int get_num_ranks() const { return num_ranks; }

private:
  int my_rank;
  int num_ranks;
};

static distribution_test_init const distrib_test_info;

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

void check_coverage(element_table const &table,
                    distribution_plan const &to_test)
{
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
        REQUIRE(coverage(row, col) == element_status::unassigned);
        coverage(row, col) = element_status::assigned;
      }
    }
  }
  for (auto const &elem : coverage)
  {
    REQUIRE(elem == element_status::assigned);
  }
}

void check_even_sizing(element_table const &table,
                       distribution_plan const &to_test)
{
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
}

void check_rowmaj_layout(distribution_plan const &to_test, int const num_cols)
{
  for (auto const &[rank, grid] : to_test)
  {
    int const my_col = rank % num_cols;
    if (my_col != 0)
    {
      REQUIRE(grid.col_start == to_test.at(rank - 1).col_stop + 1);
    }
  }
}

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

void generate_messages_test(int const num_ranks, element_table const &table)
{
  auto const plan     = get_plan(num_ranks, table);
  auto const messages = generate_messages(plan);

  // every rank should have a message list
  REQUIRE(messages.size() == plan.size());

  fk::vector<int> send_counter(plan.size());
  for (int i = 0; i < static_cast<int>(messages.size()); ++i)
  {
    auto const &message_list = messages[i];
    auto const &subgrid      = plan.at(i);

    for (int m = 0; m < static_cast<int>(message_list.size()); ++m)
    {
      auto const &message = message_list[m];

      if (message.message_dir == message_direction::send)
      {
        if (message.target != i)
        {
          send_counter(i) += 1;
        }
        // make sure the send message is inside my assigned outputs
        REQUIRE(message.range.start >= subgrid.row_start);
        REQUIRE(message.range.stop <= subgrid.row_stop);
      }
      // receive
      else
      {
        // make sure the receive message is inside my assigned inputs
        REQUIRE(message.range.start >= subgrid.col_start);
        REQUIRE(message.range.stop <= subgrid.col_stop);

        // also, check the matching send
        int const sender_rank       = message.target;
        auto const &sender_messages = messages[sender_rank];
        int match_found             = 0;
        int send_index              = 0;
        for (int j = 0; j < static_cast<int>(sender_messages.size()); ++j)
        {
          auto const &sender_message = sender_messages[j];
          if (sender_message.range == message.range &&
              sender_message.message_dir == message_direction::send &&
              sender_message.target == i)

          {
            send_index = j;
            match_found++;
          }
        }

        // want to find exactly one matching send
        REQUIRE(match_found == 1);

        if (message.target == i)
        {
          continue;
        }
        // to prevent deadlock, make sure sender doesn't have
        // a receive from me that occurs before this send,
        // UNLESS I have a send to them that occurs before that receive
        for (int j = 0; j < send_index; ++j)
        {
          auto const &sender_message = sender_messages[j];
          if (sender_message.message_dir == message_direction::receive &&
              sender_message.target == i)
          {
            bool preceding_send_found = false;
            for (int k = 0; k < m; ++k)
            {
              auto const &my_message = message_list[k];

              if (my_message.message_dir == message_direction::send &&
                  my_message.target == sender_rank)
              {
                preceding_send_found = true;
                break;
              }
            }
            REQUIRE(preceding_send_found);
            break;
          }
        }
      }
    }
  }

  // all subgrid row members have the same data;
  // they should all have the same (+/- 1) number
  // of sends queued up
  int const num_cols = get_num_subgrid_cols(plan.size());
  int const num_rows = plan.size() / num_cols;
  for (auto const &[rank, grid] : plan)
  {
    ignore(grid);

    int const my_col    = rank % num_cols;
    int const row_begin = rank - my_col;
    int const my_sends  = send_counter(rank);

    for (int i = row_begin; i < num_rows; ++i)
    {
      REQUIRE(std::abs(my_sends - send_counter(i)) < 2);
    }
  }
}

TEST_CASE("generate messages tests", "[distribution]")
{
  SECTION("one rank, small problem")
  {
    int const degree = 2;
    int const level  = 2;
    options const o  = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    auto const pde = make_PDE<double>(PDE_opts::continuity_1, level, degree);
    element_table const table(o, pde->num_dims);

    int const num_ranks = 1;
    generate_messages_test(num_ranks, table);
  }

  SECTION("one rank, larger problem")
  {
    int const degree = 4;
    int const level  = 3;
    options const o  = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);
    element_table const table(o, pde->num_dims);

    int const num_ranks = 1;
    generate_messages_test(num_ranks, table);
  }

  SECTION("perfect square number of ranks, small")
  {
    int const degree = 3;
    int const level  = 4;
    options const o  = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    auto const pde = make_PDE<double>(PDE_opts::continuity_2, level, degree);
    element_table const table(o, pde->num_dims);

    int const num_ranks = 9;
    generate_messages_test(num_ranks, table);
  }

  SECTION("perfect square number of ranks, large")
  {
    int const degree = 6;
    int const level  = 5;
    options const o  = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    auto const pde = make_PDE<double>(PDE_opts::continuity_3, level, degree);
    element_table const table(o, pde->num_dims);

    int const num_ranks = 36;
    generate_messages_test(num_ranks, table);
  }

  SECTION("even but not square, small")
  {
    int const degree = 5;
    int const level  = 8;
    options const o  = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    auto const pde = make_PDE<double>(PDE_opts::continuity_1, level, degree);
    element_table const table(o, pde->num_dims);

    int const num_ranks = 20;
    generate_messages_test(num_ranks, table);
  }

  SECTION("even but not square, large")
  {
    int const degree = 3;
    int const level  = 2;
    options const o  = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    auto const pde = make_PDE<double>(PDE_opts::continuity_6, level, degree);
    element_table const table(o, pde->num_dims);

    int const num_ranks = 32;
    generate_messages_test(num_ranks, table);
  }
}

TEMPLATE_TEST_CASE("prepare inputs tests", "[distribution]", float, double)
{
  // in this case, the source vector is simply copied into dest
  SECTION("single rank")
  {
    fk::vector<TestType> const source{1, 2, 3, 4, 5};
    fk::vector<TestType> dest(source.size());
    distribution_plan plan;
    plan.emplace(0, element_subgrid(0, 1, 2, 3));
    int const segment_size = 0;
    int const my_rank      = 0;
    exchange_results(source, dest, segment_size, plan, my_rank);
    REQUIRE(source == dest);
  }
  SECTION("multiple rank")
  {
#ifdef ASGARD_USE_MPI

    int const my_rank   = distrib_test_info.get_my_rank();
    int const num_ranks = distrib_test_info.get_num_ranks();
    if (my_rank < num_ranks)
    {
      int const degree        = 4;
      int const level         = 6;
      int const num_dims      = 2;
      auto const segment_size = static_cast<int>(std::pow(degree, num_dims));

      options const o = make_options(
          {"-l", std::to_string(level), "-d", std::to_string(degree)});

      element_table const table(o, num_dims);

      // create the system vector
      fk::vector<TestType> const fx = [&table, segment_size]() {
        fk::vector<TestType> fx(table.size() * segment_size);
        std::iota(fx.begin(), fx.end(), -1e6);
        return fx;
      }();

      auto const plan = get_plan(num_ranks, table);
      auto const grid = plan.at(my_rank);

      int const input_start  = grid.row_start * segment_size;
      int const input_end    = (grid.row_stop + 1) * segment_size - 1;
      int const output_start = grid.col_start * segment_size;
      int const output_end   = (grid.col_stop + 1) * segment_size - 1;
      fk::vector<TestType> const my_input = fx.extract(input_start, input_end);
      fk::vector<TestType> const gold = fx.extract(output_start, output_end);

      fk::vector<TestType> result(gold.size());

      exchange_results(my_input, result, segment_size, plan, my_rank);

      REQUIRE(result == gold);
    }

#else
    REQUIRE(true);
#endif
  }
}

TEMPLATE_TEST_CASE("gather results tests", "[distribution]", float, double)
{
  SECTION("single rank")
  {
    fk::vector<TestType> const source{1, 2, 3, 4, 5};
    distribution_plan plan;
    plan.emplace(0, element_subgrid(0, 1, 2, 3));
    int const segment_size = 0;
    int const my_rank      = 0;
    auto const result = gather_results(source, plan, my_rank, segment_size);
    REQUIRE(source == fk::vector<TestType>(result));
  }

  SECTION("multiple rank")
  {
#ifdef ASGARD_USE_MPI

    int const my_rank   = distrib_test_info.get_my_rank();
    int const num_ranks = distrib_test_info.get_num_ranks();

    if (my_rank < num_ranks)
    {
      int const degree   = 2;
      int const level    = 2;
      int const num_dims = 2;

      options const o = make_options(
          {"-l", std::to_string(level), "-d", std::to_string(degree)});

      element_table const table(o, num_dims);
      auto const plan         = get_plan(num_ranks, table);
      auto const segment_size = static_cast<int>(std::pow(degree, num_dims));

      // create the system vector
      fk::vector<TestType> const fx = [&table, segment_size]() {
        fk::vector<TestType> fx(table.size() * segment_size);
        std::iota(fx.begin(), fx.end(), -1e6);
        return fx;
      }();

      auto const grid = plan.at(my_rank);

      int const my_start = grid.col_start * segment_size;
      int const my_end   = (grid.col_stop + 1) * segment_size - 1;
      fk::vector<TestType> const my_input = fx.extract(my_start, my_end);

      std::vector<TestType> const results =
          gather_results(my_input, plan, my_rank, segment_size);

      if (my_rank == 0)
      {
        REQUIRE(fx == fk::vector<TestType>(results));
      }
    }
#else
    REQUIRE(true);
#endif
  }
}

TEMPLATE_TEST_CASE("gather errors tests", "[distribution]", float, double)
{
  int const my_rank   = distrib_test_info.get_my_rank();
  int const num_ranks = distrib_test_info.get_num_ranks();

  if (my_rank < num_ranks)
  {
    TestType const my_rmse      = static_cast<TestType>(my_rank);
    TestType const my_rel_error = static_cast<TestType>(my_rank + num_ranks);
    auto const [rmse_vect, relative_error_vect] =
        gather_errors(my_rmse, my_rel_error);
    if (my_rank == 0)
    {
      for (int i = 0; i < num_ranks; ++i)
      {
        REQUIRE(rmse_vect(i) == i);
        REQUIRE(relative_error_vect(i) == i + num_ranks);
      }
    }
  }
}
