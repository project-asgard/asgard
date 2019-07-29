
#include "chunk.hpp"
#include "tests_general.hpp"

// check for complete, non-overlapping element assignment
auto const validity_check = [](std::vector<element_chunk> const &chunks,
                               element_table const &table) {
  enum element_status
  {
    unassigned,
    assigned
  };

  fk::matrix<element_status> coverage(table.size(), table.size());

  // non-overlapping check
  for (element_chunk const &chunk : chunks)
  {
    for (const auto &[row, cols] : chunk)
    {
      for (int col = cols.start; col <= cols.stop; ++col)
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

// check that a given task vector occupies between 49% and 101% of the limiit
auto const size_check = [](std::vector<element_chunk> const &chunks,
                           PDE<double> const &pde, int const limit_MB,
                           bool const large_problem) {
  rank_workspace const work(pde, chunks);
  double lower_bound    = static_cast<double>(limit_MB * 0.49);
  double upper_bound    = static_cast<double>(limit_MB * 1.01);
  double workspace_size = work.size_MB();
  if (large_problem)
  {
    REQUIRE(workspace_size > lower_bound);
  }
  REQUIRE(workspace_size < upper_bound);
};

TEST_CASE("chunk convenience functions", "[chunk]")
{
  SECTION("elements in chunk - empty")
  {
    element_chunk g;
    assert(num_elements_in_chunk(g) == 0);
  }
  SECTION("elements in chunk - single row")
  {
    element_chunk g;
    g.insert({2, limits(0, 4)});
    assert(num_elements_in_chunk(g) == 5);
  }

  SECTION("elements in chunk - multiple rows")
  {
    element_chunk g;
    g.insert({3, limits(1, 2)});
    g.insert({4, limits(5, 10)});
    assert(num_elements_in_chunk(g) == 8);
  }

  SECTION("max connected in chunk - empty")
  {
    element_chunk g;
    assert(max_connected_in_chunk(g) == 0);
  }
  SECTION("max connected in chunk - single row")
  {
    element_chunk g;
    g.insert({2, limits(0, 4)});
    assert(max_connected_in_chunk(g) == 5);
  }
  SECTION("max connected in chunk - multiple rows")
  {
    element_chunk g;
    g.insert({3, limits(1, 2)});
    g.insert({4, limits(5, 10)});
    assert(max_connected_in_chunk(g) == 6);
  }

  SECTION("columns in chunk - single row")
  {
    element_chunk g;
    g.insert({2, limits(0, 4)});
    assert(columns_in_chunk(g) == limits(0, 4));
  }

  SECTION("columns in chunk - multiple rows")
  {
    element_chunk g;
    g.insert({3, limits(1, 2)});
    g.insert({4, limits(5, 10)});
    assert(columns_in_chunk(g) == limits(1, 10));
  }

  SECTION("rows in chunk - single row")
  {
    element_chunk g;
    g.insert({2, limits(0, 4)});
    assert(rows_in_chunk(g) == limits(2, 2));
  }
  SECTION("rows in chunk - multiple rows")
  {
    element_chunk g;
    g.insert({3, limits(1, 2)});
    g.insert({4, limits(5, 10)});
    assert(rows_in_chunk(g) == limits(3, 4));
  }
}

TEST_CASE("element chunk, continuity 2", "[chunk]")
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);

      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);

      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
    }
  }
}

TEST_CASE("element chunk, continuity 3", "[chunk]")
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);
      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);

      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);

      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);

      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);

      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);

      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
    }
  }
}

TEST_CASE("element chunk, continuity 6", "[chunk]")
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);
      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);
      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
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
      int const num_chunks = get_num_chunks(table, *pde, ranks, limit_MB);
      auto const chunks    = assign_elements(table, num_chunks);
      assert(chunks.size() % ranks == 0);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check(chunks, table);
      size_check(chunks, *pde, limit_MB, large_problem);
    }
  }
}

auto const test_copy_in = [](PDE<double> const &pde, element_chunk const &chunk,
                             rank_workspace<double> const &rank_space,
                             host_workspace<double> const &host_space) {
  int const elem_size  = element_segment_size(pde);
  auto const x_range   = columns_in_chunk(chunk);
  auto const num_elems = (x_range.stop - x_range.start + 1) * elem_size;

  fk::vector<double> const input_copy(rank_space.batch_input);
  for (int i = 0; i < num_elems; ++i)
  {
    REQUIRE(input_copy(i) == host_space.x(i + x_range.start * elem_size));
  }
};

auto const test_copy_out = [](PDE<double> const &pde,
                              element_chunk const &chunk,
                              rank_workspace<double> const &rank_space,
                              host_workspace<double> const &host_space,
                              fk::vector<double> const &fx_prior) {
  int const elem_size  = element_segment_size(pde);
  auto const y_range   = rows_in_chunk(chunk);
  auto const num_elems = (y_range.stop - y_range.start + 1) * elem_size;

  fk::vector<double> const output_copy(rank_space.batch_output);
  for (int i = 0; i < num_elems; ++i)
  {
    int const fx_index = i + y_range.start * elem_size;
    REQUIRE(std::abs(host_space.fx(fx_index) - fx_prior(fx_index) -
                     output_copy(i)) <
            std::numeric_limits<double>::epsilon() * num_elems);
  }
};

TEST_CASE("chunk data management functions", "[chunk]")
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
    auto const chunks  = assign_elements(
        elem_table, get_num_chunks(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, chunks);

    for (auto const &chunk : chunks)
    {
      // copy in inputs
      copy_chunk_inputs(*pde, rank_space, host_space, chunk);
      test_copy_in(*pde, chunk, rank_space, host_space);
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
    auto const chunks  = assign_elements(
        elem_table, get_num_chunks(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, chunks);

    for (auto const &chunk : chunks)
    {
      // copy in inputs
      copy_chunk_inputs(*pde, rank_space, host_space, chunk);
      test_copy_in(*pde, chunk, rank_space, host_space);
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
    auto const chunks  = assign_elements(
        elem_table, get_num_chunks(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, chunks);

    for (auto const &chunk : chunks)
    {
      // copy in inputs
      copy_chunk_inputs(*pde, rank_space, host_space, chunk);
      test_copy_in(*pde, chunk, rank_space, host_space);
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
    auto const chunks  = assign_elements(
        elem_table, get_num_chunks(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, chunks);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.batch_output.begin(),
                  rank_space.batch_output.end(), gen);
    std::generate(host_space.fx.begin(), host_space.fx.end(), gen);

    for (auto const &chunk : chunks)
    {
      fk::vector<double> fx_orig(host_space.fx);
      // copy out inputs
      copy_chunk_outputs(*pde, rank_space, host_space, chunk);
      test_copy_out(*pde, chunk, rank_space, host_space, fx_orig);
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
    auto const chunks  = assign_elements(
        elem_table, get_num_chunks(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, chunks);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.batch_output.begin(),
                  rank_space.batch_output.end(), gen);
    std::generate(host_space.fx.begin(), host_space.fx.end(), gen);

    for (auto const &chunk : chunks)
    {
      fk::vector<double> fx_orig(host_space.fx);
      // copy out inputs
      copy_chunk_outputs(*pde, rank_space, host_space, chunk);
      test_copy_out(*pde, chunk, rank_space, host_space, fx_orig);
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
    auto const chunks  = assign_elements(
        elem_table, get_num_chunks(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, chunks);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.batch_output.begin(),
                  rank_space.batch_output.end(), gen);
    std::generate(host_space.fx.begin(), host_space.fx.end(), gen);

    for (auto const &chunk : chunks)
    {
      fk::vector<double> fx_orig(host_space.fx);
      // copy out inputs
      copy_chunk_outputs(*pde, rank_space, host_space, chunk);
      test_copy_out(*pde, chunk, rank_space, host_space, fx_orig);
    }
  }
}

auto const test_reduction = [](PDE<double> const &pde,
                               element_chunk const &chunk,
                               rank_workspace<double> const &rank_space) {
  int const elem_size = element_segment_size(pde);
  auto const x_range  = columns_in_chunk(chunk);

  fk::vector<double> total_sum(rank_space.batch_output.size());
  for (auto const &[row, cols] : chunk)
  {
    int const prev_row_elems = [i = row, &chunk] {
      if (i == chunk.begin()->first)
      {
        return 0;
      }
      int prev_elems = 0;
      for (int r = chunk.begin()->first; r < i; ++r)
      {
        prev_elems += chunk.at(r).stop - chunk.at(r).start + 1;
      }
      return prev_elems;
    }();
    int const reduction_offset = prev_row_elems * pde.num_terms * elem_size;
    fk::matrix<double, mem_type::view, resource::device> const reduction_matrix(
        rank_space.reduction_space, elem_size,
        (cols.stop - cols.start + 1) * pde.num_terms, reduction_offset);

    fk::matrix<double> reduction_copy(reduction_matrix);
    fk::vector<double> sum(reduction_matrix.nrows());
    for (int i = 0; i < reduction_matrix.nrows(); ++i)
    {
      for (int j = 0; j < reduction_matrix.ncols(); ++j)
        sum(i) += reduction_copy(i, j);
    }
    int const row_this_task = row - chunk.begin()->first;
    fk::vector<double, mem_type::view> partial_sum(
        total_sum, row_this_task * elem_size,
        (row_this_task + 1) * elem_size - 1);

    partial_sum = partial_sum + sum;
  }

  fk::vector<double> output_copy(rank_space.batch_output);
  fk::vector<double> const diff = output_copy - total_sum;
  auto abs_compare              = [](double const a, double const b) {
    return (std::abs(a) < std::abs(b));
  };
  double const result =
      std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
  int const num_cols = (x_range.stop - x_range.start + 1) * pde.num_terms;
  // tol = epsilon * possible number of additions for an element * 10
  double const tol = std::numeric_limits<double>::epsilon() * num_cols * 10;
  REQUIRE(result <= tol);
};

TEST_CASE("chunk reduction function", "[chunk]")
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
    auto const chunks  = assign_elements(
        elem_table, get_num_chunks(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, chunks);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.reduction_space.begin(),
                  rank_space.reduction_space.end(), gen);

    for (auto const &chunk : chunks)
    {
      // reduce and test
      reduce_chunk(*pde, rank_space, chunk);
      test_reduction(*pde, chunk, rank_space);
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
    auto const chunks  = assign_elements(
        elem_table, get_num_chunks(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, chunks);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.reduction_space.begin(),
                  rank_space.reduction_space.end(), gen);

    for (auto const &chunk : chunks)
    {
      // reduce and test
      reduce_chunk(*pde, rank_space, chunk);
      test_reduction(*pde, chunk, rank_space);
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
    auto const chunks  = assign_elements(
        elem_table, get_num_chunks(elem_table, *pde, ranks, limit_MB));
    rank_workspace<double> rank_space(*pde, chunks);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(rank_space.reduction_space.begin(),
                  rank_space.reduction_space.end(), gen);

    for (auto const &chunk : chunks)
    {
      // reduce and test
      reduce_chunk(*pde, rank_space, chunk);
      test_reduction(*pde, chunk, rank_space);
    }
  }
}
