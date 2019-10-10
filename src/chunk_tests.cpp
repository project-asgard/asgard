
#include "chunk.hpp"
#include "distribution.hpp"
#include "tests_general.hpp"

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

//
// test helpers
//

// check for complete, non-overlapping element assignment (subgrid)
void validity_check_sub(std::vector<element_chunk> const &chunks,
                        element_subgrid const &grid)
{
  enum element_status
  {
    unassigned,
    assigned
  };

  fk::matrix<element_status> coverage(grid.nrows(), grid.ncols());

  // non-overlapping check
  for (element_chunk const &chunk : chunks)
  {
    for (auto const &[row, cols] : chunk)
    {
      for (int col = cols.start; col <= cols.stop; ++col)
      {
        int const row_l = grid.to_local_row(row);
        int const col_l = grid.to_local_col(col);
        REQUIRE(coverage(row_l, col_l) == element_status::unassigned);
        coverage(row_l, col_l) = element_status::assigned;
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
}

// check that a given chunk vector occupies between 49% and 101% of the limiit
template<typename P>
void size_check(std::vector<element_chunk> const &chunks, PDE<P> const &pde,
                int const limit_MB, bool const large_problem)
{
  rank_workspace const work(pde, chunks);
  P const lower_bound    = static_cast<P>(limit_MB * 0.49);
  P const upper_bound    = static_cast<P>(limit_MB * 1.01);
  P const workspace_size = work.size_MB();
  if (large_problem)
  {
    REQUIRE(workspace_size > lower_bound);
  }
  REQUIRE(workspace_size < upper_bound);
}

//
// main test func
//

template<typename P>
void test_chunking(PDE<P> const &pde, int const degree, int const level,
                   int const num_ranks = 1, bool const large_problem = true,
                   int const start_MB = 1000, int const stop_MB = 10000,
                   int const step_MB = 1000)
{
  // adjust to have same relative amount of space for float/double
  P const mult = std::is_same<P, float>::value ? 0.5 : 1.0;
  options const o =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree)});
  element_table const table(o, pde.num_dims);
  auto const plan = get_plan(num_ranks, table);

  for (int limit_MB = start_MB * mult; limit_MB <= stop_MB * mult;
       limit_MB *= step_MB * mult)
  {
    for (auto const &[rank, grid] : plan)
    {
      ignore(rank);
      int const num_chunks = get_num_chunks(grid, pde, limit_MB);
      auto const chunks    = assign_elements(grid, num_chunks);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check_sub(chunks, grid);
      size_check(chunks, pde, limit_MB, large_problem);
    }
  }
}

TEMPLATE_TEST_CASE("element chunk, continuity 2", "[chunk]", float, double)
{
  SECTION("1 rank, deg 5, level 6")
  {
    int const degree = 5;
    int const level  = 6;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    int const num_ranks      = 1;
    bool const large_problem = false;
    test_chunking(*pde, degree, level, num_ranks, large_problem);
  }

  SECTION("2 ranks, deg 5, level 6")
  {
    int const degree = 5;
    int const level  = 6;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    int const num_ranks      = 2;
    bool const large_problem = false;
    test_chunking(*pde, degree, level, num_ranks, large_problem);
  }
}
TEMPLATE_TEST_CASE("element chunk, continuity 3", "[chunk]", float, double)
{
  SECTION("1 rank, deg 5, level 6")
  {
    int const degree = 5;
    int const level  = 6;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    test_chunking(*pde, degree, level);
  }

  SECTION("2 ranks, deg 10, level 5")
  {
    int const degree = 10;
    int const level  = 5;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    int const num_ranks = 2;
    test_chunking(*pde, degree, level, num_ranks);
  }

  SECTION("3 ranks, deg 4, level 6")
  {
    int const degree = 4;
    int const level  = 6;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    int const num_ranks = 3;

    test_chunking(*pde, degree, level, num_ranks);
  }

  SECTION("10 ranks, deg 5, level 7")
  {
    int const degree = 5;
    int const level  = 7;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    int const num_ranks = 10;

    test_chunking(*pde, degree, level, num_ranks);
  }
}

TEMPLATE_TEST_CASE("element chunk, continuity 6", "[chunk]", float, double)
{
  SECTION("1 rank, deg 2, level 3")
  {
    int const degree = 2;
    int const level  = 3;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    int const num_ranks      = 1;
    bool const large_problem = false;
    test_chunking(*pde, degree, level, num_ranks, large_problem);
  }

  SECTION("11 ranks, deg 4, level 4")
  {
    int const degree = 4;
    int const level  = 4;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    int const num_ranks = 11;
    test_chunking(*pde, degree, level, num_ranks);
  }
}

template<typename P>
void validate_copy_in(PDE<P> const &pde, element_subgrid const &grid,
                      element_chunk const &chunk,
                      rank_workspace<P> const &rank_space,
                      host_workspace<P> const &host_space)
{
  int const elem_size  = element_segment_size(pde);
  auto const x_range   = columns_in_chunk(chunk);
  auto const num_elems = (x_range.stop - x_range.start + 1) * elem_size;

  fk::vector<P> const input_copy(rank_space.batch_input.clone_onto_host());

  for (int i = 0; i < num_elems; ++i)
  {
    REQUIRE(input_copy(i) ==
            host_space.x(i + grid.to_local_col(x_range.start) * elem_size));
  }
}

template<typename P>
void copy_in_test(int const degree, int const level, PDE<P> const &pde)
{
  options const o =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree)});

  element_table const elem_table(o, pde.num_dims);

  int const num_ranks = 1;
  auto const plan     = get_plan(num_ranks, elem_table);

  for (auto const &[rank, subgrid] : plan)
  {
    ignore(rank);
    host_workspace<P> host_space(pde, subgrid);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<P> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(host_space.x.begin(), host_space.x.end(), gen);

    int const limit_MB = 1000;
    auto const chunks =
        assign_elements(subgrid, get_num_chunks(subgrid, pde, limit_MB));
    rank_workspace<P> rank_space(pde, chunks);

    for (auto const &chunk : chunks)
    {
      // copy in inputs
      copy_chunk_inputs(pde, subgrid, rank_space, host_space, chunk);
      validate_copy_in(pde, subgrid, chunk, rank_space, host_space);
    }
  }
}

template<typename P>
void validate_copy_out(PDE<P> const &pde, element_subgrid const &grid,
                       element_chunk const &chunk,
                       rank_workspace<P> const &rank_space,
                       host_workspace<P> const &host_space,
                       fk::vector<P> const &fx_prior)
{
  int const elem_size  = element_segment_size(pde);
  auto const y_range   = rows_in_chunk(chunk);
  auto const num_elems = (y_range.stop - y_range.start + 1) * elem_size;

  fk::vector<P> const output_copy(rank_space.batch_output.clone_onto_host());
  for (int i = 0; i < num_elems; ++i)
  {
    int const fx_index = i + grid.to_local_row(y_range.start) * elem_size;
    REQUIRE(std::abs(host_space.fx(fx_index) - fx_prior(fx_index) -
                     output_copy(i)) <
            std::numeric_limits<P>::epsilon() * num_elems);
  }
}

template<typename P>
void copy_out_test(int const level, int const degree, PDE<P> const &pde)
{
  options const o =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree)});

  element_table const elem_table(o, pde.num_dims);
  int const num_ranks = 1;
  auto const plan     = get_plan(num_ranks, elem_table);

  for (auto const &[rank, grid] : plan)
  {
    ignore(rank);
    host_workspace<P> host_space(pde, grid);
    int const limit_MB = 1000;
    auto const chunks =
        assign_elements(grid, get_num_chunks(grid, pde, limit_MB));
    rank_workspace<P> rank_space(pde, chunks);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());

    std::uniform_real_distribution<P> dist(-2.0, 2.0);
    fk::vector<P> batch_out_h(rank_space.batch_output.clone_onto_host());

    auto const gen = [&dist, &mersenne_engine]() {
      return dist(mersenne_engine);
    };

    std::generate(batch_out_h.begin(), batch_out_h.end(), gen);
    rank_space.batch_output.transfer_from(batch_out_h);

    std::generate(host_space.fx.begin(), host_space.fx.end(), gen);

    for (auto const &chunk : chunks)
    {
      fk::vector<P> fx_orig(host_space.fx);
      // copy out inputs
      copy_chunk_outputs(pde, grid, rank_space, host_space, chunk);
      validate_copy_out(pde, grid, chunk, rank_space, host_space, fx_orig);
    }
  }
}

TEMPLATE_TEST_CASE("chunk data management functions", "[chunk]", float, double)
{
  SECTION("copy in deg 2/lev 4, continuity 1")
  {
    int const degree = 2;
    int const level  = 4;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    copy_in_test(degree, level, *pde);
  }

  SECTION("copy in deg 4/lev 5, continuity 3")
  {
    int const degree = 4;
    int const level  = 5;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    copy_in_test(degree, level, *pde);
  }

  SECTION("copy in deg 3/lev 2, continuity 6")
  {
    int const degree = 3;
    int const level  = 2;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    copy_in_test(degree, level, *pde);
  }

  SECTION("copy out deg 2/lev 4, continuity 1")
  {
    int const degree = 2;
    int const level  = 4;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    copy_out_test(degree, level, *pde);
  }

  SECTION("copy out deg 4/lev 5, continuity 3")
  {
    int const degree = 4;
    int const level  = 5;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    copy_out_test(degree, level, *pde);
  }

  SECTION("copy out deg 3/lev 2, continuity 6")
  {
    int const degree = 3;
    int const level  = 2;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    copy_out_test(degree, level, *pde);
  }
}

template<typename P>
void verify_reduction(PDE<P> const &pde, element_chunk const &chunk,
                      rank_workspace<P> const &rank_space)
{
  int const elem_size = element_segment_size(pde);
  auto const x_range  = columns_in_chunk(chunk);

  fk::vector<P> total_sum(rank_space.batch_output.size());
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
    fk::matrix<P, mem_type::view, resource::device> const reduction_matrix(
        rank_space.reduction_space, elem_size,
        (cols.stop - cols.start + 1) * pde.num_terms, reduction_offset);

    fk::matrix<P> reduction_copy(reduction_matrix.clone_onto_host());
    fk::vector<P> sum(reduction_matrix.nrows());

    for (int i = 0; i < reduction_matrix.nrows(); ++i)
    {
      for (int j = 0; j < reduction_matrix.ncols(); ++j)
        sum(i) += reduction_copy(i, j);
    }
    int const row_this_task = row - chunk.begin()->first;
    fk::vector<P, mem_type::view> partial_sum(
        total_sum, row_this_task * elem_size,
        (row_this_task + 1) * elem_size - 1);

    partial_sum = partial_sum + sum;
  }

  fk::vector<P> const output_copy(rank_space.batch_output.clone_onto_host());
  fk::vector<P> const diff = output_copy - total_sum;
  auto const abs_compare   = [](auto const a, auto const b) {
    return (std::abs(a) < std::abs(b));
  };
  auto const result =
      std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
  int const num_cols = (x_range.stop - x_range.start + 1) * pde.num_terms;
  // tol = epsilon * possible number of additions for an element * 10
  auto const tol = std::numeric_limits<P>::epsilon() * num_cols * 10;
  REQUIRE(result <= tol);
}

template<typename P>
void reduction_test(int const degree, int const level, PDE<P> const &pde)
{
  options const o =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree)});

  element_table const elem_table(o, pde.num_dims);

  int const num_ranks = 1;
  auto const plan     = get_plan(num_ranks, elem_table);

  for (auto const &[rank, grid] : plan)
  {
    ignore(rank);
    host_workspace<P> host_space(pde, grid);

    int const limit_MB = 1000;
    auto const chunks =
        assign_elements(grid, get_num_chunks(grid, pde, limit_MB));
    rank_workspace<P> rank_space(pde, chunks);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<P> dist(-3.0, 3.0);
    fk::vector<P> reduction_h(rank_space.reduction_space.clone_onto_host());
    auto const gen = [&dist, &mersenne_engine]() {
      return dist(mersenne_engine);
    };
    std::generate(reduction_h.begin(), reduction_h.end(), gen);
    rank_space.reduction_space.transfer_from(reduction_h);

    for (auto const &chunk : chunks)
    {
      // reduce and test
      reduce_chunk(pde, rank_space, chunk);
      verify_reduction(pde, chunk, rank_space);
    }
  }
}

TEMPLATE_TEST_CASE("chunk reduction function", "[chunk]", float, double)
{
  SECTION("reduction deg 2/lev 4, continuity 1")
  {
    int const degree = 2;
    int const level  = 4;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    reduction_test(degree, level, *pde);
  }

  SECTION("reduction deg 4/lev 5, continuity 3")
  {
    int const degree = 4;
    int const level  = 5;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    reduction_test(degree, level, *pde);
  }
  SECTION("reduction deg 3/lev 2, continuity 6")
  {
    int const degree = 3;
    int const level  = 2;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    reduction_test(degree, level, *pde);
  }
}
