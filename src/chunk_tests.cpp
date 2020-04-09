#include "batch.hpp"
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
    g.insert({2, grid_limits(0, 4)});
    assert(num_elements_in_chunk(g) == 5);
  }

  SECTION("elements in chunk - multiple rows")
  {
    element_chunk g;
    g.insert({3, grid_limits(1, 2)});
    g.insert({4, grid_limits(5, 10)});
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
    g.insert({2, grid_limits(0, 4)});
    assert(max_connected_in_chunk(g) == 5);
  }
  SECTION("max connected in chunk - multiple rows")
  {
    element_chunk g;
    g.insert({3, grid_limits(1, 2)});
    g.insert({4, grid_limits(5, 10)});
    assert(max_connected_in_chunk(g) == 6);
  }

  SECTION("columns in chunk - single row")
  {
    element_chunk g;
    g.insert({2, grid_limits(0, 4)});
    assert(columns_in_chunk(g) == grid_limits(0, 4));
  }

  SECTION("columns in chunk - multiple rows")
  {
    element_chunk g;
    g.insert({3, grid_limits(1, 2)});
    g.insert({4, grid_limits(5, 10)});
    assert(columns_in_chunk(g) == grid_limits(1, 10));
  }

  SECTION("rows in chunk - single row")
  {
    element_chunk g;
    g.insert({2, grid_limits(0, 4)});
    assert(rows_in_chunk(g) == grid_limits(2, 2));
  }
  SECTION("rows in chunk - multiple rows")
  {
    element_chunk g;
    g.insert({3, grid_limits(1, 2)});
    g.insert({4, grid_limits(5, 10)});
    assert(rows_in_chunk(g) == grid_limits(3, 4));
  }
}

// check for complete, non-overlapping element assignment
void validity_check_subgrid(std::vector<element_chunk> const &chunks,
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

  // completeness check
  for (auto const &elem : coverage)
  {
    REQUIRE(elem == element_status::assigned);
  }
}

// check that a given chunk vector occupies between 49% and 101% of the limiit
template<typename P>
void size_check_subgrid(std::vector<element_chunk> const &chunks,
                        element_subgrid const &subgrid, PDE<P> const &pde,
                        int const limit_MB)
{
  batch_workspace<P, resource::device> const work(pde, subgrid, chunks);
  P lower_bound                   = static_cast<P>(limit_MB * 0.49);
  P const upper_bound             = static_cast<P>(limit_MB * 1.01);
  P const workspace_size          = work.size_MB();
  auto const coefficients_size_MB = std::ceil(
      get_MB<P>(static_cast<uint64_t>(pde.get_coefficients(0, 0).size()) *
                pde.num_terms * pde.num_dims));
  bool const large_problem = chunks.size() > 2;
  if (large_problem)
  {
    REQUIRE(workspace_size + coefficients_size_MB > lower_bound);
  }
  REQUIRE(workspace_size + coefficients_size_MB < upper_bound);
}

template<typename P>
void test_chunking(PDE<P> const &pde, int const ranks)
{
  static auto constexpr workspace_min_MB  = 1000;
  static auto constexpr workspace_max_MB  = 4000;
  static auto constexpr workspace_step_MB = 1000;

  // FIXME assume uniform level and degree
  dimension<P> const &d = pde.get_dimensions()[0];
  int const level       = d.get_level();
  int const degree      = d.get_degree();

  options const o =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree)});
  element_table const table(o, pde.num_dims);
  auto const plan = get_plan(ranks, table);

  for (int limit_MB = workspace_min_MB; limit_MB <= workspace_max_MB;
       limit_MB += workspace_step_MB)
  {
    for (auto const &[rank, grid] : plan)
    {
      ignore(rank);
      int const num_chunks = get_num_chunks(grid, pde, limit_MB);
      auto const chunks    = assign_elements(grid, num_chunks);
      assert(static_cast<int>(chunks.size()) == num_chunks);
      validity_check_subgrid(chunks, grid);
      size_check_subgrid(chunks, grid, pde, limit_MB);
    }
  }
}

TEMPLATE_TEST_CASE("element chunk, continuity 2", "[chunk]", float, double)
{
  SECTION("1 rank, deg 5, level 6")
  {
    int const degree = 5;
    int const level  = 6;
    int const ranks  = 1;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    test_chunking(*pde, ranks);
  }
  SECTION("2 ranks, deg 5, level 6")
  {
    int const degree = 5;
    int const level  = 6;

    int const ranks = 2;
    auto const pde  = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    test_chunking(*pde, ranks);
  }
}

TEMPLATE_TEST_CASE("element chunk, continuity 3", "[chunk]", float, double)
{
  SECTION("1 rank, deg 4, level 6")
  {
    int const degree = 4;
    int const level  = 6;
    int const ranks  = 1;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    test_chunking(*pde, ranks);
  }
  SECTION("2 ranks, deg 4, level 6")
  {
    int const degree = 4;
    int const level  = 6;
    int const ranks  = 2;

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    test_chunking(*pde, ranks);
  }
  SECTION("3 ranks, deg 4, level 6")
  {
    int const degree = 4;
    int const level  = 6;
    int const ranks  = 3;

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    test_chunking(*pde, ranks);
  }
}

TEMPLATE_TEST_CASE("element chunk, continuity 6", "[chunk]", float, double)
{
  SECTION("1 rank, deg 3, level 4")
  {
    int const degree = 3;
    int const level  = 4;
    int const ranks  = 1;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    test_chunking(*pde, ranks);
  }
  SECTION("2 ranks, deg 3, level 4")
  {
    int const degree = 3;
    int const level  = 4;
    int const ranks  = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    test_chunking(*pde, ranks);
  }
  SECTION("11 ranks, deg 4, level 4")
  {
    int const degree = 4;
    int const level  = 4;
    int const ranks  = 11;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    test_chunking(*pde, ranks);
  }
}

template<typename P>
void verify_reduction(PDE<P> const &pde, element_chunk const &chunk,
                      batch_workspace<P, resource::device> const &batch_space)
{
  int const elem_size = element_segment_size(pde);
  auto const x_range  = columns_in_chunk(chunk);

  fk::vector<P> total_sum(batch_space.output.size());
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
    fk::matrix<P, mem_type::const_view, resource::device> const
        reduction_matrix(batch_space.reduction_space, elem_size,
                         (cols.stop - cols.start + 1) * pde.num_terms,
                         reduction_offset);

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

  fk::vector<P> const output_copy(batch_space.output.clone_onto_host());
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
    int const workspace_MB_limit = 4000;
    host_workspace<P> host_space(pde, grid, workspace_MB_limit);

    int const limit_MB = 1000;
    auto const chunks =
        assign_elements(grid, get_num_chunks(grid, pde, limit_MB));
    batch_workspace<P, resource::device> batch_space(pde, grid, chunks);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<P> dist(-3.0, 3.0);
    fk::vector<P> reduction_h(batch_space.reduction_space.clone_onto_host());
    auto const gen = [&dist, &mersenne_engine]() {
      return dist(mersenne_engine);
    };
    std::generate(reduction_h.begin(), reduction_h.end(), gen);
    batch_space.reduction_space.transfer_from(reduction_h);

    for (auto const &chunk : chunks)
    {
      // reduce and test
      reduce_chunk(pde, batch_space.reduction_space, batch_space.output,
                   batch_space.get_unit_vector(), grid, chunk);
      verify_reduction(pde, chunk, batch_space);
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
