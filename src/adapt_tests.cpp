#include "adapt.hpp"
#include "program_options.hpp"
#include "tests_general.hpp"

static auto constexpr adapt_thresh = 1e-4;

template<typename P>
void test_adapt(parser const &problem, std::string const &gold_base)
{
  auto const fval_orig_path    = gold_base + "orig.dat";
  auto const fval_refine_path  = gold_base + "refine.dat";
  auto const fval_coarse_path  = gold_base + "coarse.dat";
  auto const table_refine_path = gold_base + "refine_table.dat";
  auto const table_coarse_path = gold_base + "coarse_table.dat";

  auto const fval_orig =
      fk::vector<P>(read_vector_from_txt_file(fval_orig_path));
  auto const gold_coarse =
      fk::vector<P>(read_vector_from_txt_file(fval_coarse_path));
  auto const gold_refine = [fval_refine_path]() {
    auto gold = fk::vector<P>(read_vector_from_txt_file(fval_refine_path));
    for (auto i = 0; i < gold.size(); ++i)
    {
      // matlab stores new refined coefficients as 1e-15 (0 deletes from sparse
      // vect)
      if (std::abs(gold(i)) < 1e-14)
      {
        gold(i) = 0.0;
      }
    }
    return gold;
  }();

  auto const test_tables = [](elements::table const &test,
                              fk::matrix<int> const &gold) {
    for (int64_t i = 0; i < test.size(); ++i)
    {
      auto const &test_coords = test.get_coords(i);
      fk::vector<int> const gold_coords =
          gold.extract_submatrix(i, 0, 1, gold.ncols());
      REQUIRE(test_coords == gold_coords);
    }
  };

  auto const gold_coarse_table =
      fk::matrix<int>(read_matrix_from_txt_file(table_coarse_path));
  auto const gold_refine_table =
      fk::matrix<int>(read_matrix_from_txt_file(table_refine_path));

  auto const pde = make_PDE<P>(problem);
  options const opts(problem);

  adapt::distributed_grid<P> refine_grid(*pde, opts);
  auto const test_refine = refine_grid.refine(fval_orig, opts);

  adapt::distributed_grid<P> coarse_grid(*pde, opts);
  auto const test_coarse = coarse_grid.coarsen(fval_orig, opts);

  test_tables(coarse_grid.get_table(), gold_coarse_table);
  test_tables(refine_grid.get_table(), gold_refine_table);
  REQUIRE(test_refine == gold_refine);
  REQUIRE(test_coarse == gold_coarse);
}

TEMPLATE_TEST_CASE("adapt - 1d, scattered coarsen/refine", "[adapt]", double,
                   float)
{
  auto const degree = 3;
  auto const level  = 4;

  std::string const gold_base = "../testing/generated-inputs/adapt/"
                                "continuity1_l4_d3_";

  auto const pde_choice      = PDE_opts::continuity_1;
  auto const num_dims        = 1;
  auto const cfl             = parser::DEFAULT_CFL;
  auto const use_full_grid   = parser::DEFAULT_USE_FG;
  auto const max_level       = parser::DEFAULT_MAX_LEVEL;
  auto const num_steps       = parser::DEFAULT_TIME_STEPS;
  auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
  auto const do_adapt_levels = parser::DEFAULT_DO_ADAPT;
  auto const adapt_threshold = adapt_thresh;
  parser const parse(pde_choice,
                     fk::vector<int>(std::vector<int>(num_dims, level)), degree,
                     cfl, use_full_grid, max_level, num_steps, use_implicit,
                     do_adapt_levels, adapt_threshold);

  test_adapt<TestType>(parse, gold_base);
}

TEMPLATE_TEST_CASE("adapt - 2d, all zero", "[adapt]", double, float)
{
  auto const degree = 2;
  auto const level  = 5;

  std::string const gold_base = "../testing/generated-inputs/adapt/"
                                "continuity2_l5_d2_";

  auto const pde_choice      = PDE_opts::continuity_2;
  auto const num_dims        = 2;
  auto const cfl             = parser::DEFAULT_CFL;
  auto const use_full_grid   = parser::DEFAULT_USE_FG;
  auto const max_level       = parser::DEFAULT_MAX_LEVEL;
  auto const num_steps       = parser::DEFAULT_TIME_STEPS;
  auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
  auto const do_adapt_levels = parser::DEFAULT_DO_ADAPT;
  auto const adapt_threshold = adapt_thresh;
  parser const parse(pde_choice,
                     fk::vector<int>(std::vector<int>(num_dims, level)), degree,
                     cfl, use_full_grid, max_level, num_steps, use_implicit,
                     do_adapt_levels, adapt_threshold);

  test_adapt<TestType>(parse, gold_base);
}

TEMPLATE_TEST_CASE("adapt - 3d, scattered, contiguous refine/adapt", "[adapt]",
                   double, float)
{
  auto const degree = 4;
  auto const level  = 4;

  std::string const gold_base = "../testing/generated-inputs/adapt/"
                                "continuity3_l4_d4_";

  auto const pde_choice      = PDE_opts::continuity_3;
  auto const num_dims        = 3;
  auto const cfl             = parser::DEFAULT_CFL;
  auto const use_full_grid   = parser::DEFAULT_USE_FG;
  auto const max_level       = parser::DEFAULT_MAX_LEVEL;
  auto const num_steps       = parser::DEFAULT_TIME_STEPS;
  auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
  auto const do_adapt_levels = parser::DEFAULT_DO_ADAPT;
  auto const adapt_threshold = adapt_thresh;
  parser const parse(pde_choice,
                     fk::vector<int>(std::vector<int>(num_dims, level)), degree,
                     cfl, use_full_grid, max_level, num_steps, use_implicit,
                     do_adapt_levels, adapt_threshold);

  test_adapt<TestType>(parse, gold_base);
}

template<typename P>
void test_initial(parser const &problem, std::string const &gold_filepath)
{
  auto const gold = [gold_filepath]() {
    auto gold = fk::vector<P>(read_vector_from_txt_file(gold_filepath));
    for (auto i = 0; i < gold.size(); ++i)
    {
      // matlab stores new refined coefficients as 1e-15 (0 deletes from sparse
      // vect)
      if (std::abs(gold(i)) < 1e-14)
      {
        gold(i) = 0.0;
      }
    }
    return gold;
  }();
  auto const pde = make_PDE<P>(problem);
  options const opts(problem);
  auto const quiet = true;
  basis::wavelet_transform<P, resource::host> const transformer(opts, *pde,
                                                                quiet);
  adapt::distributed_grid<P> adaptive_grid(*pde, opts);
  auto const test =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  P const tol_factor = std::is_same<P, double>::value ? 1e-15 : 1e-5;
  rmse_comparison(gold, test, tol_factor);
}

TEMPLATE_TEST_CASE("initial - diffusion 1d", "[adapt]", double, float)
{
  auto const degree = 4;
  auto const level  = 3;

  std::string const gold_path = "../testing/generated-inputs/adapt/"
                                "diffusion1_l3_d4_initial.dat";

  auto const pde_choice      = PDE_opts::diffusion_1;
  auto const num_dims        = 1;
  auto const cfl             = parser::DEFAULT_CFL;
  auto const use_full_grid   = parser::DEFAULT_USE_FG;
  auto const max_level       = parser::DEFAULT_MAX_LEVEL;
  auto const num_steps       = parser::DEFAULT_TIME_STEPS;
  auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
  auto const do_adapt_levels = true;
  auto const adapt_threshold = adapt_thresh;
  parser const parse(pde_choice,
                     fk::vector<int>(std::vector<int>(num_dims, level)), degree,
                     cfl, use_full_grid, max_level, num_steps, use_implicit,
                     do_adapt_levels, adapt_threshold);

  test_initial<TestType>(parse, gold_path);
}

TEMPLATE_TEST_CASE("initial - diffusion 2d", "[adapt]", double, float)
{
  auto const degree = 3;
  auto const level  = 2;

  std::string const gold_path = "../testing/generated-inputs/adapt/"
                                "diffusion2_l2_d3_initial.dat";

  auto const pde_choice      = PDE_opts::diffusion_2;
  auto const num_dims        = 2;
  auto const cfl             = parser::DEFAULT_CFL;
  auto const use_full_grid   = parser::DEFAULT_USE_FG;
  auto const max_level       = parser::DEFAULT_MAX_LEVEL;
  auto const num_steps       = parser::DEFAULT_TIME_STEPS;
  auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
  auto const do_adapt_levels = true;
  auto const adapt_threshold = adapt_thresh;
  parser const parse(pde_choice,
                     fk::vector<int>(std::vector<int>(num_dims, level)), degree,
                     cfl, use_full_grid, max_level, num_steps, use_implicit,
                     do_adapt_levels, adapt_threshold);

  test_initial<TestType>(parse, gold_path);
}
