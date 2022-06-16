#include "adapt.hpp"
#include "coefficients.hpp"
#include "program_options.hpp"
#include "tests_general.hpp"

#include <catch2/catch_session.hpp>

static auto constexpr adapt_thresh = 1e-4;

static auto const adapt_base_dir = gold_base_dir / "adapt";

struct distribution_test_init
{
  void set_my_rank(const int rank) { my_rank = rank; }
  void set_num_ranks(const int size) { num_ranks = size; }
  int get_my_rank() const { return my_rank; }
  int get_num_ranks() const { return num_ranks; }

private:
  int my_rank;
  int num_ranks;
};
static distribution_test_init distrib_test_info;

int main(int argc, char *argv[])
{
  auto const [rank, total_ranks] = initialize_distribution();
  distrib_test_info.set_my_rank(rank);
  distrib_test_info.set_num_ranks(total_ranks);

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

template<typename P>
void test_adapt(parser const &problem, std::filesystem::path gold_base)
{
  auto const prefix = gold_base.filename().string();
  gold_base.remove_filename();
  auto const fval_orig_path    = gold_base / (prefix + "orig.dat");
  auto const fval_refine_path  = gold_base / (prefix + "refine.dat");
  auto const fval_coarse_path  = gold_base / (prefix + "coarse.dat");
  auto const table_refine_path = gold_base / (prefix + "refine_table.dat");
  auto const table_coarse_path = gold_base / (prefix + "coarse_table.dat");

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
    REQUIRE(test.size() == gold.nrows());
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
  auto const my_subgrid   = refine_grid.get_subgrid(get_rank());
  auto const segment_size = element_segment_size(*pde);
  auto const my_fval_orig =
      fval_orig.extract(my_subgrid.col_start * segment_size,
                        (my_subgrid.col_stop + 1) * segment_size - 1);

  auto const test_refine = refine_grid.refine(my_fval_orig, opts);
  adapt::distributed_grid<P> coarse_grid(*pde, opts);
  auto const test_coarse = coarse_grid.coarsen(my_fval_orig, opts);
  test_tables(coarse_grid.get_table(), gold_coarse_table);
  test_tables(refine_grid.get_table(), gold_refine_table);

  auto const refine_subgrid = refine_grid.get_subgrid(get_rank());
  fk::vector<P, mem_type::const_view> const my_gold_refine(
      gold_refine, refine_subgrid.col_start * segment_size,
      (refine_subgrid.col_stop + 1) * segment_size - 1);
  REQUIRE(test_refine == my_gold_refine);

  auto const coarsen_subgrid = coarse_grid.get_subgrid(get_rank());
  fk::vector<P, mem_type::const_view> const my_gold_coarse(
      gold_coarse, coarsen_subgrid.col_start * segment_size,
      (coarsen_subgrid.col_stop + 1) * segment_size - 1);
  REQUIRE(test_coarse == my_gold_coarse);
}

TEMPLATE_TEST_CASE("adapt - 1d, scattered coarsen/refine", "[adapt]", double,
                   float)
{
  if (!is_active())
  {
    return;
  }

  auto const degree          = 3;
  auto const level           = 4;
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

  test_adapt<TestType>(parse, adapt_base_dir / "continuity1_l4_d3_");
}

TEMPLATE_TEST_CASE("adapt - 2d, all zero", "[adapt]", double, float)
{
  if (!is_active())
  {
    return;
  }
  auto const degree          = 2;
  auto const level           = 5;
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

  // temporarily disable test for MPI due to table elements < num ranks
  if (get_num_ranks() == 1)
  {
    test_adapt<double>(parse, adapt_base_dir / "continuity2_l5_d2_");
  }
}

TEMPLATE_TEST_CASE("adapt - 3d, scattered, contiguous refine/adapt", "[adapt]",
                   double, float)
{
  if (!is_active())
  {
    return;
  }
  auto const degree          = 4;
  auto const level           = 4;
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

  test_adapt<TestType>(parse, adapt_base_dir / "continuity3_l4_d4_");
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
  generate_dimension_mass_mat<P>(*pde, transformer);

  auto const test =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  REQUIRE(gold.size() >= test.size());

  auto constexpr tol_factor = get_tolerance<P>(100);
  auto const my_subgrid     = adaptive_grid.get_subgrid(get_rank());
  auto const segment_size   = element_segment_size(*pde);
  fk::vector<P, mem_type::const_view> const my_gold(
      gold, my_subgrid.col_start * segment_size,
      (my_subgrid.col_stop + 1) * segment_size - 1);
  rmse_comparison(my_gold, test, tol_factor);
}

TEMPLATE_TEST_CASE("initial - diffusion 1d", "[adapt]", double, float)
{
  auto const degree          = 4;
  auto const level           = 3;
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

  // don't test this in the MPI case -- too small to split table
  if (get_num_ranks() == 1)
  {
    test_initial<TestType>(parse,
                           adapt_base_dir / "diffusion1_l3_d4_initial.dat");
  }
}

TEMPLATE_TEST_CASE("initial - diffusion 2d", "[adapt]", double, float)
{
  if (!is_active())
  {
    return;
  }
  auto const degree          = 3;
  auto const level           = 2;
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

  test_initial<TestType>(parse,
                         adapt_base_dir / "diffusion2_l2_d3_initial.dat");
}
