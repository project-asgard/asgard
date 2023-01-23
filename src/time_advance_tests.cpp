#include "build_info.hpp"
#include "coefficients.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>
#include <random>
#include <sstream>

using namespace asgard;

int main(int argc, char *argv[])
{
  initialize_distribution();

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

// settings for time advance testing
static auto constexpr num_steps = 5;

static auto const time_advance_base_dir = gold_base_dir / "time_advance";

template<typename P>
void time_advance_test(parser const &parse,
                       std::filesystem::path const &filepath,
                       P const tolerance_factor)
{
  auto const num_ranks = get_num_ranks();
  if (num_ranks > 1 && parse.using_implicit() &&
      parse.get_selected_solver() != solve_opts::scalapack)
  {
    // distributed implicit stepping not implemented
    return;
  }

  if (num_ranks == 1 && parse.get_selected_solver() == solve_opts::scalapack)
  {
    // don't bother using scalapack with 1 rank
    return;
  }

  auto pde = make_PDE<P>(parse);
  options const opts(parse);
  elements::table const check(opts, *pde);
  if (check.size() <= num_ranks)
  {
    // don't run tiny problems when MPI testing
    return;
  }
  adapt::distributed_grid adaptive_grid(*pde, opts);
  basis::wavelet_transform<P, resource::host> const transformer(opts, *pde);

  // -- compute dimension mass matrices
  generate_dimension_mass_mat(*pde, transformer);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  // TODO: look into issue requiring mass mats to be regenerated after init
  // cond. see problem in main.cpp
  generate_dimension_mass_mat(*pde, transformer);

  fk::vector<P> f_val(initial_condition);

  // -- time loop
  for (auto i = 0; i < opts.num_time_steps; ++i)
  {
    std::cout.setstate(std::ios_base::failbit);
    auto const time          = i * pde->get_dt();
    auto const update_system = i == 0;
    auto const method = opts.use_implicit_stepping ? time_advance::method::imp
                                                   : time_advance::method::exp;
    auto const sol =
        time_advance::adaptive_advance(method, *pde, adaptive_grid, transformer,
                                       opts, f_val, time, update_system);

    f_val.resize(sol.size()) = sol;
    std::cout.clear();

    auto const file_path =
        filepath.parent_path() /
        (filepath.filename().string() + std::to_string(i) + ".dat");
    auto const gold = fk::vector<P>(read_vector_from_txt_file(file_path));

    // each rank generates partial answer
    auto const dof =
        static_cast<int>(std::pow(parse.get_degree(), pde->num_dims));
    auto const subgrid = adaptive_grid.get_subgrid(get_rank());
    REQUIRE((subgrid.col_stop + 1) * dof - 1 <= gold.size());
    auto const my_gold = fk::vector<P, mem_type::const_view>(
        gold, subgrid.col_start * dof, (subgrid.col_stop + 1) * dof - 1);
    rmse_comparison(my_gold, f_val, tolerance_factor);
  }
}

static std::string get_level_string(fk::vector<int> const &levels)
{
  return std::accumulate(levels.begin(), levels.end(), std::string(),
                         [](std::string const &accum, int const lev) {
                           return accum + std::to_string(lev) + "_";
                         });
}

TEMPLATE_TEST_CASE("time advance - diffusion 2", "[time_advance]", double,
                   float)
{
  if (!is_active())
  {
    return;
  }

  TestType const cfl           = 0.01;
  std::string const pde_choice = "diffusion_2";
  int const num_dims           = 2;

  SECTION("diffusion2, explicit, sparse grid, level 2, degree 2")
  {
    int const degree = 2;
    int const level  = 2;

    auto constexpr tol_factor = get_tolerance<TestType>(100);

    auto const gold_base = time_advance_base_dir / "diffusion2_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit, sparse grid, level 3, degree 3")
  {
    int const degree          = 3;
    int const level           = 3;
    auto constexpr tol_factor = get_tolerance<TestType>(100);

    auto const gold_base = time_advance_base_dir / "diffusion2_sg_l3_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit, sparse grid, level 4, degree 4")
  {
    int const degree          = 4;
    int const level           = 4;
    auto constexpr tol_factor = get_tolerance<TestType>(1000000);
    auto const gold_base      = time_advance_base_dir / "diffusion2_sg_l4_d4_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit/non-uniform level, sparse grid, degree 2")
  {
    int const degree          = 2;
    auto constexpr tol_factor = get_tolerance<TestType>(100);

    fk::vector<int> const levels{4, 5};
    auto const gold_base =
        time_advance_base_dir /
        ("diffusion2_sg_l" + get_level_string(levels) + "d2_t");

    auto const full_grid = false;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEST_CASE("adaptive time advance")
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  auto const cfl = 0.01;
  SECTION("diffusion 2 implicit")
  {
    auto const tol_factor        = 1e-11;
    std::string const pde_choice = "diffusion_2";
    auto const degree            = 4;
    fk::vector<int> const levels{3, 3};
    auto const gold_base =
        time_advance_base_dir / "diffusion2_ad_implicit_sg_l3_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = true;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 0.5e-1;

    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, use_implicit,
                       do_adapt_levels, adapt_threshold);

    // temporarily disable test for MPI due to table elements < num ranks
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse, gold_base, tol_factor);
    }
#ifdef ASGARD_USE_SCALAPACK
    auto const solver_str = std::string_view("scalapack");

    parser const parse_scalapack(
        pde_choice, levels, degree, cfl, full_grid, parser::DEFAULT_MAX_LEVEL,
        num_steps, use_implicit, do_adapt_levels, adapt_threshold, solver_str);

    // temporarily disable test for MPI due to table elements < num ranks
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse_scalapack, gold_base, tol_factor);
    }
#endif
  }
  SECTION("diffusion 2 explicit")
  {
    auto const tol_factor        = 1e-11;
    std::string const pde_choice = "diffusion_2";
    auto const degree            = 4;
    fk::vector<int> const levels{3, 3};
    auto const gold_base = time_advance_base_dir / "diffusion2_ad_sg_l3_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 0.5e-1;

    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, use_implicit,
                       do_adapt_levels, adapt_threshold);
    // temporarily disable test for MPI due to table elements < num ranks
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse, gold_base, tol_factor);
    }
  }

  SECTION("fokkerplanck1_pitch_E case1 explicit")
  {
    auto constexpr tol_factor    = get_tolerance<double>(100);
    std::string const pde_choice = "fokkerplanck_1d_pitch_E_case1";
    auto const degree            = 4;
    fk::vector<int> const levels{4};
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_4p1a_ad_sg_l4_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 1e-4;

    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, use_implicit,
                       do_adapt_levels, adapt_threshold);

    // we do not gracefully handle coarsening below number of active ranks yet
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse, gold_base, tol_factor);
    }
  }

  SECTION("fokkerplanck1_pitch_E case2 explicit")
  {
    auto const tol_factor        = 1e-15;
    std::string const pde_choice = "fokkerplanck_1d_pitch_E_case2";
    auto const degree            = 4;
    fk::vector<int> const levels{4};
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_pitch_E_case2_ad_sg_l4_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 1e-4;

    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, use_implicit,
                       do_adapt_levels, adapt_threshold);

    // we do not gracefully handle coarsening below number of active ranks yet
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse, gold_base, tol_factor);
    }
  }

  SECTION("continuity 2 explicit")
  {
    auto const tol_factor        = 1e-13;
    std::string const pde_choice = "continuity_2";
    auto const degree            = 4;
    fk::vector<int> const levels{3, 3};
    auto const gold_base = time_advance_base_dir / "continuity2_ad_sg_l3_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 1e-3;

    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, use_implicit,
                       do_adapt_levels, adapt_threshold);

    time_advance_test(parse, gold_base, tol_factor);
  }
}
TEMPLATE_TEST_CASE("time advance - diffusion 1", "[time_advance]", double,
                   float)
{
  if (!is_active())
  {
    return;
  }

  TestType const cfl     = 0.01;
  std::string pde_choice = "diffusion_1";
  int const num_dims     = 1;

  SECTION("diffusion1, explicit, sparse grid, level 3, degree 3")
  {
    int const degree          = 3;
    int const level           = 3;
    auto const gold_base      = time_advance_base_dir / "diffusion1_sg_l3_d3_t";
    auto constexpr tol_factor = get_tolerance<TestType>(100);

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion1, explicit, sparse grid, level 4, degree 4")
  {
    int const degree      = 4;
    int const level       = 4;
    auto const gold_base  = time_advance_base_dir / "diffusion1_sg_l4_d4_t";
    auto const tol_factor = get_tolerance<TestType>(100000);

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 1", "[time_advance]", float,
                   double)
{
  if (!is_active())
  {
    return;
  }

  std::string const pde_choice = "continuity_1";

  TestType const cfl = 0.01;

  auto constexpr tol_factor = get_tolerance<TestType>(10);

  auto const num_dims = 1;

  SECTION("continuity1, explicit, level 2, degree 2, sparse grid")
  {
    int const degree     = 2;
    int const level      = 2;
    auto const gold_base = time_advance_base_dir / "continuity1_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity1, explicit, level 2, degree 2, full grid")
  {
    int const degree     = 2;
    int const level      = 2;
    auto const gold_base = time_advance_base_dir / "continuity1_fg_l2_d2_t";

    auto const full_grid = true;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity1, explicit, level 4, degree 3, sparse grid")
  {
    int const degree     = 3;
    int const level      = 4;
    auto const gold_base = time_advance_base_dir / "continuity1_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 2", "[time_advance]", float,
                   double)
{
  if (!is_active())
  {
    return;
  }

  std::string const pde_choice = "continuity_2";
  TestType const cfl           = 0.01;
  auto constexpr tol_factor    = get_tolerance<TestType>(10);
  auto const num_dims          = 2;

  SECTION("continuity2, explicit, level 2, degree 2, sparse grid")
  {
    int const degree     = 2;
    int const level      = 2;
    auto const gold_base = time_advance_base_dir / "continuity2_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit, level 2, degree 2, full grid")
  {
    int const degree     = 2;
    int const level      = 2;
    auto const gold_base = time_advance_base_dir / "continuity2_fg_l2_d2_t";

    auto const full_grid = true;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit, level 4, degree 3, sparse grid")
  {
    int const degree     = 3;
    int const level      = 4;
    auto const gold_base = time_advance_base_dir / "continuity2_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit/non-uniform level, full grid, degree 3")
  {
    int const degree = 3;

    fk::vector<int> const levels{3, 4};
    auto const gold_base =
        time_advance_base_dir /
        ("continuity2_fg_l" + get_level_string(levels) + "d3_t");
    auto const full_grid = true;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 3", "[time_advance]", float,
                   double)
{
  if (!is_active())
  {
    return;
  }

  std::string const pde_choice = "continuity_3";
  TestType const cfl           = 0.01;
  auto constexpr tol_factor    = get_tolerance<TestType>(10);
  auto const num_dims          = 3;

  SECTION("continuity3, explicit, level 2, degree 2, sparse grid")
  {
    int const degree     = 2;
    int const level      = 2;
    auto const gold_base = time_advance_base_dir / "continuity3_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity3, explicit, level 4, degree 3, sparse grid")
  {
    int const degree     = 3;
    int const level      = 4;
    auto const gold_base = time_advance_base_dir / "continuity3_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity3, explicit/non-uniform level, degree 4, sparse grid")
  {
    int const degree = 4;

    fk::vector<int> const levels{3, 4, 2};
    auto const gold_base =
        time_advance_base_dir /
        ("continuity3_sg_l" + get_level_string(levels) + "d4_t");
    auto const full_grid = false;

    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, get_tolerance<TestType>(10));
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 6", "[time_advance]", float,
                   double)
{
  if (!is_active())
  {
    return;
  }

  std::string const pde_choice = "continuity_6";
  TestType const cfl           = 0.01;
  auto constexpr tol_factor    = get_tolerance<TestType>(10);
  auto const num_dims          = 6;

  SECTION("continuity6, level 2, degree 3, sparse grid")
  {
    int const degree     = 3;
    int const level      = 2;
    auto const gold_base = time_advance_base_dir / "continuity6_sg_l2_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity6, explicit/non-uniform level, degree 4, sparse grid")
  {
    int const degree = 2;

    fk::vector<int> const levels{2, 3, 2, 3, 3, 2};
    auto const gold_base =
        time_advance_base_dir /
        ("continuity6_sg_l" + get_level_string(levels) + "d2_t");
    auto const full_grid = false;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_pitch_C", "[time_advance]",
                   float, double)
{
  if (!is_active())
  {
    return;
  }

  std::string const pde_choice = "fokkerplanck_1d_pitch_C";
  TestType const cfl           = 0.01;
  auto constexpr tol_factor    = get_tolerance<TestType>(200);
  auto const num_dims          = 1;

  SECTION("fokkerplanck_1d_pitch_C, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_4p2_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p3", "[time_advance]",
                   float, double)
{
  if (!is_active())
  {
    return;
  }

  std::string const pde_choice = "fokkerplanck_1d_4p3";
  TestType const cfl           = 0.01;
  auto const num_dims          = 1;

  SECTION("fokkerplanck_1d_4p3, level 2, degree 2, sparse grid")
  {
    int const degree          = 2;
    int const level           = 2;
    auto constexpr tol_factor = get_tolerance<TestType>(10);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_4p3_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_pitch_E_case1",
                   "[time_advance]", float, double)
{
  if (!is_active())
  {
    return;
  }

  std::string const pde_choice = "fokkerplanck_1d_pitch_E_case1";
  TestType const cfl           = 0.01;
  auto constexpr tol_factor    = get_tolerance<TestType>(100);
  auto const num_dims          = 1;

  SECTION("fokkerplanck_1d_pitch_E_case1, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_4p1a_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_pitch_E_case2",
                   "[time_advance]", float, double)
{
  if (!is_active())
  {
    return;
  }

  std::string const pde_choice = "fokkerplanck_1d_pitch_E_case2";
  TestType const cfl           = 0.01;
  auto constexpr tol_factor    = get_tolerance<TestType>(10);
  auto const num_dims          = 1;

  SECTION("fokkerplanck_1d_pitch_E_case2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_pitch_E_case2_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

// explicit time advance is not a fruitful approach to this problem
TEMPLATE_TEST_CASE("implicit time advance - fokkerplanck_2d_complete_case4",
                   "[time_advance]", float, double)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  TestType const cfl     = 0.01;
  std::string pde_choice = "fokkerplanck_2d_complete_case4";
  auto const num_dims    = 2;
  auto const implicit    = true;
#ifdef ASGARD_USE_SCALAPACK
  auto const do_adapt_levels = parser::DEFAULT_DO_ADAPT;
  auto const adapt_threshold = parser::DEFAULT_ADAPT_THRESH;
  auto const solver_str      = std::string_view("scalapack");
#endif
  SECTION("fokkerplanck_2d_complete_case4, level 3, degree 3, sparse grid")
  {
    int const level           = 3;
    int const degree          = 3;
    auto constexpr tol_factor = get_tolerance<TestType>(1e5);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck2_complete_implicit_sg_l3_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION("fokkerplanck_2d_complete_case4, level 4, degree 3, sparse grid")
  {
    int const level           = 4;
    int const degree          = 3;
    auto constexpr tol_factor = get_tolerance<TestType>(1e5);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck2_complete_implicit_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION("fokkerplanck_2d_complete_case4, level 5, degree 3, sparse grid")
  {
    int const level           = 5;
    int const degree          = 3;
    auto constexpr tol_factor = get_tolerance<TestType>(1e5);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck2_complete_implicit_sg_l5_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);

#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION(
      "fokkerplanck_2d_complete_case4, implicit/non-uniform level, degree 3, "
      "sparse grid")
  {
    int const degree = 3;
    fk::vector<int> const levels{2, 3};
    auto constexpr tol_factor = get_tolerance<TestType>(1e5);

    auto const gold_base =
        time_advance_base_dir / ("fokkerplanck2_complete_implicit_sg_l" +
                                 get_level_string(levels) + "d3_t");
    auto const full_grid = false;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(pde_choice, levels, degree, cfl, full_grid,
                                 parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
                                 do_adapt_levels, adapt_threshold, solver_str);
    time_advance_test(parse_scalapack, gold_base, tol_factor);

#endif
  }
}

TEMPLATE_TEST_CASE("implicit time advance - diffusion 1", "[time_advance]",
                   double, float)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  TestType const cfl        = 0.01;
  std::string pde_choice    = "diffusion_1";
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  auto const num_dims = 1;
  auto const implicit = true;
#ifdef ASGARD_USE_SCALAPACK
  auto const do_adapt_levels = parser::DEFAULT_DO_ADAPT;
  auto const adapt_threshold = parser::DEFAULT_ADAPT_THRESH;
  auto const solver_str      = std::string_view("scalapack");
#endif
  SECTION("diffusion1, implicit, sparse grid, level 4, degree 4")
  {
    int const degree = 4;
    int const level  = 4;
    auto const gold_base =
        time_advance_base_dir / "diffusion1_implicit_sg_l4_d4_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);

#endif
  }
}

TEMPLATE_TEST_CASE("implicit time advance - diffusion 2", "[time_advance]",
                   double, float)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  std::string pde_choice    = "diffusion_2";
  TestType const cfl        = 0.01;
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  auto const num_dims = 2;
  auto const implicit = true;
#ifdef ASGARD_USE_SCALAPACK
  auto const do_adapt_levels = parser::DEFAULT_DO_ADAPT;
  auto const adapt_threshold = parser::DEFAULT_ADAPT_THRESH;
  auto const solver_str      = std::string_view("scalapack");
#endif

  SECTION("diffusion2, implicit, sparse grid, level 3, degree 3")
  {
    int const degree = 3;
    int const level  = 3;
    auto const gold_base =
        time_advance_base_dir / "diffusion2_implicit_sg_l3_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION("diffusion2, implicit, sparse grid, level 4, degree 3")
  {
    int const degree = 3;
    int const level  = 4;
    auto const gold_base =
        time_advance_base_dir / "diffusion2_implicit_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION("diffusion2, implicit, sparse grid, level 5, degree 3")
  {
    int const degree = 3;
    int const level  = 5;
    auto const gold_base =
        time_advance_base_dir / "diffusion2_implicit_sg_l5_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION("diffusion2, implicit/non-uniform level, degree 2, sparse grid")
  {
    int const degree = 2;

    fk::vector<int> const levels{4, 5};
    auto const gold_base =
        time_advance_base_dir /
        ("diffusion2_implicit_sg_l" + get_level_string(levels) + "d2_t");
    auto const full_grid = false;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(pde_choice, levels, degree, cfl, full_grid,
                                 parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
                                 do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 1", "[time_advance]",
                   double)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  std::string pde_choice    = "continuity_1";
  TestType const cfl        = 0.01;
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  auto const num_dims = 1;
  auto const implicit = true;
#ifdef ASGARD_USE_SCALAPACK
  auto const do_adapt_levels = parser::DEFAULT_DO_ADAPT;
  auto const adapt_threshold = parser::DEFAULT_ADAPT_THRESH;
  auto const solver_str      = std::string_view("scalapack");
#endif
  SECTION("continuity1, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    auto const gold_base =
        time_advance_base_dir / "continuity1_implicit_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION("continuity1, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto const gold_base =
        time_advance_base_dir / "continuity1_implicit_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION("continuity1, level 4, degree 3, sparse grid, iterative")
  {
    int const degree = 3;
    int const level  = 4;
    auto const gold_base =
        time_advance_base_dir / "continuity1_implicit_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 2", "[time_advance]",
                   float, double)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  std::string pde_choice    = "continuity_2";
  TestType const cfl        = 0.01;
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  auto const num_dims = 2;
  auto const implicit = true;
#ifdef ASGARD_USE_SCALAPACK
  auto const do_adapt_levels = parser::DEFAULT_DO_ADAPT;
  auto const adapt_threshold = parser::DEFAULT_ADAPT_THRESH;
  auto const solver_str      = std::string_view("scalapack");
#endif
  SECTION("continuity2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto const gold_base =
        time_advance_base_dir / "continuity2_implicit_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION("continuity2, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto const gold_base =
        time_advance_base_dir / "continuity2_implicit_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }

  SECTION("continuity2, level 4, degree 3, sparse grid, iterative")
  {
    int const degree               = 3;
    int const level                = 4;
    auto constexpr temp_tol_factor = get_tolerance<TestType>(10);

    auto const continuity2_base_dir =
        time_advance_base_dir / "continuity2_implicit_l4_d3_t";
    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, continuity2_base_dir, temp_tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
        do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, continuity2_base_dir, temp_tol_factor);
#endif
  }
  SECTION("continuity2, implicit/non-uniform level, degree 3, full grid")
  {
    int const degree = 3;

    fk::vector<int> const levels{3, 4};
    auto const gold_base =
        time_advance_base_dir /
        ("continuity2_implicit_fg_l" + get_level_string(levels) + "d3_t");
    auto const full_grid = true;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack(pde_choice, levels, degree, cfl, full_grid,
                                 parser::DEFAULT_MAX_LEVEL, num_steps, implicit,
                                 do_adapt_levels, adapt_threshold, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }
}
