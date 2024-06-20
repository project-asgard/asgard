#include "asgard_vector.hpp"
#include "build_info.hpp"
#include "coefficients.hpp"
#include "pde.hpp"
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

// NOTE: when using this template the precision is inferred from the type
//       of the tolerance factor, make sure the type of the factor is correct
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
  if (parse.do_adapt_levels())
  {
    generate_all_coefficients_max_level(*pde, transformer);
  }
  else
  {
    generate_all_coefficients(*pde, transformer);
  }

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  // TODO: look into issue requiring mass mats to be regenerated after init
  // cond. see problem in main.cpp
  generate_dimension_mass_mat(*pde, transformer);

  fk::vector<P> f_val(initial_condition);

  asgard::matrix_list<P> operator_matrices;

  // -- time loop
  for (auto i = 0; i < opts.num_time_steps; ++i)
  {
    std::cout.setstate(std::ios_base::failbit);
    auto const time          = i * pde->get_dt();
    auto const update_system = i == 0;

    auto const method = opts.use_implicit_stepping ? time_advance::method::imp
                                                   : time_advance::method::exp;

    f_val = time_advance::adaptive_advance(method, *pde, operator_matrices,
                                           adaptive_grid, transformer, opts,
                                           f_val, time, update_system);

    std::cout.clear();

    auto const file_path =
        filepath.parent_path() /
        (filepath.filename().string() + std::to_string(i) + ".dat");
    auto const gold = read_vector_from_txt_file<P>(file_path);

    // each rank generates partial answer
    auto const dof =
        static_cast<int>(std::pow(parse.get_degree(), pde->num_dims()));
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

// The parser is constructed in one of 5 patterns,
// each is covered by a make method.
// All defaults are assumed automatically, only the adjusted variables are
// modified.
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps)
{
  parser parse(pde_choice, starting_levels);
  parser_mod::set(parse, parser_mod::degree, degree);
  parser_mod::set(parse, parser_mod::use_full_grid, full_grid);
  parser_mod::set(parse, parser_mod::num_time_steps, num_time_steps);
  parser_mod::set(parse, parser_mod::cfl, cfl);
  return parse;
}
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps, bool use_implicit)
{
  parser parse = make_basic_parser(pde_choice, starting_levels, degree, cfl,
                                   full_grid, num_time_steps);
  parser_mod::set(parse, parser_mod::use_implicit_stepping, use_implicit);
  return parse;
}
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps, bool use_implicit,
                         std::string const &solver_str)
{
  parser parse = make_basic_parser(pde_choice, starting_levels, degree, cfl,
                                   full_grid, num_time_steps, use_implicit);
  parser_mod::set(parse, parser_mod::solver_str, solver_str);
  return parse;
}
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps, bool use_implicit,
                         bool do_adapt_levels, double adapt_threshold,
                         bool use_linf_nrm)
{
  parser parse = make_basic_parser(pde_choice, starting_levels, degree, cfl,
                                   full_grid, num_time_steps, use_implicit);
  parser_mod::set(parse, parser_mod::do_adapt, do_adapt_levels);
  parser_mod::set(parse, parser_mod::adapt_threshold, adapt_threshold);
  parser_mod::set(parse, parser_mod::use_linf_nrm, use_linf_nrm);

  return parse;
}
parser make_basic_parser(std::string const &pde_choice,
                         fk::vector<int> starting_levels, int const degree,
                         double const cfl, bool full_grid,
                         int const num_time_steps, bool use_implicit,
                         bool do_adapt_levels, double adapt_threshold,
                         bool use_linf_nrm, std::string const &solver_str)
{
  parser parse = make_basic_parser(
      pde_choice, starting_levels, degree, cfl, full_grid, num_time_steps,
      use_implicit, do_adapt_levels, adapt_threshold, use_linf_nrm);
  parser_mod::set(parse, parser_mod::solver_str, solver_str);
  return parse;
}

TEMPLATE_TEST_CASE("time advance - diffusion 2", "[time_advance]", test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit, sparse grid, level 3, degree 3")
  {
    int const degree          = 3;
    int const level           = 3;
    auto constexpr tol_factor = get_tolerance<TestType>(100);

    auto const gold_base = time_advance_base_dir / "diffusion2_sg_l3_d3_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit, sparse grid, level 4, degree 4")
  {
    int const degree          = 4;
    int const level           = 4;
    auto constexpr tol_factor = get_tolerance<TestType>(1000000);
    auto const gold_base      = time_advance_base_dir / "diffusion2_sg_l4_d4_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

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
    parser const parse   = make_basic_parser(pde_choice, levels, degree, cfl,
                                           full_grid, num_steps);

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
    auto const tol_factor        = get_tolerance<default_precision>(1000);
    std::string const pde_choice = "diffusion_2";
    int const degree             = 4;
    fk::vector<int> const levels{3, 3};
    auto const gold_base =
        time_advance_base_dir / "diffusion2_ad_implicit_sg_l3_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = true;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 0.5e-1;
    auto const use_linf_nrm    = true;

    parser const parse = make_basic_parser(
        pde_choice, levels, degree, cfl, full_grid, num_steps, use_implicit,
        do_adapt_levels, adapt_threshold, use_linf_nrm);

    // temporarily disable test for MPI due to table elements < num ranks
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse, gold_base, tol_factor);
    }
#ifdef ASGARD_USE_SCALAPACK
    auto const solver_str = std::string("scalapack");

    parser const parse_scalapack = make_basic_parser(
        pde_choice, levels, degree, cfl, full_grid, num_steps, use_implicit,
        do_adapt_levels, adapt_threshold, use_linf_nrm, solver_str);

    // temporarily disable test for MPI due to table elements < num ranks
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse_scalapack, gold_base, tol_factor);
    }
#endif
  }
  SECTION("diffusion 2 explicit")
  {
    auto const tol_factor        = get_tolerance<default_precision>(1000);
    std::string const pde_choice = "diffusion_2";
    auto const degree            = 4;
    fk::vector<int> const levels{3, 3};
    auto const gold_base = time_advance_base_dir / "diffusion2_ad_sg_l3_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 0.5e-1;
    auto const use_linf_nrm    = true;

    parser const parse = make_basic_parser(
        pde_choice, levels, degree, cfl, full_grid, num_steps, use_implicit,
        do_adapt_levels, adapt_threshold, use_linf_nrm);
    // temporarily disable test for MPI due to table elements < num ranks
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse, gold_base, tol_factor);
    }
  }

  SECTION("fokkerplanck1_pitch_E case1 explicit")
  {
    auto constexpr tol_factor    = get_tolerance<default_precision>(100);
    std::string const pde_choice = "fokkerplanck_1d_pitch_E_case1";
    auto const degree            = 4;
    fk::vector<int> const levels{4};
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_4p1a_ad_sg_l4_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 1e-4;
    auto const use_linf_nrm    = true;

    parser const parse = make_basic_parser(
        pde_choice, levels, degree, cfl, full_grid, num_steps, use_implicit,
        do_adapt_levels, adapt_threshold, use_linf_nrm);

    // we do not gracefully handle coarsening below number of active ranks yet
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse, gold_base, tol_factor);
    }
  }

  SECTION("fokkerplanck1_pitch_E case2 explicit")
  {
    auto const tol_factor        = get_tolerance<default_precision>(10);
    std::string const pde_choice = "fokkerplanck_1d_pitch_E_case2";
    auto const degree            = 4;
    fk::vector<int> const levels{4};
    auto const gold_base =
        time_advance_base_dir / "fokkerplanck1_pitch_E_case2_ad_sg_l4_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 1e-4;
    auto const use_linf_nrm    = true;

    parser const parse = make_basic_parser(
        pde_choice, levels, degree, cfl, full_grid, num_steps, use_implicit,
        do_adapt_levels, adapt_threshold, use_linf_nrm);

    // we do not gracefully handle coarsening below number of active ranks yet
    if (get_num_ranks() == 1)
    {
      time_advance_test(parse, gold_base, tol_factor);
    }
  }

  SECTION("continuity 2 explicit")
  {
    auto const tol_factor        = get_tolerance<default_precision>(100);
    std::string const pde_choice = "continuity_2";
    auto const degree            = 4;
    fk::vector<int> const levels{3, 3};
    auto const gold_base = time_advance_base_dir / "continuity2_ad_sg_l3_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 1e-3;
    auto const use_linf_nrm    = true;

    parser const parse = make_basic_parser(
        pde_choice, levels, degree, cfl, full_grid, num_steps, use_implicit,
        do_adapt_levels, adapt_threshold, use_linf_nrm);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity 2 explicit, l2 norm")
  {
    // Gold data was calculated with L^{\infty} norm
    auto const tol_factor        = static_cast<default_precision>(0.00001);
    std::string const pde_choice = "continuity_2";
    auto const degree            = 4;
    fk::vector<int> const levels{3, 3};
    auto const gold_base = time_advance_base_dir / "continuity2_ad_sg_l3_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const use_linf_nrm    = false;
    auto const adapt_threshold = 1e-3;

    parser parse = make_basic_parser(pde_choice, levels, degree, cfl, full_grid,
                                     num_steps, use_implicit, do_adapt_levels,
                                     adapt_threshold, use_linf_nrm);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity 2 explicit")
  {
    auto const tol_factor        = get_tolerance<default_precision>(100);
    std::string const pde_choice = "continuity_2";
    auto const degree            = 4;
    fk::vector<int> const levels{3, 3};
    auto const gold_base = time_advance_base_dir / "continuity2_ad_sg_l3_d4_t";

    auto const full_grid       = false;
    auto const use_implicit    = parser::DEFAULT_USE_IMPLICIT;
    auto const do_adapt_levels = true;
    auto const adapt_threshold = 1e-3;
    auto const use_linf_nrm    = true;

    fk::vector<int> max_adapt_level{6, 8};

    parser parse = make_basic_parser(pde_choice, levels, degree, cfl, full_grid,
                                     num_steps, use_implicit, do_adapt_levels,
                                     adapt_threshold, use_linf_nrm);
    parser_mod::set(parse, parser_mod::max_adapt_level, max_adapt_level);
    parser_mod::set(parse, parser_mod::use_linf_nrm, use_linf_nrm);

    time_advance_test(parse, gold_base, tol_factor);
  }
}
TEMPLATE_TEST_CASE("time advance - diffusion 1", "[time_advance]", test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion1, explicit, sparse grid, level 4, degree 4")
  {
    int const degree      = 4;
    int const level       = 4;
    auto const gold_base  = time_advance_base_dir / "diffusion1_sg_l4_d4_t";
    auto const tol_factor = get_tolerance<TestType>(100000);

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 1", "[time_advance]", test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity1, explicit, level 2, degree 2, full grid")
  {
    int const degree     = 2;
    int const level      = 2;
    auto const gold_base = time_advance_base_dir / "continuity1_fg_l2_d2_t";

    auto const full_grid = true;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity1, explicit, level 4, degree 3, sparse grid")
  {
    int const degree     = 3;
    int const level      = 4;
    auto const gold_base = time_advance_base_dir / "continuity1_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 2", "[time_advance]", test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit, level 2, degree 2, full grid")
  {
    int const degree     = 2;
    int const level      = 2;
    auto const gold_base = time_advance_base_dir / "continuity2_fg_l2_d2_t";

    auto const full_grid = true;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit, level 4, degree 3, sparse grid")
  {
    int const degree     = 3;
    int const level      = 4;
    auto const gold_base = time_advance_base_dir / "continuity2_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

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
    parser const parse   = make_basic_parser(pde_choice, levels, degree, cfl,
                                           full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 3", "[time_advance]", test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity3, explicit, level 4, degree 3, sparse grid")
  {
    int const degree     = 3;
    int const level      = 4;
    auto const gold_base = time_advance_base_dir / "continuity3_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

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

    parser const parse = make_basic_parser(pde_choice, levels, degree, cfl,
                                           full_grid, num_steps);

    time_advance_test(parse, gold_base, get_tolerance<TestType>(10));
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 6", "[time_advance]", test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

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
    parser const parse   = make_basic_parser(pde_choice, levels, degree, cfl,
                                           full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_pitch_C", "[time_advance]",
                   test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p3", "[time_advance]",
                   test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_pitch_E_case1",
                   "[time_advance]", test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_pitch_E_case2",
                   "[time_advance]", test_precs)
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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

// explicit time advance is not a fruitful approach to this problem
TEMPLATE_TEST_CASE("implicit time advance - fokkerplanck_2d_complete_case4",
                   "[time_advance]", test_precs)
{
  if (!is_active() || get_num_ranks() == 2 || get_num_ranks() == 3)
  {
    return;
  }

  TestType const cfl     = 0.01;
  std::string pde_choice = "fokkerplanck_2d_complete_case4";
  auto const num_dims    = 2;
  auto const implicit    = true;
  auto const solver_str  = std::string("scalapack");

  SECTION("fokkerplanck_2d_complete_case4, level 3, degree 3, sparse grid")
  {
    int const level           = 3;
    int const degree          = 3;
    auto constexpr tol_factor = get_tolerance<TestType>(1e5);

    auto const gold_base =
        time_advance_base_dir / "fokkerplanck2_complete_implicit_sg_l3_d3_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, std::string("scalapack"));

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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);

#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
    parser const parse   = make_basic_parser(pde_choice, levels, degree, cfl,
                                           full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack =
        make_basic_parser(pde_choice, levels, degree, cfl, full_grid, num_steps,
                          implicit, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);

#endif
  }
}

TEMPLATE_TEST_CASE("implicit time advance - diffusion 1", "[time_advance]",
                   test_precs)
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

  SECTION("diffusion1, implicit, sparse grid, level 4, degree 4")
  {
    int const degree = 4;
    int const level  = 4;
    auto const gold_base =
        time_advance_base_dir / "diffusion1_implicit_sg_l4_d4_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, std::string("scalapack"));

    time_advance_test(parse_scalapack, gold_base, tol_factor);

#endif
  }
}

TEMPLATE_TEST_CASE("implicit time advance - diffusion 2", "[time_advance]",
                   test_precs)
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
  auto const solver_str = std::string("scalapack");
#endif

  SECTION("diffusion2, implicit, sparse grid, level 3, degree 3")
  {
    int const degree = 3;
    int const level  = 3;
    auto const gold_base =
        time_advance_base_dir / "diffusion2_implicit_sg_l3_d3_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
    parser const parse   = make_basic_parser(pde_choice, levels, degree, cfl,
                                           full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack =
        make_basic_parser(pde_choice, levels, degree, cfl, full_grid, num_steps,
                          implicit, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 1", "[time_advance]",
                   test_precs)
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
  auto const solver_str = std::string("scalapack");
#endif
  SECTION("continuity1, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    auto const gold_base =
        time_advance_base_dir / "continuity1_implicit_l2_d2_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 2", "[time_advance]",
                   test_precs)
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
  auto const solver_str = std::string("scalapack");
#endif
  SECTION("continuity2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto const gold_base =
        time_advance_base_dir / "continuity2_implicit_l2_d2_t";

    auto const full_grid = false;
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
    parser const parse   = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit);

    time_advance_test(parse, continuity2_base_dir, temp_tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack = make_basic_parser(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, num_steps, implicit, solver_str);

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
                       parser::DEFAULT_MAX_LEVEL, -1, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
#ifdef ASGARD_USE_SCALAPACK
    parser const parse_scalapack =
        make_basic_parser(pde_choice, levels, degree, cfl, full_grid, num_steps,
                          implicit, solver_str);

    time_advance_test(parse_scalapack, gold_base, tol_factor);
#endif
  }
}

TEMPLATE_TEST_CASE("IMEX time advance - landau", "[imex]", test_precs)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  std::string const pde_choice = "landau";
  fk::vector<int> const levels{4, 4};
  int const degree            = 3;
  static int constexpr nsteps = 100;

  TestType constexpr gmres_tol =
      std::is_same_v<TestType, double> ? 1.0e-8 : 1.0e-6;
  TestType constexpr tolerance =
      std::is_same_v<TestType, double> ? 1.0e-9 : 1.0e-3;
  // Note: not sure if the single precision test makes sense here

  parser parse(pde_choice, levels);
  parser_mod::set(parse, parser_mod::degree, degree);
  parser_mod::set(parse, parser_mod::dt, 0.019634954084936);
  parser_mod::set(parse, parser_mod::use_imex_stepping, true);
  parser_mod::set(parse, parser_mod::use_full_grid, true);
  parser_mod::set(parse, parser_mod::num_time_steps, nsteps);
  parser_mod::set(parse, parser_mod::gmres_tolerance, gmres_tol);

  auto const pde = make_PDE<TestType>(parse);

  options const opts(parse);
  elements::table const check(opts, *pde);

  adapt::distributed_grid adaptive_grid(*pde, opts);
  basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                       *pde);

  // -- compute dimension mass matrices
  generate_dimension_mass_mat(*pde, transformer);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate moments
  for (auto &m : pde->moments)
  {
    m.createFlist(*pde, opts);
    expect(m.get_fList().size() > 0);

    m.createMomentVector(*pde, parse, adaptive_grid.get_table());
    expect(m.get_vector().size() > 0);
  }

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  generate_dimension_mass_mat(*pde, transformer);

  fk::vector<TestType> f_val(initial_condition);
  asgard::matrix_list<TestType> operator_matrices;

  TestType E_pot_initial = 0.0;
  TestType E_kin_initial = 0.0;

  // -- time loop
  for (int i = 0; i < opts.num_time_steps; ++i)
  {
    std::cout.setstate(std::ios_base::failbit);
    TestType const time      = i * pde->get_dt();
    bool const update_system = i == 0;
    f_val                    = time_advance::adaptive_advance(
        asgard::time_advance::method::imex, *pde, operator_matrices,
        adaptive_grid, transformer, opts, f_val, time, update_system);

    std::cout.clear();

    // compute the E potential and kinetic energy
    fk::vector<TestType> E_field_sq(pde->E_field);
    for (auto &e : E_field_sq)
    {
      e = e * e;
    }
    dimension<TestType> &dim = pde->get_dimensions()[0];
    TestType E_pot           = calculate_integral(E_field_sq, dim);
    TestType E_kin =
        calculate_integral(pde->moments[2].get_realspace_moment(), dim);
    if (i == 0)
    {
      E_pot_initial = E_pot;
      E_kin_initial = E_kin;
    }

    // calculate the absolute relative total energy
    TestType E_relative =
        std::fabs((E_pot + E_kin) - (E_pot_initial + E_kin_initial));
    REQUIRE(E_relative <= tolerance);
  }

  parameter_manager<TestType>::get_instance().reset();
}

#ifdef ASGARD_ENABLE_DOUBLE
TEMPLATE_TEST_CASE("IMEX time advance - twostream", "[imex]", double)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  std::string const pde_choice = "two_stream";
  fk::vector<int> const levels{5, 5};
  int const degree            = 3;
  static int constexpr nsteps = 20;

  TestType constexpr tolerance =
      std::is_same_v<TestType, double> ? 1.0e-9 : 1.0e-5;

  parser parse(pde_choice, levels);
  parser_mod::set(parse, parser_mod::degree, degree);
  parser_mod::set(parse, parser_mod::dt, 6.25e-3);
  parser_mod::set(parse, parser_mod::use_imex_stepping, true);
  parser_mod::set(parse, parser_mod::use_full_grid, true);
  parser_mod::set(parse, parser_mod::num_time_steps, nsteps);

  auto const pde = make_PDE<TestType>(parse);

  options const opts(parse);
  elements::table const check(opts, *pde);

  adapt::distributed_grid adaptive_grid(*pde, opts);
  basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                       *pde);

  // -- compute dimension mass matrices
  generate_dimension_mass_mat(*pde, transformer);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate moments
  for (auto &m : pde->moments)
  {
    m.createFlist(*pde, opts);
    expect(m.get_fList().size() > 0);

    m.createMomentVector(*pde, parse, adaptive_grid.get_table());
    expect(m.get_vector().size() > 0);
  }

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  generate_dimension_mass_mat(*pde, transformer);

  fk::vector<TestType> f_val(initial_condition);
  asgard::matrix_list<TestType> operator_matrices;

  TestType E_pot_initial = 0.0;
  TestType E_kin_initial = 0.0;

  // -- time loop
  for (int i = 0; i < opts.num_time_steps; ++i)
  {
    std::cout.setstate(std::ios_base::failbit);
    TestType const time      = i * pde->get_dt();
    bool const update_system = i == 0;
    f_val                    = time_advance::adaptive_advance(
        asgard::time_advance::method::imex, *pde, operator_matrices,
        adaptive_grid, transformer, opts, f_val, time, update_system);

    std::cout.clear();

    // compute the E potential and kinetic energy
    fk::vector<TestType> E_field_sq(pde->E_field);
    for (auto &e : E_field_sq)
    {
      e = e * e;
    }
    dimension<TestType> &dim = pde->get_dimensions()[0];
    TestType E_pot           = calculate_integral(E_field_sq, dim);
    TestType E_kin =
        calculate_integral(pde->moments[2].get_realspace_moment(), dim);
    if (i == 0)
    {
      E_pot_initial = E_pot;
      E_kin_initial = E_kin;
    }

    // calculate the absolute relative total energy
    TestType E_relative =
        std::fabs((E_pot + E_kin) - (E_pot_initial + E_kin_initial));
    // REQUIRE(E_relative <= tolerance);

    // calculate integral of moments
    fk::vector<TestType> mom0 = pde->moments[0].get_realspace_moment();
    fk::vector<TestType> mom1 = pde->moments[1].get_realspace_moment();

    TestType n_total = calculate_integral(fm::scal(TestType{2.0}, mom0), dim);

    fk::vector<TestType> n_times_u(mom0.size());
    for (int j = 0; j < n_times_u.size(); j++)
    {
      n_times_u[j] = mom0[j] * mom1[j];
    }

    TestType nu_total = calculate_integral(n_times_u, dim);

    // n total should be close to 6.28
    REQUIRE((n_total - 6.283185) <= 1.0e-4);

    // n*u total should be 0
    REQUIRE(nu_total <= 1.0e-14);

    // total relative energy change drops and stabilizes around 2.0e-5
    REQUIRE(E_relative <= 5.5e-5);

    if (i > 0 && i < 100)
    {
      // check the initial slight energy decay before it stabilizes
      // Total energy at time step 1:   5.4952938
      // Total energy at time step 100: 5.4952734
      REQUIRE(E_relative >= tolerance);
    }
  }

  parameter_manager<TestType>::get_instance().reset();
}

TEMPLATE_TEST_CASE("IMEX time advance - twostream - ASG", "[imex][adapt]",
                   double)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  std::string const pde_choice = "two_stream";
  fk::vector<int> const levels{5, 5};
  int const degree            = 3;
  static int constexpr nsteps = 10;

  TestType constexpr tolerance =
      std::is_same_v<TestType, double> ? 1.0e-9 : 1.0e-5;

  parser parse(pde_choice, levels);
  parser_mod::set(parse, parser_mod::degree, degree);
  parser_mod::set(parse, parser_mod::dt, 6.25e-3);
  parser_mod::set(parse, parser_mod::use_imex_stepping, true);
  parser_mod::set(parse, parser_mod::use_full_grid, false);
  parser_mod::set(parse, parser_mod::do_adapt, true);
  parser_mod::set(parse, parser_mod::max_level, 5);
  parser_mod::set(parse, parser_mod::adapt_threshold, 1.0e-6);
  parser_mod::set(parse, parser_mod::num_time_steps, nsteps);

  auto const pde = make_PDE<TestType>(parse);

  options const opts(parse);
  elements::table const check(opts, *pde);

  adapt::distributed_grid adaptive_grid(*pde, opts);
  basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                       *pde);

  // -- compute dimension mass matrices
  generate_dimension_mass_mat(*pde, transformer);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate moments
  for (auto &m : pde->moments)
  {
    m.createFlist(*pde, opts);
    expect(m.get_fList().size() > 0);

    m.createMomentVector(*pde, parse, adaptive_grid.get_table());
    expect(m.get_vector().size() > 0);
  }

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  generate_dimension_mass_mat(*pde, transformer);

  fk::vector<TestType> f_val(initial_condition);
  asgard::matrix_list<TestType> operator_matrices;

  TestType E_pot_initial = 0.0;
  TestType E_kin_initial = 0.0;

  // number of DOF for the FG case: (degree * 2^level)^2 = 9.216e3
  int const fg_dof = std::pow(degree * fm::two_raised_to(levels[0]), 2);

  // -- time loop
  for (int i = 0; i < opts.num_time_steps; ++i)
  {
    std::cout.setstate(std::ios_base::failbit);
    TestType const time      = i * pde->get_dt();
    bool const update_system = i == 0;
    f_val                    = time_advance::adaptive_advance(
        asgard::time_advance::method::imex, *pde, operator_matrices,
        adaptive_grid, transformer, opts, f_val, time, update_system);

    std::cout.clear();

    // compute the E potential and kinetic energy
    fk::vector<TestType> E_field_sq(pde->E_field);
    for (auto &e : E_field_sq)
    {
      e = e * e;
    }
    dimension<TestType> &dim = pde->get_dimensions()[0];
    TestType E_pot           = calculate_integral(E_field_sq, dim);
    TestType E_kin =
        calculate_integral(pde->moments[2].get_realspace_moment(), dim);
    if (i == 0)
    {
      E_pot_initial = E_pot;
      E_kin_initial = E_kin;
    }

    // calculate the absolute relative total energy
    TestType E_relative =
        std::fabs((E_pot + E_kin) - (E_pot_initial + E_kin_initial));
    // REQUIRE(E_relative <= tolerance);

    // calculate integral of moments
    fk::vector<TestType> mom0 = pde->moments[0].get_realspace_moment();
    fk::vector<TestType> mom1 = pde->moments[1].get_realspace_moment();

    TestType n_total = calculate_integral(fm::scal(TestType{2.0}, mom0), dim);

    fk::vector<TestType> n_times_u(mom0.size());
    for (int j = 0; j < n_times_u.size(); j++)
    {
      n_times_u[j] = mom0[j] * mom1[j];
    }

    TestType nu_total = calculate_integral(n_times_u, dim);

    // n total should be close to 6.28
    REQUIRE((n_total - 6.283185) <= 1.0e-4);

    // n*u total should be 0
    REQUIRE(nu_total <= 1.0e-14);

    // total relative energy change drops and stabilizes around 2.0e-5
    REQUIRE(E_relative <= 5.5e-5);

    if (i > 0 && i < 100)
    {
      // check the initial slight energy decay before it stabilizes
      // Total energy at time step 1:   5.4952938
      // Total energy at time step 100: 5.4952734
      REQUIRE(E_relative >= tolerance);
    }

    // for this configuration, the DOF of ASG / DOF of FG should be between
    // 60-65%. Testing against 70% is conservative but will capture issues with
    // adaptivity
    REQUIRE(static_cast<TestType>(f_val.size()) / fg_dof <= 0.70);
  }

  parameter_manager<TestType>::get_instance().reset();
}
#endif

TEMPLATE_TEST_CASE("IMEX time advance - relaxation1x1v", "[imex]", test_precs)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  std::string const pde_choice = "relaxation_1x1v";
  fk::vector<int> const levels{0, 4};
  int const degree            = 3;
  static int constexpr nsteps = 10;

  TestType constexpr gmres_tol =
      std::is_same_v<TestType, double> ? 1.0e-10 : 1.0e-6;

  // the expected L2 from analytical solution after the maxwellian has relaxed
  TestType constexpr expected_l2 = 8.654e-4;
  // rel tolerance for comparing l2
  TestType constexpr tolerance = std::is_same_v<TestType, double> ? 1.0e-3 : 5.0e-3;

  parser parse(pde_choice, levels);
  parser_mod::set(parse, parser_mod::degree, degree);
  parser_mod::set(parse, parser_mod::dt, 5.0e-4);
  parser_mod::set(parse, parser_mod::use_imex_stepping, true);
  parser_mod::set(parse, parser_mod::use_full_grid, true);
  parser_mod::set(parse, parser_mod::num_time_steps, nsteps);
  parser_mod::set(parse, parser_mod::gmres_tolerance, gmres_tol);

  auto const pde = make_PDE<TestType>(parse);

  options const opts(parse);
  elements::table const check(opts, *pde);

  adapt::distributed_grid adaptive_grid(*pde, opts);
  basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                       *pde);

  // -- compute dimension mass matrices
  generate_dimension_mass_mat(*pde, transformer);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate moments
  for (auto &m : pde->moments)
  {
    m.createFlist(*pde, opts);
    expect(m.get_fList().size() > 0);

    m.createMomentVector(*pde, parse, adaptive_grid.get_table());
    expect(m.get_vector().size() > 0);
  }

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  generate_dimension_mass_mat(*pde, transformer);

  fk::vector<TestType> f_val(initial_condition);
  asgard::matrix_list<TestType> operator_matrices;

  // -- time loop
  for (int i = 0; i < opts.num_time_steps; ++i)
  {
    std::cout.setstate(std::ios_base::failbit);
    TestType const time      = i * pde->get_dt();
    bool const update_system = i == 0;
    fk::vector<TestType> sol = time_advance::adaptive_advance(
        asgard::time_advance::method::imex, *pde, operator_matrices,
        adaptive_grid, transformer, opts, f_val, time, update_system);

    f_val = std::move(sol);
    std::cout.clear();

    // get analytic solution at final time step to compare
    if (i == opts.num_time_steps - 1)
    {
      fk::vector<TestType> const analytic_solution = sum_separable_funcs(
          pde->exact_vector_funcs(), pde->get_dimensions(), adaptive_grid,
          transformer, degree, time + pde->get_dt());

      // calculate L2 error between simulation and analytical solution
      TestType const L2 = nrm2_dist(f_val, analytic_solution);
      TestType const relative_error =
          TestType{100.0} * (L2 / asgard::l2_norm(analytic_solution));
      auto const [l2_errors, relative_errors] =
          asgard::gather_errors<TestType>(L2, relative_error);
      expect(l2_errors.size() == relative_errors.size());
      for (int j = 0; j < l2_errors.size(); ++j)
      {
        // verify the l2 is close to the expected l2 from the analytical
        // solution
        TestType const abs_diff = std::abs(l2_errors[j] - expected_l2);
        TestType const expected =
            tolerance * std::max(std::abs(l2_errors[j]), std::abs(expected_l2));
        REQUIRE(abs_diff <= expected);
      }
    }
  }

  parameter_manager<TestType>::get_instance().reset();
}

TEMPLATE_TEST_CASE("IMEX time advance - relaxation1x2v", "[!mayfail][imex]",
                   test_precs)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  std::string const pde_choice = "relaxation_1x2v";
  fk::vector<int> const levels{0, 4, 4};
  int const degree            = 3;
  static int constexpr nsteps = 10;

  parser parse(pde_choice, levels);
  parser_mod::set(parse, parser_mod::degree, degree);
  parser_mod::set(parse, parser_mod::dt, 5.0e-4);
  parser_mod::set(parse, parser_mod::use_imex_stepping, true);
  parser_mod::set(parse, parser_mod::use_full_grid, true);
  parser_mod::set(parse, parser_mod::num_time_steps, nsteps);

  auto const pde = make_PDE<TestType>(parse);

  parameter_manager<TestType>::get_instance().reset();

  // TODO
  REQUIRE(false);
}

TEMPLATE_TEST_CASE("IMEX time advance - relaxation1x3v", "[!mayfail][imex]",
                   test_precs)
{
  // Disable test for MPI - IMEX needs to be tested further with MPI
  if (!is_active() || get_num_ranks() > 1)
  {
    return;
  }

  std::string const pde_choice = "relaxation_1x3v";
  fk::vector<int> const levels{0, 4, 4, 4};
  int const degree            = 3;
  static int constexpr nsteps = 10;

  parser parse(pde_choice, levels);
  parser_mod::set(parse, parser_mod::degree, degree);
  parser_mod::set(parse, parser_mod::dt, 5.0e-4);
  parser_mod::set(parse, parser_mod::use_imex_stepping, true);
  parser_mod::set(parse, parser_mod::use_full_grid, true);
  parser_mod::set(parse, parser_mod::num_time_steps, nsteps);

  auto const pde = make_PDE<TestType>(parse);

  parameter_manager<TestType>::get_instance().reset();

  // TODO
  REQUIRE(false);
}

/*****************************************************************************
 * Testing the ability to split a matrix into multiple calls
 *****************************************************************************/
#ifndef KRON_MODE_GLOBAL
template<typename prec>
void test_memory_mode(imex_flag imex)
{
  if (get_num_ranks() > 1) // this is a one-rank test
    return;
  // make some PDE, no need to be too specific
  fk::vector<int> levels = {5, 5};
  parser parse("two_stream", levels);
  parser_mod::set(parse, parser_mod::degree, 3);

  auto pde = make_PDE<prec>(parse);

  options const opts(parse);

  adapt::distributed_grid grid(*pde, opts);
  basis::wavelet_transform<prec, resource::host> const transformer(opts, *pde);
  generate_dimension_mass_mat(*pde, transformer);
  generate_all_coefficients(*pde, transformer);
  auto const x = grid.get_initial_condition(*pde, transformer, opts);
  generate_dimension_mass_mat(*pde, transformer);

  // one means that all data fits in memory and only one call will be made
  constexpr bool force_sparse = true;

  kron_sparse_cache spcache_null1, spcache_one;
  memory_usage memory_one =
      compute_mem_usage(*pde, grid, opts, imex, spcache_null1);
  auto mat_one              = make_kronmult_matrix(*pde, grid, opts, memory_one,
                                      imex_flag::unspecified, spcache_null1);
  memory_usage spmemory_one = compute_mem_usage(
      *pde, grid, opts, imex, spcache_one, 6, 2147483646, force_sparse);
  auto spmat_one = make_kronmult_matrix(*pde, grid, opts, spmemory_one, imex,
                                        spcache_one, force_sparse);

  kron_sparse_cache spcache_null2, spcache_multi;
  memory_usage memory_multi =
      compute_mem_usage(*pde, grid, opts, imex, spcache_null2, 0, 8000);
  auto mat_multi =
      make_kronmult_matrix(*pde, grid, opts, memory_multi, imex, spcache_null2);
  memory_usage spmemory_multi = compute_mem_usage(
      *pde, grid, opts, imex, spcache_multi, 6, 8000, force_sparse);
  auto spmat_multi = make_kronmult_matrix(*pde, grid, opts, spmemory_multi,
                                          imex, spcache_multi, force_sparse);

  REQUIRE(mat_one.is_onecall());
  REQUIRE(spmat_one.is_onecall());
  REQUIRE(not spmat_multi.is_onecall());

  fk::vector<prec> y_one(mat_one.output_size());
  fk::vector<prec> y_multi(mat_multi.output_size());
  fk::vector<prec> y_spone(spmat_one.output_size());
  fk::vector<prec> y_spmulti(spmat_multi.output_size());
  REQUIRE(y_one.size() == y_multi.size());
  REQUIRE(y_one.size() == y_spmulti.size());
  REQUIRE(y_one.size() == y_spone.size());

#ifdef ASGARD_USE_CUDA
  fk::vector<prec, mem_type::owner, resource::device> xdev(y_one.size());
  fk::vector<prec, mem_type::owner, resource::device> ydev(y_multi.size());
  mat_one.set_workspace(xdev, ydev);
  mat_multi.set_workspace(xdev, ydev);
  spmat_one.set_workspace(xdev, ydev);
  spmat_multi.set_workspace(xdev, ydev);
#endif
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  // allocate large enough vectors, total size is 24MB
  cudaStream_t load_stream;
  cudaStreamCreate(&load_stream);
  auto worka = fk::vector<int, mem_type::owner, resource::device>(1048576);
  auto workb = fk::vector<int, mem_type::owner, resource::device>(1048576);
  auto irowa = fk::vector<int, mem_type::owner, resource::device>(262144);
  auto irowb = fk::vector<int, mem_type::owner, resource::device>(262144);
  auto icola = fk::vector<int, mem_type::owner, resource::device>(262144);
  auto icolb = fk::vector<int, mem_type::owner, resource::device>(262144);
  mat_multi.set_workspace_ooc(worka, workb, load_stream);
  spmat_multi.set_workspace_ooc(worka, workb, load_stream);
  mat_multi.set_workspace_ooc_sparse(irowa, irowb, icola, icolb);
  spmat_multi.set_workspace_ooc_sparse(irowa, irowb, icola, icolb);
#endif

  mat_one.apply(2.0, x.data(), 0.0, y_one.data());
  mat_multi.apply(2.0, x.data(), 0.0, y_multi.data());
  spmat_one.apply(2.0, x.data(), 0.0, y_spone.data());
  spmat_multi.apply(2.0, x.data(), 0.0, y_spmulti.data());

  rmse_comparison(y_one, y_multi, prec{10});
  rmse_comparison(y_one, y_spone, prec{10});
  rmse_comparison(y_one, y_spmulti, prec{10});

  fk::vector<prec> y2_one(y_one);
  fk::vector<prec> y2_multi(y_multi);
  fk::vector<prec> y2_spone(y_spone);
  fk::vector<prec> y2_spmulti(y_spmulti);

  mat_one.apply(2.5, y_one.data(), 3.0, y2_one.data());
  mat_multi.apply(2.5, y_multi.data(), 3.0, y2_multi.data());
  spmat_one.apply(2.5, y_spone.data(), 3.0, y2_spone.data());
  spmat_multi.apply(2.5, y_spmulti.data(), 3.0, y2_spmulti.data());

  rmse_comparison(y2_one, y2_multi, prec{10});
  rmse_comparison(y_one, y_spone, prec{10});
  rmse_comparison(y_one, y_spmulti, prec{10});

  parameter_manager<prec>::get_instance().reset();
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  cudaStreamDestroy(load_stream);
#endif
}

TEMPLATE_TEST_CASE("testing multi imex unspecified", "unspecified", test_precs)
{
  test_memory_mode<TestType>(imex_flag::unspecified);
}

TEMPLATE_TEST_CASE("testing multi imex implicit", "imex_implicit", test_precs)
{
  test_memory_mode<TestType>(imex_flag::imex_implicit);
}

TEMPLATE_TEST_CASE("testing multi imex explicit", "imex_explicit", test_precs)
{
  test_memory_mode<TestType>(imex_flag::imex_explicit);
}
#endif
