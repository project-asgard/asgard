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

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

// settings for time advance testing
static auto constexpr num_steps = 5;

template<typename P>
void time_advance_test(parser const &parse, std::string const &filepath,
                       P const tolerance_factor = 1e-6)
{
  auto pde = make_PDE<P>(parse);

  int const my_rank   = get_rank();
  int const num_ranks = get_num_ranks();

  if (num_ranks > 1 && parse.using_implicit())
  {
    // distributed implicit stepping not implemented
    return;
  }

  options const opts(parse);
  elements::table const table(opts, *pde);

  auto const plan    = get_plan(num_ranks, table);
  auto const subgrid = plan.at(my_rank);

  basis::wavelet_transform<P, resource::host> const transformer(opts, *pde);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate initial condition vector.
  P const initial_scale = 1.0;
  std::vector<fk::vector<P>> initial_conditions;
  for (dimension<P> const &dim : pde->get_dimensions())
  {
    initial_conditions.push_back(
        forward_transform<P>(dim, dim.initial_condition, transformer));
  }
  fk::vector<P> const initial_condition =
      combine_dimensions(parse.get_degree(), table, subgrid.col_start,
                         subgrid.col_stop, initial_conditions);

  // -- generate sources.
  // these will be scaled later for time
  std::vector<fk::vector<P>> initial_sources;

  for (source<P> const &source : pde->sources)
  {
    std::vector<fk::vector<P>> initial_sources_dim;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      initial_sources_dim.push_back(forward_transform<P>(
          pde->get_dimensions()[i], source.source_funcs[i], transformer));
    }

    initial_sources.push_back(combine_dimensions(
        parse.get_degree(), table, subgrid.row_start, subgrid.row_stop,
        initial_sources_dim, initial_scale));
  }

  // generate boundary condition vectors
  // these will be scaled later similarly to the source vectors
  std::array<unscaled_bc_parts<P>, 2> unscaled_parts =
      boundary_conditions::make_unscaled_bc_parts(
          *pde, table, transformer, subgrid.row_start, subgrid.row_stop);

  fk::vector<P> f_val(initial_condition);

  // -- time loop
  for (auto i = 0; i < parse.get_time_steps(); ++i)
  {
    P const time = i * pde->get_dt();

    std::cout.setstate(std::ios_base::failbit);

    if (parse.using_implicit())
    {
      f_val =
          implicit_time_advance(*pde, table, initial_sources, unscaled_parts,
                                f_val, plan, time, parse.get_selected_solver());
    }

    else
    {
      auto const workspace_limit_MB = 4000;
      f_val = explicit_time_advance(*pde, table, opts, initial_sources,
                                    unscaled_parts, f_val, plan,
                                    workspace_limit_MB, time);
    }

    std::cout.clear();
    std::string const file_path = filepath + std::to_string(i) + ".dat";
    fk::vector<P> const gold =
        fk::vector<P>(read_vector_from_txt_file(file_path));

    // each rank generates partial answer
    auto const dof =
        static_cast<int>(std::pow(parse.get_degree(), pde->num_dims));
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
  TestType const cfl           = 0.01;
  std::string const pde_choice = "diffusion_2";
  int const num_dims           = 2;

  SECTION("diffusion2, explicit, sparse grid, level 2, degree 2")
  {
    int const degree = 2;
    int const level  = 2;

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-15 : 1e-5;

    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit, sparse grid, level 3, degree 3")
  {
    int const degree = 3;
    int const level  = 3;
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-15 : 1e-5;

    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2_sg_l3_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit, sparse grid, level 4, degree 4")
  {
    int const degree = 4;
    int const level  = 4;
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-12 : 1e-1;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2_sg_l4_d4_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, explicit/non-uniform level, sparse grid, degree 2")
  {
    int const degree = 2;
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-15 : 1e-5;

    fk::vector<int> const levels{4, 5};
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2_sg_l" +
                                  get_level_string(levels) + "d2_t";

    auto const full_grid = false;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - diffusion 1", "[time_advance]", double,
                   float)
{
  TestType const cfl     = 0.01;
  std::string pde_choice = "diffusion_1";
  int const num_dims     = 1;

  SECTION("diffusion1, explicit, sparse grid, level 3, degree 3")
  {
    int const degree            = 3;
    int const level             = 3;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion1_sg_l3_d3_t";
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-15 : 1e-5;

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion1, explicit, sparse grid, level 4, degree 4")
  {
    int const degree            = 4;
    int const level             = 4;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion1_sg_l4_d4_t";
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-11 : 1e-2;

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
  std::string const pde_choice = "continuity_1";

  TestType const cfl = 0.01;

  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-7;

  auto const num_dims = 1;

  SECTION("continuity1, explicit, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity1_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity1, explicit, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity1_fg_l2_d2_t";

    auto const full_grid = true;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity1, explicit, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity1_sg_l4_d3_t";

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
  std::string const pde_choice = "continuity_2";
  TestType const cfl           = 0.01;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-17 : 1e-7;
  auto const num_dims = 2;

  SECTION("continuity2, explicit, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity2_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity2_fg_l2_d2_t";

    auto const full_grid = true;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity2, explicit, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity2_sg_l4_d3_t";

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
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "continuity2_fg_l" +
                                  get_level_string(levels) + "d3_t";
    auto const full_grid = true;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 3", "[time_advance]", float,
                   double)
{
  std::string const pde_choice = "continuity_3";
  TestType const cfl           = 0.01;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-17 : 1e-8;
  auto const num_dims = 3;

  SECTION("continuity3, explicit, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity3_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity3, explicit, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity3_sg_l4_d3_t";

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
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "continuity3_sg_l" +
                                  get_level_string(levels) + "d4_t";
    auto const full_grid = false;

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-17 : 1e-7;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 6", "[time_advance]", float,
                   double)
{
  std::string const pde_choice = "continuity_6";
  TestType const cfl           = 0.01;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-6;
  auto const num_dims = 6;

  SECTION("continuity6, level 2, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity6_sg_l2_d3_t";

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
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "continuity6_sg_l" +
                                  get_level_string(levels) + "d2_t";
    auto const full_grid = false;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p2", "[time_advance]",
                   float, double)
{
  std::string const pde_choice = "fokkerplanck_1d_4p2";
  TestType const cfl           = 0.01;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-6;
  auto const num_dims = 1;

  SECTION("fokkerplanck_1d_4p2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/fokkerplanck1_4p2_sg_l2_d2_t";

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
  std::string const pde_choice = "fokkerplanck_1d_4p3";
  TestType const cfl           = 0.01;
  auto const num_dims          = 1;

  SECTION("fokkerplanck_1d_4p3, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-16 : 1e-6;

    std::string const gold_base =
        "../testing/generated-inputs/time_advance/fokkerplanck1_4p3_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p1a", "[time_advance]",
                   float, double)
{
  std::string const pde_choice = "fokkerplanck_1d_4p1a";
  TestType const cfl           = 0.01;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-5;
  auto const num_dims = 1;

  SECTION("fokkerplanck_1d_4p1a, level 2, degree 2, sparse grid")
  {
    int const degree            = 2;
    int const level             = 2;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "fokkerplanck1_4p1a_sg_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

// explicit time advance is not a fruitful approach to this problem
TEMPLATE_TEST_CASE("implicit time advance - fokkerplanck_2d_complete",
                   "[time_advance]", float, double)
{
  TestType const cfl     = 0.01;
  std::string pde_choice = "fokkerplanck_2d_complete";
  auto const num_dims    = 2;
  auto const implicit    = true;

  SECTION("fokkerplanck_2d_complete, level 3, degree 3, sparse grid")
  {
    int const level  = 3;
    int const degree = 3;

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-15 : 1e-6;

    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "fokkerplanck2_complete_implicit_sg_l3_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("fokkerplanck_2d_complete, level 4, degree 3, sparse grid")
  {
    int const level  = 4;
    int const degree = 3;

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-15 : 1e-5;

    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "fokkerplanck2_complete_implicit_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("fokkerplanck_2d_complete, level 5, degree 3, sparse grid")
  {
    int const level  = 5;
    int const degree = 3;

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-15 : 1e-5;

    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "fokkerplanck2_complete_implicit_sg_l5_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("fokkerplanck_2d_complete, implicit/non-uniform level, degree 3, "
          "sparse grid")
  {
    int const degree = 3;
    fk::vector<int> const levels{2, 3};

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-15 : 1e-5;

    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "fokkerplanck2_complete_implicit_sg_l" +
                                  get_level_string(levels) + "d3_t";
    auto const full_grid = false;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - diffusion 1", "[time_advance]",
                   double, float)
{
  TestType const cfl     = 0.01;
  std::string pde_choice = "diffusion_1";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-5;

  auto const num_dims = 1;
  auto const implicit = true;

  SECTION("diffusion1, implicit, sparse grid, level 4, degree 4")
  {
    int const degree            = 4;
    int const level             = 4;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion1_implicit_sg_l4_d4_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - diffusion 2", "[time_advance]",
                   double, float)
{
  std::string pde_choice = "diffusion_2";
  TestType const cfl     = 0.01;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-5;

  auto const num_dims = 2;
  auto const implicit = true;

  SECTION("diffusion2, implicit, sparse grid, level 3, degree 3")
  {
    int const degree            = 3;
    int const level             = 3;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2_implicit_sg_l3_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, implicit, sparse grid, level 4, degree 3")
  {
    int const degree            = 3;
    int const level             = 4;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2_implicit_sg_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, implicit, sparse grid, level 5, degree 3")
  {
    int const degree            = 3;
    int const level             = 5;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2_implicit_sg_l5_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("diffusion2, implicit/non-uniform level, degree 2, sparse grid")
  {
    int const degree = 2;

    fk::vector<int> const levels{4, 5};
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2_implicit_sg_l" +
                                  get_level_string(levels) + "d2_t";
    auto const full_grid = false;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 1", "[time_advance]",
                   double)
{
  std::string pde_choice = "continuity_1";
  TestType const cfl     = 0.01;

  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-8;

  auto const num_dims = 1;
  auto const implicit = true;

  SECTION("continuity1, level 2, degree 2, sparse grid")
  {
    int const degree     = 2;
    int const level      = 2;
    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity1_implicit_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity1, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity1_implicit_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity1, level 4, degree 3, sparse grid, iterative")
  {
    int const degree     = 3;
    int const level      = 4;
    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity1_implicit_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 2", "[time_advance]",
                   float, double)
{
  std::string pde_choice = "continuity_2";
  TestType const cfl     = 0.01;

  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-7;

  auto const num_dims = 2;
  auto const implicit = true;

  SECTION("continuity2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity2_implicit_l2_d2_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity2, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity2_implicit_l4_d3_t";

    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }

  SECTION("continuity2, level 4, degree 3, sparse grid, iterative")
  {
    int const degree = 3;
    int const level  = 4;

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-14 : 1e-7;

    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity2_implicit_l4_d3_t";
    auto const full_grid = false;
    parser const parse(
        pde_choice, fk::vector<int>(std::vector<int>(num_dims, level)), degree,
        cfl, full_grid, parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }
  SECTION("continuity2, implicit/non-uniform level, degree 3, full grid")
  {
    int const degree = 3;

    fk::vector<int> const levels{3, 4};
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "continuity2_implicit_fg_l" +
                                  get_level_string(levels) + "d3_t";
    auto const full_grid = true;
    parser const parse(pde_choice, levels, degree, cfl, full_grid,
                       parser::DEFAULT_MAX_LEVEL, num_steps, implicit);

    time_advance_test(parse, gold_base, tol_factor);
  }
}
