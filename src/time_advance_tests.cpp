#include "build_info.hpp"
#include "chunk.hpp"
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
void time_advance_test(int const level, int const degree, PDE<P> &pde,
                       int const num_steps, std::string const filepath,
                       bool const full_grid = false, P const tol_factor = 1e-10,
                       double const cfl = 0.01)
{
  int const my_rank   = get_rank();
  int const num_ranks = get_num_ranks();

  std::vector<std::string> const args = [level, degree, full_grid, cfl]() {
    std::string const grid_str    = full_grid ? "-f" : "";
    std::vector<std::string> args = {"-l",
                                     std::to_string(level),
                                     "-d",
                                     std::to_string(degree),
                                     grid_str,
                                     "-c",
                                     to_string_with_precision(cfl, 16)};

    return args;
  }();
  options const o = make_options(args);

  element_table const table(o, pde.num_dims);

  // can't run problem with fewer elements than ranks
  // this is asserted on in the distribution component
  if (num_ranks >= table.size())
  {
    return;
  }

  auto const plan    = get_plan(num_ranks, table);
  auto const subgrid = plan.at(my_rank);

  // -- set coeffs

  basis::wavelet_transform<P, resource::host> const transformer(level, degree);
  generate_all_coefficients(pde, transformer);

  // -- generate initial condition vector.
  fk::vector<P> const initial_condition = [&pde, &table, &transformer, &subgrid,
                                           degree]() {
    std::vector<fk::vector<P>> initial_conditions;
    for (dimension<P> const &dim : pde.get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<P>(dim, dim.initial_condition, transformer));
    }
    return combine_dimensions(degree, table, subgrid.col_start,
                              subgrid.col_stop, initial_conditions);
  }();

  // -- generate sources.
  // these will be scaled later for time
  std::vector<fk::vector<P>> const initial_sources =
      [&pde, &table, &transformer, &subgrid, degree]() {
        std::vector<fk::vector<P>> initial_sources;
        for (source<P> const &source : pde.sources)
        {
          // gather contributions from each dim for this source, in wavelet
          // space
          std::vector<fk::vector<P>> initial_sources_dim;
          for (int i = 0; i < pde.num_dims; ++i)
          {
            initial_sources_dim.push_back(forward_transform<P>(
                pde.get_dimensions()[i], source.source_funcs[i], transformer));
          }
          // combine those contributions to form the unscaled source vector
          initial_sources.push_back(
              combine_dimensions(degree, table, subgrid.row_start,
                                 subgrid.row_stop, initial_sources_dim));
        }
        return initial_sources;
      }();

  /* generate boundary condition vectors */
  /* these will be scaled later similarly to the source vectors */
  std::array<unscaled_bc_parts<P>, 2> unscaled_parts =
      boundary_conditions::make_unscaled_bc_parts(
          pde, table, transformer, subgrid.row_start, subgrid.row_stop);

  // -- prep workspace/chunks
  int const workspace_limit_MB = 4000;

  // -- time loop
  P const dt = pde.get_dt() * o.get_cfl();

  fk::vector<P> f_val(initial_condition);
  for (int i = 0; i < num_steps; ++i)
  {
    P const time = i * dt;

    std::cout.setstate(std::ios_base::failbit);
    f_val = explicit_time_advance(pde, table, initial_sources, unscaled_parts,
                                  f_val, plan, workspace_limit_MB, time, dt);
    std::cout.clear();
    std::string const file_path = filepath + std::to_string(i) + ".dat";

    int const degree       = pde.get_dimensions()[0].get_degree();
    int const segment_size = static_cast<int>(std::pow(degree, pde.num_dims));
    fk::vector<P> const gold =
        fk::vector<P>(read_vector_from_txt_file(file_path))
            .extract(subgrid.col_start * segment_size,
                     (subgrid.col_stop + 1) * segment_size - 1);
    rmse_comparison(gold, f_val, tol_factor);
  }
}

template<typename P>
void implicit_time_advance_test(int const level, int const degree, PDE<P> &pde,
                                int const num_steps, std::string const filepath,
                                bool const full_grid     = false,
                                P const tolerance_factor = 1e-6,
                                double const cfl         = 0.01,
                                solve_opts const solver  = solve_opts::direct)
{
  int const my_rank   = get_rank();
  int const num_ranks = get_num_ranks();
  if (num_ranks > 1)
  {
    // distributed implicit stepping not implemented
    ignore(level);
    ignore(degree);
    ignore(pde);
    ignore(solver);
    ignore(num_steps);
    ignore(filepath);
    ignore(full_grid);
    return;
  }

  std::string const grid_str = full_grid ? "-f" : "";
  options const o =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree),
                    "-c", std::to_string(cfl), "--implicit", grid_str});

  element_table const table(o, pde.num_dims);
  auto const plan    = get_plan(num_ranks, table);
  auto const subgrid = plan.at(my_rank);

  basis::wavelet_transform<P, resource::host> const transformer(level, degree);

  // -- set coeffs
  generate_all_coefficients(pde, transformer);

  // -- generate initial condition vector.
  P const initial_scale = 1.0;
  std::vector<fk::vector<P>> initial_conditions;
  for (dimension<P> const &dim : pde.get_dimensions())
  {
    initial_conditions.push_back(
        forward_transform<P>(dim, dim.initial_condition, transformer));
  }
  fk::vector<P> const initial_condition = combine_dimensions(
      degree, table, subgrid.col_start, subgrid.col_stop, initial_conditions);

  // -- generate sources.
  // these will be scaled later for time
  std::vector<fk::vector<P>> initial_sources;

  for (source<P> const &source : pde.sources)
  {
    std::vector<fk::vector<P>> initial_sources_dim;
    for (int i = 0; i < pde.num_dims; ++i)
    {
      initial_sources_dim.push_back(forward_transform<P>(
          pde.get_dimensions()[i], source.source_funcs[i], transformer));
    }

    initial_sources.push_back(
        combine_dimensions(degree, table, subgrid.row_start, subgrid.row_stop,
                           initial_sources_dim, initial_scale));
  }

  // generate boundary condition vectors
  // these will be scaled later similarly to the source vectors
  std::array<unscaled_bc_parts<P>, 2> unscaled_parts =
      boundary_conditions::make_unscaled_bc_parts(
          pde, table, transformer, subgrid.row_start, subgrid.row_stop);

  // -- prep workspace/chunks
  int const workspace_limit_MB            = 4000;
  std::vector<element_chunk> const chunks = assign_elements(
      subgrid, get_num_chunks(subgrid, pde, workspace_limit_MB));

  fk::vector<P> f_val(initial_condition);

  // -- time loop
  P const dt = pde.get_dt() * o.get_cfl();

  for (int i = 0; i < num_steps; ++i)
  {
    P const time = i * dt;

    std::cout.setstate(std::ios_base::failbit);
    f_val = implicit_time_advance(pde, table, initial_sources, unscaled_parts,
                                  f_val, chunks, plan, time, dt, solver);
    std::cout.clear();
    std::string const file_path = filepath + std::to_string(i) + ".dat";

    fk::vector<P> const gold =
        fk::vector<P>(read_vector_from_txt_file(file_path));

    rmse_comparison(gold, f_val, tolerance_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - diffusion 2", "[time_advance]", double,
                   float)
{
  TestType const cfl = 0.003;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-5;

  SECTION("diffusion2, explicit, sparse grid, level 2, degree 2")
  {
    int const degree     = 2;
    int const level      = 2;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2/diffusion2_e_sg_l2_d2_t";

    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor, cfl);
  }

  SECTION("diffusion2, explicit, sparse grid, level 3, degree 3")
  {
    int const degree     = 3;
    int const level      = 3;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2/diffusion2_e_sg_l3_d3_t";

    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor, cfl);
  }

  SECTION("diffusion2, explicit, sparse grid, level 4, degree 4")
  {
    int const degree     = 4;
    int const level      = 4;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2/diffusion2_e_sg_l4_d4_t";

    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor, cfl);
  }
}

TEMPLATE_TEST_CASE("time advance - diffusion 1", "[time_advance]", double,
                   float)
{
  TestType const cfl = 0.003;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-4;

  SECTION("diffusion1, explicit, sparse grid, level 2, degree 2")
  {
    int const degree     = 2;
    int const level      = 2;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion1/diffusion1_e_sg_l2_d2_t";

    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor, cfl);
  }

  SECTION("diffusion1, explicit, sparse grid, level 3, degree 3")
  {
    int const degree     = 3;
    int const level      = 3;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion1/diffusion1_e_sg_l3_d3_t";

    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor, cfl);
  }

  SECTION("diffusion1, explicit, sparse grid, level 4, degree 4")
  {
    int const degree     = 4;
    int const level      = 4;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion1/diffusion1_e_sg_l4_d4_t";

    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor, cfl);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 1", "[time_advance]", float,
                   double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-17 : 1e-8;
  SECTION("continuity1, explicit, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity1_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }

  SECTION("continuity1, explicit, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity1_fg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    bool const full_grid = true;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }

  SECTION("continuity1, explicit, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity1_sg_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 2", "[time_advance]", float,
                   double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-17 : 1e-7;
  SECTION("continuity2, explicit, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity2_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }

  SECTION("continuity2, explicit, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity2_fg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    bool const full_grid = true;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }

  SECTION("continuity2, explicit, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity2_sg_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 3", "[time_advance]", float,
                   double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-17 : 1e-8;
  SECTION("continuity3, explicit, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity3_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }

  SECTION("continuity3, explicit, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity3_sg_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 6", "[time_advance]", float,
                   double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-7;

  SECTION("continuity6, level 2, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity6_sg_l2_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p2", "[time_advance]",
                   float, double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-6;

  SECTION("fokkerplanck_1d_4p2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/fokkerplanck1_4p2_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p2, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p3", "[time_advance]",
                   float, double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-7;

  SECTION("fokkerplanck_1d_4p3, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    std::string const gold_base =
        "../testing/generated-inputs/time_advance/fokkerplanck1_4p3_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p3, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p1a", "[time_advance]",
                   float, double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-5;

  SECTION("fokkerplanck_1d_4p1a, level 2, degree 2, sparse grid")
  {
    int const degree            = 2;
    int const level             = 2;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "fokkerplanck1_4p1a_sg_l2_d2_t";

    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p1a, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_2d_complete", "[time_advance]",
                   float, double)
{
  /* FIXME - these tolerances are way too high. Different parameters are likely
     being used for gold data generation than here */
  TestType const tol_factor = std::is_same<TestType, double>::value ? 1e-6 : 1;

  SECTION("fokkerplanck_2d_complete, level 3, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 3;
    double const cfl = 1e-10;

    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "fokkerplanck2_complete_sg_l3_d3_t";
    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete, level, degree);
    bool const full_grid = false;

    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      tol_factor, cfl);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - diffusion 1", "[time_advance]",
                   double, float)
{
  TestType const cfl = 0.003;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-5;

  SECTION("diffusion1, implicit, sparse grid, level 2, degree 2")
  {
    int const degree     = 2;
    int const level      = 2;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion1/diffusion1_i_sg_l2_d2_t";

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor, cfl);
  }

  SECTION("diffusion1, implicit, sparse grid, level 3, degree 3")
  {
    int const degree     = 3;
    int const level      = 3;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion1/diffusion1_i_sg_l3_d3_t";

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor, cfl);
  }

  SECTION("diffusion1, implicit, sparse grid, level 4, degree 4")
  {
    int const degree     = 4;
    int const level      = 4;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion1/diffusion1_i_sg_l4_d4_t";

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor, cfl);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - diffusion 2", "[time_advance]",
                   double, float)
{
  TestType const cfl = 0.003;

  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-5;

  SECTION("diffusion2, implicit, sparse grid, level 2, degree 2")
  {
    int const degree     = 2;
    int const level      = 2;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2/diffusion2_i_sg_l2_d2_t";

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor, cfl);
  }

  SECTION("diffusion2, implicit, sparse grid, level 3, degree 3")
  {
    int const degree     = 3;
    int const level      = 3;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2/diffusion2_i_sg_l3_d3_t";

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor, cfl);
  }

  SECTION("diffusion2, implicit, sparse grid, level 4, degree 4")
  {
    int const degree     = 4;
    int const level      = 4;
    bool const full_grid = false;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "diffusion2/diffusion2_i_sg_l4_d4_t";

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor, cfl);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 1", "[time_advance]",
                   double)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-17 : 1e-8;
  bool const full_grid = false;

  SECTION("continuity1, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity1_implicit_l2_d2_t";

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor);
  }

  SECTION("continuity1, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity1_implicit_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor);
  }

  SECTION("continuity1, level 4, degree 3, sparse grid, iterative")
  {
    int const degree     = 3;
    int const level      = 4;
    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity1_implicit_l4_d3_t";
    double const cfl = 0.01;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor, cfl, solve_opts::gmres);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 2", "[time_advance]",
                   float, double)
{
  bool const full_grid = false;

  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-7;
  SECTION("continuity2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity2_implicit_l2_d2_t";
    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor);
  }

  SECTION("continuity2, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity2_implicit_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor);
  }

  SECTION("continuity2, level 4, degree 3, sparse grid, iterative")
  {
    int const degree = 3;
    int const level  = 4;
    double const cfl = 0.01;
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-14 : 1e-7;

    auto const gold_base = "../testing/generated-inputs/time_advance/"
                           "continuity2_implicit_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base,
                               full_grid, tol_factor, cfl, solve_opts::gmres);
  }
}
