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

template<typename P>
void relaxed_comparison(fk::vector<P> const &first, fk::vector<P> const &second)
{
  auto const diff = first - second;

  auto const abs_compare = [](P const a, P const b) {
    return (std::abs(a) < std::abs(b));
  };
  P const result =
      std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));

  P const tolerance = std::numeric_limits<P>::epsilon() *
                      (std::is_same<P, double>::value ? 1e5 : 1e3);
  REQUIRE(result <= tolerance);
}

int const num_steps          = 5;
int const workspace_limit_MB = 1000;
auto const tol_scale         = 1e4;


template<typename P>
void time_advance_test(int const level, int const degree, PDE<P> &pde,
                       int const num_steps, std::string const filepath,
                       bool const full_grid                            = false,
                       std::vector<std::string> const &additional_args = {})
{
  int const my_rank   = get_rank();
  int const num_ranks = get_num_ranks();

  std::vector<std::string> const args = [&additional_args, level, degree,
                                         full_grid]() {
    std::string const grid_str    = full_grid ? "-f" : "";
    std::vector<std::string> args = {"-l", std::to_string(level), "-d",
                                     std::to_string(degree), grid_str};
    args.insert(args.end(), additional_args.begin(), additional_args.end());
    return args;
  }();
  options const o = make_options(args);

  element_table const table(o, pde.num_dims);

  // can't run problem with fewer elements than ranks
  // this is asserted on in the distribution component
  if (num_ranks < static_cast<int>(table.size()))
  {
    return;
  }

  auto const plan    = get_plan(num_ranks, table);
  auto const subgrid = plan.at(my_rank);

  // -- set coeffs
  generate_all_coefficients(pde);

  // -- generate initial condition vector.
  fk::vector<P> const initial_condition = [&pde, &table, &subgrid, degree]() {
    std::vector<fk::vector<P>> initial_conditions;
    for (dimension<P> const &dim : pde.get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<P>(dim, dim.initial_condition));
    }
    return combine_dimensions(degree, table, subgrid.col_start,
                              subgrid.col_stop, initial_conditions);
  }();

  // -- generate sources.
  // these will be scaled later for time
  std::vector<fk::vector<P>> const initial_sources = [&pde, &table, &subgrid,
                                                      degree]() {
    std::vector<fk::vector<P>> initial_sources;
    for (source<P> const &source : pde.sources)
    {
      // gather contributions from each dim for this source, in wavelet space
      std::vector<fk::vector<P>> initial_sources_dim;
      for (int i = 0; i < pde.num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<P>(
            pde.get_dimensions()[i], source.source_funcs[i]));
      }
      // combine those contributions to form the unscaled source vector
      initial_sources.push_back(
          combine_dimensions(degree, table, subgrid.row_start, subgrid.row_stop,
                             initial_sources_dim));
    }
    return initial_sources;
  }();

  // -- prep workspace/chunks
  host_workspace<P> host_space(pde, subgrid);
  std::vector<element_chunk> const chunks = assign_elements(
      subgrid, get_num_chunks(subgrid, pde, workspace_limit_MB));
  rank_workspace<P> rank_space(pde, chunks);
  host_space.x = initial_condition;

  // -- time loop
  P const dt = pde.get_dt() * o.get_cfl();

  for (int i = 0; i < num_steps; ++i)
  {
    P const time = i * dt;
    explicit_time_advance(pde, table, initial_sources, host_space, rank_space,
                          chunks, plan, time, dt);
    std::string const file_path = filepath + std::to_string(i) + ".dat";

    int const degree       = pde.get_dimensions()[0].get_degree();
    int const segment_size = static_cast<int>(std::pow(degree, pde.num_dims));
    fk::vector<P> const gold =
        fk::vector<P>(read_vector_from_txt_file(file_path))
            .extract(subgrid.col_start * segment_size,
                     (subgrid.col_stop + 1) * segment_size - 1);
    relaxed_comparison(gold, host_space.x);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 1", "[time_advance]", float,
                   double)

{
  SECTION("continuity1, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity1_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    time_advance_test(level, degree, *pde, num_steps, gold_base);
  }
  SECTION("continuity1, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity1_fg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    bool const full_grid = true;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid);
  }

  SECTION("continuity1, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity1_sg_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid);
  }
}
TEMPLATE_TEST_CASE("time advance - continuity 2", "[time_advance]", float,
                   double)
{
  SECTION("continuity2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity2_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    time_advance_test(level, degree, *pde, num_steps, gold_base);
  }

  SECTION("continuity2, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity2_fg_l2_d2_t";
    bool const full_grid = true;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid);
  }

  SECTION("continuity2, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity2_sg_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 3", "[time_advance]", float,
                   double)
{
  SECTION("continuity3, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity3_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    time_advance_test(level, degree, *pde, num_steps, gold_base);
  }

  SECTION("continuity3, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity3_sg_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid);
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 6", "[time_advance]", float,
                   double)
{
  SECTION("continuity6, level 2, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/continuity6_sg_l2_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    time_advance_test(level, degree, *pde, num_steps, gold_base);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p2", "[time_advance]",
                   float, double)
{
  SECTION("fokkerplanck_1d_4p2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/fokkerplanck1_4p2_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p2, level, degree);
    time_advance_test(level, degree, *pde, num_steps, gold_base);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p3", "[time_advance]",
                   float, double)
{
  SECTION("fokkerplanck_1d_4p3, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    std::vector<std::string> const addtl_args = {"-c", std::to_string(0.01)};
    std::string const gold_base =
        "../testing/generated-inputs/time_advance/fokkerplanck1_4p3_sg_l2_d2_t";
    auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p3, level, degree);

    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      addtl_args);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p1a", "[time_advance]",
                   float, double)
{
  SECTION("fokkerplanck_1d_4p1a, level 2, degree 2, sparse grid")
  {
    int const degree            = 2;
    int const level             = 2;
    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "fokkerplanck1_4p1a_sg_l2_d2_t";

    std::vector<std::string> const addtl_args = {"-c", std::to_string(0.01)};
    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p1a, level, degree);

    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      addtl_args);
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_2d_complete", "[time_advance]",
                   float, double)
{
  SECTION("fokkerplanck_2d_complete, level 3, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 3;

    std::string const gold_base = "../testing/generated-inputs/time_advance/"
                                  "fokkerplanck2_complete_sg_l3_d3_t";
    std::vector<std::string> const addtl_args = {
        "-c", to_string_with_precision(1e-10, 16)};
    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete, level, degree);

    bool const full_grid = false;
    time_advance_test(level, degree, *pde, num_steps, gold_base, full_grid,
                      addtl_args);
  }
}

template<typename P>
void implicit_time_advance_test(int const level, int const degree, PDE<P> &pde,
                                int const num_steps, std::string const filepath,
                                bool const full_grid = false)
{
  int const my_rank   = get_rank();
  int const num_ranks = get_num_ranks();
  if (num_ranks > 1)
  {
    // distributed implicit stepping not implemented
    ignore(level);
    ignore(degree);
    ignore(pde);
    ignore(num_steps);
    ignore(filepath);
    ignore(full_grid);
    return;
  }

  std::string const grid_str = full_grid ? "-f" : "";
  options const o =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree),
                    "--implicit", grid_str});

  element_table const table(o, pde.num_dims);
  auto const plan    = get_plan(num_ranks, table);
  auto const subgrid = plan.at(my_rank);

  // -- set coeffs
  generate_all_coefficients(pde);

  // -- generate initial condition vector.
  P const initial_scale = 1.0;
  std::vector<fk::vector<P>> initial_conditions;
  for (dimension<P> const &dim : pde.get_dimensions())
  {
    initial_conditions.push_back(
        forward_transform<P>(dim, dim.initial_condition));
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
          pde.get_dimensions()[i], source.source_funcs[i]));
    }

    initial_sources.push_back(
        combine_dimensions(degree, table, subgrid.row_start, subgrid.row_stop,
                           initial_sources_dim, initial_scale));
  }

  // -- prep workspace/chunks
  host_workspace<P> host_space(pde, subgrid);
  std::vector<element_chunk> const chunks = assign_elements(
      subgrid, get_num_chunks(subgrid, pde, workspace_limit_MB));
  rank_workspace<P> rank_space(pde, chunks);
  host_space.x = initial_condition;

  // -- time loop
  P const dt = pde.get_dt() * o.get_cfl();

  for (int i = 0; i < num_steps; ++i)
  {
    P const time = i * dt;
    implicit_time_advance(pde, table, initial_sources, host_space, chunks, time,
                          dt);

    std::string const file_path = filepath + std::to_string(i) + ".dat";

    fk::vector<P> const gold =
        fk::vector<P>(read_vector_from_txt_file(file_path));

    relaxed_comparison(gold, host_space.x);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 1", "[time_advance]",
                   float, double)
{
  SECTION("continuity1, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    auto const gold_base = "../testing/generated-inputs/implicit_time_advance/"
                           "continuity1_implicit_l2_d2_t";
    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base);
  }

  SECTION("continuity1, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto const gold_base = "../testing/generated-inputs/implicit_time_advance/"
                           "continuity1_implicit_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base);
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 2", "[time_advance]",
                   float, double)
{
  SECTION("continuity2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    auto const gold_base = "../testing/generated-inputs/implicit_time_advance/"
                           "continuity2_implicit_l2_d2_t";
    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base);
  }

  SECTION("continuity2, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto const gold_base = "../testing/generated-inputs/implicit_time_advance/"
                           "continuity2_implicit_l4_d3_t";
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    implicit_time_advance_test(level, degree, *pde, num_steps, gold_base);
  }
}
