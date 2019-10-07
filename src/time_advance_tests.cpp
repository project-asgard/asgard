#include "chunk.hpp"
#include "coefficients.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>
#include <random>

TEMPLATE_TEST_CASE("time advance - continuity 1", "[time_advance]", float,
                   double)

{
  auto const relaxed_comparison = [](auto const &first, auto const &second) {
    auto const diff = first - second;

    auto const abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    if constexpr (std::is_same<TestType, double>::value)
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e5;
      REQUIRE(result <= tol);
    }
    else
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE(result <= tol);
    }
  };

  int const test_steps = 5;

  SECTION("continuity1, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }
    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity1_sg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }

  SECTION("continuity1, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree), "-f"});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity1_fg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
  SECTION("continuity1, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity1_sg_l4_d3_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 2", "[time_advance]", float,
                   double)

{
  auto const relaxed_comparison = [](auto const &first, auto const &second) {
    auto const diff = first - second;

    auto const abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    if constexpr (std::is_same<TestType, double>::value)
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e5;
      REQUIRE(result <= tol);
    }
    else
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE(result <= tol);
    }
  };

  int const test_steps = 5;
  SECTION("continuity2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity2_sg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }

  SECTION("continuity2, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree), "-f"});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity2_fg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
  SECTION("continuity2, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity2_sg_l4_d3_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 3", "[time_advance]", float,
                   double)
{
  auto const relaxed_comparison = [](auto const &first, auto const &second) {
    auto const diff = first - second;

    auto const abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    if constexpr (std::is_same<TestType, double>::value)
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e5;
      REQUIRE(result <= tol);
    }
    else
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE(result <= tol);
    }
  };

  int const test_steps = 5;
  SECTION("continuity3, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity3_sg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }

  SECTION("continuity3, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity3_sg_l4_d3_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
}

TEMPLATE_TEST_CASE("time advance - continuity 6", "[time_advance]", float,
                   double)
{
  auto const relaxed_comparison = [](auto const &first, auto const &second) {
    auto const diff = first - second;

    auto const abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    if constexpr (std::is_same<TestType, double>::value)
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e5;
      REQUIRE(result <= tol);
    }
    else
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE(result <= tol);
    }
  };
  int const test_steps = 5;
  SECTION("continuity6, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity6_sg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }

  SECTION("continuity6, level 2, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity6_sg_l2_d3_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p2",
                   "[time_advance,fokker]", float, double)
{
  auto const relaxed_comparison = [](auto const &first, auto const &second) {
    auto const diff = first - second;

    auto const abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    if constexpr (std::is_same<TestType, double>::value)
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e5;
      REQUIRE(result <= tol);
    }
    else
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE(result <= tol);
    }
  };

  int const test_steps = 5;

  SECTION("fokkerplanck_1_4p2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }
    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path = "../testing/generated-inputs/time_advance/"
                                    "fokkerplanck1_4p2_sg_l2_d2_t" +
                                    std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }

  SECTION("fokkerplanck_1_4p2, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree), "-f"});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType, int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path = "../testing/generated-inputs/time_advance/"
                                    "fokkerplanck1_4p2_fg_l2_d2_t" +
                                    std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p1a", "[time_advance]",
                   float, double)
{
  auto const relaxed_comparison = [](auto const &first, auto const &second) {
    auto const diff = first - second;

    auto const abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    if constexpr (std::is_same<TestType, double>::value)
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e5;
      REQUIRE(result <= tol);
    }
    else
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE(result <= tol);
    }
  };

  int const test_steps = 5;

  SECTION("fokkerplanck_1_4p1a, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p1a, level, degree);

    options const o =
        make_options({"-l", std::to_string(level), "-d", std::to_string(degree),
                      "-c", std::to_string(0.01)});

    element_table<int> const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType,int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/fokkerplanck1_4p1a_sg_l" +
          std::to_string(level) + "_d" + std::to_string(degree) + "_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
}

TEMPLATE_TEST_CASE("time advance - fokkerplanck_1d_4p3", "[time_advance]",
                   float, double)
{
  auto const relaxed_comparison = [](auto const &first, auto const &second) {
    auto const diff = first - second;

    auto const abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    if constexpr (std::is_same<TestType, double>::value)
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e5;
      REQUIRE(result <= tol);
    }
    else
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE(result <= tol);
    }
  };

  int const test_steps = 5;

  SECTION("fokkerplanck_1_4p3, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p3, level, degree);

    options const o =
        make_options({"-l", std::to_string(level), "-d", std::to_string(degree),
                      "-c", std::to_string(0.01)});

    element_table const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType,long long int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/fokkerplanck1_4p3_sg_l" +
          std::to_string(level) + "_d" + std::to_string(degree) + "_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 1", "[time_advance]",
                   float, double)

{
  auto const relaxed_comparison = [](auto const &first, auto const &second) {
    auto const diff = first - second;

    auto const abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    if constexpr (std::is_same<TestType, double>::value)
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e5;
      REQUIRE(result <= tol);
    }
    else
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE(result <= tol);
    }
  };

  int const test_steps = 5;

  SECTION("continuity1, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    options const o = make_options({"-l", std::to_string(level), "-d",
                                    std::to_string(degree), "--implicit"});

    element_table const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }
    // -- prep workspace/chunks
    host_workspace<TestType,long long int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;

      implicit_time_advance(*pde, table, initial_sources, host_space, chunks,
                            time, dt);

      std::string const file_path =
          "../testing/generated-inputs/implicit_time_advance/"
          "continuity1_implicit_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }

  SECTION("continuity1, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    options const o = make_options({"-l", std::to_string(level), "-d",
                                    std::to_string(degree), "--implicit"});

    element_table const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType,long long int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      implicit_time_advance(*pde, table, initial_sources, host_space, chunks,
                            time, dt);

      std::string const file_path =
          "../testing/generated-inputs/implicit_time_advance/"
          "continuity1_implicit_l4_d3_t" +
          std::to_string(i) + ".dat";

      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
}

TEMPLATE_TEST_CASE("implicit time advance - continuity 2", "[time_advance]",
                   float, double)
{
  auto const relaxed_comparison = [](auto const &first, auto const &second) {
    auto const diff = first - second;

    auto const abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    if constexpr (std::is_same<TestType, double>::value)
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e5;
      REQUIRE(result <= tol);
    }
    else
    {
      TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE(result <= tol);
    }
  };

  int const test_steps = 5;
  SECTION("continuity2, level 2, degree 2, sparse grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType,long long int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      implicit_time_advance(*pde, table, initial_sources, host_space, chunks,
                            time, dt);

      std::string const file_path =
          "../testing/generated-inputs/implicit_time_advance/"
          "continuity2_implicit_l2_d2_t" +
          std::to_string(i) + ".dat";

      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }

  SECTION("continuity2, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    options const o = make_options({"-l", std::to_string(level), "-d",
                                    std::to_string(degree), "--implicit"});

    element_table const table(o, pde->num_dims);

    // set coeffs
    generate_all_coefficients(*pde);

    // -- generate initial condition vector.
    TestType const initial_scale = 1.0;
    std::vector<fk::vector<TestType>> initial_conditions;
    for (dimension<TestType> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<TestType>(dim, dim.initial_condition));
    }
    fk::vector<TestType> const initial_condition =
        combine_dimensions(degree, table, initial_conditions, initial_scale);

    // -- generate sources.
    // these will be scaled later for time
    std::vector<fk::vector<TestType>> initial_sources;

    for (source<TestType> const &source : pde->sources)
    {
      std::vector<fk::vector<TestType>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<TestType>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }

      initial_sources.push_back(combine_dimensions(
          degree, table, initial_sources_dim, initial_scale));
    }

    // -- prep workspace/chunks
    host_workspace<TestType,long long int> host_space(*pde, table);
    std::vector<element_chunk> const chunks =
        assign_elements(table, get_num_chunks(table, *pde));
    rank_workspace<TestType> rank_space(*pde, chunks);
    host_space.x = initial_condition;

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      implicit_time_advance(*pde, table, initial_sources, host_space, chunks,
                            time, dt);

      std::string const file_path =
          "../testing/generated-inputs/implicit_time_advance/"
          "continuity2_implicit_l4_d3_t" +
          std::to_string(i) + ".dat";

      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, host_space.fx);
    }
  }
}
