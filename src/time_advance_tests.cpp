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

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity1_sg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
    }
  }

  SECTION("continuity1, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree), "-f"});

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity1_fg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
    }
  }
  SECTION("continuity1, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity1_sg_l4_d3_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
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

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity2_sg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
    }
  }

  SECTION("continuity2, level 2, degree 2, full grid")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree), "-f"});

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity2_fg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
    }
  }
  SECTION("continuity2, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity2_sg_l4_d3_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
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

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity3_sg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
    }
  }

  SECTION("continuity3, level 4, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 4;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity3_sg_l4_d3_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
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

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }
    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity6_sg_l2_d2_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
    }
  }

  SECTION("continuity6, level 2, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const table(o, pde->num_dims);

    // set coeffs
    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }
    explicit_system<TestType> system(*pde, table);
    auto const batches = build_work_set(*pde, table, system);

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

    system.batch_input = initial_condition;

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

    // -- time loop
    TestType const dt = pde->get_dt() * o.get_cfl();

    for (int i = 0; i < test_steps; ++i)
    {
      TestType const time = i * dt;
      explicit_time_advance(*pde, initial_sources, system, batches, time, dt);

      std::string const file_path =
          "../testing/generated-inputs/time_advance/continuity6_sg_l2_d3_t" +
          std::to_string(i) + ".dat";
      fk::vector<TestType> const gold =
          fk::vector<TestType>(read_vector_from_txt_file(file_path));

      relaxed_comparison(gold, system.batch_output);
    }
  }
}
