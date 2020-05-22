#include "matlab_utilities.hpp"
#include "pde_factory.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"
#include <vector>

// determined empirically 11/19
// lowest epsilon multiplier for which component tests pass
static auto const pde_eps_multiplier = 1e2;

template<typename P>
void test_initial_condition(PDE<P> const &pde, std::string const base_dir,
                            P const test_val)
{
  fk::vector<P> const x = {test_val};

  for (int i = 0; i < pde.num_dims; ++i)
  {
    P const gold = read_scalar_from_txt_file(base_dir + "initial_dim" +
                                             std::to_string(i) + ".dat");

    P const fx = pde.get_dimensions()[i].initial_condition(x, 0)(0);
    relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
  }
}

template<typename P>
void test_exact_solution(PDE<P> const &pde, std::string const base_dir,
                         P const test_val)
{
  fk::vector<P> const x = {test_val};

  for (int i = 0; i < pde.num_dims; ++i)
  {
    P const gold = read_scalar_from_txt_file(base_dir + "exact_dim" +
                                             std::to_string(i) + ".dat");
    P const fx   = pde.exact_vector_funcs[i](x, 0)(0);
    relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
  }

  P const gold = read_scalar_from_txt_file(base_dir + "exact_time.dat");
  P const fx   = pde.exact_time(x(0));
  relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
}

TEMPLATE_TEST_CASE("testing diffusion 2 implementations", "[pde]", double,
                   float)
{
  auto const pde             = make_PDE<TestType>(PDE_opts::diffusion_2, 3, 2);
  std::string const base_dir = "../testing/generated-inputs/pde/diffusion_2_";
  fk::vector<TestType> const x = {4.2};

  SECTION("diffusion 2 initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, 4.2);
  }

  SECTION("diffusion 2 exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, 4.2);
  }

  SECTION("diffusion 2 dt")
  {
    TestType const gold = read_scalar_from_txt_file(base_dir + "dt.dat");
    TestType const dt   = pde->get_dt();
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing diffusion 1 implementations", "[pde]", double,
                   float)
{
  auto const pde             = make_PDE<TestType>(PDE_opts::diffusion_1, 3, 2);
  std::string const base_dir = "../testing/generated-inputs/pde/diffusion_1_";
  fk::vector<TestType> const x = {4.2};

  SECTION("diffusion 1 initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, 4.2);
  }

  SECTION("diffusion 1 exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, 4.2);
  }

  SECTION("diffusion 1 source functions")
  {
    for (int i = 0; i < pde->num_sources; ++i)
    {
      std::string const source_string =
          base_dir + "source" + std::to_string(i) + "_";
      for (int j = 0; j < pde->num_dims; ++j)
      {
        std::string const full_path =
            source_string + "dim" + std::to_string(j) + ".dat";
        TestType const gold = read_scalar_from_txt_file(full_path);
        TestType const fx   = pde->sources[i].source_funcs[j](x, 0)(0);
        relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
      }
      TestType const gold =
          read_scalar_from_txt_file(source_string + "time.dat");
      TestType const fx = pde->sources[i].time_func(x(0));
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("diffusion 1 dt")
  {
    TestType const gold = read_scalar_from_txt_file(base_dir + "dt.dat");
    TestType const dt   = pde->get_dt();
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing contuinity 1 implementations", "[pde]", double,
                   float)
{
  auto const pde             = make_PDE<TestType>(PDE_opts::continuity_1);
  std::string const base_dir = "../testing/generated-inputs/pde/continuity_1_";
  fk::vector<TestType> const x = {1.1};

  SECTION("continuity 1 initial condition functions")
  {
    for (int i = 0; i < pde->num_dims; ++i)
    {
      TestType const gold = read_scalar_from_txt_file(
          base_dir + "initial_dim" + std::to_string(i) + ".dat");
      TestType const fx = pde->get_dimensions()[i].initial_condition(x, 0)(0);
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("continuity 1 exact solution functions")
  {
    for (int i = 0; i < pde->num_dims; ++i)
    {
      TestType const gold = read_scalar_from_txt_file(
          base_dir + "exact_dim" + std::to_string(i) + ".dat");
      TestType const fx = pde->exact_vector_funcs[i](x, 0)(0);
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
    TestType const gold =
        read_scalar_from_txt_file(base_dir + "exact_time.dat");
    TestType const fx = pde->exact_time(x(0));
    relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
  }
  SECTION("continuity 1 source functions")
  {
    for (int i = 0; i < pde->num_sources; ++i)
    {
      std::string const source_string =
          base_dir + "source" + std::to_string(i) + "_";
      for (int j = 0; j < pde->num_dims; ++j)
      {
        std::string const full_path =
            source_string + "dim" + std::to_string(j) + ".dat";
        TestType const gold = read_scalar_from_txt_file(full_path);
        TestType const fx   = pde->sources[i].source_funcs[j](x, 0)(0);
        relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
      }
      TestType const gold =
          read_scalar_from_txt_file(source_string + "time.dat");
      TestType const fx = pde->sources[i].time_func(x(0));
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("continuity 1 dt")
  {
    TestType const gold = read_scalar_from_txt_file(base_dir + "dt.dat");
    TestType const dt   = pde->get_dt();
    REQUIRE(dt == gold);
  }
}
TEMPLATE_TEST_CASE("testing contuinity 2 implementations", "[pde]", double,
                   float)
{
  auto const pde             = make_PDE<TestType>(PDE_opts::continuity_2);
  std::string const base_dir = "../testing/generated-inputs/pde/continuity_2_";
  fk::vector<TestType> const x = {2.2};

  SECTION("continuity 2 initial condition functions")
  {
    for (int i = 0; i < pde->num_dims; ++i)
    {
      TestType const gold = read_scalar_from_txt_file(
          base_dir + "initial_dim" + std::to_string(i) + ".dat");
      TestType const fx = pde->get_dimensions()[i].initial_condition(x, 0)(0);
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("continuity 2 exact solution functions")
  {
    for (int i = 0; i < pde->num_dims; ++i)
    {
      TestType const gold = read_scalar_from_txt_file(
          base_dir + "exact_dim" + std::to_string(i) + ".dat");
      TestType const fx = pde->exact_vector_funcs[i](x, 0)(0);
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
    TestType const gold =
        read_scalar_from_txt_file(base_dir + "exact_time.dat");
    TestType const fx = pde->exact_time(x(0));
    relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
  }
  SECTION("continuity 2 source functions")
  {
    for (int i = 0; i < pde->num_sources; ++i)
    {
      std::string const source_string =
          base_dir + "source" + std::to_string(i) + "_";
      for (int j = 0; j < pde->num_dims; ++j)
      {
        std::string const full_path =
            source_string + "dim" + std::to_string(j) + ".dat";
        TestType const gold = read_scalar_from_txt_file(full_path);
        TestType const fx   = pde->sources[i].source_funcs[j](x, 0)(0);
        relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
      }
      TestType const gold =
          read_scalar_from_txt_file(source_string + "time.dat");
      TestType const fx = pde->sources[i].time_func(x(0));
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("continuity 2 dt")
  {
    TestType const gold = read_scalar_from_txt_file(base_dir + "dt.dat");
    TestType const dt   = pde->get_dt();
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing contuinity 3 implementations", "[pde]", double,
                   float)
{
  auto const pde             = make_PDE<TestType>(PDE_opts::continuity_3);
  std::string const base_dir = "../testing/generated-inputs/pde/continuity_3_";
  fk::vector<TestType> const x = {3.3};

  SECTION("continuity 3 initial condition functions")
  {
    for (int i = 0; i < pde->num_dims; ++i)
    {
      TestType const gold = read_scalar_from_txt_file(
          base_dir + "initial_dim" + std::to_string(i) + ".dat");

      TestType const fx = pde->get_dimensions()[i].initial_condition(x, 0)(0);
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("continuity 3 exact solution functions")
  {
    for (int i = 0; i < pde->num_dims; ++i)
    {
      TestType const gold = read_scalar_from_txt_file(
          base_dir + "exact_dim" + std::to_string(i) + ".dat");
      TestType const fx = pde->exact_vector_funcs[i](x, 0)(0);
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
    TestType const gold =
        read_scalar_from_txt_file(base_dir + "exact_time.dat");
    TestType const fx = pde->exact_time(x(0));
    relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
  }
  SECTION("continuity 3 source functions")
  {
    for (int i = 0; i < pde->num_sources; ++i)
    {
      std::string const source_string =
          base_dir + "source" + std::to_string(i) + "_";
      for (int j = 0; j < pde->num_dims; ++j)
      {
        std::string const full_path =
            source_string + "dim" + std::to_string(j) + ".dat";
        TestType const gold = read_scalar_from_txt_file(full_path);
        TestType const fx   = pde->sources[i].source_funcs[j](x, 0)(0);
        relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
      }

      TestType const gold =
          read_scalar_from_txt_file(source_string + "time.dat");
      TestType const fx = pde->sources[i].time_func(x(0));
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("continuity 3 dt")
  {
    TestType const gold = read_scalar_from_txt_file(base_dir + "dt.dat");
    TestType const dt   = pde->get_dt();
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing contuinity 6 implementations", "[pde]", double,
                   float)
{
  auto const pde             = make_PDE<TestType>(PDE_opts::continuity_6);
  std::string const base_dir = "../testing/generated-inputs/pde/continuity_6_";
  fk::vector<TestType> const x = {6.6};

  SECTION("continuity 6 initial condition functions")
  {
    for (int i = 0; i < pde->num_dims; ++i)
    {
      TestType const gold = read_scalar_from_txt_file(
          base_dir + "initial_dim" + std::to_string(i) + ".dat");

      TestType const fx = pde->get_dimensions()[i].initial_condition(x, 0)(0);
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("continuity 6 exact solution functions")
  {
    for (int i = 0; i < pde->num_dims; ++i)
    {
      TestType const gold = read_scalar_from_txt_file(
          base_dir + "exact_dim" + std::to_string(i) + ".dat");
      TestType const fx = pde->exact_vector_funcs[i](x, 0)(0);
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
    TestType const gold =
        read_scalar_from_txt_file(base_dir + "exact_time.dat");
    TestType const fx = pde->exact_time(x(0));
    relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
  }
  SECTION("continuity 6 source functions")
  {
    for (int i = 0; i < pde->num_sources; ++i)
    {
      std::string const source_string =
          base_dir + "source" + std::to_string(i) + "_";
      for (int j = 0; j < pde->num_dims; ++j)
      {
        std::string const full_path =
            source_string + "dim" + std::to_string(j) + ".dat";
        TestType const gold = read_scalar_from_txt_file(full_path);
        TestType const fx   = pde->sources[i].source_funcs[j](x, 0)(0);
        relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
      }

      TestType const gold =
          read_scalar_from_txt_file(source_string + "time.dat");
      TestType const fx = pde->sources[i].time_func(x(0));
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("continuity 6 dt")
  {
    TestType const gold = read_scalar_from_txt_file(base_dir + "dt.dat");
    TestType const dt   = pde->get_dt();
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing fokkerplanck2_complete implementations", "[pde]",
                   double, float)
{
  auto const pde = make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete);
  std::string const base_dir =
      "../testing/generated-inputs/pde/fokkerplanck2_complete_";
  fk::vector<TestType> const x = {0.5};

  SECTION("fokkerplanck2_complete initial condition functions")
  {
    for (int i = 0; i < pde->num_dims; ++i)
    {
      TestType const gold = read_scalar_from_txt_file(
          base_dir + "initial_dim" + std::to_string(i) + ".dat");
      TestType const fx = pde->get_dimensions()[i].initial_condition(x, 0)(0);
      relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
    }
  }

  SECTION("fokkerplanck2_complete dt")
  {
    TestType const gold = read_scalar_from_txt_file(base_dir + "dt.dat");
    TestType const dt   = pde->get_dt();
    REQUIRE(dt == gold);
  }
}

template<typename P>
void test_coefficients(PDE<P> &pde, std::string const gold_path,
                       P const tol_factor = 1e-15, bool const rotate = true)
{
  // FIXME assume uniform level and degree
  dimension<P> const &d           = pde.get_dimensions()[0];
  int const level                 = d.get_level();
  int const degree                = d.get_degree();
  std::string const filename_base = gold_path + "_l" + std::to_string(level) +
                                    "_d" + std::to_string(degree) + "_";

  P const time = 1.0;
  pde.regenerate_coefficients(time, rotate);

  for (int t = 0; t < pde.num_terms; ++t)
  {
    for (int d = 0; d < pde.num_dims; ++d)
    {
      std::string const filename = filename_base + std::to_string(t + 1) + "_" +
                                   std::to_string(d + 1) + ".dat";

      fk::matrix<P> const gold =
          fk::matrix<P>(read_matrix_from_txt_file(filename));
      fk::matrix<P> const &test = pde.get_coefficients(t, d).clone_onto_host();

      rmse_comparison(gold, test, tol_factor);
    }
  }
}

TEMPLATE_TEST_CASE("diffusion 2 (single term)", "[coefficients]", double, float)
{
  std::string const gold_path =
      "../testing/generated-inputs/coefficients/diffusion2/coefficients";

  TestType const tol_factor = 1e-13;

  SECTION("level 2, degree 2")
  {
    int const level  = 2;
    int const degree = 2;

    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);

    test_coefficients<TestType>(*pde, gold_path, tol_factor);
  }

  SECTION("level 4, degree 4")
  {
    int const level  = 4;
    int const degree = 4;

    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);

    test_coefficients<TestType>(*pde, gold_path, tol_factor);
  }

  SECTION("level 5, degree 5")
  {
    int const level  = 5;
    int const degree = 5;

    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);

    test_coefficients<TestType>(*pde, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("diffusion 1 (single term)", "[coefficients]", double, float)
{
  std::string const gold_path =
      "../testing/generated-inputs/coefficients/diffusion1/coefficients";

  TestType const tol_factor = 1e-13;

  SECTION("level 2, degree 2")
  {
    int const level  = 2;
    int const degree = 2;

    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    std::string const gold_path =
        "../testing/generated-inputs/coefficients/diffusion1/coefficients";

    test_coefficients<TestType>(*pde, gold_path, tol_factor);
  }

  SECTION("level 4, degree 4")
  {
    int const level  = 4;
    int const degree = 4;

    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    std::string const gold_path =
        "../testing/generated-inputs/coefficients/diffusion1/coefficients";

    test_coefficients<TestType>(*pde, gold_path, tol_factor);
  }

  SECTION("level 5, degree 5")
  {
    int const level  = 5;
    int const degree = 5;

    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    std::string const gold_path =
        "../testing/generated-inputs/coefficients/diffusion1/coefficients";

    test_coefficients<TestType>(*pde, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 1 (single term)", "[coefficients]", double,
                   float)
{
  auto pde = make_PDE<TestType>(PDE_opts::continuity_1);
  std::string const gold_path =
      "../testing/generated-inputs/coefficients/continuity1/coefficients";

  TestType const tol_factor = 1e-15;

  test_coefficients<TestType>(*pde, gold_path, tol_factor);
}

TEMPLATE_TEST_CASE("continuity 2 terms", "[coefficients]", double, float)
{
  int const level  = 4;
  int const degree = 3;
  auto pde         = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
  std::string const gold_path =
      "../testing/generated-inputs/coefficients/continuity2_coefficients";

  TestType const tol_factor = 1e-14;

  test_coefficients(*pde, gold_path, tol_factor);
}

TEMPLATE_TEST_CASE("continuity 3 terms - norotate", "[coefficients]", double,
                   float)
{
  int const level  = 4;
  int const degree = 4;
  auto pde         = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
  std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                "continuity3_coefficients_norotate";
  bool const rotate         = false;
  TestType const tol_factor = 1e-15;

  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("continuity 3 terms", "[coefficients]", double, float)
{
  int const level  = 4;
  int const degree = 4;
  auto pde         = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
  std::string const gold_path =
      "../testing/generated-inputs/coefficients/continuity3_coefficients";
  bool const rotate         = true;
  TestType const tol_factor = 1e-14;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("continuity 6 terms", "[coefficients]", double, float)
{
  int const level  = 2;
  int const degree = 4;
  auto pde         = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
  std::string const gold_path =
      "../testing/generated-inputs/coefficients/continuity6_coefficients";
  bool const rotate = true;

  TestType const tol_factor = 1e-14;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p2 terms", "[coefficients]", double, float)
{
  int const level  = 3;
  int const degree = 4;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p2, level, degree);
  std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p2_coefficients";
  bool const rotate = true;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-7;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p2 terms - norotate", "[coefficients]",
                   double, float)
{
  int const level  = 3;
  int const degree = 4;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p2, level, degree);
  std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p2_coefficients_norotate";
  bool const rotate = false;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-7;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p3 terms", "[coefficients]", double, float)
{
  int const level  = 4;
  int const degree = 3;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p3, level, degree);
  std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p3_coefficients";
  bool const rotate = false;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-7;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p4 terms", "[coefficients]", double, float)
{
  int const level  = 3;
  int const degree = 4;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p4, level, degree);
  std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p4_coefficients";
  bool const rotate = false;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-7;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p5 terms", "[coefficients]", double, float)
{
  int const level  = 5;
  int const degree = 2;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p5, level, degree);
  std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p5_coefficients";
  bool const rotate = false;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-7;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck2_complete terms", "[coefficients]", double,
                   float)
{
  int const level  = 3;
  int const degree = 4;
  auto pde =
      make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete, level, degree);
  std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck2_complete_coefficients";
  bool const rotate = true;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-12 : 1e-5;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck2_complete terms - norotate", "[coefficients]",
                   double, float)
{
  int const level  = 3;
  int const degree = 4;
  auto pde =
      make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete, level, degree);

  std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck2_complete_coefficients_norotate";
  bool const rotate = false;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-12 : 1e-5;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}
