#include "coefficients.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

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
  generate_all_coefficients(pde, time, rotate);

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

/* Captain! */
TEMPLATE_TEST_CASE("fokkerplanck2_complete terms", "[coefficients]", double,
                   float)
{
  SECTION("fokkerplanck_2d_complete, level 3, degree 3")
  {
    int const level  = 3;
    int const degree = 3;
    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete, level, degree);
    std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                  "fokkerplanck2_complete_coefficients";
    bool const rotate = true;
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-10 : 1e-6;
    test_coefficients(*pde, gold_path, tol_factor, rotate);
  }

  SECTION("fokkerplanck_2d_complete, level 4, degree 4")
  {
    int const level  = 4;
    int const degree = 4;
    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete, level, degree);
    std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                  "fokkerplanck2_complete_coefficients";
    bool const rotate = true;
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-11 : 1e-4;
    test_coefficients(*pde, gold_path, tol_factor, rotate);
  }

  SECTION("fokkerplanck_2d_complete, level 5, degree 5")
  {
    int const level  = 5;
    int const degree = 5;
    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete, level, degree);
    std::string const gold_path = "../testing/generated-inputs/coefficients/"
                                  "fokkerplanck2_complete_coefficients";
    bool const rotate = true;
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-10 : 1e-4;
    test_coefficients(*pde, gold_path, tol_factor, rotate);
  }
}
