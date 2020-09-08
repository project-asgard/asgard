#include "coefficients.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

template<typename P>
void test_coefficients(PDE<P> &pde, std::string const gold_path,
                       P const tol_factor = 1e-15, bool const rotate = true)
{
  // FIXME this test assumes uniform level and degree
  auto const &d                 = pde.get_dimensions()[0];
  auto const level              = d.get_level();
  auto const degree             = d.get_degree();
  auto const degrees_freedom_1d = degree * fm::two_raised_to(level);

  auto const filename_base = gold_path + "_l" + std::to_string(level) + "_d" +
                             std::to_string(degree) + "_";

  P const time = 1.0;

  auto const opts =
      make_options({"-d", std::to_string(degree), "-l", std::to_string(level)});

  basis::wavelet_transform<P, resource::host> const transformer(opts, pde);
  generate_all_coefficients(pde, transformer, time, rotate);
  for (auto t = 0; t < pde.num_terms; ++t)
  {
    for (auto d = 0; d < pde.num_dims; ++d)
    {
      auto const filename = filename_base + std::to_string(t + 1) + "_" +
                                   std::to_string(d + 1) + ".dat";

      auto const gold =
          fk::matrix<P>(read_matrix_from_txt_file(filename));
      auto const full_coeff = pde.get_coefficients(t, d).clone_onto_host();
      fk::matrix<P, mem_type::const_view> const test(
          full_coeff, 0, degrees_freedom_1d-1, 0, degrees_freedom_1d-1);
      //fk::matrix<P, mem_type::const_view>(gold, 0, 7, 0, 7).print("gold");
      //test.print("test");
      rmse_comparison(gold, test, tol_factor);
    }
  }
}

TEMPLATE_TEST_CASE("diffusion 2 (single term)", "[coefficients]", double,
float)
{
  auto const gold_path =
      "../testing/generated-inputs/coefficients/diffusion2_coefficients";

  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-14 : 1e-4;

  SECTION("level 3, degree 5")
  {
    auto const level  = 3;
    auto const degree = 5;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);
    test_coefficients<TestType>(*pde, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("diffusion 1 (single term)", "[coefficients]", double, float)
{
  auto const gold_path =
      "../testing/generated-inputs/coefficients/diffusion1_coefficients";

  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-12 : 1e-3;

  SECTION("level 5, degree 6")
  {
    auto const level  = 5;
    auto const degree = 6;
    auto pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    test_coefficients<TestType>(*pde, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 1 (single term)", "[coefficients]", double,
                   float)
{
  auto pde = make_PDE<TestType>(PDE_opts::continuity_1);
  auto const gold_path =
      "../testing/generated-inputs/coefficients/continuity1_coefficients";
  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-14 : 1e-5;
  test_coefficients<TestType>(*pde, gold_path, tol_factor);
}

TEMPLATE_TEST_CASE("continuity 2 terms", "[coefficients]", double, float)
{
  auto const level  = 4;
  auto const degree = 3;
  auto pde         = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
  auto const gold_path =
      "../testing/generated-inputs/coefficients/continuity2_coefficients";

  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-14 : 1e-6;

  test_coefficients(*pde, gold_path, tol_factor);
}

TEMPLATE_TEST_CASE("continuity 3 terms", "[coefficients]", double, float)
{
  auto const level  = 4;
  auto const degree = 4;
  auto pde         = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
  auto const gold_path =
      "../testing/generated-inputs/coefficients/continuity3_coefficients";
  auto const rotate         = true;
  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-14 : 1e-5;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("continuity 6 terms", "[coefficients]", double, float)
{
  auto const level  = 2;
  auto const degree = 4;
  auto pde         = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
  auto const gold_path =
      "../testing/generated-inputs/coefficients/continuity6_coefficients";
  auto const rotate = true;

  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-14 : 1e-5;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}


TEMPLATE_TEST_CASE("fokkerplanck1_4p1a terms", "[coefficients]", double, float)
{
  auto const level  = 4;
  auto const degree = 3;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p1a, level, degree);
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p1a_coefficients";
  auto const rotate = true;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-5;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p2 terms", "[coefficients]", double, float)
{
  auto const level  = 5;
  auto const degree = 2;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p2, level, degree);
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p2_coefficients";
  auto const rotate = true;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-5;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p3 terms", "[coefficients]", double, float)
{
  auto const level  = 2;
  auto const degree = 5;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p3, level, degree);
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p3_coefficients";
  auto const rotate = true;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-4;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p4 terms", "[coefficients]", double, float)
{
  auto const level  = 5;
  auto const degree = 3;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p4, level, degree);
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p4_coefficients";
  auto const rotate = true;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-6;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p5 terms", "[coefficients]", double, float)
{
  auto const level  = 3;
  auto const degree = 5;
  auto pde = make_PDE<TestType>(PDE_opts::fokkerplanck_1d_4p5, level, degree);
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck1_4p5_coefficients";
  auto const rotate = true;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-4;
  test_coefficients(*pde, gold_path, tol_factor, rotate);
}

TEMPLATE_TEST_CASE("fokkerplanck2_complete terms", "[coefficients]", double,
                   float)
{
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                                "fokkerplanck2_complete_coefficients";

  SECTION("fokkerplanck_2d_complete, level 3, degree 3")
  {
    auto const level   = 3;
    auto const degree  = 3;
    auto const rotate = true;

    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete, level, degree);

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-13 : 1e-3;

    test_coefficients(*pde, gold_path, tol_factor, rotate);
  }

  SECTION("fokkerplanck_2d_complete, level 4, degree 4")
  {
    auto const level   = 4;
    auto const degree  = 4;
    auto const rotate = true;

    auto pde =
        make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete, level, degree);

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-13 : 1e-3;

    test_coefficients(*pde, gold_path, tol_factor, rotate);
  }
}
