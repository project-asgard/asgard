#include "coefficients.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

static auto const coefficients_base_dir = gold_base_dir / "coefficients";

template<typename P>
void test_coefficients(parser const &parse, std::string const &gold_path,
                       P const tol_factor = get_tolerance<P>(10),
                       bool const rotate  = true)
{
  auto pde = make_PDE<P>(parse);
  options const opts(parse);
  basis::wavelet_transform<P, resource::host> const transformer(opts, *pde);
  P const time = 1.0;
  generate_dimension_mass_mat(*pde, transformer);
  generate_all_coefficients(*pde, transformer, time, rotate);

  auto const lev_string = std::accumulate(
      pde->get_dimensions().begin(), pde->get_dimensions().end(), std::string(),
      [](std::string const &accum, dimension<P> const &dim) {
        return accum + std::to_string(dim.get_level()) + "_";
      });

  // FIXME assume uniform degree across dimensions here
  auto const filename_base = gold_path + "_l" + lev_string + "d" +
                             std::to_string(parse.get_degree()) + "_";

  for (auto d = 0; d < pde->num_dims; ++d)
  {
    for (auto t = 0; t < pde->num_terms; ++t)
    {
      auto const filename = filename_base + std::to_string(t + 1) + "_" +
                            std::to_string(d + 1) + ".dat";
      auto const gold = fk::matrix<P>(read_matrix_from_txt_file(filename));

      auto const full_coeff = pde->get_coefficients(t, d).clone_onto_host();

      auto const &dim = pde->get_dimensions()[d];
      auto const degrees_freedom_1d =
          dim.get_degree() * fm::two_raised_to(dim.get_level());
      fk::matrix<P, mem_type::const_view> const test(
          full_coeff, 0, degrees_freedom_1d - 1, 0, degrees_freedom_1d - 1);

      rmse_comparison(gold, test, tol_factor);
    }
  }
}

TEMPLATE_TEST_CASE("diffusion 2 (single term)", "[coefficients]", double, float)
{
  auto const pde_choice     = PDE_opts::diffusion_2;
  auto const gold_path      = coefficients_base_dir / "diffusion2_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  SECTION("level 3, degree 5")
  {
    auto const degree = 5;
    auto const levels = fk::vector<int>{3, 3};
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non-uniform level: levels 2, 3, degree 5")
  {
    auto const degree = 5;
    auto const levels = fk::vector<int>{2, 3};
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("diffusion 1 (single term)", "[coefficients]", double, float)
{
  auto const pde_choice     = PDE_opts::diffusion_1;
  auto const gold_path      = coefficients_base_dir / "diffusion1_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10000);

  SECTION("level 5, degree 6")
  {
    auto const levels = fk::vector<int>{5};
    auto const degree = 6;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 1 (single term)", "[coefficients]", double,
                   float)
{
  auto const pde_choice = PDE_opts::continuity_1;
  auto const gold_path  = coefficients_base_dir / "continuity1_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  SECTION("level 2, degree 2 (default)")
  {
    auto const levels = fk::vector<int>{2};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 2 terms", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::continuity_2;
  auto const gold_path  = coefficients_base_dir / "continuity2_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("level 4, degree 3")
  {
    auto const levels = fk::vector<int>{4, 4};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non-uniform level: levels 4, 5, degree 3")
  {
    auto const levels = fk::vector<int>{4, 5};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 3 terms", "[coefficients]", double, float)
{
  auto const gold_path  = coefficients_base_dir / "continuity3_coefficients";
  auto const pde_choice = PDE_opts::continuity_3;
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  SECTION("level 4, degree 4")
  {
    auto const levels = fk::vector<int>{4, 4, 4};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non uniform level: levels 2, 3, 2, degree 4")
  {
    auto const levels = fk::vector<int>{2, 3, 2};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 6 terms", "[coefficients]", double, float)
{
  auto const gold_path  = coefficients_base_dir / "continuity6_coefficients";
  auto const pde_choice = PDE_opts::continuity_6;
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  SECTION("level 2, degree 4")
  {
    auto const levels = fk::vector<int>{2, 2, 2, 2, 2, 2};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non uniform level: levels 2, 3, 3, 3, 2, 4, degree 4")
  {
    auto const levels = fk::vector<int>{2, 3, 3, 3, 2, 4};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_E case1 terms", "[coefficients]",
                   double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_pitch_E_case1;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p1a_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("level 4, degree 3")
  {
    auto const levels = fk::vector<int>{4};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_E case2 terms", "[coefficients]",
                   double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_pitch_E_case2;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_pitch_E_case2_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("level 4, degree 3")
  {
    auto const levels = fk::vector<int>{4};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_C terms", "[coefficients]", double,
                   float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_pitch_C;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p2_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-5;

  SECTION("level 5, degree 2")
  {
    auto const levels = fk::vector<int>{5};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p3 terms", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_4p3;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p3_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-4;

  SECTION("level 2, degree 5")
  {
    auto const levels = fk::vector<int>{2};
    auto const degree = 5;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p4 terms", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_4p4;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p4_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-6;

  SECTION("level 5, degree 3")
  {
    auto const levels = fk::vector<int>{5};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p5 terms", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_4p5;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p5_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-4;

  SECTION("level 3, degree 5")
  {
    auto const levels = fk::vector<int>{3};
    auto const degree = 5;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck2_complete_case4 terms", "[coefficients]",
                   double, float)
{
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck2_complete_coefficients";

  auto const pde_choice = PDE_opts::fokkerplanck_2d_complete_case4;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-12 : 1e-3;

  SECTION("level 3, degree 3")
  {
    auto const levels = fk::vector<int>{3, 3};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("level 4, degree 4")
  {
    auto const levels = fk::vector<int>{4, 4};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
  SECTION("non-uniform levels: 2, 3, deg 3")
  {
    auto const levels = fk::vector<int>{2, 3};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non-uniform levels: 4, 2, deg 4")
  {
    auto const levels = fk::vector<int>{4, 2};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("pterm lhs mass")
  {
    auto const gold = fk::matrix<TestType>(
        read_matrix_from_txt_file(std::string(gold_path) + "_lhsmass.dat"));

    auto constexpr tol_factor = get_tolerance<TestType>(100);
    auto const levels         = fk::vector<int>{4, 4};
    int const degree          = 4;

    parser const test_parse(pde_choice, levels, degree);
    auto pde = make_PDE<TestType>(test_parse);
    options const opts(test_parse);

    basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                         *pde);
    TestType const time = 1.0;
    generate_dimension_mass_mat(*pde, transformer);
    generate_all_coefficients(*pde, transformer, time, true);

    int row = 0;
    for (auto i = 0; i < pde->num_dims; ++i)
    {
      for (auto j = 0; j < pde->num_terms; ++j)
      {
        auto const &term_1D       = pde->get_terms()[j][i];
        auto const &partial_terms = term_1D.get_partial_terms();
        for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
        {
          int const dof = degree * fm::two_raised_to(levels(i));

          auto const mass =
              partial_terms[k].get_lhs_mass().extract_submatrix(0, 0, dof, dof);

          fk::matrix<TestType> gold_mass(
              gold.extract_submatrix(row, 0, dof, dof));

          rmse_comparison(mass, gold_mass, tol_factor);

          row += dof;
        }
      }
    }
  }
}
