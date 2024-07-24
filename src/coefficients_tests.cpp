#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "basis.hpp"
#include "coefficients.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "pde/pde_base.hpp"
#include "tests_general.hpp"

static auto const coefficients_base_dir = gold_base_dir / "coefficients";

using namespace asgard;

template<typename P>
void test_coefficients(parser const &parse, std::string const &gold_path,
                       P const tol_factor = get_tolerance<P>(10),
                       bool const rotate  = true)
{
  auto pde = make_PDE<P>(parse);
  options const opts(parse);
  basis::wavelet_transform<P, resource::host> const transformer(opts, *pde);
  P const time = 0.0;
  generate_dimension_mass_mat(*pde, transformer);
  generate_all_coefficients(*pde, transformer, time, rotate);

  auto const lev_string = std::accumulate(
      pde->get_dimensions().begin(), pde->get_dimensions().end(), std::string(),
      [](std::string const &accum, dimension<P> const &dim) {
        return accum + std::to_string(dim.get_level()) + "_";
      });

  auto const filename_base = gold_path + "_l" + lev_string + "d" +
                             std::to_string(parse.get_degree() + 1) + "_";

  for (auto d = 0; d < pde->num_dims(); ++d)
  {
    for (auto t = 0; t < pde->num_terms(); ++t)
    {
      auto const filename = filename_base + std::to_string(t + 1) + "_" +
                            std::to_string(d + 1) + ".dat";
      fk::matrix<P> const gold = read_matrix_from_txt_file<P>(filename);

      auto const full_coeff = pde->get_coefficients(t, d);

      auto const &dim = pde->get_dimensions()[d];
      auto const degrees_freedom_1d =
          (dim.get_degree() + 1) * fm::two_raised_to(dim.get_level());
      fk::matrix<P, mem_type::const_view> const test(
          full_coeff, 0, degrees_freedom_1d - 1, 0, degrees_freedom_1d - 1);

      rmse_comparison(gold, test, tol_factor);
    }
  }
}

TEMPLATE_TEST_CASE("diffusion 2 (single term)", "[coefficients]", test_precs)
{
  auto const pde_choice     = PDE_opts::diffusion_2;
  auto const gold_path      = coefficients_base_dir / "diffusion2_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  SECTION("level 3, degree 4")
  {
    auto const degree = 4;
    auto const levels = fk::vector<int>{3, 3};
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non-uniform level: levels 2, 3, degree 4")
  {
    auto const degree = 4;
    auto const levels = fk::vector<int>{2, 3};
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("diffusion 1 (single term)", "[coefficients]", test_precs)
{
  auto const pde_choice     = PDE_opts::diffusion_1;
  auto const gold_path      = coefficients_base_dir / "diffusion1_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10000);

  SECTION("level 5, degree 5")
  {
    auto const levels = fk::vector<int>{5};
    auto const degree = 5;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 1 (single term)", "[coefficients]", test_precs)
{
  auto const pde_choice = PDE_opts::continuity_1;
  auto const gold_path  = coefficients_base_dir / "continuity1_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  SECTION("level 2, degree 1 (default)")
  {
    auto const levels = fk::vector<int>{2};
    auto const degree = 1;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 2 terms", "[coefficients]", test_precs)
{
  auto const pde_choice = PDE_opts::continuity_2;
  auto const gold_path  = coefficients_base_dir / "continuity2_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  SECTION("level 4, degree 2")
  {
    auto const levels = fk::vector<int>{4, 4};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non-uniform level: levels 4, 5, degree 2")
  {
    auto const levels = fk::vector<int>{4, 5};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 3 terms", "[coefficients]", test_precs)
{
  auto const gold_path  = coefficients_base_dir / "continuity3_coefficients";
  auto const pde_choice = PDE_opts::continuity_3;
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  SECTION("level 4, degree 3")
  {
    auto const levels = fk::vector<int>{4, 4, 4};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non uniform level: levels 2, 3, 2, degree 3")
  {
    auto const levels = fk::vector<int>{2, 3, 2};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 6 terms", "[coefficients]", test_precs)
{
  auto const gold_path  = coefficients_base_dir / "continuity6_coefficients";
  auto const pde_choice = PDE_opts::continuity_6;
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  SECTION("level 2, degree 3")
  {
    auto const levels = fk::vector<int>{2, 2, 2, 2, 2, 2};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non uniform level: levels 2, 3, 3, 3, 2, 4, degree 3")
  {
    auto const levels = fk::vector<int>{2, 3, 3, 3, 2, 4};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_E case1 terms", "[coefficients]",
                   test_precs)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_pitch_E_case1;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p1a_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("level 4, degree 2")
  {
    auto const levels = fk::vector<int>{4};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_E case2 terms", "[coefficients]",
                   test_precs)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_pitch_E_case2;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_pitch_E_case2_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("level 4, degree 2")
  {
    auto const levels = fk::vector<int>{4};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_C terms", "[coefficients]", test_precs)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_pitch_C;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p2_coefficients";
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-14 : 1e-5;

  SECTION("level 5, degree 1")
  {
    auto const levels = fk::vector<int>{5};
    auto const degree = 1;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p3 terms", "[coefficients]", test_precs)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_4p3;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p3_coefficients";
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-13 : 1e-4;

  SECTION("level 2, degree 4")
  {
    auto const levels = fk::vector<int>{2};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p4 terms", "[coefficients]", test_precs)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_4p4;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p4_coefficients";
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-14 : 1e-6;

  SECTION("level 5, degree 2")
  {
    auto const levels = fk::vector<int>{5};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p5 terms", "[coefficients]", test_precs)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_4p5;
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p5_coefficients";
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-14 : 1e-4;

  SECTION("level 3, degree 4")
  {
    auto const levels = fk::vector<int>{3};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck2_complete_case4 terms", "[coefficients]",
                   test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck2_complete_coefficients";

  auto const pde_choice     = PDE_opts::fokkerplanck_2d_complete_case4;
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-12 : 1e-3;

  SECTION("level 3, degree 2")
  {
    auto const levels = fk::vector<int>{3, 3};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("level 4, degree 3")
  {
    auto const levels = fk::vector<int>{4, 4};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
  SECTION("non-uniform levels: 2, 3, degree 2")
  {
    auto const levels = fk::vector<int>{2, 3};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non-uniform levels: 4, 2, degree 3")
  {
    auto const levels = fk::vector<int>{4, 2};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("pterm lhs mass")
  {
    fk::matrix<TestType> const gold = read_matrix_from_txt_file<TestType>(
        std::string(gold_path) + "_lhsmass.dat");

    auto const levels = fk::vector<int>{4, 4};
    int const degree  = 3;

    parser const test_parse(pde_choice, levels, degree);
    auto pde = make_PDE<TestType>(test_parse);
    options const opts(test_parse);

    basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                         *pde);
    TestType const time = 0.0;
    generate_dimension_mass_mat(*pde, transformer);
    generate_all_coefficients(*pde, transformer, time, true);

    int row = 0;
    for (auto i = 0; i < pde->num_dims(); ++i)
    {
      for (auto j = 0; j < pde->num_terms(); ++j)
      {
        auto const &term_1D       = pde->get_terms()[j][i];
        auto const &partial_terms = term_1D.get_partial_terms();
        for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
        {
          int const dof = (degree + 1) * fm::two_raised_to(levels(i));

          auto const mass =
              partial_terms[k].get_lhs_mass().extract_submatrix(0, 0, dof, dof);

          fk::matrix<TestType> gold_mass(
              gold.extract_submatrix(row, 0, dof, dof));

          rmse_comparison(mass, gold_mass, get_tolerance<TestType>(100));

          row += dof;
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE("vlasov terms", "[coefficients]", test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "vlasov_lb_full_f_coefficients";

  auto const pde_choice     = PDE_opts::vlasov_lb_full_f;
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-12 : 1e-3;

  auto const cfl                  = 0.01;
  auto const full_grid            = true;
  static auto constexpr num_steps = 1;
  auto const use_implicit         = false;
  auto const do_adapt_levels      = false;
  auto const adapt_threshold      = 0.5e-1;

  SECTION("level [4,3], degree 2")
  {
    auto const levels = fk::vector<int>{4, 3};
    auto const degree = 2;

    parser const test_parse(pde_choice, levels, degree, cfl, full_grid,
                            parser::DEFAULT_MAX_LEVEL, num_steps, use_implicit,
                            do_adapt_levels, adapt_threshold);

    // parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("penalty check", "[coefficients]", test_precs)
{
  vector_func<TestType> ic = {partial_term<TestType>::null_vector_func};
  g_func_type<TestType> gfunc;

  SECTION("level 4, degree 2")
  {
    int const level  = 4;
    int const degree = 2;
    basis::wavelet_transform<TestType, resource::host> waves(level, degree,
                                                             true);

    dimension<TestType> dim(0.0, 1.0, level, degree, ic, gfunc, "x");
    partial_term<TestType> central(
        coefficient_type::div, gfunc, gfunc, flux_type::central,
        boundary_condition::periodic, boundary_condition::periodic);

    partial_term<TestType> penalty(
        coefficient_type::penalty, gfunc, gfunc, flux_type::downwind,
        boundary_condition::periodic, boundary_condition::periodic);

    partial_term<TestType> downwind(
        coefficient_type::div, gfunc, gfunc, flux_type::downwind,
        boundary_condition::periodic, boundary_condition::periodic);

    auto central_mat =
        generate_coefficients(dim, central, waves, level, TestType{0.0}, true);
    auto penalty_mat =
        generate_coefficients(dim, penalty, waves, level, TestType{0.0}, true);
    auto downwind_mat =
        generate_coefficients(dim, downwind, waves, level, TestType{0.0}, true);

    rmse_comparison(central_mat + penalty_mat, downwind_mat,
                    get_tolerance<TestType>(10));
  }
}
