#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"
#include <numeric>
#include <random>

template<typename P>
void test_multiwavelet_gen(int const degree)
{
  std::string const out_base =
      "../testing/generated-inputs/transformations/multiwavelet_" +
      std::to_string(degree) + "_";

  auto const [m_h0, m_h1, m_g0, m_g1, m_phi_co, m_scale_co] =
      generate_multi_wavelets<P>(degree);

  auto const [h0, h1, g0, g1, scale_co] = [&out_base, degree]() {
    std::string const h0_string    = out_base + "h0.dat";
    std::string const h1_string    = out_base + "h1.dat";
    std::string const g0_string    = out_base + "g0.dat";
    std::string const g1_string    = out_base + "g1.dat";
    std::string const scale_string = out_base + "scale_co.dat";

    if (degree < 2)
    {
      auto const h0 =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(h0_string))}};
      auto const h1 =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(h1_string))}};
      auto const g0 =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(g0_string))}};
      auto const g1 =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(g1_string))}};
      auto const scale_co = fk::matrix<P>{
          {static_cast<P>(read_scalar_from_txt_file(scale_string))}};
      return std::array<fk::matrix<P>, 5>{h0, h1, g0, g1, scale_co};
    }
    else
    {
      fk::matrix<P> const h0 =
          fk::matrix<P>(read_matrix_from_txt_file(h0_string));
      fk::matrix<P> const h1 =
          fk::matrix<P>(read_matrix_from_txt_file(h1_string));
      fk::matrix<P> const g0 =
          fk::matrix<P>(read_matrix_from_txt_file(g0_string));
      fk::matrix<P> const g1 =
          fk::matrix<P>(read_matrix_from_txt_file(g1_string));
      fk::matrix<P> const scale_co =
          fk::matrix<P>(read_matrix_from_txt_file(scale_string));
      return std::array<fk::matrix<P>, 5>{h0, h1, g0, g1, scale_co};
    }
  }();

  std::string const phi_string = out_base + "phi_co.dat";
  fk::matrix<P> const phi_co =
      fk::matrix<P>(read_matrix_from_txt_file(phi_string));

  P const tol_factor = std::is_same<P, double>::value ? 1e-14 : 1e-5;

  rmse_comparison(h0, m_h0, tol_factor);
  rmse_comparison(h1, m_h1, tol_factor);
  rmse_comparison(g0, m_g0, tol_factor);
  rmse_comparison(g1, m_g1, tol_factor);
  rmse_comparison(phi_co, m_phi_co, tol_factor);
  rmse_comparison(scale_co, m_scale_co, tol_factor);
}

TEMPLATE_TEST_CASE("Multiwavelet", "[transformations]", double, float)
{
  SECTION("Multiwavelet generation, degree = 1")
  {
    int const degree = 1;
    test_multiwavelet_gen<TestType>(degree);
  }

  SECTION("Multiwavelet generation, degree = 3")
  {
    int const degree = 3;
    test_multiwavelet_gen<TestType>(degree);
  }
}

template<typename P>
void test_operator_two_scale(int const levels, int const degree)
{
  fk::matrix<P> const gold = fk::matrix<P>(read_matrix_from_txt_file(
      "../testing/generated-inputs/transformations/operator_two_scale_" +
      std::to_string(degree) + "_" + std::to_string(levels) + ".dat"));
  fk::matrix<P> const test = operator_two_scale<P>(degree, levels);

  P const tol_factor = 1e-14;

  rmse_comparison(gold, test, tol_factor);
}

TEMPLATE_TEST_CASE("operator_two_scale function working appropriately",
                   "[transformations]", double)
{
  SECTION("operator_two_scale(2, 2)")
  {
    int const degree = 2;
    int const levels = 2;
    test_operator_two_scale<TestType>(levels, degree);
  }
  SECTION("operator_two_scale(2, 3)")
  {
    int const degree = 2;
    int const levels = 3;
    test_operator_two_scale<TestType>(levels, degree);
  }
  SECTION("operator_two_scale(4, 3)")
  {
    int const degree = 4;
    int const levels = 3;
    test_operator_two_scale<TestType>(levels, degree);
  }
  SECTION("operator_two_scale(5, 5)")
  {
    int const degree = 5;
    int const levels = 5;
    test_operator_two_scale<TestType>(levels, degree);
  }

  SECTION("operator_two_scale(2, 6)")
  {
    int const degree = 2;
    int const levels = 6;
    test_operator_two_scale<TestType>(levels, degree);
  }
}

template<typename P>
void test_apply_fmwt(int const levels, int const degree)
{
  int const degrees_freedom = degree * pow(2, levels);

  P const tol_factor = std::is_same<P, double>::value ? 1e-15 : 1e-7;

  std::random_device rd;
  std::mt19937 mersenne_engine(rd());
  std::uniform_real_distribution<P> dist(-2.0, 2.0);
  auto const gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };
  fk::matrix<P> const to_transform = [&gen, degrees_freedom]() {
    fk::matrix<P> matrix(degrees_freedom, degrees_freedom);
    std::generate(matrix.begin(), matrix.end(), gen);
    return matrix;
  }();

  fk::matrix<P> const fmwt = operator_two_scale<P>(degree, levels);

  auto const left_gold = fmwt * to_transform;
  auto const left_test = apply_left_fmwt<P>(fmwt, to_transform, degree, levels);
  rmse_comparison(left_test, left_gold, tol_factor);

  fk::matrix<P> const fmwt_transpose  = fk::matrix<P>(fmwt).transpose();
  fk::matrix<P> const left_trans_gold = fmwt_transpose * to_transform;
  auto const left_trans_test =
      apply_left_fmwt_transposed<P>(fmwt, to_transform, degree, levels);
  rmse_comparison(left_trans_test, left_trans_gold, tol_factor);

  auto const right_gold = to_transform * fmwt;
  auto const right_test =
      apply_right_fmwt<P>(fmwt, to_transform, degree, levels);
  rmse_comparison(right_test, right_gold, tol_factor);

  auto const right_trans_gold = to_transform * fmwt_transpose;
  auto const right_trans_test =
      apply_right_fmwt_transposed<P>(fmwt, to_transform, degree, levels);
  rmse_comparison(right_trans_test, right_trans_gold, tol_factor);
}

TEMPLATE_TEST_CASE("apply_fmwt", "[apply_fmwt]", double, float)
{
  // compare optimized apply fmwt against full matrix multiplication
  SECTION("degree=2 levels=2")
  {
    int const degree = 2;
    int const levels = 2;
    test_apply_fmwt<TestType>(levels, degree);
  }

  SECTION("degree=4, levels=5")
  {
    int degree = 4;
    int levels = 5;
    test_apply_fmwt<TestType>(levels, degree);
  }
}
