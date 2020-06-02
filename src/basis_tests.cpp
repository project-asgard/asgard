#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"
#include <numeric>
#include <random>

template<typename P>
void test_multiwavelet_gen(int const degree, P const tol_factor)
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

  rmse_comparison(h0, m_h0, tol_factor);
  rmse_comparison(h1, m_h1, tol_factor);
  rmse_comparison(g0, m_g0, tol_factor);
  rmse_comparison(g1, m_g1, tol_factor);
  rmse_comparison(phi_co, m_phi_co, tol_factor);
  rmse_comparison(scale_co, m_scale_co, tol_factor);
}

TEMPLATE_TEST_CASE("Multiwavelet", "[transformations]", double, float)
{
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-14 : 1e-4;

  SECTION("Multiwavelet generation, degree = 1")
  {
    int const degree = 1;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }

  SECTION("Multiwavelet generation, degree = 2")
  {
    int const degree = 2;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }

  SECTION("Multiwavelet generation, degree = 3")
  {
    int const degree = 3;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }

  SECTION("Multiwavelet generation, degree = 4")
  {
    int const degree = 4;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }
}

template<typename P>
void test_operator_two_scale(int const levels, int const degree)
{
  fk::matrix<P> const gold = fk::matrix<P>(read_matrix_from_txt_file(
      "../testing/generated-inputs/transformations/operator_two_scale_" +
      std::to_string(degree) + "_" + std::to_string(levels) + ".dat"));
  fk::matrix<P> const test = operator_two_scale<P>(degree, levels);

  P const tol_factor = std::is_same_v<P, double> ? 1e-13 : 1e-8;

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

  P tol_factor = std::is_same<P, double>::value ? 1e-14 : 1e-5;

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

template<typename P, resource resrc>
void test_fmwt_block_generation(int const level, int const degree)
{
  P constexpr tol = std::is_same_v<P, float> ? 1e-4 : 1e-14;

  basis::wavelet_transform<P, resrc> const forward_transform(level, degree);

  auto const &blocks = forward_transform.get_blocks();

  // check odd/non-level unique blocks
  for (auto i = 1; i < level + 1; ++i)
  {
    std::string const gold_str =
        "../testing/generated-inputs/basis/transform_blocks_l" +
        std::to_string(level) + "_d" + std::to_string(degree) + "_" +
        std::to_string(i + 1) + ".dat";
    auto const gold   = fk::matrix<P>(read_matrix_from_txt_file(gold_str));
    auto const &block = blocks[(i - 1) * 2 + 1];
    if constexpr (resrc == resource::host)
    {
      rmse_comparison(gold, block, tol);
    }
    else
    {
      rmse_comparison(gold, block.clone_onto_host(), tol);
    }
  }

  // check even/level unique blocks
  for (auto i = 0; i < 2 * level; i += 2)
  {
    std::string const gold_str =
        "../testing/generated-inputs/basis/transform_blocks_l" +
        std::to_string((level - i / 2)) + "_d" + std::to_string(degree) + "_" +
        "1.dat";
    auto const gold   = fk::matrix<P>(read_matrix_from_txt_file(gold_str));
    auto const &block = blocks[i];
    if constexpr (resrc == resource::host)
    {
      rmse_comparison(gold, block, tol);
    }
    else
    {
      rmse_comparison(gold, block.clone_onto_host(), tol);
    }
  }
}

TEMPLATE_TEST_CASE_SIG("wavelet constructor", "[basis]",

                       ((typename TestType, resource resrc), TestType, resrc),
                       (double, resource::host), (double, resource::device),
                       (float, resource::host), (float, resource::device))
{
  SECTION("level 2 degree 2")
  {
    auto const degree = 2;
    auto const levels = 2;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
  SECTION("level 2 degree 5")
  {
    auto const degree = 5;
    auto const levels = 2;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
  SECTION("level 5 degree 2")
  {
    auto const degree = 2;
    auto const levels = 5;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
  SECTION("level 5 degree 5")
  {
    auto const degree = 5;
    auto const levels = 5;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
  SECTION("level 12 degree 2")
  {
    auto const degree = 2;
    auto const levels = 12;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
  SECTION("level 12 degree 5")
  {
    auto const degree = 5;
    auto const levels = 12;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
}

// tests transform across all supported levels
template<typename P, resource resrc>
void test_fmwt_application(
    basis::wavelet_transform<P, resrc> const &forward_transform)
{
  P constexpr tol = std::is_same<P, double>::value ? 1e-15 : 1e-7;

  std::random_device rd;
  std::mt19937 mersenne_engine(rd());
  std::uniform_real_distribution<P> dist(-2.0, 2.0);
  auto const gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };

  for (auto level = 2; level <= forward_transform.max_level; ++level)
  {
    auto const degrees_freedom =
        fm::two_raised_to(level) * forward_transform.degree;
    auto const to_transform = [&gen, degrees_freedom]() {
      fk::matrix<P> matrix(degrees_freedom, degrees_freedom);
      std::generate(matrix.begin(), matrix.end(), gen);
      return matrix;
    }();

    auto const fmwt = operator_two_scale<P>(forward_transform.degree, level);
    auto const fmwt_transpose = fk::matrix<P>(fmwt).transpose();

    auto const left_gold        = fmwt * to_transform;
    auto const right_gold       = to_transform * fmwt;
    auto const left_trans_gold  = fmwt_transpose * to_transform;
    auto const right_trans_gold = to_transform * fmwt_transpose;

    if constexpr (resrc == resource::host)
    {
      auto const left_test = forward_transform.apply(
          to_transform, level, basis::side::left, basis::transpose::no_trans);
      auto const right_test = forward_transform.apply(
          to_transform, level, basis::side::right, basis::transpose::no_trans);
      auto const left_trans_test = forward_transform.apply(
          to_transform, level, basis::side::left, basis::transpose::trans);
      auto const right_trans_test = forward_transform.apply(
          to_transform, level, basis::side::right, basis::transpose::trans);

      rmse_comparison(left_test, left_gold, tol);
      rmse_comparison(right_test, right_gold, tol);
      rmse_comparison(left_trans_test, left_trans_gold, tol);
      rmse_comparison(right_trans_test, right_trans_gold, tol);
    }
    else
    {
      auto const transform_d = to_transform.clone_onto_device();
      auto const left_test   = forward_transform
                                 .apply(transform_d, level, basis::side::left,
                                        basis::transpose::no_trans)
                                 .clone_onto_host();
      auto const right_test = forward_transform
                                  .apply(transform_d, level, basis::side::right,
                                         basis::transpose::no_trans)
                                  .clone_onto_host();
      auto const left_trans_test =
          forward_transform
              .apply(transform_d, level, basis::side::left,
                     basis::transpose::trans)
              .clone_onto_host();
      auto const right_trans_test =
          forward_transform
              .apply(transform_d, level, basis::side::right,
                     basis::transpose::trans)
              .clone_onto_host();
      rmse_comparison(left_test, left_gold, tol);
      rmse_comparison(right_test, right_gold, tol);
      rmse_comparison(left_trans_test, left_trans_gold, tol);
      rmse_comparison(right_trans_test, right_trans_gold, tol);
    }
  }
}

TEMPLATE_TEST_CASE_SIG("wavelet transform", "[basis]",

                       ((typename TestType, resource resrc), TestType, resrc),
                       (double, resource::host), (double, resource::device),
                       (float, resource::host), (float, resource::device))
{
  SECTION("level 2 degree 2")
  {
    auto const degree = 2;
    auto const levels = 2;
    basis::wavelet_transform<TestType, resrc> const forward_transform(levels,
                                                                      degree);
    test_fmwt_application<TestType, resrc>(forward_transform);
  }
  SECTION("level 2 degree 5")
  {
    auto const degree = 5;
    auto const levels = 2;
    basis::wavelet_transform<TestType, resrc> const forward_transform(levels,
                                                                      degree);
    test_fmwt_application<TestType, resrc>(forward_transform);
  }
  SECTION("level 5 degree 3")
  {
    auto const degree = 3;
    auto const levels = 5;
    basis::wavelet_transform<TestType, resrc> const forward_transform(levels,
                                                                      degree);
    test_fmwt_application<TestType, resrc>(forward_transform);
  }
  SECTION("level 5 degree 6")
  {
    auto const degree = 6;
    auto const levels = 5;
    basis::wavelet_transform<TestType, resrc> const forward_transform(levels,
                                                                      degree);
    test_fmwt_application<TestType, resrc>(forward_transform);
  }

  SECTION("level 8 degree 4")
  {
    auto const degree = 4;
    auto const levels = 8;
    basis::wavelet_transform<TestType, resrc> const forward_transform(levels,
                                                                      degree);
    test_fmwt_application<TestType, resrc>(forward_transform);
  }
  SECTION("level 8 degree 5")
  {
    auto const degree = 5;
    auto const levels = 8;
    basis::wavelet_transform<TestType, resrc> const forward_transform(levels,
                                                                      degree);
    test_fmwt_application<TestType, resrc>(forward_transform);
  }
}
