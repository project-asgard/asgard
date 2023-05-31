#include "matlab_utilities.hpp"
#include "quadrature.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"

static auto const quadrature_base_dir = gold_base_dir / "quadrature";

using namespace asgard;

TEMPLATE_TEST_CASE("legendre/legendre derivative function", "[matlab]",
                   test_precs)
{
  SECTION("legendre(-1,0)")
  {
    fk::matrix<TestType> const poly_gold  = {{1.0}};
    fk::matrix<TestType> const deriv_gold = {{0.0}};

    fk::vector<TestType> const in = {-1.0};
    int const degree              = 0;
    auto const [poly, deriv]      = legendre(in, degree);

    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }
  SECTION("legendre(-1, 2)")
  {
    fk::matrix<TestType> const poly_gold = read_matrix_from_txt_file<TestType>(
        quadrature_base_dir / "legendre_poly_neg1_2.dat");

    fk::matrix<TestType> const deriv_gold = read_matrix_from_txt_file<TestType>(
        quadrature_base_dir / "legendre_deriv_neg1_2.dat");

    fk::vector<TestType> const in = {-1.0};
    int const degree              = 2;
    auto const [poly, deriv]      = legendre(in, degree);

    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }

  SECTION("legendre(linspace (-2.5, 3.0, 11), 5)")
  {
    fk::matrix<TestType> const poly_gold = read_matrix_from_txt_file<TestType>(
        quadrature_base_dir / "legendre_poly_linspace_5.dat");

    fk::matrix<TestType> const deriv_gold = read_matrix_from_txt_file<TestType>(
        quadrature_base_dir / "legendre_deriv_linspace_5.dat");

    fk::vector<TestType> const in = linspace<TestType>(-2.5, 3.0, 11);

    int const degree         = 5;
    auto const [poly, deriv] = legendre(in, degree);

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-15 : 1e-6;

    rmse_comparison(poly, poly_gold, tol_factor);
    rmse_comparison(deriv, deriv_gold, tol_factor);
  }
}

TEMPLATE_TEST_CASE("legendre weights and roots function", "[matlab]",
                   test_precs)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-6;

  SECTION("legendre_weights(10, -1, 1)")
  {
    fk::matrix<TestType> const roots_gold = read_matrix_from_txt_file<TestType>(
        quadrature_base_dir / "lgwt_roots_10_neg1_1.dat");

    fk::matrix<TestType> const weights_gold =
        read_matrix_from_txt_file<TestType>(quadrature_base_dir /
                                            "lgwt_weights_10_neg1_1.dat");

    int const n                = 10;
    TestType const a           = -1;
    TestType const b           = 1;
    bool const use_degree_quad = true;
    auto const [roots, weights] =
        legendre_weights<TestType>(n, a, b, use_degree_quad);

    rmse_comparison(roots, fk::vector<TestType>(roots_gold), tol_factor);
    rmse_comparison(weights, fk::vector<TestType>(weights_gold), tol_factor);
  }

  SECTION("legendre_weights(32, -5, 2)")
  {
    fk::matrix<TestType> const roots_gold = read_matrix_from_txt_file<TestType>(
        quadrature_base_dir / "lgwt_roots_32_neg5_2.dat");
    fk::matrix<TestType> const weights_gold =
        read_matrix_from_txt_file<TestType>(quadrature_base_dir /
                                            "lgwt_weights_32_neg5_2.dat");

    int const n                = 32;
    TestType const a           = -5;
    TestType const b           = 2;
    bool const use_degree_quad = true;
    auto const [roots, weights] =
        legendre_weights<TestType>(n, a, b, use_degree_quad);

    rmse_comparison(roots, fk::vector<TestType>(roots_gold), tol_factor);
    rmse_comparison(weights, fk::vector<TestType>(weights_gold), tol_factor);
  }
}
