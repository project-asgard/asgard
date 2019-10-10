#include "matlab_utilities.hpp"
#include "quadrature.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("legendre/legendre derivative function", "[matlab]", double,
                   float)
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
    fk::matrix<TestType> const poly_gold =
        fk::matrix<TestType>(read_matrix_from_txt_file(
            "../testing/generated-inputs/quadrature/legendre_poly_neg1_2.dat"));

    fk::matrix<TestType> const deriv_gold = fk::matrix<
        TestType>(read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/legendre_deriv_neg1_2.dat"));

    fk::vector<TestType> const in = {-1.0};
    int const degree              = 2;
    auto const [poly, deriv]      = legendre(in, degree);

    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }

  SECTION("legendre(linspace (-2.5, 3.0, 11), 5)")
  {
    fk::matrix<TestType> const poly_gold = fk::matrix<
        TestType>(read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/legendre_poly_linspace_5.dat"));

    fk::matrix<TestType> const deriv_gold = fk::matrix<TestType>(
        read_matrix_from_txt_file("../testing/generated-inputs/quadrature/"
                                  "legendre_deriv_linspace_5.dat"));

    fk::vector<TestType> const in = linspace<TestType>(-2.5, 3.0, 11);

    int const degree         = 5;
    auto const [poly, deriv] = legendre(in, degree);

    relaxed_comparison(poly, poly_gold);
    relaxed_comparison(deriv, deriv_gold);
  }
}

TEMPLATE_TEST_CASE("legendre weights and roots function", "[matlab]", double,
                   float)
{

  SECTION("legendre_weights(10, -1, 1)")
  {
    fk::matrix<TestType> const roots_gold =
        fk::matrix<TestType>(read_matrix_from_txt_file(
            "../testing/generated-inputs/quadrature/lgwt_roots_10_neg1_1.dat"));

    fk::matrix<TestType> const weights_gold = fk::matrix<
        TestType>(read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/lgwt_weights_10_neg1_1.dat"));

    int const n                 = 10;
    int const a                 = -1;
    int const b                 = 1;
    auto const [roots, weights] = legendre_weights<TestType>(n, a, b);
    relaxed_comparison(roots, roots_gold);
    relaxed_comparison(weights, weights_gold);
  }

  SECTION("legendre_weights(32, -5, 2)")
  {
    fk::matrix<TestType> const roots_gold =
        fk::matrix<TestType>(read_matrix_from_txt_file(
            "../testing/generated-inputs/quadrature/lgwt_roots_32_neg5_2.dat"));
    fk::matrix<TestType> const weights_gold = fk::matrix<
        TestType>(read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/lgwt_weights_32_neg5_2.dat"));

    int const n                 = 32;
    int const a                 = -5;
    int const b                 = 2;
    auto const [roots, weights] = legendre_weights<TestType>(n, a, b);

    relaxed_comparison(roots, roots_gold);
    relaxed_comparison(weights, weights_gold);
  }
}
