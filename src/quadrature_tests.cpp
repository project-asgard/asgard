#include "matlab_utilities.hpp"
#include "quadrature.hpp"

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
    auto [deriv, poly]            = legendre(in, degree);

    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }
  SECTION("legendre(-1, 2)")
  {
    fk::matrix<TestType> const poly_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/legendre_poly_neg1_2.dat");

    fk::matrix<TestType> const deriv_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/legendre_deriv_neg1_2.dat");

    fk::vector<TestType> const in = {-1.0};
    int const degree              = 2;
    auto [deriv, poly]            = legendre(in, degree);

    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }

  SECTION("legendre(linspace (-2, 2, 20), 5)")
  {
    fk::matrix<TestType> const poly_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/legendre_poly_linspace_5.dat");

    fk::matrix<TestType> const deriv_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/legendre_deriv_linspace_5.dat");

    fk::vector<TestType> const in = linspace<TestType>(-2.0, 2.0, 20);
    int const degree              = 5;
    auto [deriv, poly]            = legendre(in, degree);

    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }
}
