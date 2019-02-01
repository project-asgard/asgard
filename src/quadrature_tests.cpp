#include "quadrature.hpp"

#include "tests_general.hpp"

TEMPLATE_TEST_CASE("legendre/legendre derivative function", "[matlab]", double,
                   float)
{
  fk::vector<TestType> in = {-1};

  SECTION("legendre(-1,0)")
  {
    fk::matrix<TestType> poly_gold  = {{1.0}};
    fk::matrix<TestType> deriv_gold = {{0.0}};
    int const degree                = 0;
    auto [poly, deriv]              = legendre(in, degree);
    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }
  SECTION("legendre(-1,1")
  {
    fk::matrix<TestType> poly_gold  = {{1.0}};
    fk::matrix<TestType> deriv_gold = {{0.0}};
    int const degree                = 1;
    auto [poly, deriv]              = legendre(in, degree);
    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }
  SECTION("legendre([-0.5, 0.8], 3)")
  {
    fk::matrix<TestType> poly_gold = {
        {1.0, -0.866025403784439, -0.279508497187474},
        {1.0, 1.385640646055102, 1.028591269649904}};
    fk::matrix<TestType> deriv_gold = {
        {0.0, 1.732050807568877, -3.354101966249685},
        {0.0, 1.732050807568877, 5.366563145999496}};
    fk::vector<TestType> input = {-0.5, 0.8};
    int const degree           = 3;
    auto [poly, deriv]         = legendre(input, degree);
    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }
}
