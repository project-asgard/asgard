#include "matlab_utilities.hpp"
#include "quadrature.hpp"

#include "tests_general.hpp"

TEMPLATE_TEST_CASE("legendre/legendre derivative function", "[matlab]", double,
                   float)
{
  auto const relaxed_comparison = [](auto const first, auto const second) {
    auto first_it = first.begin();
    std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
      REQUIRE(Approx(*first_it++)
                  .epsilon(std::numeric_limits<TestType>::epsilon() * 1e2) ==
              second_elem);
    });
  };

  SECTION("legendre(-1,0)")
  {
    fk::matrix<TestType> const poly_gold  = {{1.0}};
    fk::matrix<TestType> const deriv_gold = {{0.0}};

    fk::vector<TestType> const in = {-1.0};
    int const degree              = 0;
    auto [poly, deriv]            = legendre(in, degree);

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
    auto [poly, deriv]            = legendre(in, degree);

    REQUIRE(poly == poly_gold);
    REQUIRE(deriv == deriv_gold);
  }

  SECTION("legendre(linspace (-2.5, 3.0, 11), 5)")
  {
    fk::matrix<TestType> const poly_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/legendre_poly_linspace_5.dat");

    fk::matrix<TestType> const deriv_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/legendre_deriv_linspace_5.dat");

    fk::vector<TestType> const in = linspace<TestType>(-2.5, 3.0, 11);

    int const degree              = 5;
    auto [poly, deriv]            = legendre(in, degree);
    
    // the relaxed comparison is not because of difference in precision in calculations
    // c++ float vs matlab. the reordered operations make very subtle differences requiring
    // relaxed comparison for certain inputs (like these) 
    relaxed_comparison(poly, poly_gold);
    relaxed_comparison(deriv, deriv_gold);
  }
}
