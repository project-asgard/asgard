#include "matlab_utilities.hpp"
#include "quadrature.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("legendre/legendre derivative function", "[matlab]", double,
                   float)
{
  // the relaxed comparison is due to:
  // 1) difference in precision in calculations
  // (c++ float/double vs matlab always double)
  // 2) the reordered operations make very subtle differences
  // requiring relaxed comparison for certain inputs
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
    auto const [poly, deriv]      = legendre(in, degree);

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
    auto const [poly, deriv]      = legendre(in, degree);

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

    int const degree         = 5;
    auto const [poly, deriv] = legendre(in, degree);

    relaxed_comparison(poly, poly_gold);
    relaxed_comparison(deriv, deriv_gold);
  }
}

TEMPLATE_TEST_CASE("legendre weights and roots function", "[matlab]", double,
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

  SECTION("lgwt(10, -1, 1)")
  {
    fk::matrix<TestType> const roots_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/lgwt_roots_10_neg1_1.dat");

    fk::matrix<TestType> const weights_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/lgwt_weights_10_neg1_1.dat");

    const int n                 = 10;
    const int a                 = -1;
    const int b                 = 1;
    auto const [roots, weights] = legendre_weights<TestType>(n, a, b);

    relaxed_comparison(roots, roots_gold);
    relaxed_comparison(weights, weights_gold);
  }

  SECTION("lgwt(32, -5, 2)")
  {
    fk::matrix<TestType> const roots_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/lgwt_roots_32_neg5_2.dat");

    fk::matrix<TestType> const weights_gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/quadrature/lgwt_weights_32_neg5_2.dat");

    const int n                 = 32;
    const int a                 = -5;
    const int b                 = 2;
    auto const [roots, weights] = legendre_weights<TestType>(n, a, b);

    relaxed_comparison(roots, roots_gold);
    relaxed_comparison(weights, weights_gold);
  }
}
