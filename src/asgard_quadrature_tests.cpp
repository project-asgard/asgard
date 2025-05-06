#include "asgard_test_macros.hpp"

// static auto const quadrature_base_dir = gold_base_dir / "quadrature";

using namespace asgard;

template<typename P>
void test_quad() {
  P constexpr tol = (std::is_same_v<P, double>) ? 1.E-15 : 5.E-7;

  {
    current_test<P> name_("legendre quadrature 1pnt");
    auto [p, w] = legendre_weights<double>(0, -1, 1, quadrature_mode::use_degree);

    tassert(p.size() == w.size());
    tassert(p.size() == 1u);

    tcheckless(0, std::abs(p[0]), tol);
    tcheckless(0, std::abs(w[0] - 2), tol);
  }{
    current_test<P> name_("legendre quadrature 2pnt");
    auto [p, w] = legendre_weights<double>(1, -1, 1, quadrature_mode::use_degree);

    tassert(p.size() == w.size());
    tassert(p.size() == 2u);

    std::vector<double> pref = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    std::vector<double> wref = {1, 1};

    for (size_t i = 0; i < p.size(); i++) {
      tcheckless(i, std::abs(p[i] - pref[i]), tol);
      tcheckless(i, std::abs(w[i] - wref[i]), tol);
    }
  }{
    current_test<P> name_("legendre quadrature 3pnt");
    auto [p, w] = legendre_weights<double>(2, -1, 1, quadrature_mode::use_degree);

    tassert(p.size() == 3u);

    std::vector<double> pref = {-std::sqrt(3.0 / 5.0), 0.0, std::sqrt(3.0 / 5.0)};
    std::vector<double> wref = {5.0 / 9.0, 8.0 / 9.0, 5.0/9.0};

    for (size_t i = 0; i < p.size(); i++) {
      tcheckless(i, std::abs(p[i] - pref[i]), tol);
      tcheckless(i, std::abs(w[i] - wref[i]), tol);
    }
  }{
    current_test<P> name_("legendre quadrature - shift points");
    auto [p, w] = legendre_weights<double>(1, 2, 4, quadrature_mode::use_degree);

    tassert(p.size() == w.size());
    tassert(p.size() == 2u);

    std::vector<double> pref = {3.0 -1.0 / std::sqrt(3.0), 3.0 + 1.0 / std::sqrt(3.0)};
    std::vector<double> wref = {1, 1};

    for (size_t i = 0; i < p.size(); i++) {
      tcheckless(i, std::abs(p[i] - pref[i]), tol);
      tcheckless(i, std::abs(w[i] - wref[i]), tol);
    }
  }{
    current_test<P> name_("legendre quadrature - scale weights");
    auto [p, w] = legendre_weights<double>(2, 0, 1, quadrature_mode::use_degree);

    std::vector<double> wref = {5.0 / 9.0, 8.0 / 9.0, 5.0/9.0};

    for (size_t i = 0; i < p.size(); i++) {
      tcheckless(i, std::abs(w[i] - 0.5 * wref[i]), tol);
    }
  }{
    current_test<P> name_("legendre quadrature - sin(x)^2");
    P constexpr pi2 = P{0.5} * PI;
    auto [p, w] = legendre_weights<double>(3, -pi2, pi2);

    double q = 0;
    for (size_t i = 0; i < p.size(); i++) {
      P const s = std::sin(p[i]);
      q += w[i] * s * s;
    }
    // using lower precision, q is an approximation and not exact to machine eps
    tcheckless(0, std::abs(q - pi2), 10 * tol);
  }
}

void all_quad_tests() {

  test_quad<double>();

  #ifdef ASGARD_ENABLE_FLOAT
  test_quad<float>();
  #endif
}

int main(int, char**) {

  all_tests global_("quadrature", " Gauss-Legendre operations");

  all_quad_tests();

  return 0;
}

// TEST_CASE("quadrature")
// {
//   auto [p, w]   = legendre_weights<double>(3, -1, 1);
//   auto [p2, w2] = legendre_weightsv2<double>(3, -1, 1);

//   std::cout << " sizes = " << p.size() << "   " << p2.size() << "\n";

//   double perr = 0, werr = 0;
//   for (auto i : indexof(p)) {
//     perr = std::max(perr, std::abs(p[i] - p2[i]));
//     werr = std::max(werr, std::abs(w[i] - w2[i]));
//     std::cout << w[i] << "    " << p[i] << "  ||  " << w2[i] << "    " << p2[i] << "\n";
//   }

//   std::cout << " err = " << perr << "  " << werr << "\n";



// }

// TEMPLATE_TEST_CASE("linspace() matches matlab implementation", "[matlab]",
//                    test_precs)
// {
//   SECTION("linspace(0,1) returns 100 elements")
//   {
//     fk::vector<TestType> const test = linspace<TestType>(0, 1);
//     REQUIRE(test.size() == 100);
//   }
//   SECTION("linspace(-1,1,9)")
//   {
//     fk::vector<TestType> const gold = {-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1};
//     REQUIRE(gold.size() == 9);
//     fk::vector<TestType> const test = linspace<TestType>(-1, 1, 9);
//     REQUIRE(test == gold);
//   }
//   SECTION("linspace(1,-1,9)")
//   {
//     fk::vector<TestType> const gold = {1, 0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75, -1};
//     REQUIRE(gold.size() == 9);
//     fk::vector<TestType> const test = linspace<TestType>(1, -1, 9);
//     REQUIRE(test == gold);
//   }
//   SECTION("linspace(-1,1,8)")
//   {
//     fk::vector<TestType> const gold = {-1, -0.7142857142857143, -0.4285714285714285, -0.1428571428571428, 0.1428571428571428, 0.4285714285714285, 0.7142857142857143, 1};
//     REQUIRE(gold.size() == 8);
//     fk::vector<TestType> const test = linspace<TestType>(-1, 1, 8);
//     REQUIRE(test == gold);
//   }
// }

// TEMPLATE_TEST_CASE("legendre/legendre derivative function", "[matlab]",
//                    test_precs)
// {
//   SECTION("legendre(-1,0)")
//   {
//     fk::matrix<TestType> const poly_gold  = {{1.0}};
//     fk::matrix<TestType> const deriv_gold = {{0.0}};

//     fk::vector<TestType> const in = {-1.0};
//     // TODO: understand why was this set to 0 before the change to degree?
//     int const degree         = 0;
//     auto const [poly, deriv] = legendre(in, degree);

//     REQUIRE(poly == poly_gold);
//     REQUIRE(deriv == deriv_gold);
//   }
//   SECTION("legendre(-1, 2)")
//   {
//     fk::matrix<TestType> const poly_gold = read_matrix_from_txt_file<TestType>(
//         quadrature_base_dir / "legendre_poly_neg1_2.dat");

//     fk::matrix<TestType> const deriv_gold = read_matrix_from_txt_file<TestType>(
//         quadrature_base_dir / "legendre_deriv_neg1_2.dat");

//     fk::vector<TestType> const in = {-1.0};
//     int const degree              = 1;
//     auto const [poly, deriv]      = legendre(in, degree);

//     REQUIRE(poly == poly_gold);
//     REQUIRE(deriv == deriv_gold);
//   }

//   SECTION("legendre(linspace (-2.5, 3.0, 11), 5)")
//   {
//     fk::matrix<TestType> const poly_gold = read_matrix_from_txt_file<TestType>(
//         quadrature_base_dir / "legendre_poly_linspace_5.dat");

//     fk::matrix<TestType> const deriv_gold = read_matrix_from_txt_file<TestType>(
//         quadrature_base_dir / "legendre_deriv_linspace_5.dat");

//     fk::vector<TestType> const in = linspace<TestType>(-2.5, 3.0, 11);

//     int const degree         = 4;
//     auto const [poly, deriv] = legendre(in, degree);

//     TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-15 : 1e-6;

//     rmse_comparison(poly, poly_gold, tol_factor);
//     rmse_comparison(deriv, deriv_gold, tol_factor);
//   }
// }

// TEMPLATE_TEST_CASE("legendre weights and roots function", "[matlab]",
//                    test_precs)
// {
//   TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-15 : 1e-6;

//   SECTION("legendre_weights(9, -1, 1)")
//   {
//     fk::matrix<TestType> const roots_gold = read_matrix_from_txt_file<TestType>(
//         quadrature_base_dir / "lgwt_roots_10_neg1_1.dat");

//     fk::matrix<TestType> const weights_gold =
//         read_matrix_from_txt_file<TestType>(quadrature_base_dir /
//                                             "lgwt_weights_10_neg1_1.dat");

//     int const n                     = 9;
//     TestType const a                = -1;
//     TestType const b                = 1;
//     quadrature_mode const quad_mode = quadrature_mode::use_degree;
//     auto const [roots, weights] =
//         legendre_weights<TestType>(n, a, b, quad_mode);

//     rmse_comparison(roots, fk::vector<TestType>(roots_gold), tol_factor);
//     rmse_comparison(weights, fk::vector<TestType>(weights_gold), tol_factor);
//   }

//   SECTION("legendre_weights(31, -5, 2)")
//   {
//     fk::matrix<TestType> const roots_gold = read_matrix_from_txt_file<TestType>(
//         quadrature_base_dir / "lgwt_roots_32_neg5_2.dat");
//     fk::matrix<TestType> const weights_gold =
//         read_matrix_from_txt_file<TestType>(quadrature_base_dir /
//                                             "lgwt_weights_32_neg5_2.dat");

//     int const n                     = 31;
//     TestType const a                = -5;
//     TestType const b                = 2;
//     quadrature_mode const quad_mode = quadrature_mode::use_degree;
//     auto const [roots, weights] =
//         legendre_weights<TestType>(n, a, b, quad_mode);

//     rmse_comparison(roots, fk::vector<TestType>(roots_gold), tol_factor);
//     rmse_comparison(weights, fk::vector<TestType>(weights_gold), tol_factor);
//   }
// }
