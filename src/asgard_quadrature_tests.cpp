#include "asgard_test_macros.hpp"

// static auto const quadrature_base_dir = gold_base_dir / "quadrature";

using namespace asgard;

void test_quad() {
  double constexpr tol = 1.E-15;

  {
    current_test<double> name_("legendre quadrature 1pnt");
    auto [p, w] = legendre_weights(0, -1, 1, quadrature_mode::use_degree);

    tassert(p.size() == w.size());
    tassert(p.size() == 1u);

    tcheckless(0, std::abs(p[0]), tol);
    tcheckless(0, std::abs(w[0] - 2), tol);
  }{
    current_test<double> name_("legendre quadrature 2pnt");
    auto [p, w] = legendre_weights(1, -1, 1, quadrature_mode::use_degree);

    tassert(p.size() == w.size());
    tassert(p.size() == 2u);

    std::vector<double> pref = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    std::vector<double> wref = {1, 1};

    for (size_t i = 0; i < p.size(); i++) {
      tcheckless(i, std::abs(p[i] - pref[i]), tol);
      tcheckless(i, std::abs(w[i] - wref[i]), tol);
    }
  }{
    current_test<double> name_("legendre quadrature 3pnt");
    auto [p, w] = legendre_weights(2, -1, 1, quadrature_mode::use_degree);

    tassert(p.size() == 3u);

    std::vector<double> pref = {-std::sqrt(3.0 / 5.0), 0.0, std::sqrt(3.0 / 5.0)};
    std::vector<double> wref = {5.0 / 9.0, 8.0 / 9.0, 5.0/9.0};

    for (size_t i = 0; i < p.size(); i++) {
      tcheckless(i, std::abs(p[i] - pref[i]), tol);
      tcheckless(i, std::abs(w[i] - wref[i]), tol);
    }
  }{
    current_test<double> name_("legendre quadrature - shift points");
    auto [p, w] = legendre_weights(1, 2, 4, quadrature_mode::use_degree);

    tassert(p.size() == w.size());
    tassert(p.size() == 2u);

    std::vector<double> pref = {3.0 -1.0 / std::sqrt(3.0), 3.0 + 1.0 / std::sqrt(3.0)};
    std::vector<double> wref = {1, 1};

    for (size_t i = 0; i < p.size(); i++) {
      tcheckless(i, std::abs(p[i] - pref[i]), tol);
      tcheckless(i, std::abs(w[i] - wref[i]), tol);
    }
  }{
    current_test<double> name_("legendre quadrature - scale weights");
    auto [p, w] = legendre_weights(2, 0, 1, quadrature_mode::use_degree);

    std::vector<double> wref = {5.0 / 9.0, 8.0 / 9.0, 5.0/9.0};

    for (size_t i = 0; i < p.size(); i++) {
      tcheckless(i, std::abs(w[i] - 0.5 * wref[i]), tol);
    }
  }{
    current_test<double> name_("legendre quadrature - sin(x)^2");
    double constexpr pi2 = 0.5 * PI;
    auto [p, w] = legendre_weights(3, -pi2, pi2);

    double q = 0;
    for (size_t i = 0; i < p.size(); i++) {
      double const s = std::sin(p[i]);
      q += w[i] * s * s;
    }
    // using lower precision, q is an approximation and not exact to machine eps
    tcheckless(0, std::abs(q - pi2),  10 * tol);
  }{
    current_test<double> name_("legendre quadrature - exp(x)");
    double const ex = std::exp(3.0) - 1.0;
    auto [p, w] = legendre_weights(7, 0, 3);

    double q = 0;
    for (size_t i = 0; i < p.size(); i++)
      q += w[i] * std::exp(p[i]);

    // using lower precision, q is an approximation and not exact to machine eps
    tcheckless(0, std::abs(q - ex),  5 * tol);
  }
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("quadrature", " Gauss-Legendre operations");

  test_quad();

  return 0;
}
