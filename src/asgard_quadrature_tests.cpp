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
