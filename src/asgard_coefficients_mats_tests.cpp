#include "tests_general.hpp"

#include "asgard_coefficients_mats.hpp"

using namespace asgard;

// functors have signature P(P)
template<typename P, typename func_lhs, typename func_rhs>
std::array<P, 2> test_mass_coeff_moments(func_lhs flhs, func_rhs frhs,
                                         P xleft, P xright, int level, int degree)
{
  dimension<P> const dim(xleft, xright, level, degree, nullptr, nullptr, "dim");
  hierarchy_manipulator<P> hier(degree, 1, {xleft, }, {xright, });

  block_diag_matrix<P> ref; // reference matrix
  gen_diag_cmat<P, coefficient_type::mass>(dim, level, 0, [&](int, P x, P)->P{ return frhs(x); }, ref);

  partial_term<P> const pterm(coefficient_type::mass, nullptr,
                              [&](P x, P)->P{ return flhs(x); });
  level_mass_matrces<P> mass;
  generate_partial_mass<P>(0, dim, pterm, hier, 0, mass);

  if (mass.has_level(level))
    invert_mass(degree + 1, mass[level], ref);

  // project the lhs and rhs functions
  std::vector<P> vlhs = hier.cell_project(
    [&](std::vector<P> const &x, std::vector<P> &fx) {
      for (auto i : indexof(x))
        fx[i] = flhs(x[i]);
    }, nullptr, level);
  std::vector<P> vrhs = hier.cell_project(
    [&](std::vector<P> const &x, std::vector<P> &fx) {
      for (auto i : indexof(x))
        fx[i] = frhs(x[i]);
    }, nullptr, level);

  // combine the two vectors into one, use two versions for the moments
  int const pdof      = degree + 1;
  int const num_cells = fm::ipow2(level);
  std::vector<P> mom2(2 * pdof * num_cells), mom3(3 * pdof * num_cells);
  for (int i : iindexof(num_cells))
  {
    std::copy_n(vlhs.data() + i * pdof, pdof, mom2.data() + 2 * i * pdof);
    std::copy_n(vlhs.data() + i * pdof, pdof, mom3.data() + 3 * i * pdof);

    std::copy_n(vrhs.data() + i * pdof, pdof, mom2.data() + 2 * i * pdof + pdof);
    std::copy_n(vrhs.data() + i * pdof, pdof, mom3.data() + 3 * i * pdof + 2 * pdof);
  }

  block_diag_matrix<P> comp2, comp3;
  gen_diag_mom_cmat<P, coefficient_type::mass, 1>(dim, level, 0, 2, mom2, comp2);
  gen_diag_mom_cmat<P, coefficient_type::mass, 2>(dim, level, 0, 3, mom3, comp3);

  block_diag_matrix<P> comp2_div, comp3_div;
  gen_diag_mom_cmat_div<P, coefficient_type::mass, 1>(dim, level, 0, 2, mom2, comp2_div);
  gen_diag_mom_cmat_div<P, coefficient_type::mass, 2>(dim, level, 0, 3, mom3, comp3_div);

  // std::cout << " ref \n";
  // ref.to_full().print(std::cout);
  // std::cout << " comp2 \n";
  // comp2.to_full().print(std::cout);
  // std::cout << " comp3 \n";
  // comp2_div.to_full().print(std::cout);

  P const err2 = std::max(ref.to_full().max_diff(comp2.to_full()),
                          ref.to_full().max_diff(comp2_div.to_full()));
  P const err3 = std::max(ref.to_full().max_diff(comp3.to_full()),
                          ref.to_full().max_diff(comp3_div.to_full()));

  // std::cout << " err = " << err2 << "  " << err3 << "\n";

  return {err2, err3};
}

TEMPLATE_TEST_CASE("projected mass", "[mass-coefficients]", test_precs)
{
  // compare the construction between a mass term constructed from simple function
  // as opposed to the projection of the functions onto the Legendre basis
  using P = TestType;
  P constexpr tol = (std::is_same_v<P, double>) ? 1.E-11 : 1.E-4;

  std::array<P, 2> err = {0, 0};

  err = test_mass_coeff_moments<P>([](P)->P{ return 1; }, [](P)->P{ return -1; }, 0, 1, 2, 0);
  REQUIRE(std::max(err[0], err[1]) < tol);

  err = test_mass_coeff_moments<P>([](P)->P{ return 1; }, [](P)->P{ return -1; }, 0, 1, 5, 2);
  REQUIRE(std::max(err[0], err[1]) < tol);

  err = test_mass_coeff_moments<P>([](P)->P{ return 2; }, [](P)->P{ return 1; }, 0, 1, 5, 2);
  REQUIRE(std::max(err[0], err[1]) < tol);

  err = test_mass_coeff_moments<P>([](P)->P{ return 1; },
                                   [](P x)->P{ return std::sin(x); },
                                   -3.14, 3.14, 3, 0);
  REQUIRE(std::max(err[0], err[1]) < tol);

  err = test_mass_coeff_moments<P>([](P x)->P{ return 1 + 0.1 * std::sin(x); },
                                   [](P x)->P{ return 1 + x + 0.1 *std::cos(x); },
                                   -3.14, 3.14, 7, 1);
  REQUIRE(std::max(err[0], err[1]) < 5.E-5);
}
