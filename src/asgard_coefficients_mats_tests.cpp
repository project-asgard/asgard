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

  partial_term<P> const pterm{pt_mass, nullptr,
                              [&](P x, P)->P{ return flhs(x); }};
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

  partial_term<P> const ptermc2(mass_moment_over_density{1});
  partial_term<P> const ptermc3(mass_moment_over_density{2});

  constexpr pterm_dependence dep = pterm_dependence::moment_divided_by_density;

  block_diag_matrix<P> comp2, comp3;
  gen_diag_mom_by_mom0<P, 1, dep>(dim, ptermc2, level, 0, mom2, comp2);
  gen_diag_mom_by_mom0<P, 1, dep>(dim, ptermc3, level, 0, mom3, comp3);

  P const err2 = ref.to_full().max_diff(comp2.to_full());
  P const err3 = ref.to_full().max_diff(comp3.to_full());

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

TEMPLATE_TEST_CASE("simple div", "[div]", test_precs)
{
  using P = TestType;

  int const level = 3;

  legendre_basis<P> const basis(0); // zero order

  rhs_raw_data<P> rhs_raw;

  block_tri_matrix<P> mat;

  gen_tri_cmat<P, operation_type::div, rhs_type::is_const>(
      basis, 0, 1, level, nullptr, 1, flux_type::upwind, boundary_type::periodic, rhs_raw, mat);

  for (int i = 0; i < 8; i++) {
    REQUIRE(mat.lower(i)[0] == -8);
    REQUIRE(mat.diag(i)[0] == 8);
    REQUIRE(mat.upper(i)[0] == 0);
  }

  auto cc = [](std::vector<P> const &x, std::vector<P> &fx)
    -> void {
      for (auto i : indexof(x))
        fx[i] = 1;
    };

  gen_tri_cmat<P, operation_type::div, rhs_type::is_func>(
      basis, 0, 1, level + 1, cc, 0, flux_type::upwind, boundary_type::periodic, rhs_raw, mat);

  for (int i = 0; i < 16; i++) {
    REQUIRE(mat.lower(i)[0] == -16);
    REQUIRE(mat.diag(i)[0] == 16);
    REQUIRE(mat.upper(i)[0] == 0);
  }

  gen_tri_cmat<P, operation_type::div, rhs_type::is_const>(
      basis, 0, 1, level, nullptr, 1, flux_type::central, boundary_type::free, rhs_raw, mat);

  std::vector<P> const ref = {0, -4, 4, -4, 0, 4, -4, 0, 4, -4, 0, 4, -4, 0, 4,
                              -4, 0, 4, -4, 0, 4, -4, 4, 0};
  for (int i = 0; i < 8; i++) {
    REQUIRE(mat.lower(i)[0] == ref[3 * i]);
    REQUIRE(mat.diag(i)[0] == ref[3 * i + 1]);
    REQUIRE(mat.upper(i)[0] == ref[3 * i + 2]);
  }
}

TEMPLATE_TEST_CASE("simple mass", "[mass]", test_precs)
{
  using P = TestType;

  P constexpr tol = (std::is_same_v<P, double>) ? 1.E-13 : 1.E-4;

  int const level = 3;

  int const pdof = 3;
  legendre_basis<P> const basis(pdof - 1); // zero order

  block_diag_matrix<P> mat;

  gen_diag_cmat<P, operation_type::mass>(basis, level, 1, mat);

  for (int i = 0; i < 8; i++) {
    std::vector<P> ref = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int k = 0; k < pdof * pdof; k++)
     REQUIRE(std::abs(ref[k] - mat[i][k]) < tol);
  }

  auto cc = [](std::vector<P> const &x, std::vector<P> &fx)
    -> void {
      for (auto i : indexof(x))
        fx[i] = -3.5;
    };

  gen_diag_cmat<P, operation_type::mass>(
      basis, 0, 1, level, cc, nullptr, mat);

  for (int i = 0; i < 8; i++) {
    std::vector<P> ref = {-3.5, 0, 0, 0, -3.5, 0, 0, 0, -3.5};
    for (int k = 0; k < pdof * pdof; k++)
     REQUIRE(std::abs(ref[k] - mat[i][k]) < tol);
  }
}
