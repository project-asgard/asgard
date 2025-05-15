#include "asgard_test_macros.hpp"

#include "asgard_coefficients_mats.hpp"

using namespace asgard;

template<typename P>
void test_div_matrix()
{
  current_test<P> name_("div matrix");

  int const level = 3;

  legendre_basis<P> const basis(0); // zero order

  rhs_raw_data<P> rhs_raw;

  block_tri_matrix<P> mat;

  gen_tri_cmat<P, operation_type::div, rhs_type::is_const>(
      basis, 0, 1, level, nullptr, 1, flux_type::upwind, boundary_type::periodic, rhs_raw, mat);

  for (int i = 0; i < 8; i++) {
    tassert(mat.lower(i)[0] == -8);
    tassert(mat.diag(i)[0] == 8);
    tassert(mat.upper(i)[0] == 0);
  }

  auto cc = [](std::vector<P> const &x, std::vector<P> &fx)
    -> void {
      for (auto i : indexof(x))
        fx[i] = 1;
    };

  gen_tri_cmat<P, operation_type::div, rhs_type::is_func>(
      basis, 0, 1, level + 1, cc, 0, flux_type::upwind, boundary_type::periodic, rhs_raw, mat);

  for (int i = 0; i < 16; i++) {
    tassert(mat.lower(i)[0] == -16);
    tassert(mat.diag(i)[0] == 16);
    tassert(mat.upper(i)[0] == 0);
  }

  gen_tri_cmat<P, operation_type::div, rhs_type::is_const>(
      basis, 0, 1, level, nullptr, 1, flux_type::central, boundary_type::none, rhs_raw, mat);

  std::vector<P> const ref = {0, -4, 4, -4, 0, 4, -4, 0, 4, -4, 0, 4, -4, 0, 4,
                              -4, 0, 4, -4, 0, 4, -4, 4, 0};
  for (int i = 0; i < 8; i++) {
    tassert(mat.lower(i)[0] == ref[3 * i]);
    tassert(mat.diag(i)[0] == ref[3 * i + 1]);
    tassert(mat.upper(i)[0] == ref[3 * i + 2]);
  }
}
template<typename P>
void test_volume_matrix()
{
  current_test<P> name_("volume matrix");

  P constexpr tol = (std::is_same_v<P, double>) ? 1.E-13 : 1.E-4;

  int const level = 3;

  int const pdof = 3;
  legendre_basis<P> const basis(pdof - 1); // zero order

  block_diag_matrix<P> mat;

  gen_diag_cmat<P, operation_type::volume>(basis, level, 1, mat);

  for (int i = 0; i < 8; i++) {
    std::vector<P> ref = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int k = 0; k < pdof * pdof; k++)
     tassert(std::abs(ref[k] - mat[i][k]) < tol);
  }

  auto cc = [](std::vector<P> const &x, std::vector<P> &fx)
    -> void {
      for (auto i : indexof(x))
        fx[i] = -3.5;
    };

  rhs_raw_data<P> dummy;
  gen_diag_cmat<P, operation_type::volume>(
      basis, 0, 1, level, cc, dummy, mat);

  for (int i = 0; i < 8; i++) {
    std::vector<P> ref = {-3.5, 0, 0, 0, -3.5, 0, 0, 0, -3.5};
    for (int k = 0; k < pdof * pdof; k++)
     tassert(std::abs(ref[k] - mat[i][k]) < tol);
  }
}

template<typename P>
void all_templated_tests()
{
  test_div_matrix<P>();
  test_volume_matrix<P>();
}

int main(int argc, char **argv)
{
  libasgard_runtime running_(argc, argv);

  all_tests global_("coefficient-tests", " construction of coefficient matrices");

  #ifdef ASGARD_ENABLE_DOUBLE
  all_templated_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  all_templated_tests<float>();
  #endif

  return 0;
}
