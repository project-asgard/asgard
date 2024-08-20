#include "tests_general.hpp"

#include "asgard_testpdes_interpolation.hpp"

using namespace asgard;

#ifdef KRON_MODE_GLOBAL

/////////////////////////////////////////////////////////////////////
//  Testing the loaded interpolation nodes
/////////////////////////////////////////////////////////////////////
TEMPLATE_TEST_CASE("1d interpolation nodes", "[linear]", test_precs)
{
  constexpr TestType tol = (std::is_same_v<TestType, double>) ? 1.E-15 : 1.E-7;

  constexpr int order = 1;
  connect_1d conn(1, connect_1d::hierarchy::volume);
  wavelet_interp1d<order, TestType> wavint(&conn);

  REQUIRE(wavint.nodes().size() == 4);

  std::array<TestType, 4> lev1 = {1.0 / 3.0, 2.0 / 3.0, 1.0 / 6.0, 5.0 / 6.0};
  for (int i = 0; i < 4; i++)
    REQUIRE(std::abs(wavint.node(i) - lev1[i]) < tol);

  conn = connect_1d(2, connect_1d::hierarchy::volume);
  wavint = wavelet_interp1d<order, TestType>(&conn);
  std::array<TestType, 8> lev2 = {1.0 / 3.0, 2.0 / 3.0, 1.0 / 6.0, 5.0 / 6.0,
                                  1.0 / 12.0, 5.0 / 12.0, 7.0 / 12.0, 11.0 / 12.0};
  for (int i = 0; i < 8; i++)
    REQUIRE(std::abs(wavint.node(i) - lev2[i]) < tol);
}

/////////////////////////////////////////////////////////////////////
//  Testing the evaluation of the projection onto the interp. nodes
/////////////////////////////////////////////////////////////////////
template<int degree, typename precision, typename fcall_type>
void project_inver(int num_levels, fcall_type fcall)
{
  constexpr precision tol = (std::is_same_v<precision, double>) ? 1.E-12 : 1.E-5;
  constexpr int pdof = degree + 1; // polynomial terms are one more than the degree

  auto ffunc = [&](fk::vector<precision> const &x, precision = 0)
      -> fk::vector<precision> {
    fk::vector<precision> fx(x.size());
    for (auto i : indexof(x))
      fx[i] = fcall(x[i]);
    return fx;
  };

  auto disc = make_discretize<precision>(1, num_levels, degree, {{ffunc, }, });

  auto const &proj = disc.current_state();

  vector2d<int> cells = get_cells<precision>(1, disc.get_grid());
  dimension_sort dsort(cells);
  kronmult::permutes perms(1);

  connect_1d conn(num_levels, connect_1d::hierarchy::volume);
  wavelet_interp1d<degree, precision> wavint(&conn);
  kronmult::block_global_workspace<precision> workspace;

  std::vector<precision> nodal(pdof * cells.num_strips());
  kronmult::global_cpu(1, pdof, pdof, cells, dsort, perms, conn,
                       wavint.proj2node(), proj.data(), nodal.data(),
                       workspace);

  for (int i = 0; i < cells.num_strips(); i++)
  {
    int const idx = cells[i][0];
    for (int j = 0; j < pdof; j++)
      REQUIRE(std::abs(fcall(wavint.node(idx * pdof + j)) - nodal[i * pdof + j]) < tol);
  }
}

TEMPLATE_TEST_CASE("simple 1d interpolation", "[linear]", test_precs)
{
  for (int l = 1; l < 8; l++)
    project_inver<1, TestType>(l, test_functions<TestType>::one);

  for (int l = 1; l < 8; l++)
    project_inver<1, TestType>(l, test_functions<TestType>::lag1);

  for (int l = 1; l < 8; l++)
    project_inver<1, TestType>(l, test_functions<TestType>::lin);

  for (int l = 1; l < 8; l++)
    project_inver<1, TestType>(l, test_functions<TestType>::lin1);

  for (int l = 2; l < 8; l++)
    project_inver<1, TestType>(l, test_functions<TestType>::lin2);

  for (int l = 3; l < 9; l++)
    project_inver<1, TestType>(l, test_functions<TestType>::lin3);
}

/////////////////////////////////////////////////////////////////////
//  Testing the computation of the hierarchical interp. coefficients
/////////////////////////////////////////////////////////////////////
template<int order, typename precision, typename fcall_type>
void project_inver(int num_levels, int exact_basis, fcall_type fcall)
{
  constexpr precision tol = (std::is_same_v<precision, double>) ? 1.E-12 : 1.E-5;
  constexpr int pterms = order + 1; // polynomial terms are one more than the degree

  vector2d<int> cells(1, fm::ipow(2, num_levels));
  for (int i = 0; i < cells.num_strips(); i++)
    cells[i][0] = i;

  dimension_sort dsort(cells);

  connect_1d conn(num_levels, connect_1d::hierarchy::volume);
  wavelet_interp1d<order, precision> wavint(&conn);
  kronmult::block_global_workspace<precision> workspace;

  std::vector<precision> vals(fm::ipow(2, num_levels) * pterms);
  for (size_t i = 0; i < vals.size(); i++)
    vals[i] = fcall(wavint.node(i));

  kronmult::globalsv_cpu(1, pterms, cells, dsort, conn,
                         wavint.node2hier(), vals.data(), workspace);

  REQUIRE(std::abs(vals[exact_basis] - 1) < tol);

  precision nrm = 0;
  for (auto v : vals)
    nrm += v * v;

  REQUIRE(std::abs(nrm - 1) < tol);
}

TEMPLATE_TEST_CASE("simple 1d hierarchial coefficients", "[linear]", test_precs)
{
  for (int l = 2; l < 5; l++)
    project_inver<1, TestType>(l, 0, test_functions<TestType>::ibasis0);
  for (int l = 2; l < 5; l++)
    project_inver<1, TestType>(l, 1, test_functions<TestType>::ibasis1);
  for (int l = 2; l < 6; l++)
    project_inver<1, TestType>(l, 2, test_functions<TestType>::ibasis2);
  for (int l = 2; l < 6; l++)
    project_inver<1, TestType>(l, 3, test_functions<TestType>::ibasis3);
}

/////////////////////////////////////////////////////////////////////
//  Testing the loaded integrals
/////////////////////////////////////////////////////////////////////
TEMPLATE_TEST_CASE("1d integration blocks", "[linear]", test_precs)
{
  constexpr TestType tol = (std::is_same_v<TestType, double>) ? 1.E-15 : 1.E-5;

  // no-scaling case, using only canonical integrals
  constexpr int order = 1;
  connect_1d conn(1, connect_1d::hierarchy::volume);
  wavelet_interp1d<order, TestType> wavint(&conn);

  std::array<double, 16> lev1 = {
    1.0/2.0, -std::sqrt(3.0) / 2.0, 1.0/2.0, std::sqrt(3.0) / 2.0,
    1.0/4.0, -std::sqrt(3.0) / 4.0, 1.0/4.0, std::sqrt(3.0) / 4.0,
    0.0, 0.0, 0.0, 0.0,
    std::sqrt(3.0) / 4.0, -1.0 / 4.0, std::sqrt(3.0) / 4.0, 1.0 / 4.0,
  };

  for (size_t i = 0; i < lev1.size(); i++)
    REQUIRE(std::abs(wavint.hier2proj()[i] - lev1[i]) < tol);

  connect_1d conn2(2, connect_1d::hierarchy::volume);
  wavelet_interp1d<order, TestType> wavint2(&conn2);

  std::array<double, 56> lev2 = {
      1.0/2.0, -std::sqrt(3.0) / 2.0, 1.0/2.0, std::sqrt(3.0) / 2.0, // cell 0-0
      1.0/4.0, -std::sqrt(3.0) / 4.0, 1.0/4.0, std::sqrt(3.0) / 4.0, // cell 0-1
      1.0/8.0, -std::sqrt(3.0) / 8.0, 1.0/8.0, 0, // cell 0-2
      1.0/8.0, 0, 1.0/8.0, std::sqrt(3.0) / 8.0, // cell 0-3
      // second row of the large block-sparse matrix
      0.0, 0.0, 0.0, 0.0, // cell 1-0 (connected but the basis is orthogonal)
      std::sqrt(3.0) / 4.0, -1.0 / 4.0, std::sqrt(3.0) / 4.0, 1.0 / 4.0, // cell 1-1
      std::sqrt(3.0) / 8.0, -1.0/8.0, -std::sqrt(3.0) / 8.0, 1.0 / 4.0, // cell 1-2
      -std::sqrt(3.0) / 8.0, -1.0/4.0, std::sqrt(3.0) / 8.0, 1.0/8.0, // cell 1-3
      // third row of the large block-sparse matrix
      0.0, 0.0, 0.0, 0.0, // cell 2-0
      0.0, 0.0, 0.0, 0.0, // cell 2-1
      std::sqrt(2.0) * std::sqrt(3.0) / 8.0, -std::sqrt(2.0) / 8.0, std::sqrt(2.0) * std::sqrt(3.0) / 8.0, std::sqrt(2.0) / 8.0,
      // fourth row of the large block-sparse matrix
      0.0, 0.0, 0.0, 0.0, // cell 3-0
      0.0, 0.0, 0.0, 0.0, // cell 3-1
      std::sqrt(2.0) * std::sqrt(3.0) / 8.0, -std::sqrt(2.0) / 8.0, std::sqrt(2.0) * std::sqrt(3.0) / 8.0, std::sqrt(2.0) / 8.0,
  };
  for (size_t i = 0; i < lev2.size(); i++)
    REQUIRE(std::abs(wavint2.hier2proj()[i] - lev2[i]) < tol);
}

#else
TEST_CASE("interpolation disabled", "[disabled]")
{
  REQUIRE(true);
}
#endif // KRON_MODE_GLOBAL
