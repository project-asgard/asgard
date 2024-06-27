#include "tests_general.hpp"

#include "asgard_interpolation.hpp"
#include "asgard_interptest_common.hpp"

using namespace asgard;

#ifdef KRON_MODE_GLOBAL_BLOCK
void make_cellsd2p5(int *c)
{
  std::array<int, 10> i = {0, 0, 0, 1, 0, 2, 0, 3, 1, 0};
  std::copy(i.begin(), i.end(), c);
}

template<typename precision>
std::array<precision, 40> make_nodesd2p5()
{
  return {
      // (0, 0)
      precision{1} / 3, precision{1} / 3, precision{1} / 3, precision{2} / 3,
      precision{2} / 3, precision{1} / 3, precision{2} / 3, precision{2} / 3,
      // (0, 1)
      precision{1} / 3, precision{1} / 6, precision{1} / 3, precision{5} / 6,
      precision{2} / 3, precision{1} / 6, precision{2} / 3, precision{5} / 6,
      // (0, 2)
      precision{1} / 3, precision{1} / 12, precision{1} / 3, precision{5} / 12,
      precision{2} / 3, precision{1} / 12, precision{2} / 3, precision{5} / 12,
      // (0, 3)
      precision{1} / 3, precision{7} / 12, precision{1} / 3, precision{11} / 12,
      precision{2} / 3, precision{7} / 12, precision{2} / 3, precision{11} / 12,
      // (1, 0)
      precision{1} / 6, precision{1} / 3, precision{1} / 6, precision{2} / 3,
      precision{5} / 6, precision{1} / 3, precision{5} / 6, precision{2} / 3,
  };
}

/////////////////////////////////////////////////////////////////////
//  Testing the loaded interpolation nodes
/////////////////////////////////////////////////////////////////////
TEMPLATE_TEST_CASE("md interpolation nodes", "[linear]", test_precs)
{
  constexpr TestType tol = (std::is_same_v<TestType, double>) ? 1.E-15 : 1.E-7;

  //constexpr int order = 1;
  connect_1d conn(2, connect_1d::hierarchy::volume);
  kronmult::block_global_workspace<TestType> workspace;
  interpolation<TestType> interp(2, &conn, &workspace);

  vector2d<int> cells(2, 5);
  make_cellsd2p5(cells[0]);

  vector2d<TestType> nodes = interp.get_nodes(cells);
  auto gold = make_nodesd2p5<TestType>();

  REQUIRE(nodes.num_strips() == 20);

  for (int i = 0; i < nodes.num_strips(); i++)
    for (int j = 0; j < 2; j++)
      REQUIRE(std::abs(nodes[i][j] - gold[i * 2 + j]) < tol);
}

/////////////////////////////////////////////////////////////////////
//  Testing reconstruction of function values
/////////////////////////////////////////////////////////////////////
template<typename precision>
void project_inver_md(int num_dimensions, int num_levels,
                      std::vector<std::function<precision(precision)>> fcalls)
{
  constexpr precision tol = (std::is_same_v<precision, double>) ? 1.E-12 : 1.E-5;

  constexpr int pterms = 2;

  std::vector<vector_func<precision>> funcs;
  for (int d = 0; d < num_dimensions; d++)
    funcs.push_back([&, d](fk::vector<precision> const &x, precision const)
                        -> fk::vector<precision>
                    {
                      fk::vector<precision> fx(x.size());
                      for (int i = 0; i < fx.size(); i++)
                        fx[i] = fcalls[d](x[i]);
                      return fx;
                    });

  std::vector<dimension<precision>> dims;
  dims.reserve(num_dimensions);
  for (int d = 0; d < num_dimensions; d++)
    dims.emplace_back(0, 1, num_levels, pterms, funcs[d],
                      nullptr, std::string("dim") + std::to_string(d));

  connect_1d conn(num_levels, connect_1d::hierarchy::volume);
  kronmult::block_global_workspace<precision> workspace;
  interpolation<precision> interp(num_dimensions, &conn, &workspace);

  parser const cli_input = make_empty_parser();
  bool constexpr quiet = true;
  asgard::basis::wavelet_transform<precision, asgard::resource::host>
      transformer(cli_input, pterms, quiet);

  adapt::distributed_grid<precision> grid(cli_input, dims);

  vector2d<int> cells = get_cells<precision>(num_dimensions, grid);
  dimension_sort dsort(cells);
  vector2d<precision> nodes = interp.get_nodes(cells);

  // project the function onto the wavelet basis
  fk::vector<precision> proj(cells.num_strips() * fm::ipow(2, num_dimensions));
  grid.get_initial_condition(
      options(cli_input), dims, funcs,
      1.0, transformer, fk::vector<precision, mem_type::view>(proj));

  // for (auto x : proj) std::cout << x << "\n";

  std::vector<precision> nodal(proj.size());
  interp.get_nodal_values(cells, dsort, proj.data(), nodal.data());

  REQUIRE(nodes.num_strips() == static_cast<int64_t>(nodal.size()));

  //for (auto x : nodal) std::cout << x << "\n";
  for (int i = 0; i < nodes.num_strips(); i++)
  {
    precision val = fcalls[0](nodes[i][0]);
    for (int d = 1; d < num_dimensions; d++)
      val *= fcalls[d](nodes[i][d]);
    //std::cout << " -- " << val << "   " << nodal[i] << '\n';
    REQUIRE(std::abs(val - nodal[i]) < tol);
  }
}

TEMPLATE_TEST_CASE("md nodal value reconstruction", "[linear]", test_precs)
{
  // the level for each test should match the sum of the levels of the functions
  // one, lag1 and lin have level 0, the rest uses the number (lin1, lin2, lin3)
  project_inver_md<TestType>(2, 1, {test_functions<TestType>::one, test_functions<TestType>::one});
  project_inver_md<TestType>(2, 2, {test_functions<TestType>::one, test_functions<TestType>::one});

  project_inver_md<TestType>(2, 2, {test_functions<TestType>::one, test_functions<TestType>::lag1});
  project_inver_md<TestType>(2, 2, {test_functions<TestType>::lag1, test_functions<TestType>::one});

  project_inver_md<TestType>(3, 2, {test_functions<TestType>::lag1, test_functions<TestType>::lin,
                                    test_functions<TestType>::lin1});

  project_inver_md<TestType>(4, 3, {test_functions<TestType>::lin1, test_functions<TestType>::lin2,
                                    test_functions<TestType>::lag1, test_functions<TestType>::lin});

  project_inver_md<TestType>(4, 6, {test_functions<TestType>::lin1, test_functions<TestType>::lin2,
                                    test_functions<TestType>::lin3, test_functions<TestType>::lin});
}

/////////////////////////////////////////////////////////////////////
//  Testing reconstruction of hierarchical coefficients
/////////////////////////////////////////////////////////////////////
template<int order, typename precision, typename fcall_type>
void project_inver2d(int exact_basis, fcall_type fcall)
{
  constexpr precision tol = (std::is_same_v<precision, double>) ? 1.E-12 : 1.E-5;

  vector2d<int> cells(2, 5);
  make_cellsd2p5(cells[0]);
  dimension_sort dsort(cells);

  connect_1d conn(2, connect_1d::hierarchy::volume);
  kronmult::block_global_workspace<precision> workspace;
  interpolation interp(2, &conn, &workspace);

  vector2d<precision> nodes = interp.get_nodes(cells);
  std::vector<precision> vals(nodes.num_strips());
  for (size_t i = 0; i < vals.size(); i++)
    vals[i] = fcall(nodes[i][0], nodes[i][1]);

  interp.compute_hierarchical_coeffs(cells, dsort, vals);

  // for (auto v : vals) std::cout << v << "\n";

  REQUIRE(std::abs(vals[exact_basis] - 1) < tol);

  precision nrm = 0;
  for (auto v : vals)
    nrm += v * v;

  REQUIRE(std::abs(nrm - 1) < tol);
}

TEMPLATE_TEST_CASE("2d hierarchial coefficients", "[linear]", test_precs)
{
  project_inver2d<1, TestType>(0,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis0(x0) * test_functions<TestType>::ibasis0(x1);
      });
  project_inver2d<1, TestType>(1,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis0(x0) * test_functions<TestType>::ibasis1(x1);
      });
  project_inver2d<1, TestType>(2,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis1(x0) * test_functions<TestType>::ibasis0(x1);
      });
  project_inver2d<1, TestType>(3,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis1(x0) * test_functions<TestType>::ibasis1(x1);
      });

  project_inver2d<1, TestType>(4,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis0(x0) * test_functions<TestType>::ibasis2(x1);
      });
  project_inver2d<1, TestType>(5,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis0(x0) * test_functions<TestType>::ibasis3(x1);
      });
  project_inver2d<1, TestType>(6,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis1(x0) * test_functions<TestType>::ibasis2(x1);
      });
  project_inver2d<1, TestType>(7,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis1(x0) * test_functions<TestType>::ibasis3(x1);
      });

  project_inver2d<1, TestType>(16,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis2(x0) * test_functions<TestType>::ibasis0(x1);
      });
  project_inver2d<1, TestType>(17,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis2(x0) * test_functions<TestType>::ibasis1(x1);
      });
  project_inver2d<1, TestType>(18,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis3(x0) * test_functions<TestType>::ibasis0(x1);
      });
  project_inver2d<1, TestType>(19,
      [](TestType x0, TestType x1)->TestType{
          return test_functions<TestType>::ibasis3(x0) * test_functions<TestType>::ibasis1(x1);
      });
}

/////////////////////////////////////////////////////////////////////
//  Testing projection/inversion resulting in an identity
/////////////////////////////////////////////////////////////////////
template<typename precision>
void proj_interp_md(int num_dimensions, int num_levels,
                    std::vector<std::function<precision(precision)>> fcalls)
{
  constexpr precision tol = (std::is_same_v<precision, double>) ? 1.E-11 : 1.E-4;

  constexpr int pterms = 2;

  std::vector<vector_func<precision>> funcs;
  for (int d = 0; d < num_dimensions; d++)
    funcs.push_back([&, d](fk::vector<precision> const &x, precision const)
                        -> fk::vector<precision>
                    {
                      fk::vector<precision> fx(x.size());
                      for (int i = 0; i < fx.size(); i++)
                        fx[i] = fcalls[d](x[i]);
                      return fx;
                    });

  std::vector<dimension<precision>> dims;
  dims.reserve(num_dimensions);
  for (int d = 0; d < num_dimensions; d++)
    dims.emplace_back(0, 1, num_levels, pterms, funcs[d],
                      nullptr, std::string("dim") + std::to_string(d));

  connect_1d conn(num_levels, connect_1d::hierarchy::volume);
  kronmult::block_global_workspace<precision> workspace;
  interpolation<precision> interp(num_dimensions, &conn, &workspace);

  parser const cli_input = make_empty_parser();
  bool constexpr quiet = true;
  asgard::basis::wavelet_transform<precision, asgard::resource::host>
      transformer(cli_input, pterms, quiet);

  adapt::distributed_grid<precision> grid(cli_input, dims);

  vector2d<int> cells = get_cells<precision>(num_dimensions, grid);
  dimension_sort dsort(cells);
  vector2d<precision> nodes = interp.get_nodes(cells);

  // project the function onto the wavelet basis
  fk::vector<precision> proj(cells.num_strips() * fm::ipow(2, num_dimensions));
  grid.get_initial_condition(
      options(cli_input), dims, funcs,
      1.0, transformer, fk::vector<precision, mem_type::view>(proj));

  // for (auto x : proj) std::cout << x << "\n";

  std::vector<precision> nodal(proj.size());
  interp.get_nodal_values(cells, dsort, proj.data(), nodal.data());

  REQUIRE(nodes.num_strips() == static_cast<int64_t>(nodal.size()));

  //for (auto x : nodal) std::cout << x << "\n";
  for (int i = 0; i < nodes.num_strips(); i++)
  {
    precision val = fcalls[0](nodes[i][0]);
    for (int d = 1; d < num_dimensions; d++)
      val *= fcalls[d](nodes[i][d]);
    REQUIRE(std::abs(val - nodal[i]) < tol);
  }

  interp.compute_hierarchical_coeffs(cells, dsort, nodal);

  // this should give us back the projection coefficients
  std::vector<precision> iproj(proj.size());
  interp.get_projection_coeffs(cells, dsort, nodal.data(), iproj.data());

  for (int i = 0; i < proj.size(); i++)
    REQUIRE(std::abs(iproj[i] - proj[i]) < tol);
}

TEMPLATE_TEST_CASE("md projeciton-interpolation", "[linear]", test_precs)
{
  // the level for each test should match the sum of the levels of the functions
  // one, lag1 and lin have level 0, the rest uses the number (lin1, lin2, lin3)
  proj_interp_md<TestType>(2, 1, {test_functions<TestType>::one, test_functions<TestType>::one});
  proj_interp_md<TestType>(2, 2, {test_functions<TestType>::one, test_functions<TestType>::one});
  proj_interp_md<TestType>(2, 2, {test_functions<TestType>::one, test_functions<TestType>::lag1});

  proj_interp_md<TestType>(3, 2, {test_functions<TestType>::lag1, test_functions<TestType>::lin,
                                  test_functions<TestType>::lin1});

  proj_interp_md<TestType>(1, 2, {test_functions<TestType>::lin2,});

  proj_interp_md<TestType>(2, 2, {test_functions<TestType>::lag1, test_functions<TestType>::lin2});

  proj_interp_md<TestType>(3, 2, {test_functions<TestType>::lag1, test_functions<TestType>::lin,
                                  test_functions<TestType>::lin2});

  proj_interp_md<TestType>(4, 3, {test_functions<TestType>::lin1, test_functions<TestType>::lin2,
                                  test_functions<TestType>::lag1, test_functions<TestType>::lin});

  proj_interp_md<TestType>(4, 5, {test_functions<TestType>::lin1, test_functions<TestType>::lin2,
                                  test_functions<TestType>::lin2, test_functions<TestType>::lin});
}
#else
TEST_CASE("interpolation disabled", "[disabled]")
{
  REQUIRE(true);
}
#endif
