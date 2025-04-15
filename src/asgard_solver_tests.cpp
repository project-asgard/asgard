#include "tests_general.hpp"

using namespace asgard;

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEMPLATE_TEST_CASE("simple GMRES", "[solver]", test_precs)
{
  fk::matrix<TestType> const A_gold{
      {3.383861628748717e+00, 1.113343240310116e-02, 2.920740795411032e+00},
      {3.210305545769361e+00, 3.412141162288144e+00, 3.934494120167269e+00},
      {1.723479266939425e+00, 1.710451084172946e+00, 4.450671104482062e+00}};

  fk::matrix<TestType> const precond{{3.383861628748717e+00, 0.0, 0.0},
                                     {0.0, 3.412141162288144e+00, 0.0},
                                     {0.0, 0.0, 4.450671104482062e+00}};

  fk::vector<TestType> const b_gold{
      2.084406360034887e-01, 6.444769305362776e-01, 3.687335330031937e-01};

  fk::vector<TestType> const x_gold{
      4.715561567725287e-02, 1.257695999382253e-01, 1.625351700791827e-02};

  fk::vector<TestType> const b_gold_2{
      9.789303188021963e-01, 8.085725142873675e-01, 7.370498473207234e-01};
  fk::vector<TestType> const x_gold_2{
      1.812300946484165e-01, -7.824949213916167e-02, 1.254969087137521e-01};

  SECTION("gmres test case 1")
  {
    fk::vector<TestType> test(x_gold.size());

    std::cout.setstate(std::ios_base::failbit);
    gmres_info<TestType> const gmres_output = solvers::simple_gmres(
        A_gold, test, b_gold, fk::matrix<TestType>(), A_gold.ncols(),
        A_gold.ncols(), std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(gmres_output.error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold);
  }

  SECTION("test case 1, point jacobi preconditioned")
  {
    fk::vector<TestType> test(x_gold.size());

    std::cout.setstate(std::ios_base::failbit);
    gmres_info<TestType> const gmres_output = solvers::simple_gmres(
        A_gold, test, b_gold, precond, A_gold.ncols(), A_gold.ncols(),
        std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(gmres_output.error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold);
  }

  SECTION("gmres test case 2")
  {
    fk::vector<TestType> test(x_gold_2.size());

    std::cout.setstate(std::ios_base::failbit);
    gmres_info<TestType> const gmres_output = solvers::simple_gmres(
        A_gold, test, b_gold_2, fk::matrix<TestType>(), A_gold.ncols(),
        A_gold.ncols(), std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(gmres_output.error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold_2);
  }

  SECTION("test case 2, point jacobi preconditioned")
  {
    fk::vector<TestType> test(x_gold_2.size());
    std::cout.setstate(std::ios_base::failbit);
    gmres_info<TestType> const gmres_output = solvers::simple_gmres(
        A_gold, test, b_gold_2, precond, A_gold.ncols(), A_gold.ncols(),
        std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(gmres_output.error < std::numeric_limits<TestType>::epsilon());
    rmse_comparison(x_gold_2, test, get_tolerance<TestType>(10));
  }
}

// solves u_xx = rhs over (xleft, xright), if bc is Dirichlet, dleft/dright are the boundary cond
// returns the result from comparison against the du_ref, which should be u_x
template<typename P>
P test_poisson(std::function<P(P)> du_ref, std::function<P(P)> rhs, P xleft, P xright,
               P dleft, P dright, solvers::poisson_bc const bc, int degree, int level)
{
  solvers::poisson<P> solver(degree, xleft, xright, level);

  // construct the cell-by-cell Legenre expansion of the rhs
  // we must switch to std::vector functions
  auto lrhs = [&](std::vector<P> const &x, std::vector<P> &fx)
      -> void {
          for (auto i : indexof(x))
            fx[i] = - rhs(x[i]);
      };
  auto rref = [&](std::vector<P> const &x, std::vector<P> &fx)
      -> void {
          // the solver computes the gevative-gradient
          for (auto i : indexof(x))
            fx[i] = - du_ref(x[i]);
      };

  // the hierarchy manipulatro can do the projection
  hierarchy_manipulator<P> hier(degree, 1, {xleft, }, {xright, });

  std::vector<P> vrhs = hier.cell_project(lrhs, nullptr, level);
  std::vector<P> sv; // will hold the output

  solver.solve(vrhs, dleft, dright, bc, sv);

  // the output sv holds the cell-by-cell constant values of the gradient
  // comput reference expansion of the provided reference gradient
  hierarchy_manipulator<P> hier0(0, 1, {xleft, }, {xright, });

  std::vector<P> vref = hier0.cell_project(rref, nullptr, level);

  // vref is the pw-constant expansion of rref over the non-hierarchical cells
  // the Legenre polynomials are scaled to unit norm, to get the point-wise values
  // we must rescale back
  P const scale = std::sqrt(fm::ipow2(level) / (xright - xleft));
  for (auto &v : vref)
    v *= scale;

  return fm::diff_inf(sv, vref);
}

TEMPLATE_TEST_CASE("poisson solver projected", "[solver]", test_precs)
{
  TestType tol = (std::is_same_v<TestType, double>) ? 1.E-14 : 1.E-5;

  SECTION("constant gradient, low degree")
  {
    int const degree = 0;
    int const level  = 3;

    // example 1, u = x over (-2, 3), du = 1, ddu = 0
    auto rhs = [](TestType)->TestType { return TestType{0}; };
    auto du  = [](TestType)->TestType { return TestType{1}; };

    TestType err = test_poisson<TestType>(
        du, rhs, -2, 3, -2, 3, solvers::poisson_bc::dirichlet, degree, level);

    REQUIRE(err < tol);
  }

  SECTION("constant gradient, high degree")
  {
    int const degree = 2;
    int const level  = 5;

    // example 1, using higher degree and level
    auto rhs = [](TestType)->TestType { return TestType{0}; };
    auto du  = [](TestType)->TestType { return TestType{1}; };

    TestType err = test_poisson<TestType>(
        du, rhs, -2, 3, -2, 3, solvers::poisson_bc::dirichlet, degree, level);

    REQUIRE(err < tol);
  }

  SECTION("variable gradient")
  {
    int const degree = 1;
    int const level  = 4;

    // example 1, u = x over (-2, 3), du = 1, ddu = 0
    auto rhs = [](TestType)->TestType { return TestType{2}; };
    auto du  = [](TestType x)->TestType { return TestType{2} * x; };

    TestType err = test_poisson<TestType>(
        du, rhs, -2, 3, 4, 9, solvers::poisson_bc::dirichlet, degree, level);

    REQUIRE(err < tol);
  }

  SECTION("messy gradient, high degree")
  {
    // do not attempt this in single precision
    if (std::is_same_v<TestType, float>)
      return;

    TestType constexpr pi = 3.141592653589793;

    int const degree = 2;
    int const level  = 9;

    // example 2, u = sin(pi * x) over (-1, 1), du = pi * cos(pi * x),
    //            ddu = -pi^2 * sin(pi * x), ddu = 0
    auto rhs = [](TestType x)->TestType { return -pi * pi * std::sin(pi * x) - 1; };
    auto du  = [](TestType x)->TestType { return pi * std::cos(pi * x); };

    TestType err = test_poisson<TestType>(
        du, rhs, -1, 1, 5, 11, solvers::poisson_bc::periodic, degree, level);

    // std::cout << " error = " << err << "\n";

    REQUIRE(err < 1.E-8);
  }
}
