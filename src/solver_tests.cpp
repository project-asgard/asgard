#include "solver.hpp"
#include "tests_general.hpp"

using namespace asgard;

TEMPLATE_TEST_CASE("simple GMRES", "[solver]", float, double)
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
    TestType const error = solver::simple_gmres(
        A_gold, test, b_gold, fk::matrix<TestType>(), A_gold.ncols(),
        A_gold.ncols(), std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold);
  }

  SECTION("test case 1, point jacobi preconditioned")
  {
    fk::vector<TestType> test(x_gold.size());

    std::cout.setstate(std::ios_base::failbit);
    TestType const error = solver::simple_gmres(
        A_gold, test, b_gold, precond, A_gold.ncols(), A_gold.ncols(),
        std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold);
  }

  SECTION("gmres test case 2")
  {
    fk::vector<TestType> test(x_gold_2.size());

    std::cout.setstate(std::ios_base::failbit);
    TestType const error = solver::simple_gmres(
        A_gold, test, b_gold_2, fk::matrix<TestType>(), A_gold.ncols(),
        A_gold.ncols(), std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold_2);
  }

  SECTION("test case 2, point jacobi preconditioned")
  {
    fk::vector<TestType> test(x_gold_2.size());
    std::cout.setstate(std::ios_base::failbit);
    TestType const error = solver::simple_gmres(
        A_gold, test, b_gold_2, precond, A_gold.ncols(), A_gold.ncols(),
        std::numeric_limits<TestType>::epsilon());
    std::cout.clear();
    REQUIRE(error < std::numeric_limits<TestType>::epsilon());
    REQUIRE(test == x_gold_2);
  }
}
