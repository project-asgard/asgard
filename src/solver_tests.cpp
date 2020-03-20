#include "solver.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("simple GMRES", "[solver]", double)
{
  fk::matrix<TestType> const A_gold{
      {3.383861628748717e+00, 1.113343240310116e-02, 2.920740795411032e+00},
      {3.210305545769361e+00, 3.412141162288144e+00, 3.934494120167269e+00},
      {1.723479266939425e+00, 1.710451084172946e+00, 4.450671104482062e+00}};

  fk::vector<TestType> const B_gold{
      2.084406360034887e-01, 6.444769305362776e-01, 3.687335330031937e-01};

  fk::vector<TestType> const X_gold{
      4.715561567725287e-02, 1.257695999382253e-01, 1.625351700791827e-02};

  fk::vector<TestType> const B1_gold{
      9.789303188021963e-01, 8.085725142873675e-01, 7.370498473207234e-01};
  fk::vector<TestType> const X1_gold{
      1.812300946484165e-01, -7.824949213916167e-02, 1.254969087137521e-01};

  SECTION("gmres test case 1")
  {
    fk::vector<TestType> test(X_gold.size());
    solver::simple_gmres(A_gold, test, B_gold, fk::matrix<TestType>(),
                         A_gold.ncols(), A_gold.ncols(),
                         static_cast<TestType>(1e-15));
    REQUIRE(test == X_gold);
  }

  SECTION("gmres test case 2") {}
}
