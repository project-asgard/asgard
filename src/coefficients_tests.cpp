#include "coefficients.hpp"
#include "pde.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("stub", "[coefficients]", double, float)
{
  auto continuity1 = make_PDE<TestType>(PDE_opts::continuity_1);
  REQUIRE(generate_coefficients(continuity1->dimensions[0],
                                continuity1->terms[0]) ==
          fk::matrix<TestType>());
}
