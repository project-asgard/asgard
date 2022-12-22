#include "asgard_dimension.hpp"

#include "tests_general.hpp"

using namespace asgard;

template<typename precision>
struct volume
{
  static precision jacobi(precision const, precision const) { return 1.0; }
};

TEMPLATE_TEST_CASE("testing construction of a basic field_discretization",
                   "[grid]", float, double)
{
  parser const cli_input = make_empty_parser();

  TestType min0 = 0.0, min1 = 1.0;
  int level = 2, degree = 2;

  dimension_description<TestType> dim_0 = dimension_description<TestType>(
      min0, min1, level, degree, "x", volume<TestType>::jacobi);
  dimension_description<TestType> dim_1 =
      dimension_description<TestType>(min0, min1, level, degree, "y");

  REQUIRE_THROWS_WITH(dimension_set<TestType>(cli_input, {dim_0, dim_0}),
                      "dimensions should have unique names");

  dimension_set<TestType> dset(cli_input, {dim_0, dim_1});

  REQUIRE_THROWS_WITH(
      dset("invalid name"),
      "invalid dimension name: 'invalid name', has not been defined.");
}
