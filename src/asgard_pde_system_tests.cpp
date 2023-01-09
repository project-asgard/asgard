#include "asgard_pde_system.hpp"

#include "tests_general.hpp"

using namespace asgard;

template<typename precision>
struct volume
{
  static precision jacobi(precision const, precision const) { return 1.0; }
};

template<typename precision>
struct initial
{
  static fk::vector<precision>
  x(fk::vector<precision> const &x, precision const = 0)
  {
    fk::vector<precision> fx(x.size());
    for (int i = 0; i < fx.size(); i++)
      fx[i] = 1.0;
    return fx;
  }
  static fk::vector<precision>
  y(fk::vector<precision> const &x, precision const = 0)
  {
    fk::vector<precision> fx(x.size());
    for (int i = 0; i < fx.size(); i++)
      fx[i] = x[i];
    return fx;
  }
  static fk::vector<precision>
  v(fk::vector<precision> const &x, precision const = 0)
  {
    fk::vector<precision> fx(x.size());
    for (int i = 0; i < fx.size(); i++)
      fx[i] = x[i] * x[i];
    return fx;
  }
};

int main(int argc, char *argv[])
{
  initialize_distribution();

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

TEMPLATE_TEST_CASE("testing construction of a basic pde_system", "[pde]", float,
                   double)
{
  parser const cli_input = make_empty_parser();

  TestType min0 = 0.0, min1 = 1.0;
  int level = 2, degree = 2;

  dimension_description<TestType> dim_0 =
      dimension_description<TestType>(min0, min1, level, degree, "x");
  dimension_description<TestType> dim_1 = dimension_description<TestType>(
      min0, min1, level, degree, "y", volume<TestType>::jacobi);
  dimension_description<TestType> dim_2 =
      dimension_description<TestType>(min0, min1, level, degree, "vx");
  dimension_description<TestType> dim_3 =
      dimension_description<TestType>(min0, min1, level, degree, "vy");

  field_description<TestType> field_1d(field_mode::evolution, "x",
                                       initial<TestType>::x,
                                       initial<TestType>::x, "projected to 1d");
  field_description<TestType> pos_field(
      field_mode::evolution, {"x", "y"},
      {initial<TestType>::x, initial<TestType>::y},
      {initial<TestType>::x, initial<TestType>::y}, "position");
  field_description<TestType> vel_only_field(
      field_mode::closure, {"vx", "vy"},
      {initial<TestType>::v, initial<TestType>::v},
      {initial<TestType>::v, initial<TestType>::v}, "vel-only");
  field_description<TestType> mixed_field(
      field_mode::closure, std::vector<std::string>{"x", "y", "vx", "vy"},
      {initial<TestType>::x, initial<TestType>::y, initial<TestType>::v,
       initial<TestType>::v},
      {initial<TestType>::x, initial<TestType>::y, initial<TestType>::v,
       initial<TestType>::v},
      "mixed-field");

  REQUIRE_THROWS_WITH(pde_system<TestType>(cli_input,
                                           {dim_0, dim_1, dim_2, dim_3},
                                           {field_1d, field_1d, pos_field,
                                            vel_only_field, mixed_field}),
                      "pde-system created with repeated fields (same names)");

  REQUIRE_NOTHROW(
      pde_system<TestType>(cli_input, {dim_0, dim_1, dim_2, dim_3},
                           {field_1d, pos_field, vel_only_field, mixed_field}));
  pde_system<TestType> physics(
      cli_input, {dim_0, dim_1, dim_2, dim_3},
      {field_1d, pos_field, vel_only_field, mixed_field});

  physics.load_initial_conditions();

  REQUIRE_NOTHROW(physics.get_field("projected to 1d"));

  REQUIRE_THROWS_WITH(physics.get_field("invalid name"),
                      "field name 'invalid name' is not unrecognized");
}
