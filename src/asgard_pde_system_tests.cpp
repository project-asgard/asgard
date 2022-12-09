#include "asgard_pde_system.hpp"

#include "tests_general.hpp"

using namespace asgard;

static fk::vector<float> ic_x(fk::vector<float> const &x,
                              float const = 0)
{
  fk::vector<float> fx(x.size());
  for(int i=0; i<fx.size(); i++) fx[i] = 1.0;
  return fx;
}
static fk::vector<float> ic_y(fk::vector<float> const &x,
                             float const = 0)
{
  fk::vector<float> fx(x.size());
  for(int i=0; i<fx.size(); i++) fx[i] = x[i];
  return fx;
}
static fk::vector<float> ic_vel(fk::vector<float> const &x,
                             float const = 0)
{
  fk::vector<float> fx(x.size());
  for(int i=0; i<fx.size(); i++) fx[i] = x[i] * x[i];
  return fx;
}

static float vol_jac_dV(float const, float const)
{
  return 1.0;
}

TEMPLATE_TEST_CASE("testing construction of a basic pde_system", "[pde]", float)
{

  std::vector<char> ename = {'a', 's'};
  char *ename_data = ename.data();
  parser const cli_input(1, &ename_data);

  TestType min0 = 0.0, min1 = 1.0;
  int level = 2, degree = 2;

  dimension_description<TestType> dim_0 =
      dimension_description<TestType>(min0, min1, level, degree, "x");
  dimension_description<TestType> dim_1 =
      dimension_description<TestType>(min0, min1, level, degree, "y", vol_jac_dV);
  dimension_description<TestType> dim_2 =
      dimension_description<TestType>(min0, min1, level, degree, "vx");
  dimension_description<TestType> dim_3 =
      dimension_description<TestType>(min0, min1, level, degree, "vy");

  field_description<TestType> field_1d(field_mode::evolution, "x", ic_x, ic_x, "projected to 1d");
  field_description<TestType> pos_field(field_mode::evolution, {"x", "y"}, {ic_x, ic_y}, {ic_x, ic_y}, "position");
  field_description<TestType> vel_only_field(field_mode::closure, {"vx", "vy"}, {ic_vel, ic_vel}, {ic_vel, ic_vel}, "vel-only");
  field_description<TestType> mixed_field(field_mode::closure,
                                       std::vector<std::string>{"x", "y", "vx", "vy"},
                                       {ic_x, ic_y, ic_vel, ic_vel},
                                       {ic_x, ic_y, ic_vel, ic_vel},
                                       "mixed-field");

  REQUIRE_THROWS_WITH(
    pde_system<TestType>(cli_input, {dim_0, dim_1, dim_2, dim_3}, {field_1d, field_1d, pos_field, vel_only_field, mixed_field}),
    "pde-system created with repeated fields (same names)" );

  REQUIRE_NOTHROW(
    pde_system<TestType>(cli_input, {dim_0, dim_1, dim_2, dim_3}, {field_1d, pos_field, vel_only_field, mixed_field})
  );
  pde_system<TestType> physics(cli_input, {dim_0, dim_1, dim_2, dim_3}, {field_1d, pos_field, vel_only_field, mixed_field});

  physics.load_initial_conditions();

  REQUIRE_NOTHROW(physics.get_field("projected to 1d"));

  REQUIRE_THROWS_WITH(
    physics.get_field("invalid name"),
    "field name 'invalid name' is not unrecognized" );
}
