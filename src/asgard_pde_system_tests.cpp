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

int main(int argc, char *argv[])
{

  parser const cli_input(argc, argv);

  float min0 = 0.0, min1 = 1.0;
  int level = 2, degree = 2;

  dimension_description<float> dim_0 =
      dimension_description<float>(min0, min1, level, degree, "x");
  dimension_description<float> dim_1 =
      dimension_description<float>(min0, min1, level, degree, "y", vol_jac_dV);
  dimension_description<float> dim_2 =
      dimension_description<float>(min0, min1, level, degree, "vx");
  dimension_description<float> dim_3 =
      dimension_description<float>(min0, min1, level, degree, "vy");

  field_description<float> field_1d(field_mode::evolution, "x", ic_x, ic_x, "projected to 1d");
  field_description<float> pos_field(field_mode::evolution, {"x", "y"}, {ic_x, ic_y}, {ic_x, ic_y}, "position");
  field_description<float> vel_only_field(field_mode::conservation, {"vx", "vy"}, {ic_vel, ic_vel}, {ic_vel, ic_vel}, "vel-only");
  field_description<float> mixed_field(field_mode::conservation,
                                       std::vector<std::string>{"x", "y", "vx", "vy"},
                                       {ic_x, ic_y, ic_vel, ic_vel},
                                       {ic_x, ic_y, ic_vel, ic_vel},
                                       "mixed-field");

  try {
    pde_system<float> broken_physics(cli_input, {dim_0, dim_1, dim_2, dim_3}, {field_1d, field_1d, pos_field, vel_only_field, mixed_field});
    REQUIRE(false); // the code below should not be reached, the line above should trow
    std::cout << "error: somehow created a pde_system with two fields of the same name " << std::endl;
    return 1;
  }catch(std::runtime_error &e){}

  pde_system<float> physics(cli_input, {dim_0, dim_1, dim_2, dim_3}, {field_1d, pos_field, vel_only_field, mixed_field});

  return 0;
}
