#include "asgard_field.hpp"

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
      dimension_description<float>(min0, min1, level, degree, "y");
  dimension_description<float> dim_2 =
      dimension_description<float>(min0, min1, level, degree, "vx");
  dimension_description<float> dim_3 =
      dimension_description<float>(min0, min1, level, degree, "vy");

  dimension_set<float> dim_set(cli_input, {dim_0, dim_1, dim_2, dim_3});

  field_description<float> field_1d("x", ic_x, ic_x, vol_jac_dV, "projected to 1d");
  field_description<float> pos_field({"x", "y"}, {ic_x, ic_y}, {ic_x, ic_y}, {vol_jac_dV, vol_jac_dV}, "position");
  field_description<float> vel_only_field({"vx", "vy"}, {ic_vel, ic_vel}, {ic_vel, ic_vel}, {vol_jac_dV, vol_jac_dV}, "vel-only");
  field_description<float> mixed_field(std::vector<std::string>{"x", "y", "vx", "vy"},
                                       {ic_x, ic_y, ic_vel, ic_vel},
                                       {ic_x, ic_y, ic_vel, ic_vel},
                                       {vol_jac_dV, vol_jac_dV, vol_jac_dV, vol_jac_dV},
                                       "mixed-field");

  field<float> m(dim_set, field_1d);
  field<float> u(dim_set, pos_field);
  field<float> v(dim_set, vel_only_field);
  field<float> w(dim_set, mixed_field);

  return 0;
}
