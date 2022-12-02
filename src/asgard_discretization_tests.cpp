#include "asgard_discretization.hpp"

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

int main(int argc, char *argv[])
{

  parser const cli_input(argc, argv);

  float min0 = 0.0, min1 = 1.0;
  int level = 2, degree = 2;

  dimension_description<float> dim_0 =
      dimension_description<float>(min0, min1, level, degree, "x");
  dimension_description<float> dim_1 =
      dimension_description<float>(min0, min1, level, degree, "y");

  field_description<float> pos_field(field_mode::evolution, {"x", "y"}, {ic_x, ic_y}, {}, "position");

  dimension_set<float> dims(cli_input, {dim_0, dim_1});

  field_discretization<float> grid(cli_input, dims, pos_field.d_names);

  auto init = grid.get_initial_conditions(pos_field);

  return 0;
}
