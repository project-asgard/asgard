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

  bool const quiet = false;
  asgard::basis::wavelet_transform<float, asgard::resource::host>
      transformer(cli_input, degree, quiet);

  field_discretization<float, asgard::resource::host> grid(cli_input, dims, transformer, pos_field.d_names);

  fk::vector<float> init(grid.size());

  grid.get_initial_conditions(pos_field, fk::vector<float, mem_type::view>(init));

  // Note to Steve/Cole, here we have the fk::vector with initial conditions
  // the goal is to plot it somehow
  for(int i=0; i<init.size(); i++) {
    std::cout << init[i] << std::endl;
  }

  return 0;
}
