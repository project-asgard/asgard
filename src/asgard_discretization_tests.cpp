#include "asgard_discretization.hpp"

#include "tests_general.hpp"

using namespace asgard;

template<typename precision>
struct initial {
  static fk::vector<precision> x(fk::vector<precision> const &x,
                                 precision const = 0)
  {
    fk::vector<precision> fx(x.size());
    for(int i=0; i<fx.size(); i++) fx[i] = 1.0;
    return fx;
  }
  static fk::vector<precision> y(fk::vector<precision> const &x,
                                 precision const = 0)
  {
    fk::vector<precision> fx(x.size());
    for(int i=0; i<fx.size(); i++) fx[i] = x[i];
    return fx;
  }
};

TEMPLATE_TEST_CASE("testing construction of a basic field_discretization", "[grid]", float, double)
{

  std::vector<char> ename = {'a', 's'};
  char *ename_data = ename.data();
  parser const cli_input(1, &ename_data); // dummy parser

  TestType min0 = 0.0, min1 = 1.0;
  int level = 2, degree = 2;

  dimension_description<TestType> dim_0 =
      dimension_description<TestType>(min0, min1, level, degree, "x");
  dimension_description<TestType> dim_1 =
      dimension_description<TestType>(min0, min1, level, degree, "y");

  field_description<TestType> pos_field(field_mode::evolution, {"x", "y"},
                                        {initial<TestType>::x, initial<TestType>::y},
                                        {}, "position");

  dimension_set<TestType> dims(cli_input, {dim_0, dim_1});

  bool const quiet = false;
  asgard::basis::wavelet_transform<TestType, asgard::resource::host>
      transformer(cli_input, degree, quiet);

  field_discretization<TestType, asgard::resource::host> grid(cli_input, transformer, dims, pos_field.d_names);

  fk::vector<TestType> init = grid.get_initial_conditions(pos_field);

  REQUIRE(init.size() == 32);
}
