#include "asgard_field.hpp"

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

TEMPLATE_TEST_CASE("testing construction of a basic field_description",
                   "[field]", float, double)
{
  parser const cli_input = make_empty_parser();

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

  REQUIRE_THROWS_WITH(
      field_description<TestType>(field_mode::closure, {"vx", "vx"},
                                  {initial<TestType>::v, initial<TestType>::v},
                                  "bad vel"),
      "repeated dimensions in the field definition");
}
