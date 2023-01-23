#include "asgard_discretization.hpp"
#include "build_info.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"

using namespace asgard;

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
};

int main(int argc, char *argv[])
{
  initialize_distribution();

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

TEMPLATE_TEST_CASE("testing construction of a basic field_discretization",
                   "[grid]", float, double)
{
  parser const cli_input = make_empty_parser();

  TestType min0 = 0.0, min1 = 1.0;
  int level = 2, degree = 2;

  dimension_description<TestType> dim_0 =
      dimension_description<TestType>(min0, min1, level, degree, "x");
  dimension_description<TestType> dim_1 =
      dimension_description<TestType>(min0, min1, level, degree, "y");

  field_description<TestType> pos_field(
      field_mode::evolution, {"x", "y"},
      {initial<TestType>::x, initial<TestType>::y}, {}, "position");

  dimension_set<TestType> dims(cli_input, {dim_0, dim_1});

  bool const quiet = false;
  asgard::basis::wavelet_transform<TestType, asgard::resource::host>
      transformer(cli_input, degree, quiet);

  field_discretization<TestType, asgard::resource::host> grid(
      cli_input, transformer, dims, pos_field.d_names);

  fk::vector<TestType> init = grid.get_initial_conditions(pos_field);

  REQUIRE(init.size() == 32);

  auto const dense_size = dense_space_size(dims.list);
  fk::vector<TestType> real_space(dense_size);
  // temporary workspaces for the transform
  fk::vector<TestType, mem_type::owner, resource::host> workspace(dense_size *
                                                                  2);
  std::array<fk::vector<TestType, mem_type::view, resource::host>, 2>
      tmp_workspace = {fk::vector<TestType, mem_type::view, resource::host>(
                           workspace, 0, dense_size),
                       fk::vector<TestType, mem_type::view, resource::host>(
                           workspace, dense_size, dense_size * 2 - 1)};

  // transform initial condition to realspace
  wavelet_to_realspace(dims.list, init, grid.grid->get_table(), transformer,
                       tmp_workspace, real_space);

  REQUIRE(real_space.size() == dense_size);

  std::array<TestType, 8> soln{0.0528312, 0.197169, 0.302831, 0.447169,
                               0.552831,  0.697169, 0.802831, 0.947169};

  int dim0_size = dims.list[0].degree * fm::two_raised_to(dims.list[0].level);
  REQUIRE(dim0_size == 8);
  for (int i = 0; i < dim0_size; ++i)
  {
    int dim1_size = dims.list[1].degree * fm::two_raised_to(dims.list[1].level);
    REQUIRE(dim1_size == 8);
    for (int j = 0; j < dim1_size; ++j)
    {
      REQUIRE_THAT(real_space[i * 8 + j],
                   Catch::Matchers::WithinRel(soln[j], TestType{0.00001}));
    }
  }
}
