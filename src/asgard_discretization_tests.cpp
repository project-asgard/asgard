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

struct distribution_test_init
{
  void set_my_rank(const int rank) { my_rank = rank; }
  void set_num_ranks(const int size) { num_ranks = size; }
  int get_my_rank() const { return my_rank; }
  int get_num_ranks() const { return num_ranks; }

private:
  int my_rank;
  int num_ranks;
};
static distribution_test_init distrib_test_info;

int main(int argc, char *argv[])
{
  auto const [rank, total_ranks] = initialize_distribution();
  distrib_test_info.set_my_rank(rank);
  distrib_test_info.set_num_ranks(total_ranks);

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

TEMPLATE_TEST_CASE("testing construction of a basic field_discretization", "[grid]", float, double)
{

  parser const cli_input = make_empty_parser();

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
