#include "tests_general.hpp"

static auto const matlab_utilities_base_dir =
    gold_base_dir / "matlab_utilities";

using namespace asgard;

// using widening conversions for golden data in order to test integers
// FIXME look for another way
TEMPLATE_TEST_CASE("generate eye(5)", "[matlab]",
                   test_precs)
{
  fk::matrix<TestType> const gold{
    {1, 0, 0, 0, 0},
    {0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 0, 1},
  };
  fk::matrix<TestType> const test = eye<TestType>(5);
  REQUIRE(test == gold);
}

TEMPLATE_TEST_CASE("horizontal matrix concatenation", "[matlab]", test_precs,
                   int)
{
  // clang-format off
  fk::matrix<TestType> const gold {{3, 2, 1},
                                   {1, 2, 3},
                                   {2, 1, 3}};
  // clang-format on

  SECTION("horz_matrix_concat(single element)")
  {
    REQUIRE(horz_matrix_concat<TestType>({gold}) == gold);
  }
  SECTION("horz_matrix_concat(multiple elements)")
  {
    fk::matrix<TestType> const column_one{{3}, {1}, {2}};
    fk::matrix<TestType> const column_two{{2}, {2}, {1}};
    fk::matrix<TestType> const column_three{{1}, {3}, {3}};

    std::vector<fk::matrix<TestType>> const test(
        {column_one, column_two, column_three});
    REQUIRE(horz_matrix_concat<TestType>(test) == gold);
  }
}

TEMPLATE_TEST_CASE("find function", "[matlab]", test_precs, int)
{
  fk::vector<TestType> haystack{2, 3, 4, 5, 6};

  int const needle = 7;
  // not capturing "needle" because https://stackoverflow.com/a/43468519
  auto const greater_eq = [](TestType i) { return i >= needle; };
  auto const is_even    = [](TestType i) {
    return (static_cast<int>(i) % 2) == 0;
  };

  SECTION("empty find -- vector")
  {
    fk::vector<int> const gold;
    REQUIRE(find(haystack, greater_eq) == gold);
  }
  SECTION("find a chunk -- vector")
  {
    fk::vector<int> const gold = {0, 2, 4};
    REQUIRE(find(haystack, is_even) == gold);
  }

  fk::matrix<TestType> const haystack_mat{{2, 3}, {4, 5}, {6, 6}};
  SECTION("empty find -- matrix")
  {
    fk::matrix<int> const gold;
    REQUIRE(find(haystack_mat, greater_eq) == gold);
  }
  SECTION("find a chunk -- vector")
  {
    fk::matrix<int> const gold = {{0, 0}, {1, 0}, {2, 0}, {2, 1}};
    REQUIRE(find(haystack_mat, is_even) == gold);
  }
}

TEST_CASE("read_vector_from_bin_file returns expected vector", "[matlab]")
{
  SECTION("read_vector_from_bin_file gets 100-element row vector")
  {
    auto const gold = linspace<default_precision>(-1, 1);
    auto const test = read_vector_from_bin_file<default_precision>(
        matlab_utilities_base_dir / "read_vector_bin_neg1_1_100.dat");
    REQUIRE(test == gold);
  }
  SECTION("read_vector_from_bin_file gets 100-element column vector")
  {
    auto const gold = linspace<default_precision>(-1, 1);
    auto const test = read_vector_from_bin_file<default_precision>(
        matlab_utilities_base_dir / "read_vector_bin_neg1_1_100T.dat");
    REQUIRE(test == gold);
  }
}

TEST_CASE("read_vector_from_txt_file returns expected vector", "[matlab]")
{
  SECTION("read_vector_from_txt_file gets 100-element row vector")
  {
    fk::vector<default_precision> gold = linspace<default_precision>(-1, 1);
    fk::vector<default_precision> test =
        read_vector_from_txt_file<default_precision>(
            matlab_utilities_base_dir / "read_vector_txt_neg1_1_100.dat");
    REQUIRE(test == gold);
  }
  SECTION("read_vector_from_txt_file gets 100-element column vector")
  {
    fk::vector<default_precision> const gold =
        linspace<default_precision>(-1, 1);
    fk::vector<default_precision> const test =
        read_vector_from_txt_file<default_precision>(
            matlab_utilities_base_dir / "read_vector_txt_neg1_1_100T.dat");
    REQUIRE(test == gold);
  }
}

TEST_CASE("read_matrix_from_txt_file returns expected vector", "[matlab]")
{
  SECTION("read_matrix_from_txt_file gets 5,5 matrix")
  {
    auto gold = fk::matrix<default_precision>(5, 5);
    // generate the golden matrix
    for (int i = 0; i < 5; i++)
      for (int j = 0; j < 5; j++)
        gold(i, j) = 17.0 / (i + 1 + j);

    fk::matrix<default_precision> const test =
        read_matrix_from_txt_file<default_precision>(matlab_utilities_base_dir /
                                                     "read_matrix_txt_5x5.dat");
    REQUIRE(test == gold);
  }
}

TEST_CASE("read_scalar_from_txt_file returns expected value", "[matlab]")
{
  SECTION("get stored scalar")
  {
    double const gold = 42;
    double const test = read_scalar_from_txt_file(matlab_utilities_base_dir /
                                                  "read_scalar_42.dat");
    REQUIRE(gold == test);
  }
}

TEMPLATE_TEST_CASE("interp1", "[matlab]", test_precs)
{
  SECTION("basic")
  {
    fk::vector<TestType> grid{0.0, 1.0, 2.0, 3.0, 4.0};

    fk::vector<TestType> vals(grid.size());
    for (int i = 0; i < grid.size(); ++i)
    {
      vals[i] = std::cos(grid[i]);
    }
    fk::vector<TestType> expected{vals[1], vals[2], vals[0], vals[4]};

    fk::vector<TestType> queries{1.49, 1.51, -0.1, 4.5};
    fk::vector<TestType> output = interp1(grid, vals, queries);
    REQUIRE(output == expected);
  }
}
