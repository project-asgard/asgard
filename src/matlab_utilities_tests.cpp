#include "matlab_utilities.hpp"

#include "tests_general.hpp"
#include <vector>

TEMPLATE_TEST_CASE("linspace() matches matlab implementation", "[matlab]",
                   float, double)
{
  SECTION("linspace(0,1) returns 100 elements")
  {
    fk::vector<TestType> test = linspace<TestType>(0, 1);
    REQUIRE(test.size() == 100);
  }
  SECTION("linspace(-1,1,9)")
  {
    fk::vector<TestType> gold = fk::vector<TestType>(read_vector_from_txt_file(
        "../testing/generated-inputs/matlab_utilities/linspace_neg1_1_9.dat"));
    REQUIRE(gold.size() == 9);
    fk::vector<TestType> test = linspace<TestType>(-1, 1, 9);
    REQUIRE(test == gold);
  }
  SECTION("linspace(1,-1,9)")
  {
    fk::vector<TestType> gold = fk::vector<TestType>(read_vector_from_txt_file(
        "../testing/generated-inputs/matlab_utilities/linspace_1_neg1_9.dat"));
    REQUIRE(gold.size() == 9);
    fk::vector<TestType> test = linspace<TestType>(1, -1, 9);
    REQUIRE(test == gold);
  }
  SECTION("linspace(-1,1,8)")
  {
    fk::vector<TestType> gold = fk::vector<TestType>(read_vector_from_txt_file(
        "../testing/generated-inputs/matlab_utilities/linspace_neg1_1_8.dat"));
    REQUIRE(gold.size() == 8);
    fk::vector<TestType> test = linspace<TestType>(-1, 1, 8);
    REQUIRE(test == gold);
  }
}

// using widening conversions for golden data in order to test integers
// FIXME look for another way
TEMPLATE_TEST_CASE("eye() matches matlab implementation", "[matlab]", float,
                   double, int)
{
  SECTION("eye()")
  {
    fk::matrix<TestType> gold{{1}};
    fk::matrix<TestType> test = eye<TestType>();
    REQUIRE(test == gold);
  }
  SECTION("eye(5)")
  {
    // clang-format off
    fk::matrix<TestType> gold{
      {1, 0, 0, 0, 0},
      {0, 1, 0, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 0, 1, 0},
      {0, 0, 0, 0, 1},
    }; // clang-format on
    fk::matrix<TestType> test = eye<TestType>(5);
    REQUIRE(test == gold);
  }
  SECTION("eye(5,5)")
  {
    // clang-format off
    fk::matrix<TestType> gold{
      {1, 0, 0, 0, 0},
      {0, 1, 0, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 0, 1, 0},
      {0, 0, 0, 0, 1},
    }; // clang-format on
    fk::matrix<TestType> test = eye<TestType>(5, 5);
    REQUIRE(test == gold);
  }
  SECTION("eye(5,3)")
  {
    // clang-format off
    fk::matrix<TestType> gold{
      {1, 0, 0},
      {0, 1, 0},
      {0, 0, 1},
      {0, 0, 0},
      {0, 0, 0},
    }; // clang-format on
    fk::matrix<TestType> test = eye<TestType>(5, 3);
    REQUIRE(test == gold);
  }
  SECTION("eye(3,5)")
  {
    // clang-format off
    fk::matrix<TestType> gold{
      {1, 0, 0, 0, 0},
      {0, 1, 0, 0, 0},
      {0, 0, 1, 0, 0},
    }; // clang-format on
    fk::matrix<TestType> test = eye<TestType>(3, 5);
    REQUIRE(test == gold);
  }
}

TEMPLATE_TEST_CASE("polynomial evaluation functions", "[matlab]", float, double,
                   int)
{
  SECTION("polyval(p = [3,2,1], x = [5,7,9])")
  {
    fk::vector<TestType> p{3, 2, 1};
    fk::vector<TestType> x{5, 7, 9};
    fk::vector<TestType> gold{86, 162, 262};
    fk::vector<TestType> test = polyval(p, x);
    REQUIRE(test == gold);
  }
  SECTION("polyval(p = [4, 0, 1, 2], x = 2")
  {
    fk::vector<TestType> p{4, 0, 1, 2};
    TestType x    = 2;
    TestType gold = 36;
    TestType test = polyval(p, x);
    REQUIRE(test == gold);
  }
  SECTION("polyval(p = [4, 0, 1, 2], x = 0")
  {
    fk::vector<TestType> p{4, 0, 1, 2};
    TestType x    = 0;
    TestType gold = 2;
    TestType test = polyval(p, x);
    REQUIRE(test == gold);
  }
}

TEMPLATE_TEST_CASE("horizontal matrix concatenation", "[matlab]", float, double,
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

TEST_CASE("meshgrid", "[matlab]")
{
  SECTION("length 1 case")
  {
    fk::matrix<int> const gold{{-4}};
    int const start  = -4;
    int const length = 1;
    REQUIRE(meshgrid(start, length) == gold);
  }
  SECTION("longer case")
  {
    // clang-format off
    fk::matrix<int> const gold {{-3, -2, -1},
			        {-3, -2, -1},
				{-3, -2, -1}};
    // clang-format on
    int const start  = -3;
    int const length = 3;
    REQUIRE(meshgrid(start, length) == gold);
  }
}

TEMPLATE_TEST_CASE("find function", "[matlab]", float, double, int)
{
  fk::vector<TestType> haystack{2, 3, 4, 5, 6};

  int const needle = 7;
  // not capturing "needle" because https://stackoverflow.com/a/43468519
  auto greater_eq = [](TestType i) { return i >= needle; };

  auto is_even = [](TestType i) { return (static_cast<int>(i) % 2) == 0; };
  SECTION("empty find -- vector")
  {
    fk::vector<int> gold;
    REQUIRE(find(haystack, greater_eq) == gold);
  }
  SECTION("find a group -- vector")
  {
    fk::vector<int> gold = {0, 2, 4};
    REQUIRE(find(haystack, is_even) == gold);
  }

  fk::matrix<TestType> haystack_mat{{2, 3}, {4, 5}, {6, 6}};
  SECTION("empty find -- matrix")
  {
    fk::matrix<int> gold;
    REQUIRE(find(haystack_mat, greater_eq) == gold);
  }
  SECTION("find a group -- vector")
  {
    //clang-format off
    fk::matrix<int> gold = {{0, 0}, {1, 0}, {2, 0}, {2, 1}};
    //clang-format on
    REQUIRE(find(haystack_mat, is_even) == gold);
  }
}

TEMPLATE_TEST_CASE("l2_norm function", "[matlab]", float, double)
{
  SECTION("zeros -- vector")
  {
    fk::vector<TestType> const vec{0, 0, 0, 0, 0};
    TestType const gold = 0;
    REQUIRE(l2_norm(vec) == gold);
  }

  SECTION("plus and minus -- vector")
  {
    fk::vector<TestType> const vec{1, -1, 1, -1};
    TestType const gold = 2;
    REQUIRE(l2_norm(vec) == gold);
  }
}

TEMPLATE_TEST_CASE("inf_norm function", "[matlab]", float, double)
{
  SECTION("zeros -- vector")
  {
    fk::vector<TestType> const vec{0, 0, 0, 0, 0};
    TestType const gold = 0;
    REQUIRE(inf_norm(vec) == gold);
  }

  SECTION("plus and minus -- vector")
  {
    fk::vector<TestType> const vec{1, -1, 1, -4};
    TestType const gold = 4;
    REQUIRE(inf_norm(vec) == gold);
  }
}

TEST_CASE("read_vector_from_bin_file returns expected vector", "[matlab]")
{
  SECTION("read_vector_from_bin_file gets 100-element row vector")
  {
    fk::vector<double> gold = linspace<double>(-1, 1);
    fk::vector<double> test = read_vector_from_bin_file(
        "../testing/generated-inputs/matlab_utilities/"
        "read_vector_bin_neg1_1_100.dat");
    REQUIRE(test == gold);
  }
  SECTION("read_vector_from_bin_file gets 100-element column vector")
  {
    fk::vector<double> gold = linspace<double>(-1, 1);
    fk::vector<double> test = read_vector_from_bin_file(
        "../testing/generated-inputs/matlab_utilities/"
        "read_vector_bin_neg1_1_100T.dat");
    REQUIRE(test == gold);
  }
  SECTION("read_vector_from_bin_file fails on non-existent path")
  {
    fk::vector<double> test =
        read_vector_from_bin_file("this/path/does/not/exist");
    REQUIRE(test.size() == 0);
  }
}

TEST_CASE("read_vector_from_txt_file returns expected vector", "[matlab]")
{
  SECTION("read_vector_from_txt_file gets 100-element row vector")
  {
    fk::vector<double> gold = linspace<double>(-1, 1);
    fk::vector<double> test = read_vector_from_txt_file(
        "../testing/generated-inputs/matlab_utilities/"
        "read_vector_txt_neg1_1_100.dat");
    REQUIRE(test == gold);
  }
  SECTION("read_vector_from_txt_file gets 100-element column vector")
  {
    fk::vector<double> gold = linspace<double>(-1, 1);
    fk::vector<double> test = read_vector_from_txt_file(
        "../testing/generated-inputs/matlab_utilities/"
        "read_vector_txt_neg1_1_100T.dat");
    REQUIRE(test == gold);
  }
  SECTION("read_vector_from_txt_file fails on non-existent path")
  {
    fk::vector<double> test =
        read_vector_from_txt_file("this/path/does/not/exist");
    REQUIRE(test.size() == 0);
  }
}

TEST_CASE("read_matrix_from_txt_file returns expected vector", "[matlab]")
{
  SECTION("read_matrix_from_txt_file gets 5,5 matrix")
  {
    auto gold = fk::matrix<double>(5, 5);
    // generate the golden matrix
    for (int i = 0; i < 5; i++)
      for (int j = 0; j < 5; j++)
        gold(i, j) = 17.0 / (i + 1 + j);

    fk::matrix<double> test = read_matrix_from_txt_file(
        "../testing/generated-inputs/matlab_utilities/read_matrix_txt_5x5.dat");
    REQUIRE(test == gold);
  }
  SECTION("read_matrix_from_txt_file fails on non-existent path")
  {
    fk::matrix<double> test =
        read_matrix_from_txt_file("this/path/does/not/exist");
    REQUIRE(test.size() == 0);
  }
}

TEST_CASE("read_scalar_from_txt_file returns expected value", "[matlab]")
{
  SECTION("get stored scalar")
  {
    double const gold = 42;
    double const test = read_scalar_from_txt_file(
        "../testing/generated-inputs/matlab_utilities/read_scalar_42.dat");
    REQUIRE(gold == test);
  }
}

TEMPLATE_TEST_CASE(
    "reshape() matches matlab implementation for 2d matrices ony", "[matlab]",
    float, double, int)
{
  SECTION("reshape 2x2 to 1x4")
  {
    fk::matrix<TestType> matrix{{1, 3}, {2, 4}};
    fk::matrix<TestType> test = reshape<TestType>(matrix, 1, 4);
    fk::matrix<TestType> gold{{1, 2, 3, 4}};
    REQUIRE(test == gold);
  }

  SECTION("reshape 4x6 to 2x12")
  {
    fk::matrix<TestType> matrix{{1, 5, 9, 13, 17, 21},
                                {2, 6, 10, 14, 18, 22},
                                {3, 7, 11, 15, 19, 23},
                                {4, 8, 12, 16, 20, 24}};
    fk::matrix<TestType> test = reshape<TestType>(matrix, 2, 12);
    fk::matrix<TestType> gold{{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23},
                              {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}};
    REQUIRE(test == gold);
  }
}
