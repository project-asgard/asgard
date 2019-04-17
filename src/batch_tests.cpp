#include "tests_general.hpp"

#include "batch.hpp"

#include "tensors.hpp"

TEMPLATE_TEST_CASE("batch_list: constructors, copy/move", "[batch]", float,
                   double)
{
  // clang-format off
  fk::matrix<TestType> const first {
         {12, 22, 32},
         {13, 23, 33},
         {14, 24, 34},
         {15, 25, 35},
         {16, 26, 36},
  };
  fk::matrix<TestType> const second {
         {17, 27, 37},
         {18, 28, 38},
         {19, 29, 39},
         {20, 30, 40},
         {21, 31, 41},
  };
  fk::matrix<TestType> const third {
         {22, 32, 42},
         {23, 33, 43},
         {24, 34, 44},
         {25, 35, 45},
         {26, 36, 46},
  }; // clang-format on

  int const start_row = 0;
  int const stop_row  = 3;
  int const nrows     = stop_row - start_row + 1;
  int const start_col = 1;
  int const stop_col  = 2;
  int const ncols     = stop_col - start_col + 1;
  int const stride    = first.nrows();

  fk::matrix<TestType, mem_type::view> const first_v(first, start_row, stop_row,
                                                     start_col, stop_col);
  fk::matrix<TestType, mem_type::view> const second_v(
      second, start_row, stop_row, start_col, stop_col);
  fk::matrix<TestType, mem_type::view> const third_v(third, start_row, stop_row,
                                                     start_col, stop_col);

  int const num_batch             = 3;
  batch_list<TestType> const gold = [&] {
    batch_list<TestType> builder(num_batch, nrows, ncols, stride);

    builder.insert(first_v, 0);
    builder.insert(second_v, 1);
    builder.insert(third_v, 2);

    return builder;
  }();

  SECTION("constructor")
  {
    batch_list<TestType> const empty(num_batch, nrows, ncols, stride);
    REQUIRE(empty.num_batch == num_batch);
    REQUIRE(empty.nrows == nrows);
    REQUIRE(empty.ncols == ncols);
    REQUIRE(empty.stride == stride);

    for (TestType *const ptr : empty)
    {
      REQUIRE(ptr == nullptr);
    }
  }

  SECTION("copy construction")
  {
    batch_list<TestType> const gold_copy(gold);
    REQUIRE(gold_copy == gold);
  }

  SECTION("copy assignment")
  {
    batch_list<TestType> test(num_batch, nrows, ncols, stride);
    test = gold;
    REQUIRE(test == gold);
  }

  SECTION("move construction")
  {
    batch_list<TestType> gold_copy(gold);
    batch_list const test = std::move(gold_copy);
    REQUIRE(test == gold);
    REQUIRE(gold_copy.get_list() == nullptr);
  }

  SECTION("move assignment")
  {
    batch_list<TestType> test(num_batch, nrows, ncols, stride);
    batch_list<TestType> gold_copy(gold);
    test = std::move(gold_copy);
    REQUIRE(test == gold);
    REQUIRE(gold_copy.get_list() == nullptr);
  }
}

TEMPLATE_TEST_CASE("batch_list: insert/remove", "[batch]", float, double)
{
  SECTION("insert") {}

  SECTION("remove") {}
}

TEMPLATE_TEST_CASE("batch_list: getter", "[batch]", float, double) {}

TEMPLATE_TEST_CASE("batch_list: utility functions", "[batch]", float, double)
{
  SECTION("is_filled") {}

  SECTION("clear_all") {}

  SECTION("iterators") {}
}

TEMPLATE_TEST_CASE("free function: execute gemm", "[batch]", float, double)
{
  REQUIRE(true);
}
