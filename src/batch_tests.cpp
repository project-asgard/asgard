#include "tests_general.hpp"

#include "batch.hpp"
#include "tensors.hpp"

TEMPLATE_TEST_CASE("batch_list", "[batch]", float, double)
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
  SECTION("batch_list: constructors, copy/move")
  {
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
    }

    SECTION("move assignment")
    {
      batch_list<TestType> test(num_batch, nrows, ncols, stride);
      batch_list<TestType> gold_copy(gold);
      test = std::move(gold_copy);
      REQUIRE(test == gold);
    }
  }

  SECTION("batch_list: insert/clear and getters")
  {
    SECTION("insert/getter")
    {
      batch_list<TestType> test(num_batch, nrows, ncols, stride);
      test.insert(first_v, 0);
      TestType *const *ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == nullptr);
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      test.insert(third_v, 2);
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      test.insert(second_v, 1);
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == second_v.data());
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));
    }

    SECTION("clear")
    {
      batch_list<TestType> test(gold);

      // clear should return true when
      // an element was assigned to that index
      REQUIRE(test.clear(0));
      TestType *const *ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == nullptr);
      REQUIRE(ptr_list[1] == second_v.data());
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      // clear should return false when
      // no element was assigned to that index
      REQUIRE(!test.clear(0));

      REQUIRE(test.clear(1));
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == nullptr);
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      REQUIRE(!test.clear(1));

      REQUIRE(test.clear(2));
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == nullptr);
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == nullptr);
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      REQUIRE(!test.clear(2));
    }
  }

  SECTION("batch_list: utility functions")
  {
    SECTION("is_filled")
    {
      REQUIRE(gold.is_filled());
      batch_list<TestType> test(gold);
      test.clear(0);
      REQUIRE(!test.is_filled());
    }

    SECTION("clear_all")
    {
      batch_list<TestType> const test = [&] {
        batch_list<TestType> gold_copy(gold);
        return gold_copy.clear_all();
      }();

      for (TestType *const ptr : test)
      {
        REQUIRE(ptr == nullptr);
      }
    }

    SECTION("const iterator")
    {
      TestType **const test = new TestType *[num_batch]();
      int counter           = 0;
      for (TestType *const ptr : gold)
      {
        test[counter++] = ptr;
      }

      TestType *const *const gold_list = gold.get_list();

      for (int i = 0; i < num_batch; ++i)
      {
        REQUIRE(test[i] == gold_list[i]);
      }
    }
  }
}
