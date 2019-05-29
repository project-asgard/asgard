#include "batch.hpp"
#include "coefficients.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"
#include <numeric>
#include <random>
TEMPLATE_TEST_CASE("batch", "[batch]", float, double)
{
  bool const do_trans = true;

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

  int const num_batch        = 3;
  batch<TestType> const gold = [&] {
    batch<TestType> builder(num_batch, nrows, ncols, stride, do_trans);

    builder.assign_entry(first_v, 0);
    builder.assign_entry(second_v, 1);
    builder.assign_entry(third_v, 2);

    return builder;
  }();

  SECTION("batch: constructors, copy/move")
  {
    SECTION("constructor")
    {
      batch<TestType> const empty(num_batch, nrows, ncols, stride, do_trans);
      REQUIRE(empty.num_entries() == num_batch);
      REQUIRE(empty.nrows() == nrows);
      REQUIRE(empty.ncols() == ncols);
      REQUIRE(empty.get_stride() == stride);
      REQUIRE(empty.get_trans() == do_trans);

      for (TestType *const ptr : empty)
      {
        REQUIRE(ptr == nullptr);
      }
    }

    SECTION("copy construction")
    {
      batch<TestType> const gold_copy(gold);
      REQUIRE(gold_copy == gold);
    }

    SECTION("copy assignment")
    {
      batch<TestType> test(num_batch, nrows, ncols, stride, do_trans);
      test = gold;
      REQUIRE(test == gold);
    }

    SECTION("move construction")
    {
      batch<TestType> gold_copy(gold);
      batch const test = std::move(gold_copy);
      REQUIRE(test == gold);
    }

    SECTION("move assignment")
    {
      batch<TestType> test(num_batch, nrows, ncols, stride, do_trans);
      batch<TestType> gold_copy(gold);
      test = std::move(gold_copy);
      REQUIRE(test == gold);
    }
  }

  SECTION("batch: insert/clear and getters")
  {
    SECTION("insert/getter")
    {
      batch<TestType> test(num_batch, nrows, ncols, stride, do_trans);
      test.assign_entry(first_v, 0);
      TestType *const *ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == nullptr);
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      test.assign_entry(third_v, 2);
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      test.assign_entry(second_v, 1);
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
      batch<TestType> test(gold);

      // clear should return true when
      // an element was assigned to that index
      REQUIRE(test.clear_entry(0));
      TestType *const *ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == nullptr);
      REQUIRE(ptr_list[1] == second_v.data());
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      // clear should return false when
      // no element was assigned to that index
      REQUIRE(!test.clear_entry(0));

      REQUIRE(test.clear_entry(1));
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == nullptr);
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      REQUIRE(!test.clear_entry(1));

      REQUIRE(test.clear_entry(2));
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == nullptr);
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == nullptr);
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      REQUIRE(!test.clear_entry(2));
    }
  }

  SECTION("batch: utility functions")
  {
    SECTION("is_filled")
    {
      REQUIRE(gold.is_filled());
      batch<TestType> test(gold);
      test.clear_entry(0);
      REQUIRE(!test.is_filled());
    }

    SECTION("clear_all")
    {
      batch<TestType> const test = [&] {
        batch<TestType> gold_copy(gold);
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

TEMPLATE_TEST_CASE("batched gemm", "[batch]", float, double)
{
  int const num_batch = 3;
  // clang-format off
  fk::matrix<TestType> const a1 {
         {12, 22, 32},
         {13, 23, 33},
         {14, 24, 34},
         {15, 25, 35},
         {16, 26, 36},
  };
  fk::matrix<TestType> const a2 {
         {17, 27, 37},
         {18, 28, 38},
         {19, 29, 39},
         {20, 30, 40},
         {21, 31, 41},
  };
  fk::matrix<TestType> const a3 {
         {22, 32, 42},
         {23, 33, 43},
         {24, 34, 44},
         {25, 35, 45},
         {26, 36, 46},
  };  

  fk::matrix<TestType> const b1 {
         {27, 37, 47},
         {28, 38, 48},
         {29, 39, 49},
         {30, 40, 50},
  };
  fk::matrix<TestType> const b2 {
         {31, 41, 51},
         {32, 42, 52},
         {33, 43, 53},
         {34, 44, 54},
  };
  fk::matrix<TestType> const b3 {
         {35, 45, 55},
         {36, 46, 56},
         {37, 47, 57},
         {38, 48, 58},
  };
  // clang-format on

  SECTION("batched gemm: no trans, no trans, alpha = 1.0, beta = 0.0")
  {
    bool const trans_a = false;
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int const a_nrows     = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int const a_ncols     = a_stop_col - a_start_col + 1;
    int const a_stride    = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    batch<TestType> const a_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, a_ncols, a_stride, trans_a);

      builder.assign_entry(a1_v, 0);
      builder.assign_entry(a2_v, 1);
      builder.assign_entry(a3_v, 2);

      return builder;
    }();

    bool const trans_b = false;
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_nrows     = b_stop_row - b_start_row + 1;
    int const b_start_col = 2;
    int const b_stop_col  = 2;
    int const b_ncols     = b_stop_col - b_start_col + 1;
    int const b_stride    = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    batch<TestType> const b_batch = [&] {
      batch<TestType> builder(num_batch, b_nrows, b_ncols, b_stride, trans_b);

      builder.assign_entry(b1_v, 0);
      builder.assign_entry(b2_v, 1);
      builder.assign_entry(b3_v, 2);

      return builder;
    }();

    // make 2x1 "c" views
    fk::matrix<TestType> c(6, 1);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 0);

    batch<TestType> const c_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, b_ncols, c.nrows(), false);
      builder.assign_entry(c1_v, 0);
      builder.assign_entry(c2_v, 1);
      builder.assign_entry(c3_v, 2);
      return builder;
    }();

    // do the math to create gold matrix
    fk::matrix<TestType> gold(6, 1);
    fk::matrix<TestType, mem_type::view> gold1_v(gold, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> gold2_v(gold, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> gold3_v(gold, 4, 5, 0, 0);
    gold1_v = a1_v * b1_v;
    gold2_v = a2_v * b2_v;
    gold3_v = a3_v * b3_v;

    // call batched gemm
    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a_batch, b_batch, c_batch, alpha, beta);

    // compare
    REQUIRE(c == gold);
  }

  auto get_trans =
      [](fk::matrix<TestType, mem_type::view> orig) -> fk::matrix<TestType> {
    fk::matrix<TestType> builder(orig);
    return builder.transpose();
  };

  SECTION("batched gemm: trans a, no trans b, alpha = 1.0, beta = 0.0")
  {
    bool const trans_a = true;
    // make 3x2 (pre-trans) "a" views
    int const a_start_row = 1;
    int const a_stop_row  = 3;
    int const a_nrows     = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 1;
    int const a_ncols     = a_stop_col - a_start_col + 1;
    int const a_stride    = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    // make the transposed versions for forming golden value
    fk::matrix<TestType> const a1_t = get_trans(a1_v);
    fk::matrix<TestType> const a2_t = get_trans(a2_v);
    fk::matrix<TestType> const a3_t = get_trans(a3_v);

    batch<TestType> const a_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, a_ncols, a_stride, trans_a);

      builder.assign_entry(a1_v, 0);
      builder.assign_entry(a2_v, 1);
      builder.assign_entry(a3_v, 2);

      return builder;
    }();

    bool const trans_b = false;
    // make 3x2 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_nrows     = b_stop_row - b_start_row + 1;
    int const b_start_col = 1;
    int const b_stop_col  = 2;
    int const b_ncols     = b_stop_col - b_start_col + 1;
    int const b_stride    = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    batch<TestType> const b_batch = [&] {
      batch<TestType> builder(num_batch, b_nrows, b_ncols, b_stride, trans_b);

      builder.assign_entry(b1_v, 0);
      builder.assign_entry(b2_v, 1);
      builder.assign_entry(b3_v, 2);

      return builder;
    }();

    // make 3x2 "c" views
    fk::matrix<TestType> c(6, 2);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 1);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 1);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 1);

    batch<TestType> const c_batch = [&] {
      batch<TestType> builder(num_batch, a_ncols, b_ncols, c.nrows(), false);
      builder.assign_entry(c1_v, 0);
      builder.assign_entry(c2_v, 1);
      builder.assign_entry(c3_v, 2);
      return builder;
    }();

    // do the math to create gold matrix
    fk::matrix<TestType> gold(6, 2);
    fk::matrix<TestType, mem_type::view> gold1_v(gold, 0, 1, 0, 1);
    fk::matrix<TestType, mem_type::view> gold2_v(gold, 2, 3, 0, 1);
    fk::matrix<TestType, mem_type::view> gold3_v(gold, 4, 5, 0, 1);

    gold1_v = a1_t * b1_v;
    gold2_v = a2_t * b2_v;
    gold3_v = a3_t * b3_v;

    // call batched gemm
    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a_batch, b_batch, c_batch, alpha, beta);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans a, trans b, alpha = 1.0, beta = 0.0")
  {
    bool const trans_a = false;
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int const a_nrows     = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int const a_ncols     = a_stop_col - a_start_col + 1;
    int const a_stride    = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    batch<TestType> const a_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, a_ncols, a_stride, trans_a);

      builder.assign_entry(a1_v, 0);
      builder.assign_entry(a2_v, 1);
      builder.assign_entry(a3_v, 2);

      return builder;
    }();

    bool const trans_b = true;
    // make 2x3 (pre trans) "b" views
    int const b_start_row = 0;
    int const b_stop_row  = 1;
    int const b_nrows     = b_stop_row - b_start_row + 1;
    int const b_start_col = 0;
    int const b_stop_col  = 2;
    int const b_ncols     = b_stop_col - b_start_col + 1;
    int const b_stride    = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    batch<TestType> const b_batch = [&] {
      batch<TestType> builder(num_batch, b_nrows, b_ncols, b_stride, trans_b);

      builder.assign_entry(b1_v, 0);
      builder.assign_entry(b2_v, 1);
      builder.assign_entry(b3_v, 2);

      return builder;
    }();

    // make the transposed versions for forming golden value
    fk::matrix<TestType> const b1_t = get_trans(b1_v);
    fk::matrix<TestType> const b2_t = get_trans(b2_v);
    fk::matrix<TestType> const b3_t = get_trans(b3_v);

    // make 2x2 "c" views
    fk::matrix<TestType> c(6, 2);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 1);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 1);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 1);

    batch<TestType> const c_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, b_nrows, c.nrows(), false);

      builder.assign_entry(c1_v, 0);
      builder.assign_entry(c2_v, 1);
      builder.assign_entry(c3_v, 2);
      return builder;
    }();

    // do the math to create gold matrix
    fk::matrix<TestType> gold(6, 2);
    fk::matrix<TestType, mem_type::view> gold1_v(gold, 0, 1, 0, 1);
    fk::matrix<TestType, mem_type::view> gold2_v(gold, 2, 3, 0, 1);
    fk::matrix<TestType, mem_type::view> gold3_v(gold, 4, 5, 0, 1);
    gold1_v = a1_v * b1_t;
    gold2_v = a2_v * b2_t;
    gold3_v = a3_v * b3_t;

    // call batched gemm
    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a_batch, b_batch, c_batch, alpha, beta);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: trans a, trans b, alpha = 1.0, beta = 0.0")
  {
    bool const trans_a = true;
    // make 3x2 (pre-trans) "a" views
    int const a_start_row = 1;
    int const a_stop_row  = 3;
    int const a_nrows     = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 1;
    int const a_ncols     = a_stop_col - a_start_col + 1;
    int const a_stride    = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    // make the transposed versions for forming golden value
    fk::matrix<TestType> const a1_t = get_trans(a1_v);
    fk::matrix<TestType> const a2_t = get_trans(a2_v);
    fk::matrix<TestType> const a3_t = get_trans(a3_v);

    batch<TestType> const a_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, a_ncols, a_stride, trans_a);

      builder.assign_entry(a1_v, 0);
      builder.assign_entry(a2_v, 1);
      builder.assign_entry(a3_v, 2);

      return builder;
    }();

    bool const trans_b = true;
    // make 2x3 (pre trans) "b" views
    int const b_start_row = 0;
    int const b_stop_row  = 1;
    int const b_nrows     = b_stop_row - b_start_row + 1;
    int const b_start_col = 0;
    int const b_stop_col  = 2;
    int const b_ncols     = b_stop_col - b_start_col + 1;
    int const b_stride    = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    batch<TestType> const b_batch = [&] {
      batch<TestType> builder(num_batch, b_nrows, b_ncols, b_stride, trans_b);

      builder.assign_entry(b1_v, 0);
      builder.assign_entry(b2_v, 1);
      builder.assign_entry(b3_v, 2);

      return builder;
    }();

    // make the transposed versions for forming golden value
    fk::matrix<TestType> const b1_t = get_trans(b1_v);
    fk::matrix<TestType> const b2_t = get_trans(b2_v);
    fk::matrix<TestType> const b3_t = get_trans(b3_v);

    // make 2x2 "c" views
    fk::matrix<TestType> c(6, 2);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 1);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 1);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 1);

    batch<TestType> const c_batch = [&] {
      batch<TestType> builder(num_batch, a_ncols, b_nrows, c.nrows(), false);
      builder.assign_entry(c1_v, 0);
      builder.assign_entry(c2_v, 1);
      builder.assign_entry(c3_v, 2);
      return builder;
    }();

    // do the math to create gold matrix
    fk::matrix<TestType> gold(6, 2);
    fk::matrix<TestType, mem_type::view> gold1_v(gold, 0, 1, 0, 1);
    fk::matrix<TestType, mem_type::view> gold2_v(gold, 2, 3, 0, 1);
    fk::matrix<TestType, mem_type::view> gold3_v(gold, 4, 5, 0, 1);
    gold1_v = a1_t * b1_t;
    gold2_v = a2_t * b2_t;
    gold3_v = a3_t * b3_t;

    // call batched gemm
    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a_batch, b_batch, c_batch, alpha, beta);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 1.0, beta = 1.0")
  {
    bool const trans_a = false;
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int const a_nrows     = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int const a_ncols     = a_stop_col - a_start_col + 1;
    int const a_stride    = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    batch<TestType> const a_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, a_ncols, a_stride, trans_a);

      builder.assign_entry(a1_v, 0);
      builder.assign_entry(a2_v, 1);
      builder.assign_entry(a3_v, 2);

      return builder;
    }();

    bool const trans_b = false;
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_nrows     = b_stop_row - b_start_row + 1;
    int const b_start_col = 2;
    int const b_stop_col  = 2;
    int const b_ncols     = b_stop_col - b_start_col + 1;
    int const b_stride    = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    batch<TestType> const b_batch = [&] {
      batch<TestType> builder(num_batch, b_nrows, b_ncols, b_stride, trans_b);

      builder.assign_entry(b1_v, 0);
      builder.assign_entry(b2_v, 1);
      builder.assign_entry(b3_v, 2);

      return builder;
    }();

    // make 2x1 "c" views
    // clang-format off
    fk::matrix<TestType> c {
	    {3548},
  	    {3695},
  	    {4631},
  	    {4790},
  	    {5834},
  	    {6005},
    }; // clang-format on
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 0);

    batch<TestType> const c_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, b_ncols, c.nrows(), false);
      builder.assign_entry(c1_v, 0);
      builder.assign_entry(c2_v, 1);
      builder.assign_entry(c3_v, 2);
      return builder;
    }();

    // do the math to create gold matrix
    fk::matrix<TestType> gold(6, 1);
    fk::matrix<TestType, mem_type::view> gold1_v(gold, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> gold2_v(gold, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> gold3_v(gold, 4, 5, 0, 0);
    gold1_v = (a1_v * b1_v) * 2.0;
    gold2_v = (a2_v * b2_v) * 2.0;
    gold3_v = (a3_v * b3_v) * 2.0;

    // call batched gemm
    TestType const alpha = 1.0;
    TestType const beta  = 1.0;
    batched_gemm(a_batch, b_batch, c_batch, alpha, beta);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 3.0, beta = 0.0")
  {
    bool const trans_a = false;
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int const a_nrows     = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int const a_ncols     = a_stop_col - a_start_col + 1;
    int const a_stride    = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    batch<TestType> const a_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, a_ncols, a_stride, trans_a);

      builder.assign_entry(a1_v, 0);
      builder.assign_entry(a2_v, 1);
      builder.assign_entry(a3_v, 2);

      return builder;
    }();

    bool const trans_b = false;
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_nrows     = b_stop_row - b_start_row + 1;
    int const b_start_col = 2;
    int const b_stop_col  = 2;
    int const b_ncols     = b_stop_col - b_start_col + 1;
    int const b_stride    = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    batch<TestType> const b_batch = [&] {
      batch<TestType> builder(num_batch, b_nrows, b_ncols, b_stride, trans_b);

      builder.assign_entry(b1_v, 0);
      builder.assign_entry(b2_v, 1);
      builder.assign_entry(b3_v, 2);

      return builder;
    }();

    // make 2x1 "c" views
    fk::matrix<TestType> c(6, 1);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 0);

    batch<TestType> const c_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, b_ncols, c.nrows(), false);
      builder.assign_entry(c1_v, 0);
      builder.assign_entry(c2_v, 1);
      builder.assign_entry(c3_v, 2);
      return builder;
    }();

    // do the math to create gold matrix
    fk::matrix<TestType> gold(6, 1);
    fk::matrix<TestType, mem_type::view> gold1_v(gold, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> gold2_v(gold, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> gold3_v(gold, 4, 5, 0, 0);
    gold1_v = (a1_v * b1_v) * 3.0;
    gold2_v = (a2_v * b2_v) * 3.0;
    gold3_v = (a3_v * b3_v) * 3.0;

    // call batched gemm
    TestType const alpha = 3.0;
    TestType const beta  = 0.0;
    batched_gemm(a_batch, b_batch, c_batch, alpha, beta);

    // compare
    REQUIRE(c == gold);
  }
}

TEMPLATE_TEST_CASE("batched gemv", "[batch]", float, double)
{
  int const num_batch = 3;
  // clang-format off
  fk::matrix<TestType> const a1 {
         {12, 22, 32},
         {13, 23, 33},
         {14, 24, 34},
         {15, 25, 35},
         {16, 26, 36},
  };
  fk::matrix<TestType> const a2 {
         {17, 27, 37},
         {18, 28, 38},
         {19, 29, 39},
         {20, 30, 40},
         {21, 31, 41},
  };
  fk::matrix<TestType> const a3 {
         {22, 32, 42},
         {23, 33, 43},
         {24, 34, 44},
         {25, 35, 45},
         {26, 36, 46},
  };  

  fk::vector<TestType> const b1 {
         {1},
         {1},
	 {1},
         {1},
  };
  // clang-format on

  // test batched gemv as reduction tool w/ unit vector
  SECTION("batched gemv: no trans, alpha = 1.0, beta = 0.0")
  {
    bool const trans_a = false;
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int const a_nrows     = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int const a_ncols     = a_stop_col - a_start_col + 1;
    int const a_stride    = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    batch<TestType> const a_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, a_ncols, a_stride, trans_a);

      builder.assign_entry(a1_v, 0);
      builder.assign_entry(a2_v, 1);
      builder.assign_entry(a3_v, 2);

      return builder;
    }();

    bool const trans_b = false;
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_nrows     = b_stop_row - b_start_row + 1;
    int const b_ncols     = 1;
    int const b_stride    = 1;

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_nrows, b_ncols, 0);

    batch<TestType> const b_batch = [&] {
      batch<TestType> builder(num_batch, b_nrows, b_ncols, b_stride, trans_b);

      builder.assign_entry(b1_v, 0);
      builder.assign_entry(b1_v, 1);
      builder.assign_entry(b1_v, 2);

      return builder;
    }();

    // make 2x1 "c" views
    fk::matrix<TestType> c(6, 1);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 0);

    batch<TestType> const c_batch = [&] {
      batch<TestType> builder(num_batch, a_nrows, b_ncols, 1, false);
      builder.assign_entry(c1_v, 0);
      builder.assign_entry(c2_v, 1);
      builder.assign_entry(c3_v, 2);
      return builder;
    }();

    // do the math to create gold matrix
    fk::matrix<TestType> gold(6, 1);
    fk::matrix<TestType, mem_type::view> gold1_v(gold, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> gold2_v(gold, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> gold3_v(gold, 4, 5, 0, 0);
    gold1_v = a1_v * b1_v;
    gold2_v = a2_v * b1_v;
    gold3_v = a3_v * b1_v;

    // call batched gemv
    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemv(a_batch, b_batch, c_batch, alpha, beta);

    // compare
    REQUIRE(c == gold);
  }
}
TEMPLATE_TEST_CASE("batch allocator", "[batch]", float, double)
{
  SECTION("1d, deg 3")
  {
    int const level     = 2;
    int const degree    = 3;
    int const num_elems = 60;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    int const stride = pde->get_coefficients(0, 0).stride();

    int const gold_size = pde->num_terms * num_elems;

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == 1);
    batch_operands_set<TestType> const batches_dim0 = batches[0];

    int const gold_rows_a   = degree;
    int const gold_cols_a   = degree;
    int const gold_stride_a = stride;
    bool const gold_trans_a = false;
    assert(batches_dim0[0].num_entries() == gold_size);
    assert(batches_dim0[0].nrows() == gold_rows_a);
    assert(batches_dim0[0].ncols() == gold_cols_a);
    assert(batches_dim0[0].get_stride() == gold_stride_a);
    assert(batches_dim0[0].get_trans() == gold_trans_a);

    int const gold_rows_b   = degree;
    int const gold_cols_b   = std::pow(degree, pde->num_dims - 1);
    int const gold_stride_b = degree;
    bool const gold_trans_b = false;
    assert(batches_dim0[1].num_entries() == gold_size);
    assert(batches_dim0[1].nrows() == gold_rows_b);
    assert(batches_dim0[1].ncols() == gold_cols_b);
    assert(batches_dim0[1].get_stride() == gold_stride_b);
    assert(batches_dim0[1].get_trans() == gold_trans_b);

    int const gold_rows_c   = gold_rows_a;
    int const gold_cols_c   = gold_cols_b;
    int const gold_stride_c = gold_rows_a;
    assert(batches_dim0[2].num_entries() == gold_size);
    assert(batches_dim0[2].nrows() == gold_rows_c);
    assert(batches_dim0[2].ncols() == gold_cols_c);
    assert(batches_dim0[2].get_stride() == gold_stride_c);
    assert(batches_dim0[2].get_trans() == false);
  }

  SECTION("1d, deg 6")
  {
    int const level     = 2;
    int const degree    = 6;
    int const num_elems = 400;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    int const stride = pde->get_coefficients(0, 0).stride();

    int const gold_size = pde->num_terms * num_elems;

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == 1);
    batch_operands_set<TestType> const batches_dim0 = batches[0];

    int const gold_rows_a   = degree;
    int const gold_cols_a   = degree;
    int const gold_stride_a = stride;
    bool const gold_trans_a = false;
    assert(batches_dim0[0].num_entries() == gold_size);
    assert(batches_dim0[0].nrows() == gold_rows_a);
    assert(batches_dim0[0].ncols() == gold_cols_a);
    assert(batches_dim0[0].get_stride() == gold_stride_a);
    assert(batches_dim0[0].get_trans() == gold_trans_a);

    int const gold_rows_b   = degree;
    int const gold_cols_b   = std::pow(degree, pde->num_dims - 1);
    int const gold_stride_b = degree;
    bool const gold_trans_b = false;
    assert(batches_dim0[1].num_entries() == gold_size);
    assert(batches_dim0[1].nrows() == gold_rows_b);
    assert(batches_dim0[1].ncols() == gold_cols_b);
    assert(batches_dim0[1].get_stride() == gold_stride_b);
    assert(batches_dim0[1].get_trans() == gold_trans_b);

    int const gold_rows_c   = gold_rows_a;
    int const gold_cols_c   = gold_cols_b;
    int const gold_stride_c = gold_rows_a;
    assert(batches_dim0[2].num_entries() == gold_size);
    assert(batches_dim0[2].nrows() == gold_rows_c);
    assert(batches_dim0[2].ncols() == gold_cols_c);
    assert(batches_dim0[2].get_stride() == gold_stride_c);
    assert(batches_dim0[2].get_trans() == false);
  }

  SECTION("2d, deg 2")
  {
    int const level      = 2;
    int const degree     = 2;
    int const num_elems  = 101;
    int const dimensions = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == dimensions);

    int const gold_size = pde->num_terms * num_elems;
    for (int i = 0; i < dimensions; ++i)
    {
      int const stride = pde->get_coefficients(0, i).stride();
      batch_operands_set<TestType> const batch_dim = batches[i];
      int const gold_rows_a   = i == 0 ? degree : std::pow(degree, i);
      int const gold_cols_a   = degree;
      int const gold_stride_a = i == 0 ? stride : gold_rows_a;
      bool const gold_trans_a = false;
      assert(batch_dim[0].num_entries() == gold_size);
      assert(batch_dim[0].nrows() == gold_rows_a);
      assert(batch_dim[0].ncols() == gold_cols_a);
      assert(batch_dim[0].get_stride() == gold_stride_a);
      assert(batch_dim[0].get_trans() == gold_trans_a);

      int const gold_rows_b = degree;
      int const gold_cols_b =
          i == 0 ? std::pow(degree, pde->num_dims - 1) : degree;
      int const gold_stride_b = i == 0 ? degree : stride;
      bool const gold_trans_b = i == 0 ? false : true;
      assert(batch_dim[1].num_entries() == gold_size);
      assert(batch_dim[1].nrows() == gold_rows_b);
      assert(batch_dim[1].ncols() == gold_cols_b);
      assert(batch_dim[1].get_stride() == gold_stride_b);
      assert(batch_dim[1].get_trans() == gold_trans_b);

      int const gold_rows_c   = gold_rows_a;
      int const gold_cols_c   = gold_cols_b;
      int const gold_stride_c = gold_rows_a;
      assert(batch_dim[2].num_entries() == gold_size);
      assert(batch_dim[2].nrows() == gold_rows_c);
      assert(batch_dim[2].ncols() == gold_cols_c);
      assert(batch_dim[2].get_stride() == gold_stride_c);
      assert(batch_dim[2].get_trans() == false);
    }
  }

  SECTION("2d, deg 5")
  {
    int const level      = 2;
    int const degree     = 5;
    int const num_elems  = 251;
    int const dimensions = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == dimensions);

    int const gold_size = pde->num_terms * num_elems;
    for (int i = 0; i < dimensions; ++i)
    {
      int const stride = pde->get_coefficients(0, i).stride();
      batch_operands_set<TestType> const batch_dim = batches[i];
      int const gold_rows_a   = i == 0 ? degree : std::pow(degree, i);
      int const gold_cols_a   = degree;
      int const gold_stride_a = i == 0 ? stride : gold_rows_a;
      bool const gold_trans_a = false;
      assert(batch_dim[0].num_entries() == gold_size);
      assert(batch_dim[0].nrows() == gold_rows_a);
      assert(batch_dim[0].ncols() == gold_cols_a);
      assert(batch_dim[0].get_stride() == gold_stride_a);
      assert(batch_dim[0].get_trans() == gold_trans_a);

      int const gold_rows_b = degree;
      int const gold_cols_b =
          i == 0 ? std::pow(degree, pde->num_dims - 1) : degree;
      int const gold_stride_b = i == 0 ? degree : stride;
      bool const gold_trans_b = i == 0 ? false : true;
      assert(batch_dim[1].num_entries() == gold_size);
      assert(batch_dim[1].nrows() == gold_rows_b);
      assert(batch_dim[1].ncols() == gold_cols_b);
      assert(batch_dim[1].get_stride() == gold_stride_b);
      assert(batch_dim[1].get_trans() == gold_trans_b);

      int const gold_rows_c   = gold_rows_a;
      int const gold_cols_c   = gold_cols_b;
      int const gold_stride_c = gold_rows_a;
      assert(batch_dim[2].num_entries() == gold_size);
      assert(batch_dim[2].nrows() == gold_rows_c);
      assert(batch_dim[2].ncols() == gold_cols_c);
      assert(batch_dim[2].get_stride() == gold_stride_c);
      assert(batch_dim[2].get_trans() == false);
    }
  }
  SECTION("6d, deg 4")
  {
    int const level      = 3;
    int const degree     = 4;
    int const num_elems  = 100;
    int const dimensions = 6;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == dimensions);

    for (int i = 0; i < dimensions; ++i)
    {
      int const gold_size = [&] {
        if (i == 0 || i == dimensions - 1)
        {
          return pde->num_terms * num_elems;
        }
        return static_cast<int>(std::pow(degree, (dimensions - i - 1))) *
               pde->num_terms * num_elems;
      }();
      int const stride = pde->get_coefficients(0, i).stride();
      batch_operands_set<TestType> const batch_dim = batches[i];
      int const gold_rows_a   = i == 0 ? degree : std::pow(degree, i);
      int const gold_cols_a   = degree;
      int const gold_stride_a = i == 0 ? stride : gold_rows_a;
      bool const gold_trans_a = false;

      assert(batch_dim[0].num_entries() == gold_size);
      assert(batch_dim[0].nrows() == gold_rows_a);
      assert(batch_dim[0].ncols() == gold_cols_a);
      assert(batch_dim[0].get_stride() == gold_stride_a);
      assert(batch_dim[0].get_trans() == gold_trans_a);

      int const gold_rows_b = degree;
      int const gold_cols_b =
          i == 0 ? std::pow(degree, pde->num_dims - 1) : degree;
      int const gold_stride_b = i == 0 ? degree : stride;
      bool const gold_trans_b = i == 0 ? false : true;
      assert(batch_dim[1].num_entries() == gold_size);
      assert(batch_dim[1].nrows() == gold_rows_b);
      assert(batch_dim[1].ncols() == gold_cols_b);
      assert(batch_dim[1].get_stride() == gold_stride_b);
      assert(batch_dim[1].get_trans() == gold_trans_b);

      int const gold_rows_c   = gold_rows_a;
      int const gold_cols_c   = gold_cols_b;
      int const gold_stride_c = gold_rows_a;
      assert(batch_dim[2].num_entries() == gold_size);
      assert(batch_dim[2].nrows() == gold_rows_c);
      assert(batch_dim[2].ncols() == gold_cols_c);
      assert(batch_dim[2].get_stride() == gold_stride_c);
      assert(batch_dim[2].get_trans() == false);
    }
  }
}

TEMPLATE_TEST_CASE("kronmult batching", "[batch]", float, double)
{
  SECTION("1 element, 1d, 1 term")
  {
    int const degree    = 4;
    int const level     = 2;
    int const num_elems = 1;

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    fk::matrix<TestType> coefficient_matrix = pde->get_coefficients(0, 0);

    // clang-format off
    fk::matrix<TestType> const A {
        { 2,  3,  4,  5}, 
	{ 6,  7,  8,  9}, 
	{10, 11, 12, 13}, 
	{14, 15, 16, 17}};
    // clang-format on

    coefficient_matrix.set_submatrix(0, 0, A);
    fk::vector<TestType> x{18, 19, 20, 21};
    fk::vector<TestType> const gold = A * x;

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);
    fk::matrix<TestType, mem_type::view> A_view(coefficient_matrix, 0,
                                                degree - 1, 0, degree - 1);

    std::vector<fk::matrix<TestType, mem_type::view>> const As = {A_view};
    fk::vector<TestType, mem_type::view> x_view(x);
    fk::vector<TestType> y_own(degree);
    fk::vector<TestType, mem_type::view> y(y_own);
    std::vector<fk::vector<TestType, mem_type::view>> work_set = {};
    int const batch_offset                                     = 0;

    kronmult_to_batch_sets(As, x_view, y, work_set, batches, batch_offset,
                           *pde);

    batch<TestType> const a = batches[0][0];
    batch<TestType> const b = batches[0][1];
    batch<TestType> const c = batches[0][2];

    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a, b, c, alpha, beta);

    REQUIRE(gold == y);
  }

  SECTION("2 elements, 1d, 1 term")
  {
    int const degree    = 4;
    int const level     = 2;
    int const num_elems = 2;

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    fk::matrix<TestType> coefficient_matrix = pde->get_coefficients(0, 0);

    // clang-format off
    fk::matrix<TestType> const A {
        { 2,  3,  4,  5,  6,  7}, 
	{ 8,  9, 10, 11, 12, 13}, 
	{14, 15, 16, 17, 18, 19}, 
	{20, 21, 22, 23, 24, 25}, 
	{26, 27, 28, 29, 30, 31}, 
	{32, 33, 34, 35, 36, 37}};
    // clang-format on
    coefficient_matrix.set_submatrix(0, 0, A);
    fk::vector<TestType> x{18, 19, 20, 21};

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // each element addresses a slightly different part of the underlying
    // coefficients
    fk::matrix<TestType, mem_type::view> A_view_e0(coefficient_matrix, 0,
                                                   degree - 1, 0, degree - 1);
    fk::matrix<TestType, mem_type::view> A_view_e1(
        coefficient_matrix, 2, 2 + degree - 1, 2, 2 + degree - 1);

    fk::vector<TestType> const gold_e0 = A_view_e0 * x;
    fk::vector<TestType> const gold_e1 = A_view_e1 * x;
    fk::vector<TestType> const gold    = gold_e0 + gold_e1;

    fk::vector<TestType, mem_type::view> x_view(x);
    fk::vector<TestType> y_own(degree * num_elems);

    // schedule gemms for both elements
    int batch_offset                                              = 0;
    std::vector<fk::matrix<TestType, mem_type::view>> const As_e0 = {A_view_e0};
    fk::vector<TestType, mem_type::view> y_e0(y_own, 0, degree - 1);
    std::vector<fk::vector<TestType, mem_type::view>> work_set_e0 = {};
    kronmult_to_batch_sets(As_e0, x_view, y_e0, work_set_e0, batches,
                           batch_offset, *pde);

    batch_offset                                                  = 1;
    std::vector<fk::matrix<TestType, mem_type::view>> const As_e1 = {A_view_e1};
    fk::vector<TestType, mem_type::view> y_e1(y_own, degree, y_own.size() - 1);
    std::vector<fk::vector<TestType, mem_type::view>> work_set_e1 = {};

    kronmult_to_batch_sets(As_e1, x_view, y_e1, work_set_e1, batches,
                           batch_offset, *pde);

    batch<TestType> const a = batches[0][0];
    batch<TestType> const b = batches[0][1];
    batch<TestType> const c = batches[0][2];

    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a, b, c, alpha, beta);

    REQUIRE(gold_e0 == y_e0);
    REQUIRE(gold_e1 == y_e1);
    REQUIRE(gold == (y_e0 + y_e1));
  }

  SECTION("2 elements, 2d, 2 terms")
  {
    int const degree    = 5;
    int const level     = 2;
    int const num_elems = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    int const num_terms = 2;
    int const num_dims  = 2;
    // first, create example coefficient matrices in the pde
    int const dof = degree * std::pow(2, level);
    std::array<fk::matrix<TestType>, num_terms *num_dims> A_mats = {
        fk::matrix<TestType>(dof, dof), fk::matrix<TestType>(dof, dof),
        fk::matrix<TestType>(dof, dof), fk::matrix<TestType>(dof, dof)};

    // create different matrices for each term/dim pairing
    int start = 1;
    for (fk::matrix<TestType> &mat : A_mats)
    {
      std::iota(mat.begin(), mat.end(), start);
      start += dof;
    }

    // create input vector
    int const x_size = static_cast<int>(std::pow(degree, pde->num_dims));
    fk::vector<TestType> x(x_size);
    std::iota(x.begin(), x.end(), 1);

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // create intermediate workspaces
    // and output vectors
    fk::vector<TestType, mem_type::view> x_view(x);
    fk::vector<TestType> y_own(x_size * num_elems * num_terms);
    fk::vector<TestType> gold(x_size * num_elems * num_terms);
    fk::vector<TestType> work_own(x_size * num_elems * num_terms *
                                  std::min(num_dims - 1, 2));

    for (int i = 0; i < num_elems; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        // linearize index
        int const kron_index = i * num_terms + j;

        // address y space
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * std::min(num_dims - 1, 2);
        fk::vector<TestType, mem_type::view> y_view(y_own, y_index,
                                                    y_index + x_size - 1);
        fk::vector<TestType, mem_type::view> gold_view(gold, y_index,
                                                       y_index + x_size - 1);

        // intermediate workspace
        std::vector<fk::vector<TestType, mem_type::view>> work_views = {
            fk::vector<TestType, mem_type::view>(work_own, work_index,
                                                 work_index + x_size - 1)};

        // create A_views
        std::vector<fk::matrix<TestType, mem_type::view>> A_views;
        for (int k = 0; k < pde->num_dims; ++k)
        {
          int const start_row = degree * i;
          int const stop_row  = degree * (i + 1) - 1;
          int const start_col = 0;
          int const stop_col  = degree - 1;
          A_views.push_back(fk::matrix<TestType, mem_type::view>(
              A_mats[j * num_dims + k], start_row, stop_row, start_col,
              stop_col));
        }

        int const batch_offset = kron_index;
        kronmult_to_batch_sets(A_views, x_view, y_view, work_views, batches,
                               batch_offset, *pde);

        gold_view = (A_views[1].kron(A_views[0])) * x;
      }
    }

    for (int k = 0; k < pde->num_dims; ++k)
    {
      batch<TestType> const a = batches[k][0];
      batch<TestType> const b = batches[k][1];
      batch<TestType> const c = batches[k][2];
      TestType const alpha    = 1.0;
      TestType const beta     = 0.0;
      batched_gemm(a, b, c, alpha, beta);
    }

    REQUIRE(y_own == gold);
  }

  SECTION("1 element, 3d, 3 terms")
  {
    int const degree    = 5;
    int const level     = 2;
    int const num_elems = 1;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);

    int const num_terms = 3;
    int const num_dims  = 3;
    // first, create example coefficient matrices in the pde
    int const dof = degree * std::pow(2, level);
    std::vector<fk::matrix<TestType>> A_mats;
    for (int i = 0; i < num_terms * num_dims; ++i)
    {
      A_mats.push_back(fk::matrix<TestType>(dof, dof));
    }

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    // create different matrices for each term/dim pairing
    for (fk::matrix<TestType> &mat : A_mats)
    {
      std::generate(mat.begin(), mat.end(), gen);
    }

    // create input vector
    int const x_size = static_cast<int>(std::pow(degree, pde->num_dims));
    fk::vector<TestType> x(x_size);
    std::generate(x.begin(), x.end(), gen);

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // create intermediate workspaces
    // and output vectors
    fk::vector<TestType, mem_type::view> x_view(x);
    fk::vector<TestType> y_own(x_size * num_elems * num_terms);
    fk::vector<TestType> gold(x_size * num_elems * num_terms);
    fk::vector<TestType> work_own(x_size * num_elems * num_terms *
                                  std::min(num_dims - 1, 2));
    std::fill(work_own.begin(), work_own.end(), 0.0);
    std::fill(y_own.begin(), y_own.end(), 0.0);
    for (int i = 0; i < num_elems; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        // linearize index
        int const kron_index = pde->num_terms * i + j;

        // address y space
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * std::min(num_dims - 1, 2);
        fk::vector<TestType, mem_type::view> y_view(y_own, y_index,
                                                    y_index + x_size - 1);
        fk::vector<TestType, mem_type::view> gold_view(gold, y_index,
                                                       y_index + x_size - 1);

        // intermediate workspace
        std::vector<fk::vector<TestType, mem_type::view>> work_views = {
            fk::vector<TestType, mem_type::view>(work_own, work_index,
                                                 work_index + x_size - 1),
            fk::vector<TestType, mem_type::view>(work_own, work_index + x_size,
                                                 work_index + x_size * 2 - 1)};

        // create A_views
        std::vector<fk::matrix<TestType, mem_type::view>> A_views;
        for (int k = 0; k < pde->num_dims; ++k)
        {
          int const start_row = degree * i;
          int const stop_row  = degree * (i + 1) - 1;
          int const start_col = 0;
          int const stop_col  = degree - 1;

          A_views.push_back(fk::matrix<TestType, mem_type::view>(
              A_mats[j * num_dims + k], start_row, stop_row, start_col,
              stop_col));
        }

        int const batch_offset = kron_index;
        kronmult_to_batch_sets(A_views, x_view, y_view, work_views, batches,
                               batch_offset, *pde);

        gold_view = (A_views[2].kron(A_views[1].kron(A_views[0]))) * x;
      }
    }

    for (int k = 0; k < pde->num_dims; ++k)
    {
      batch<TestType> const a = batches[k][0];
      batch<TestType> const b = batches[k][1];
      batch<TestType> const c = batches[k][2];
      TestType const alpha    = 1.0;
      TestType const beta     = 0.0;
      batched_gemm(a, b, c, alpha, beta);
    }

    // this method of computing "correctness" borrowed from ed's tests:
    // https://code.ornl.gov/lmm/DG-SparseGrid/blob/reference/Kronmult/test1_batch.m
    fk::vector<TestType> const diff = gold - y_own;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        *std::max_element(diff.begin(), diff.end(), abs_compare);
    TestType const tol = std::numeric_limits<TestType>::epsilon();
    REQUIRE(result <= tol * gold.size());
  }

  SECTION("3 elements, 6d, 6 terms")
  {
    int const degree    = 2;
    int const level     = 2;
    int const num_elems = 3;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    int const num_terms = 6;
    int const num_dims  = 6;
    // first, create example coefficient matrices in the pde
    int const dof = degree * std::pow(2, level);
    std::vector<fk::matrix<TestType>> A_mats;
    for (int i = 0; i < num_terms * num_dims; ++i)
    {
      A_mats.push_back(fk::matrix<TestType>(dof, dof));
    }

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    // create different matrices for each term/dim pairing
    for (fk::matrix<TestType> &mat : A_mats)
    {
      std::generate(mat.begin(), mat.end(), gen);
    }

    // create input vector
    int const x_size = static_cast<int>(std::pow(degree, pde->num_dims));
    fk::vector<TestType> x(x_size);
    std::generate(x.begin(), x.end(), gen);

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // create intermediate workspaces
    // and output vectors
    fk::vector<TestType, mem_type::view> x_view(x);
    fk::vector<TestType> y_own(x_size * num_elems * num_terms);
    fk::vector<TestType> gold(x_size * num_elems * num_terms);
    fk::vector<TestType> work_own(x_size * num_elems * num_terms *
                                  std::min(num_dims - 1, 2));
    std::fill(work_own.begin(), work_own.end(), 0.0);
    std::fill(y_own.begin(), y_own.end(), 0.0);
    for (int i = 0; i < num_elems; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        // linearize index
        int const kron_index = pde->num_terms * i + j;

        // address y space
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * std::min(num_dims - 1, 2);
        fk::vector<TestType, mem_type::view> y_view(y_own, y_index,
                                                    y_index + x_size - 1);
        fk::vector<TestType, mem_type::view> gold_view(gold, y_index,
                                                       y_index + x_size - 1);

        // intermediate workspace
        std::vector<fk::vector<TestType, mem_type::view>> work_views = {
            fk::vector<TestType, mem_type::view>(work_own, work_index,
                                                 work_index + x_size - 1),
            fk::vector<TestType, mem_type::view>(work_own, work_index + x_size,
                                                 work_index + x_size * 2 - 1)};

        // create A_views
        std::vector<fk::matrix<TestType, mem_type::view>> A_views;
        for (int k = 0; k < pde->num_dims; ++k)
        {
          int const start_row = degree * i;
          int const stop_row  = degree * (i + 1) - 1;
          int const start_col = 0;
          int const stop_col  = degree - 1;

          A_views.push_back(fk::matrix<TestType, mem_type::view>(
              A_mats[j * num_dims + k], start_row, stop_row, start_col,
              stop_col));
        }

        int const batch_offset = kron_index;
        kronmult_to_batch_sets(A_views, x_view, y_view, work_views, batches,
                               batch_offset, *pde);

        gold_view = A_views[5].kron(A_views[4].kron(A_views[3].kron(
                        A_views[2].kron(A_views[1].kron(A_views[0]))))) *
                    x;
      }
    }

    for (int k = 0; k < pde->num_dims; ++k)
    {
      batch<TestType> const a = batches[k][0];
      batch<TestType> const b = batches[k][1];
      batch<TestType> const c = batches[k][2];
      TestType const alpha    = 1.0;
      TestType const beta     = 0.0;
      batched_gemm(a, b, c, alpha, beta);
    }

    // this method of computing "correctness" borrowed from ed's tests:
    // https://code.ornl.gov/lmm/DG-SparseGrid/blob/reference/Kronmult/test1_batch.m
    fk::vector<TestType> const diff = gold - y_own;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        *std::max_element(diff.begin(), diff.end(), abs_compare);
    TestType const tol = std::numeric_limits<TestType>::epsilon();
    REQUIRE(result <= tol * gold.size());
  }
}

TEMPLATE_TEST_CASE("batch builder", "[batch]", float, double)
{
  SECTION("1d, 1 term, degree 2, level 2")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    fk::matrix<TestType> coefficient_matrix = pde->get_coefficients(0, 0);
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(coefficient_matrix.begin(), coefficient_matrix.end(), gen);
    pde->set_coefficients(coefficient_matrix, 0, 0);

    int const elem_size =
        std::pow(pde->get_dimensions()[0].get_degree(), pde->num_dims);
    fk::vector<TestType> const x = [&] {
      fk::vector<TestType> builder(elem_table.size() * elem_size);
      std::generate(builder.begin(), builder.end(), gen);
      return builder;
    }();

    fk::vector<TestType> const gold = coefficient_matrix * x;

    fk::vector<TestType> const y(x.size() * elem_table.size() * pde->num_terms);
    fk::vector<TestType> const work(0);

    int const items_to_reduce              = pde->num_terms * elem_table.size();
    fk::vector<TestType> const unit_vector = [&] {
      fk::vector<TestType> builder(items_to_reduce);
      std::fill(builder.begin(), builder.end(), 1.0);
      return builder;
    }();

    fk::vector<TestType> const fx(x.size());

    std::vector<batch_operands_set<TestType>> batches =
        build_batches(*pde, elem_table, x, y, work, unit_vector, fx);

    batch<TestType> const a = batches[0][0];
    batch<TestType> const b = batches[0][1];
    batch<TestType> const c = batches[0][2];

    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a, b, c, alpha, beta);

    batch<TestType> const r_a = batches[1][0];
    batch<TestType> const r_b = batches[1][1];
    batch<TestType> const r_c = batches[1][2];
    batched_gemv(r_a, r_b, r_c, alpha, beta);

    fk::vector<TestType> const diff = gold - fx;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        *std::max_element(diff.begin(), diff.end(), abs_compare);
    TestType const tol = std::numeric_limits<TestType>::epsilon();
    REQUIRE(result <= tol * gold.size());
  }

  SECTION("1d, 1 term, degree 4, level 3")
  {
    int const degree = 4;
    int const level  = 3;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    fk::matrix<TestType> coefficient_matrix = pde->get_coefficients(0, 0);
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(coefficient_matrix.begin(), coefficient_matrix.end(), gen);
    pde->set_coefficients(coefficient_matrix, 0, 0);

    int const elem_size =
        std::pow(pde->get_dimensions()[0].get_degree(), pde->num_dims);
    fk::vector<TestType> const x = [&] {
      fk::vector<TestType> builder(elem_table.size() * elem_size);
      std::generate(builder.begin(), builder.end(), gen);
      return builder;
    }();

    fk::vector<TestType> const gold = coefficient_matrix * x;

    fk::vector<TestType> const y(x.size() * elem_table.size() * pde->num_terms);
    fk::vector<TestType> const work(0);

    int const items_to_reduce              = pde->num_terms * elem_table.size();
    fk::vector<TestType> const unit_vector = [&] {
      fk::vector<TestType> builder(items_to_reduce);
      std::fill(builder.begin(), builder.end(), 1.0);
      return builder;
    }();

    fk::vector<TestType> const fx(x.size());

    std::vector<batch_operands_set<TestType>> batches =
        build_batches(*pde, elem_table, x, y, work, unit_vector, fx);

    batch<TestType> const a = batches[0][0];
    batch<TestType> const b = batches[0][1];
    batch<TestType> const c = batches[0][2];

    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a, b, c, alpha, beta);

    batch<TestType> const r_a = batches[1][0];
    batch<TestType> const r_b = batches[1][1];
    batch<TestType> const r_c = batches[1][2];
    batched_gemv(r_a, r_b, r_c, alpha, beta);

    fk::vector<TestType> const diff = gold - fx;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        *std::max_element(diff.begin(), diff.end(), abs_compare);

    TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e2;
    REQUIRE(result <= tol);
  }

  SECTION("2d, 2 terms, level 2, degree 2")
  {
    int const degree = 2;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    // generate initial
    int const elem_size =
        std::pow(pde->get_dimensions()[0].get_degree(), pde->num_dims);
    fk::vector<TestType> x(elem_table.size() * elem_size);
    std::fill(x.begin(), x.end(), 1.0);

    // setup output/intermediate output spaces for batched gemm
    fk::vector<TestType> const y(x.size() * elem_table.size() * pde->num_terms);
    fk::vector<TestType> const work(elem_size * elem_table.size() *
                                    elem_table.size() * pde->num_terms *
                                    std::min(pde->num_dims - 1, 2));
    fk::vector<TestType> const fx(x.size());

    // setup reduction vector
    int const items_to_reduce              = pde->num_terms * elem_table.size();
    fk::vector<TestType> const unit_vector = [&] {
      fk::vector<TestType> builder(items_to_reduce);
      std::fill(builder.begin(), builder.end(), 1.0);
      return builder;
    }();

    // call to build batches
    std::vector<batch_operands_set<TestType>> batches =
        build_batches(*pde, elem_table, x, y, work, unit_vector, fx);

    // batched gemm
    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      batch<TestType> const a = batches[i][0];
      batch<TestType> const b = batches[i][1];
      batch<TestType> const c = batches[i][2];
      batched_gemm(a, b, c, alpha, beta);
    }

    // reduce
    batch<TestType> const r_a = batches[pde->num_dims][0];
    batch<TestType> const r_b = batches[pde->num_dims][1];
    batch<TestType> const r_c = batches[pde->num_dims][2];
    batched_gemv(r_a, r_b, r_c, alpha, beta);

    std::string const file_path =
        "../testing/generated-inputs/batch/continuity2_sg_l2_d2_t1.dat";
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(file_path));

    fk::vector<TestType> const diff = gold - fx;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        *std::max_element(diff.begin(), diff.end(), abs_compare);
    TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
    REQUIRE(result <= tol);
  }

  SECTION("2d, 2 terms, level 3, degree 4, full grid")
  {
    int const degree = 4;
    int const level  = 3;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree), "-f"});

    element_table const elem_table(o, pde->num_dims);

    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    // generate initial
    int const elem_size =
        std::pow(pde->get_dimensions()[0].get_degree(), pde->num_dims);
    fk::vector<TestType> x(elem_table.size() * elem_size);
    std::fill(x.begin(), x.end(), 1.0);

    // setup output/intermediate output spaces for batched gemm
    fk::vector<TestType> const y(x.size() * elem_table.size() * pde->num_terms);
    fk::vector<TestType> const work(elem_size * elem_table.size() *
                                    elem_table.size() * pde->num_terms *
                                    std::min(pde->num_dims - 1, 2));
    fk::vector<TestType> const fx(x.size());

    // setup reduction vector
    int const items_to_reduce              = pde->num_terms * elem_table.size();
    fk::vector<TestType> const unit_vector = [&] {
      fk::vector<TestType> builder(items_to_reduce);
      std::fill(builder.begin(), builder.end(), 1.0);
      return builder;
    }();

    // call to build batches
    std::vector<batch_operands_set<TestType>> batches =
        build_batches(*pde, elem_table, x, y, work, unit_vector, fx);

    // batched gemm
    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      batch<TestType> const a = batches[i][0];
      batch<TestType> const b = batches[i][1];
      batch<TestType> const c = batches[i][2];
      batched_gemm(a, b, c, alpha, beta);
    }

    // reduce
    batch<TestType> const r_a = batches[pde->num_dims][0];
    batch<TestType> const r_b = batches[pde->num_dims][1];
    batch<TestType> const r_c = batches[pde->num_dims][2];
    batched_gemv(r_a, r_b, r_c, alpha, beta);

    std::string const file_path =
        "../testing/generated-inputs/batch/continuity2_fg_l3_d4_t1.dat";
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(file_path));

    fk::vector<TestType> const diff = gold - fx;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        *std::max_element(diff.begin(), diff.end(), abs_compare);
    TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
    REQUIRE(result <= tol);
  }

  SECTION("3d, 3 terms, level 3, degree 4, sparse grid")
  {
    int const degree = 4;
    int const level  = 3;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    // generate initial
    int const elem_size =
        std::pow(pde->get_dimensions()[0].get_degree(), pde->num_dims);
    fk::vector<TestType> x(elem_table.size() * elem_size);
    std::fill(x.begin(), x.end(), 1.0);

    // setup output/intermediate output spaces for batched gemm
    fk::vector<TestType> const y(x.size() * elem_table.size() * pde->num_terms);
    fk::vector<TestType> const work(elem_size * elem_table.size() *
                                    elem_table.size() * pde->num_terms *
                                    std::min(pde->num_dims - 1, 2));
    fk::vector<TestType> const fx(x.size());

    // setup reduction vector
    int const items_to_reduce              = pde->num_terms * elem_table.size();
    fk::vector<TestType> const unit_vector = [&] {
      fk::vector<TestType> builder(items_to_reduce);
      std::fill(builder.begin(), builder.end(), 1.0);
      return builder;
    }();

    // call to build batches
    std::vector<batch_operands_set<TestType>> batches =
        build_batches(*pde, elem_table, x, y, work, unit_vector, fx);

    // batched gemm
    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      batch<TestType> const a = batches[i][0];
      batch<TestType> const b = batches[i][1];
      batch<TestType> const c = batches[i][2];
      batched_gemm(a, b, c, alpha, beta);
    }

    // reduce
    batch<TestType> const r_a = batches[pde->num_dims][0];
    batch<TestType> const r_b = batches[pde->num_dims][1];
    batch<TestType> const r_c = batches[pde->num_dims][2];
    batched_gemv(r_a, r_b, r_c, alpha, beta);

    std::string const file_path =
        "../testing/generated-inputs/batch/continuity3_sg_l3_d4_t1.dat";
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(file_path));

    fk::vector<TestType> const diff = gold - fx;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        *std::max_element(diff.begin(), diff.end(), abs_compare);
    TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
    REQUIRE(result <= tol);
  }

  SECTION("6d, 6 terms, level 2, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 2;

    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    options const o = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    element_table const elem_table(o, pde->num_dims);

    TestType const init_time = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        auto term                     = pde->get_terms()[j][i];
        dimension<TestType> const dim = pde->get_dimensions()[i];
        fk::matrix<TestType> coeffs =
            fk::matrix<TestType>(generate_coefficients(dim, term, init_time));
        pde->set_coefficients(coeffs, j, i);
      }
    }

    // generate initial
    int const elem_size =
        std::pow(pde->get_dimensions()[0].get_degree(), pde->num_dims);
    fk::vector<TestType> x(elem_table.size() * elem_size);
    std::fill(x.begin(), x.end(), 1.0);

    // setup output/intermediate output spaces for batched gemm
    fk::vector<TestType> const y(x.size() * elem_table.size() * pde->num_terms);
    fk::vector<TestType> const work(elem_size * elem_table.size() *
                                    elem_table.size() * pde->num_terms *
                                    std::min(pde->num_dims - 1, 2));
    fk::vector<TestType> const fx(x.size());

    // setup reduction vector
    int const items_to_reduce              = pde->num_terms * elem_table.size();
    fk::vector<TestType> const unit_vector = [&] {
      fk::vector<TestType> builder(items_to_reduce);
      std::fill(builder.begin(), builder.end(), 1.0);
      return builder;
    }();

    // call to build batches
    std::vector<batch_operands_set<TestType>> batches =
        build_batches(*pde, elem_table, x, y, work, unit_vector, fx);

    // batched gemm
    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      batch<TestType> const a = batches[i][0];
      batch<TestType> const b = batches[i][1];
      batch<TestType> const c = batches[i][2];
      batched_gemm(a, b, c, alpha, beta);
    }

    // reduce
    batch<TestType> const r_a = batches[pde->num_dims][0];
    batch<TestType> const r_b = batches[pde->num_dims][1];
    batch<TestType> const r_c = batches[pde->num_dims][2];
    batched_gemv(r_a, r_b, r_c, alpha, beta);

    std::string const file_path =
        "../testing/generated-inputs/batch/continuity6_sg_l2_d3_t1.dat";
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(file_path));

    fk::vector<TestType> const diff = gold - fx;

    auto abs_compare = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        *std::max_element(diff.begin(), diff.end(), abs_compare);
    TestType const tol = std::numeric_limits<TestType>::epsilon() * 1e3;
    REQUIRE(result <= tol);
  }
}
