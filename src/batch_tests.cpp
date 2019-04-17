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

    batch_list<TestType> const a_batch = [&] {
      batch_list<TestType> builder(num_batch, a_nrows, a_ncols, a_stride);

      builder.insert(a1_v, 0);
      builder.insert(a2_v, 1);
      builder.insert(a3_v, 2);

      return builder;
    }();

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

    batch_list<TestType> const b_batch = [&] {
      batch_list<TestType> builder(num_batch, b_nrows, b_ncols, b_stride);

      builder.insert(b1_v, 0);
      builder.insert(b2_v, 1);
      builder.insert(b3_v, 2);

      return builder;
    }();

    // make 2x1 "c" views
    fk::matrix<TestType> c(6, 1);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 0);

    batch_list<TestType> const c_batch = [&] {
      batch_list<TestType> builder(num_batch, a_nrows, b_ncols, c.nrows());
      builder.insert(c1_v, 0);
      builder.insert(c2_v, 1);
      builder.insert(c3_v, 2);
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
    TestType alpha = 1.0;
    TestType beta  = 0.0;
    bool trans_a   = false;
    bool trans_b   = false;

    batched_gemm(a_batch, b_batch, c_batch, alpha, beta, trans_a, trans_b);

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

    batch_list<TestType> const a_batch = [&] {
      batch_list<TestType> builder(num_batch, a_nrows, a_ncols, a_stride);

      builder.insert(a1_v, 0);
      builder.insert(a2_v, 1);
      builder.insert(a3_v, 2);

      return builder;
    }();

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

    batch_list<TestType> const b_batch = [&] {
      batch_list<TestType> builder(num_batch, b_nrows, b_ncols, b_stride);

      builder.insert(b1_v, 0);
      builder.insert(b2_v, 1);
      builder.insert(b3_v, 2);

      return builder;
    }();

    // make 3x2 "c" views
    fk::matrix<TestType> c(6, 2);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 1);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 1);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 1);

    batch_list<TestType> const c_batch = [&] {
      batch_list<TestType> builder(num_batch, a_ncols, b_ncols, c.nrows());
      builder.insert(c1_v, 0);
      builder.insert(c2_v, 1);
      builder.insert(c3_v, 2);
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
    TestType alpha = 1.0;
    TestType beta  = 0.0;
    bool trans_a   = true;
    bool trans_b   = false;

    batched_gemm(a_batch, b_batch, c_batch, alpha, beta, trans_a, trans_b);

    // compare
    REQUIRE(c == gold);
  }
}
