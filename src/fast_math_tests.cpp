#include "fast_math.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("fm::gemm", "[fast_math]", float, double, int)
{
  // clang-format off
    fk::matrix<TestType> const ans{
        {360,  610,  860},
        {710, 1210, 1710},
    };

    fk::matrix<TestType> const in1{
        {3, 4,  5,  6,  7},
        {8, 9, 10, 11, 12},
    };

    fk::matrix<TestType> const in2{
        {12, 22, 32},
	{13, 23, 33},
	{14, 24, 34},
	{15, 25, 35},
	{16, 26, 36},
    };
  // clang-format on

  SECTION("no transpose")
  {
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    fm::gemm(in1, in2, result);
    REQUIRE(result == ans);
  }

  SECTION("transpose a")
  {
    fk::matrix<TestType> const in1_t = fk::matrix<TestType>(in1).transpose();
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    bool const trans_A = true;
    fm::gemm(in1_t, in2, result, trans_A);
    REQUIRE(result == ans);
  }
  SECTION("transpose b")
  {
    fk::matrix<TestType> const in2_t = fk::matrix<TestType>(in2).transpose();
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    bool const trans_A = false;
    bool const trans_B = true;
    fm::gemm(in1, in2_t, result, trans_A, trans_B);
    REQUIRE(result == ans);
  }

  SECTION("both transpose")
  {
    fk::matrix<TestType> const in1_t = fk::matrix<TestType>(in1).transpose();
    fk::matrix<TestType> const in2_t = fk::matrix<TestType>(in2).transpose();
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    bool const trans_A = true;
    bool const trans_B = true;
    fm::gemm(in1_t, in2_t, result, trans_A, trans_B);
    REQUIRE(result == ans);
  }

  SECTION("test scaling")
  {
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    std::fill(result.begin(), result.end(), 1.0);

    fk::matrix<TestType> const gold = [&] {
      fk::matrix<TestType> ans = (in1 * in2) * 2.0;
      std::transform(
          ans.begin(), ans.end(), ans.begin(),
          [](TestType const elem) -> TestType { return elem + 1.0; });

      return ans;
    }();

    bool const trans_A   = false;
    bool const trans_B   = false;
    TestType const alpha = 2.0;
    TestType const beta  = 1.0;

    fm::gemm(in1, in2, result, trans_A, trans_B, alpha, beta);
    REQUIRE(result == gold);
  }

  // test the case where lda doesn't equal the number of rows in
  // arguments - this often occurs when using views.
  SECTION("lda =/= nrows")
  {
    fk::matrix<TestType> in1_extended(in1.nrows() + 1, in1.ncols());

    fk::matrix<TestType, mem_type::view> in1_view(
        in1_extended, 0, in1.nrows() - 1, 0, in1.ncols() - 1);
    in1_view = in1;

    fk::matrix<TestType> in2_extended(in2.nrows() + 1, in2.ncols());

    fk::matrix<TestType, mem_type::view> in2_view(
        in2_extended, 0, in2.nrows() - 1, 0, in2.ncols() - 1);
    in2_view = in2;

    fk::matrix<TestType> result(in1.nrows() + 2, in2.ncols());
    fk::matrix<TestType, mem_type::view> result_view(result, 0, in1.nrows() - 1,
                                                     0, in2.ncols() - 1);

    fm::gemm(in1_view, in2_view, result_view);
    REQUIRE(result_view == ans);
  }
}

TEMPLATE_TEST_CASE("fm::gemv", "[fast_math]", float, double, int)
{
  // clang-format off
    fk::vector<TestType> const ans
      {27, 42, 69, 71, -65};

    fk::matrix<TestType> const A {
     {-10, -2, -7},
     {  6, -5, -5},
     {  7,  6, -7},
     {  8, -1, -8},
     { -9,  9,  8},
    };

    fk::vector<TestType> const x
     {2, 1, -7};
  // clang-format on

  SECTION("no transpose")
  {
    fk::vector<TestType> result(ans.size());
    fm::gemv(A, x, result);
    REQUIRE(result == ans);
  }

  SECTION("transpose A")
  {
    fk::matrix<TestType> const A_trans = fk::matrix<TestType>(A).transpose();
    fk::vector<TestType> result(ans.size());
    bool const trans_A = true;
    fm::gemv(A_trans, x, result, trans_A);
    REQUIRE(result == ans);
  }

  SECTION("test scaling")
  {
    fk::vector<TestType> result(ans.size());
    std::fill(result.begin(), result.end(), 1.0);

    fk::vector<TestType> const gold = [&] {
      fk::vector<TestType> ans = (A * x) * 2.0;
      std::transform(
          ans.begin(), ans.end(), ans.begin(),
          [](TestType const elem) -> TestType { return elem + 1.0; });

      return ans;
    }();

    bool const trans_A   = false;
    TestType const alpha = 2.0;
    TestType const beta  = 1.0;

    fm::gemv(A, x, result, trans_A, alpha, beta);
    REQUIRE(result == gold);
  }
}

TEMPLATE_TEST_CASE("other vector routines", "[fast_math]", float, double, int)
{
  fk::vector<TestType> const gold = {2, 3, 4, 5, 6};
  SECTION("vector scale and accumulate (fm::axpy)")
  {
    TestType const scale = 2.0;

    fk::vector<TestType> test(gold);
    fk::vector<TestType> test_own(gold);
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType> rhs{7, 8, 9, 10, 11};
    fk::vector<TestType> rhs_own(rhs);
    fk::vector<TestType, mem_type::view> rhs_view(rhs_own);

    fk::vector<TestType> const ans = {16, 19, 22, 25, 28};

    REQUIRE(fm::axpy(rhs, test, scale) == ans);
    test = gold;
    REQUIRE(fm::axpy(rhs_view, test, scale) == ans);

    REQUIRE(fm::axpy(rhs, test_view, scale) == ans);
    REQUIRE(test_own == ans);
    test_view = gold;
    REQUIRE(fm::axpy(rhs_view, test_view, scale) == ans);
    REQUIRE(test_own == ans);
  }

  SECTION("vector copy (fm::copy)")
  {
    fk::vector<TestType> test(gold.size());
    fk::vector<TestType> test_own(gold.size());
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType, mem_type::view> const gold_view(gold);

    REQUIRE(fm::copy(gold, test) == gold);
    test.scale(0);
    REQUIRE(fm::copy(gold_view, test) == gold);

    REQUIRE(fm::copy(gold, test_view) == gold);
    REQUIRE(test_own == gold);
    test_own.scale(0);
    REQUIRE(fm::copy(gold_view, test_view) == gold);
    REQUIRE(test_own == gold);
  }

  SECTION("vector scale (fm::scal)")
  {
    TestType const x = 2.0;
    fk::vector<TestType> test(gold);
    fk::vector<TestType> test_own(gold);
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType> const ans = {4, 6, 8, 10, 12};

    REQUIRE(fm::scal(x, test) == ans);
    REQUIRE(fm::scal(x, test_view) == ans);
    REQUIRE(test_own == ans);

    test     = gold;
    test_own = gold;

    TestType const x2 = 0.0;
    fk::vector<TestType> const zeros(gold.size());

    REQUIRE(fm::scal(x2, test) == zeros);
    REQUIRE(fm::scal(x2, test_view) == zeros);
    REQUIRE(test_own == zeros);
  }
}
