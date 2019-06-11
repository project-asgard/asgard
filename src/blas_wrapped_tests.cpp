#include "blas_wrapped.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

// in general, these component tests are only applied to functions
// with non-blas call paths to support integer operations.
//
// direct calls into BLAS are not covered for now
//
TEMPLATE_TEST_CASE("matrix-matrix multiply (gemm)", "[blas_wrapped]", float,
                   double, int)
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

    TestType alpha     = 1.0;
    TestType beta      = 0.0;
    int m              = in1.nrows();
    int k              = in1.ncols();
    int n              = in2.ncols();
    int lda            = in1.stride();
    int ldb            = in2.stride();
    int ldc            = result.stride();
    char const trans_a = 'n';
    char const trans_b = 'n';
    gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1.data(), &lda, in2.data(),
         &ldb, &beta, result.data(), &ldc);
    REQUIRE(result == ans);
  }

  SECTION("transpose a")
  {
    fk::matrix<TestType> const in1_t = fk::matrix<TestType>(in1).transpose();
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());

    TestType alpha     = 1.0;
    TestType beta      = 0.0;
    int m              = in1_t.ncols();
    int k              = in1_t.nrows();
    int n              = in2.ncols();
    int lda            = in1_t.stride();
    int ldb            = in2.stride();
    int ldc            = result.stride();
    char const trans_a = 't';
    char const trans_b = 'n';
    gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1_t.data(), &lda, in2.data(),
         &ldb, &beta, result.data(), &ldc);
    REQUIRE(result == ans);
  }
  SECTION("transpose b")
  {
    fk::matrix<TestType> const in2_t = fk::matrix<TestType>(in2).transpose();
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());

    TestType alpha     = 1.0;
    TestType beta      = 0.0;
    int m              = in1.nrows();
    int k              = in1.ncols();
    int n              = in2_t.nrows();
    int lda            = in1.stride();
    int ldb            = in2_t.stride();
    int ldc            = result.stride();
    char const trans_a = 'n';
    char const trans_b = 't';
    gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1.data(), &lda, in2_t.data(),
         &ldb, &beta, result.data(), &ldc);
    REQUIRE(result == ans);
  }

  SECTION("both transpose")
  {
    fk::matrix<TestType> const in1_t = fk::matrix<TestType>(in1).transpose();
    fk::matrix<TestType> const in2_t = fk::matrix<TestType>(in2).transpose();
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());

    TestType alpha     = 1.0;
    TestType beta      = 0.0;
    int m              = in1_t.ncols();
    int k              = in1_t.nrows();
    int n              = in2_t.nrows();
    int lda            = in1_t.stride();
    int ldb            = in2_t.stride();
    int ldc            = result.stride();
    char const trans_a = 't';
    char const trans_b = 't';
    gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1_t.data(), &lda,
         in2_t.data(), &ldb, &beta, result.data(), &ldc);
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

    TestType alpha     = 2.0;
    TestType beta      = 1.0;
    int m              = in1.nrows();
    int k              = in1.ncols();
    int n              = in2.ncols();
    int lda            = in1.stride();
    int ldb            = in2.stride();
    int ldc            = result.stride();
    char const trans_a = 'n';
    char const trans_b = 'n';
    gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1.data(), &lda, in2.data(),
         &ldb, &beta, result.data(), &ldc);
    REQUIRE(result == gold);
  }

  SECTION("lda =/= nrows")
  {
    fk::matrix<TestType> const in1_extended = [&] {
      fk::matrix<TestType> builder(in1.nrows() + 1, in1.ncols());
      fk::matrix<TestType, mem_type::view> window(builder, 0, in1.nrows() - 1,
                                                  0, in1.ncols() - 1);
      window = in1;
      return builder;
    }();

    fk::matrix<TestType> const in2_extended = [&] {
      fk::matrix<TestType> builder(in2.nrows() + 1, in2.ncols());
      fk::matrix<TestType, mem_type::view> window(builder, 0, in2.nrows() - 1,
                                                  0, in2.ncols() - 1);
      window = in2;
      return builder;
    }();

    fk::matrix<TestType> result(in1.nrows() + 2, in2.ncols());
    fk::matrix<TestType, mem_type::view> result_view(result, 0, in1.nrows() - 1,
                                                     0, in2.ncols() - 1);
    TestType alpha     = 1.0;
    TestType beta      = 0.0;
    int m              = in1.nrows();
    int k              = in1.ncols();
    int n              = in2.ncols();
    int lda            = in1_extended.stride();
    int ldb            = in2_extended.stride();
    int ldc            = result.stride();
    char const trans_a = 'n';
    char const trans_b = 'n';
    gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1_extended.data(), &lda,
         in2_extended.data(), &ldb, &beta, result.data(), &ldc);
    REQUIRE(result_view == ans);
  }
}

TEMPLATE_TEST_CASE("matrix-vector multiply (gemv)", "[blas_wrapped]", float,
                   double, int)
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

    TestType alpha     = 1.0;
    TestType beta      = 0.0;
    int m              = A.nrows();
    int n              = A.ncols();
    int lda            = A.stride();
    int inc            = 1;
    char const trans_a = 'n';

    gemv(&trans_a, &m, &n, &alpha, A.data(), &lda, x.data(), &inc, &beta,
         result.data(), &inc);
    REQUIRE(result == ans);
  }

  SECTION("transpose A")
  {
    fk::matrix<TestType> const A_trans = fk::matrix<TestType>(A).transpose();
    fk::vector<TestType> result(ans.size());

    TestType alpha     = 1.0;
    TestType beta      = 0.0;
    int m              = A_trans.nrows();
    int n              = A_trans.ncols();
    int lda            = A_trans.stride();
    int inc            = 1;
    char const trans_a = 't';

    gemv(&trans_a, &m, &n, &alpha, A_trans.data(), &lda, x.data(), &inc, &beta,
         result.data(), &inc);
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
    TestType alpha     = 2.0;
    TestType beta      = 1.0;
    int m              = A.nrows();
    int n              = A.ncols();
    int lda            = A.stride();
    int inc            = 1;
    char const trans_a = 'n';

    gemv(&trans_a, &m, &n, &alpha, A.data(), &lda, x.data(), &inc, &beta,
         result.data(), &inc);
    REQUIRE(result == gold);
  }

  SECTION("inc =/= 1")
  {
    fk::vector<TestType> result(ans.size() * 2 - 1);

    fk::vector<TestType> const gold     = {27, 0, 42, 0, 69, 0, 71, 0, -65};
    fk::vector<TestType> const x_padded = {2, 0, 0, 1, 0, 0, -7};

    TestType alpha     = 1.0;
    TestType beta      = 0.0;
    int m              = A.nrows();
    int n              = A.ncols();
    int lda            = A.stride();
    int incx           = 3;
    int incy           = 2;
    char const trans_a = 'n';

    gemv(&trans_a, &m, &n, &alpha, A.data(), &lda, x_padded.data(), &incx,
         &beta, result.data(), &incy);
    REQUIRE(result == gold);
  }
}

TEMPLATE_TEST_CASE("scale and copy routines (scal/copy)", "[blas_wrapped]",
                   float, double, int)
{
  fk::vector<TestType> const x         = {1, 2, 3, 4, 5};
  fk::vector<TestType> const x_tripled = {3, 6, 9, 12, 15};
  TestType const scale                 = 3;
  SECTION("scal - incx = 1")
  {
    fk::vector<TestType> test(x);
    int n          = x.size();
    TestType alpha = scale;
    int incx       = 1;
    scal(&n, &alpha, test.data(), &incx);
    REQUIRE(test == x_tripled);
  }
  SECTION("scal - incx =/= 1")
  {
    fk::vector<TestType> test{1, 0, 2, 0, 3, 0, 4, 0, 5};
    fk::vector<TestType> const gold{3, 0, 6, 0, 9, 0, 12, 0, 15};
    int n          = x.size();
    TestType alpha = scale;
    int incx       = 2;
    scal(&n, &alpha, test.data(), &incx);
    REQUIRE(test == gold);
  }
  SECTION("copy - inc = 1")
  {
    fk::vector<TestType> x_test(x);
    fk::vector<TestType> y_test(x.size());
    int n   = x.size();
    int inc = 1;
    copy(&n, x_test.data(), &inc, y_test.data(), &inc);
    REQUIRE(y_test == x);
  }
  SECTION("copy - inc =/= 1")
  {
    fk::vector<TestType> x_test{1, 0, 2, 0, 3, 0, 4, 0, 5};
    fk::vector<TestType> const gold{1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5};
    fk::vector<TestType> y_test(gold.size());

    int n    = x.size();
    int incx = 2;
    int incy = 3;
    copy(&n, x_test.data(), &incx, y_test.data(), &incy);
    REQUIRE(y_test == gold);
  }
}

TEMPLATE_TEST_CASE("scale/accumulate (axpy)", "[blas_wrapped]", float, double,
                   int)
{}

TEMPLATE_TEST_CASE("dot product (dot)", "[blas_wrapped]", float, double, int) {}
