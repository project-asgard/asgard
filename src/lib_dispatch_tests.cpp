#include "lib_dispatch.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

// in general, these component tests are only applied to functions
// with non-blas call paths to support integer operations.
//
// direct calls into BLAS are not covered for now
//
// exception: all device code paths are tested
//
TEMPLATE_TEST_CASE("matrix-matrix multiply (lib_dispatch::gemm)",
                   "[lib_dispatch]", float, double, int)
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

  fk::matrix<TestType, mem_type::owner, resource::device> const in1_d(
      in1.clone_onto_device());
  fk::matrix<TestType, mem_type::owner, resource::device> const in2_d(
      in2.clone_onto_device());

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
    lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1.data(), &lda,
                       in2.data(), &ldb, &beta, result.data(), &ldc);
    REQUIRE(result == ans);
  }
  SECTION("no transpose, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          in1.nrows(), in2.ncols());

      TestType alpha     = 1.0;
      TestType beta      = 0.0;
      int m              = in1.nrows();
      int k              = in1.ncols();
      int n              = in2.ncols();
      int lda            = in1.stride();
      int ldb            = in2.stride();
      int ldc            = result_d.stride();
      char const trans_a = 'n';
      char const trans_b = 'n';

      lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1_d.data(),
                         &lda, in2_d.data(), &ldb, &beta, result_d.data(), &ldc,
                         resource::device);
      fk::matrix<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
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
    lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1_t.data(),
                       &lda, in2.data(), &ldb, &beta, result.data(), &ldc);
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
    lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1.data(), &lda,
                       in2_t.data(), &ldb, &beta, result.data(), &ldc);
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
    lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1_t.data(),
                       &lda, in2_t.data(), &ldb, &beta, result.data(), &ldc);
    REQUIRE(result == ans);
  }

  SECTION("both transpose, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType> const in1_t = fk::matrix<TestType>(in1).transpose();
      fk::matrix<TestType> const in2_t = fk::matrix<TestType>(in2).transpose();
      fk::matrix<TestType, mem_type::owner, resource::device> const in1_t_d(
          in1_t.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> const in2_t_d(
          in2_t.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          in1.nrows(), in2.ncols());

      TestType alpha     = 1.0;
      TestType beta      = 0.0;
      int m              = in1_t.ncols();
      int k              = in1_t.nrows();
      int n              = in2_t.nrows();
      int lda            = in1_t.stride();
      int ldb            = in2_t.stride();
      int ldc            = result_d.stride();
      char const trans_a = 't';
      char const trans_b = 't';
      lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1_t_d.data(),
                         &lda, in2_t_d.data(), &ldb, &beta, result_d.data(),
                         &ldc, resource::device);
      fk::matrix<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
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
    lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1.data(), &lda,
                       in2.data(), &ldb, &beta, result.data(), &ldc);
    REQUIRE(result == gold);
  }

  SECTION("test scaling, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType> result(in1.nrows(), in2.ncols());
      std::fill(result.begin(), result.end(), 1.0);
      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          result.clone_onto_device());

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
      lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha, in1_d.data(),
                         &lda, in2_d.data(), &ldb, &beta, result_d.data(), &ldc,
                         resource::device);
      result.transfer_from(result_d);
      REQUIRE(result == gold);
    }
  }

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
    lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha,
                       in1_extended.data(), &lda, in2_extended.data(), &ldb,
                       &beta, result.data(), &ldc);
    REQUIRE(result_view == ans);
  }

  SECTION("lda =/= nrows, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType, mem_type::owner, resource::device> in1_extended_d(
          in1.nrows() + 1, in1.ncols());
      fk::matrix<TestType, mem_type::view, resource::device> in1_view_d(
          in1_extended_d, 0, in1.nrows() - 1, 0, in1.ncols() - 1);
      in1_view_d = in1_d;

      fk::matrix<TestType, mem_type::owner, resource::device> in2_extended_d(
          in2.nrows() + 1, in2.ncols());
      fk::matrix<TestType, mem_type::view, resource::device> in2_view_d(
          in2_extended_d, 0, in2.nrows() - 1, 0, in2.ncols() - 1);
      in2_view_d = in2_d;

      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          in1.nrows() + 2, in2.ncols());
      fk::matrix<TestType, mem_type::view, resource::device> result_view_d(
          result_d, 0, in1.nrows() - 1, 0, in2.ncols() - 1);

      TestType alpha     = 1.0;
      TestType beta      = 0.0;
      int m              = in1.nrows();
      int k              = in1.ncols();
      int n              = in2.ncols();
      int lda            = in1_extended_d.stride();
      int ldb            = in2_extended_d.stride();
      int ldc            = result_d.stride();
      char const trans_a = 'n';
      char const trans_b = 'n';

      lib_dispatch::gemm(&trans_a, &trans_b, &m, &n, &k, &alpha,
                         in1_extended_d.data(), &lda, in2_extended_d.data(),
                         &ldb, &beta, result_d.data(), &ldc, resource::device);
      fk::matrix<TestType> const result(result_view_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }
}

TEMPLATE_TEST_CASE("matrix-vector multiply (lib_dispatch::gemv)",
                   "[lib_dispatch]", float, double, int)
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
    fk::matrix<TestType, mem_type::owner, resource::device> const A_d(A.clone_onto_device());
    fk::vector<TestType> const x
     {2, 1, -7};
    fk::vector<TestType, mem_type::owner, resource::device> const x_d(x.clone_onto_device());
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

    lib_dispatch::gemv(&trans_a, &m, &n, &alpha, A.data(), &lda, x.data(), &inc,
                       &beta, result.data(), &inc);
    REQUIRE(result == ans);
  }
  SECTION("no transpose, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> result_d(
          ans.size());

      TestType alpha     = 1.0;
      TestType beta      = 0.0;
      int m              = A.nrows();
      int n              = A.ncols();
      int lda            = A.stride();
      int inc            = 1;
      char const trans_a = 'n';

      lib_dispatch::gemv(&trans_a, &m, &n, &alpha, A_d.data(), &lda, x_d.data(),
                         &inc, &beta, result_d.data(), &inc, resource::device);

      fk::vector<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
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

    lib_dispatch::gemv(&trans_a, &m, &n, &alpha, A_trans.data(), &lda, x.data(),
                       &inc, &beta, result.data(), &inc);
    REQUIRE(result == ans);
  }

  SECTION("transpose A, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType> const A_trans = fk::matrix<TestType>(A).transpose();
      fk::matrix<TestType, mem_type::owner, resource::device> const A_trans_d(
          A_trans.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> result_d(
          ans.size());

      TestType alpha     = 1.0;
      TestType beta      = 0.0;
      int m              = A_trans.nrows();
      int n              = A_trans.ncols();
      int lda            = A_trans.stride();
      int inc            = 1;
      char const trans_a = 't';

      lib_dispatch::gemv(&trans_a, &m, &n, &alpha, A_trans_d.data(), &lda,
                         x_d.data(), &inc, &beta, result_d.data(), &inc,
                         resource::device);
      fk::vector<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
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

    lib_dispatch::gemv(&trans_a, &m, &n, &alpha, A.data(), &lda, x.data(), &inc,
                       &beta, result.data(), &inc);
    REQUIRE(result == gold);
  }

  SECTION("test scaling, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType> result(ans.size());
      std::fill(result.begin(), result.end(), 1.0);
      fk::vector<TestType, mem_type::owner, resource::device> result_d(
          result.clone_onto_device());

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

      lib_dispatch::gemv(&trans_a, &m, &n, &alpha, A_d.data(), &lda, x_d.data(),
                         &inc, &beta, result_d.data(), &inc, resource::device);
      result.transfer_from(result_d);
      REQUIRE(result == gold);
    }
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

    lib_dispatch::gemv(&trans_a, &m, &n, &alpha, A.data(), &lda,
                       x_padded.data(), &incx, &beta, result.data(), &incy);
    REQUIRE(result == gold);
  }

  SECTION("inc =/= 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> result_d(
          ans.size() * 2 - 1);

      fk::vector<TestType> const gold     = {27, 0, 42, 0, 69, 0, 71, 0, -65};
      fk::vector<TestType> const x_padded = {2, 0, 0, 1, 0, 0, -7};
      fk::vector<TestType, mem_type::owner, resource::device> const x_padded_d(
          x_padded.clone_onto_device());

      TestType alpha     = 1.0;
      TestType beta      = 0.0;
      int m              = A.nrows();
      int n              = A.ncols();
      int lda            = A.stride();
      int incx           = 3;
      int incy           = 2;
      char const trans_a = 'n';

      lib_dispatch::gemv(&trans_a, &m, &n, &alpha, A_d.data(), &lda,
                         x_padded_d.data(), &incx, &beta, result_d.data(),
                         &incy, resource::device);
      fk::vector<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == gold);
    }
  }
}

TEMPLATE_TEST_CASE(
    "scale and copy routines (lib_dispatch::scal/lib_dispatch::copy)",
    "[lib_dispatch]", float, double, int)
{
  fk::vector<TestType> const x         = {1, 2, 3, 4, 5};
  fk::vector<TestType> const x_tripled = {3, 6, 9, 12, 15};
  TestType const scale                 = 3;
  SECTION("lib_dispatch::scal - incx = 1")
  {
    fk::vector<TestType> test(x);
    int n          = x.size();
    TestType alpha = scale;
    int incx       = 1;
    lib_dispatch::scal(&n, &alpha, test.data(), &incx);
    REQUIRE(test == x_tripled);
  }

  SECTION("lib_dispatch::scal - inc = 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> test(
          x.clone_onto_device());
      int n          = x.size();
      TestType alpha = scale;
      int incx       = 1;

      lib_dispatch::scal(&n, &alpha, test.data(), &incx, resource::device);

      fk::vector<TestType, mem_type::owner, resource::host> const test_h(
          test.clone_onto_host());
      REQUIRE(test_h == x_tripled);
    }
  }
  SECTION("lib_dispatch::scal - incx =/= 1")
  {
    fk::vector<TestType> test{1, 0, 2, 0, 3, 0, 4, 0, 5};
    fk::vector<TestType> const gold{3, 0, 6, 0, 9, 0, 12, 0, 15};
    int n          = x.size();
    TestType alpha = scale;
    int incx       = 2;

    lib_dispatch::scal(&n, &alpha, test.data(), &incx);

    REQUIRE(test == gold);
  }

  SECTION("lib_dispatch::scal - incx =/= 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> test{
          1, 0, 2, 0, 3, 0, 4, 0, 5};
      fk::vector<TestType> const gold{3, 0, 6, 0, 9, 0, 12, 0, 15};
      int n          = x.size();
      TestType alpha = scale;
      int incx       = 2;

      lib_dispatch::scal(&n, &alpha, test.data(), &incx, resource::device);

      fk::vector<TestType, mem_type::owner, resource::host> const test_h(
          test.clone_onto_host());
      REQUIRE(test_h == gold);
    }
  }
  SECTION("lib_dispatch::copy - inc = 1")
  {
    fk::vector<TestType> const x_test(x);
    fk::vector<TestType> y_test(x.size());
    int n   = x.size();
    int inc = 1;
    lib_dispatch::copy(&n, x_test.data(), &inc, y_test.data(), &inc);
    REQUIRE(y_test == x);
  }
  SECTION("lib_dispatch::copy - inc = 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> const x_test(
          x.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> y_test(x.size());
      int n   = x.size();
      int inc = 1;

      lib_dispatch::copy(&n, x_test.data(), &inc, y_test.data(), &inc,
                         resource::device);

      fk::vector<TestType, mem_type::owner, resource::host> const y(
          y_test.clone_onto_host());
      REQUIRE(y == x);
    }
  }
  SECTION("lib_dispatch::copy - inc =/= 1")
  {
    fk::vector<TestType> x_test{1, 0, 2, 0, 3, 0, 4, 0, 5};
    fk::vector<TestType> const gold{1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5};
    fk::vector<TestType> y_test(gold.size());

    int n    = x.size();
    int incx = 2;
    int incy = 3;
    lib_dispatch::copy(&n, x_test.data(), &incx, y_test.data(), &incy);
    REQUIRE(y_test == gold);
  }

  SECTION("lib_dispatch::copy - inc =/= 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType> x_test{1, 0, 2, 0, 3, 0, 4, 0, 5};
      fk::vector<TestType, mem_type::owner, resource::device> const x_d(
          x_test.clone_onto_device());
      fk::vector<TestType> const gold{1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5};
      fk::vector<TestType, mem_type::owner, resource::device> y_d(gold.size());

      int n    = x.size();
      int incx = 2;
      int incy = 3;
      lib_dispatch::copy(&n, x_d.data(), &incx, y_d.data(), &incy,
                         resource::device);
      fk::vector<TestType> const y_h(y_d.clone_onto_host());
      REQUIRE(y_h == gold);
    }
  }
}

TEMPLATE_TEST_CASE("scale/accumulate (lib_dispatch::axpy)", "[lib_dispatch]",
                   float, double, int)
{
  fk::vector<TestType> const x = {1, 2, 3, 4, 5};
  fk::vector<TestType, mem_type::owner, resource::device> const x_d(
      x.clone_onto_device());

  TestType const scale            = 3;
  fk::vector<TestType> const gold = {4, 7, 10, 13, 16};

  SECTION("lib_dispatch::axpy - inc = 1")
  {
    fk::vector<TestType> y(x.size());
    std::fill(y.begin(), y.end(), 1.0);

    int n          = x.size();
    TestType alpha = scale;
    int inc        = 1;
    lib_dispatch::axpy(&n, &alpha, x.data(), &inc, y.data(), &inc);
    REQUIRE(y == gold);
  }

  SECTION("lib_dispatch::axpy - inc = 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType> y(x.size());
      std::fill(y.begin(), y.end(), 1.0);

      int n          = x.size();
      TestType alpha = scale;
      int inc        = 1;
      fk::vector<TestType, mem_type::owner, resource::device> y_d(
          y.clone_onto_device());
      lib_dispatch::axpy(&n, &alpha, x_d.data(), &inc, y_d.data(), &inc,
                         resource::device);
      y.transfer_from(y_d);
      REQUIRE(y == gold);
    }
  }

  SECTION("lib_dispatch::axpy - inc =/= 1")
  {
    fk::vector<TestType> y         = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    fk::vector<TestType> const ans = {4, 0, 0, 7, 0, 0, 10, 0, 0, 13, 0, 0, 16};
    fk::vector<TestType> const x_extended = {1, 0, 2, 0, 3, 0, 4, 0, 5};

    int n          = x.size();
    TestType alpha = scale;
    int incx       = 2;
    int incy       = 3;
    lib_dispatch::axpy(&n, &alpha, x_extended.data(), &incx, y.data(), &incy);
    REQUIRE(y == ans);
  }
  SECTION("lib_dispatch::axpy - inc =/= 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> y_d = {
          1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
      fk::vector<TestType> const ans = {4, 0, 0,  7, 0, 0, 10,
                                        0, 0, 13, 0, 0, 16};
      fk::vector<TestType, mem_type::owner, resource::device> const x_extended =
          {1, 0, 2, 0, 3, 0, 4, 0, 5};

      int n          = x.size();
      TestType alpha = scale;
      int incx       = 2;
      int incy       = 3;
      lib_dispatch::axpy(&n, &alpha, x_extended.data(), &incx, y_d.data(),
                         &incy, resource::device);
      fk::vector<TestType> const y(y_d.clone_onto_host());
      REQUIRE(y == ans);
    }
  }
}

TEMPLATE_TEST_CASE("dot product (lib_dispatch::dot)", "[lib_dispatch]", float,
                   double, int)
{
  fk::vector<TestType> const x = {1, 2, 3, 4, 5};
  fk::vector<TestType, mem_type::owner, resource::device> const x_d(
      x.clone_onto_device());
  fk::vector<TestType> const y = {2, 4, 6, 8, 10};
  fk::vector<TestType, mem_type::owner, resource::device> const y_d(
      y.clone_onto_device());
  TestType const gold = 110;

  SECTION("lib_dispatch::dot - inc = 1")
  {
    int n              = x.size();
    int inc            = 1;
    TestType const ans = lib_dispatch::dot(&n, x.data(), &inc, y.data(), &inc);
    REQUIRE(ans == gold);
  }
  SECTION("lib_dispatch::dot - inc = 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      int n              = x.size();
      int inc            = 1;
      TestType const ans = lib_dispatch::dot(&n, x_d.data(), &inc, y_d.data(),
                                             &inc, resource::device);
      REQUIRE(ans == gold);
    }
  }
  SECTION("lib_dispatch::dot - inc =/= 1")
  {
    fk::vector<TestType> const x_extended = {1, 0, 2, 0, 3, 0, 4, 0, 5};

    fk::vector<TestType> const y_extended = {2, 0, 0, 4, 0, 0, 6,
                                             0, 0, 8, 0, 0, 10};
    int n                                 = x.size();
    int incx                              = 2;
    int incy                              = 3;
    TestType const ans = lib_dispatch::dot(&n, x_extended.data(), &incx,
                                           y_extended.data(), &incy);
    REQUIRE(ans == gold);
  }
  SECTION("lib_dispatch::dot - inc =/= 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> const
          x_extended_d = {1, 0, 2, 0, 3, 0, 4, 0, 5};

      fk::vector<TestType, mem_type::owner, resource::device> const
          y_extended_d = {2, 0, 0, 4, 0, 0, 6, 0, 0, 8, 0, 0, 10};

      int n    = x.size();
      int incx = 2;
      int incy = 3;
      TestType const ans =
          lib_dispatch::dot(&n, x_extended_d.data(), &incx, y_extended_d.data(),
                            &incy, resource::device);
      REQUIRE(ans == gold);
    }
  }
}

// this test is cublas specific - out of place inversion
TEMPLATE_TEST_CASE("device inversion test (lib_dispatch::getrf/getri)",
                   "[lib_dispatch]", float, double)
{
#ifdef ASGARD_USE_CUDA

  fk::matrix<TestType> const test{{0.767135868133925, -0.641484652834663},
                                  {0.641484652834663, 0.767135868133926}};

  fk::matrix<TestType, mem_type::owner, resource::device> test_d(
      test.clone_onto_device());
  fk::vector<int, mem_type::owner, resource::device> ipiv_d(test_d.nrows());
  fk::vector<int, mem_type::owner, resource::device> info_d(10);

  int m   = test.nrows();
  int n   = test.ncols();
  int lda = test.stride();

  lib_dispatch::getrf(&m, &n, test_d.data(), &lda, ipiv_d.data(), info_d.data(),
                      resource::device);

  auto stat = cudaDeviceSynchronize();
  REQUIRE(stat == 0);
  fk::vector<int> const info_check(info_d.clone_onto_host());
  REQUIRE(info_check(0) == 0);

  m = test.nrows();
  n = test.ncols();
  fk::matrix<TestType, mem_type::owner, resource::device> work(2, 2);
  int size = m * n;
  lib_dispatch::getri(&n, test_d.data(), &lda, ipiv_d.data(), work.data(),
                      &size, info_d.data(), resource::device);

  stat = cudaDeviceSynchronize();
  REQUIRE(stat == 0);
  fk::vector<int> const info_check_2(info_d.clone_onto_host());
  REQUIRE(info_check_2(0) == 0);

  fk::matrix<TestType> const test_copy(work.clone_onto_host());

  // A * inv(A) == I
  REQUIRE((test * test_copy) == eye<TestType>(2));

#endif
}

TEMPLATE_TEST_CASE("batched gemm", "[lib_dispatch]", float, double)
{
  int num_batch = 3;
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
  fk::matrix<TestType, mem_type::owner, resource::device> const a1_d(
      a1.clone_onto_device());
  fk::matrix<TestType, mem_type::owner, resource::device> const a2_d(
      a2.clone_onto_device());
  fk::matrix<TestType, mem_type::owner, resource::device> const a3_d(
      a3.clone_onto_device());

  fk::matrix<TestType, mem_type::owner, resource::device> const b1_d(
      b1.clone_onto_device());
  fk::matrix<TestType, mem_type::owner, resource::device> const b2_d(
      b2.clone_onto_device());
  fk::matrix<TestType, mem_type::owner, resource::device> const b3_d(
      b3.clone_onto_device());

  SECTION("batched gemm: no trans, no trans, alpha = 1.0, beta = 0.0")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v.data());
      builder.push_back(a2_v.data());
      builder.push_back(a3_v.data());

      return builder;
    }();

    char const trans_b = 'n';
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_start_col = 2;
    int const b_stop_col  = 2;
    int b_ncols           = b_stop_col - b_start_col + 1;
    int b_stride          = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v.data());
      builder.push_back(b2_v.data());
      builder.push_back(b3_v.data());

      return builder;
    }();

    // make 2x1 "c" views
    fk::matrix<TestType> c(6, 1);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 0);
    int c_stride                   = c.stride();
    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v.data());
      builder.push_back(c2_v.data());
      builder.push_back(c3_v.data());
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

    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_nrows, &b_ncols, &a_ncols, &alpha, &beta,
        &num_batch, resource::host);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 1.0, beta = 0.0, device")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const a1_v_d(
        a1_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a2_v_d(
        a2_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a3_v_d(
        a3_d, a_start_row, a_stop_row, a_start_col, a_stop_col);

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v_d.data());
      builder.push_back(a2_v_d.data());
      builder.push_back(a3_v_d.data());

      return builder;
    }();

    char const trans_b = 'n';
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_start_col = 2;
    int const b_stop_col  = 2;
    int b_ncols           = b_stop_col - b_start_col + 1;
    int b_stride          = b1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const b1_v_d(
        b1_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b2_v_d(
        b2_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b3_v_d(
        b3_d, b_start_row, b_stop_row, b_start_col, b_stop_col);

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v_d.data());
      builder.push_back(b2_v_d.data());
      builder.push_back(b3_v_d.data());

      return builder;
    }();

    // make 2x1 "c" views
    fk::matrix<TestType, mem_type::owner, resource::device> c_d(6, 1);
    fk::matrix<TestType, mem_type::view, resource::device> c1_v_d(c_d, 0, 1, 0,
                                                                  0);
    fk::matrix<TestType, mem_type::view, resource::device> c2_v_d(c_d, 2, 3, 0,
                                                                  0);
    fk::matrix<TestType, mem_type::view, resource::device> c3_v_d(c_d, 4, 5, 0,
                                                                  0);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v_d.data());
      builder.push_back(c2_v_d.data());
      builder.push_back(c3_v_d.data());
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
    int c_stride   = c_d.stride();
    TestType alpha = 1.0;
    TestType beta  = 0.0;

    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_nrows, &b_ncols, &a_ncols, &alpha, &beta,
        &num_batch, resource::device);

    // compare
    fk::matrix<TestType> const c(c_d.clone_onto_host());
    REQUIRE(c == gold);
  }

  auto const get_trans =
      [](fk::matrix<TestType, mem_type::view> orig) -> fk::matrix<TestType> {
    fk::matrix<TestType> builder(orig);
    return builder.transpose();
  };

  SECTION("batched gemm: trans a, no trans b, alpha = 1.0, beta = 0.0")
  {
    char const trans_a = 't';
    // make 3x2 (pre-trans) "a" views
    int const a_start_row = 1;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 1;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

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

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v.data());
      builder.push_back(a2_v.data());
      builder.push_back(a3_v.data());

      return builder;
    }();

    char const trans_b = 'n';
    // make 3x2 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_start_col = 1;
    int const b_stop_col  = 2;
    int b_ncols           = b_stop_col - b_start_col + 1;
    int b_stride          = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v.data());
      builder.push_back(b2_v.data());
      builder.push_back(b3_v.data());

      return builder;
    }();

    // make 3x2 "c" views
    fk::matrix<TestType> c(6, 2);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 1);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 1);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 1);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v.data());
      builder.push_back(c2_v.data());
      builder.push_back(c3_v.data());
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
    int c_stride   = c.stride();

    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_ncols, &b_ncols, &a_nrows, &alpha, &beta,
        &num_batch, resource::host);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: trans a, no trans b, alpha = 1.0, beta = 0.0, device")
  {
    char const trans_a = 't';
    // make 3x2 (pre-trans) "a" views
    int const a_start_row = 1;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 1;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const a1_v_d(
        a1_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a2_v_d(
        a2_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a3_v_d(
        a3_d, a_start_row, a_stop_row, a_start_col, a_stop_col);

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

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v_d.data());
      builder.push_back(a2_v_d.data());
      builder.push_back(a3_v_d.data());

      return builder;
    }();

    char const trans_b = 'n';
    // make 3x2 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_start_col = 1;
    int const b_stop_col  = 2;
    int b_ncols           = b_stop_col - b_start_col + 1;
    int b_stride          = b1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const b1_v_d(
        b1_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b2_v_d(
        b2_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b3_v_d(
        b3_d, b_start_row, b_stop_row, b_start_col, b_stop_col);

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v_d.data());
      builder.push_back(b2_v_d.data());
      builder.push_back(b3_v_d.data());

      return builder;
    }();

    // make 3x2 "c" views
    fk::matrix<TestType, mem_type::owner, resource::device> c_d(6, 2);
    fk::matrix<TestType, mem_type::view, resource::device> c1_v_d(c_d, 0, 1, 0,
                                                                  1);
    fk::matrix<TestType, mem_type::view, resource::device> c2_v_d(c_d, 2, 3, 0,
                                                                  1);
    fk::matrix<TestType, mem_type::view, resource::device> c3_v_d(c_d, 4, 5, 0,
                                                                  1);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v_d.data());
      builder.push_back(c2_v_d.data());
      builder.push_back(c3_v_d.data());
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

    int c_stride = c_d.stride();
    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_ncols, &b_ncols, &a_nrows, &alpha, &beta,
        &num_batch, resource::device);

    fk::matrix<TestType> const c(c_d.clone_onto_host());
    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans a, trans b, alpha = 1.0, beta = 0.0")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v.data());
      builder.push_back(a2_v.data());
      builder.push_back(a3_v.data());

      return builder;
    }();

    char const trans_b = 't';
    // make 2x3 (pre trans) "b" views
    int const b_start_row = 0;
    int const b_stop_row  = 1;
    int const b_start_col = 0;
    int const b_stop_col  = 2;
    int b_nrows           = b_stop_row - b_start_row + 1;

    int b_stride = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v.data());
      builder.push_back(b2_v.data());
      builder.push_back(b3_v.data());

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

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(c1_v.data());
      builder.push_back(c2_v.data());
      builder.push_back(c3_v.data());
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
    TestType alpha = 1.0;
    TestType beta  = 0.0;
    int c_stride   = c.stride();
    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_nrows, &b_nrows, &a_ncols, &alpha, &beta,
        &num_batch, resource::host);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans a, trans b, alpha = 1.0, beta = 0.0, device")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const a1_v_d(
        a1_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a2_v_d(
        a2_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a3_v_d(
        a3_d, a_start_row, a_stop_row, a_start_col, a_stop_col);

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v_d.data());
      builder.push_back(a2_v_d.data());
      builder.push_back(a3_v_d.data());

      return builder;
    }();

    char const trans_b = 't';
    // make 2x3 (pre trans) "b" views
    int const b_start_row = 0;
    int const b_stop_row  = 1;
    int const b_start_col = 0;
    int const b_stop_col  = 2;
    int b_nrows           = b_stop_row - b_start_row + 1;

    int b_stride = b1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const b1_v_d(
        b1_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b2_v_d(
        b2_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b3_v_d(
        b3_d, b_start_row, b_stop_row, b_start_col, b_stop_col);

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v_d.data());
      builder.push_back(b2_v_d.data());
      builder.push_back(b3_v_d.data());

      return builder;
    }();

    // make the transposed versions for forming golden value
    fk::matrix<TestType> const b1_t = get_trans(b1_v);
    fk::matrix<TestType> const b2_t = get_trans(b2_v);
    fk::matrix<TestType> const b3_t = get_trans(b3_v);

    // make 2x2 "c" views
    fk::matrix<TestType, mem_type::owner, resource::device> c_d(6, 2);
    fk::matrix<TestType, mem_type::view, resource::device> c1_v_d(c_d, 0, 1, 0,
                                                                  1);
    fk::matrix<TestType, mem_type::view, resource::device> c2_v_d(c_d, 2, 3, 0,
                                                                  1);
    fk::matrix<TestType, mem_type::view, resource::device> c3_v_d(c_d, 4, 5, 0,
                                                                  1);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(c1_v_d.data());
      builder.push_back(c2_v_d.data());
      builder.push_back(c3_v_d.data());
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
    TestType alpha = 1.0;
    TestType beta  = 0.0;
    int c_stride   = c_d.stride();
    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_nrows, &b_nrows, &a_ncols, &alpha, &beta,
        &num_batch, resource::device);

    // compare
    fk::matrix<TestType> const c(c_d.clone_onto_host());
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: trans a, trans b, alpha = 1.0, beta = 0.0")
  {
    char const trans_a = 't';
    // make 3x2 (pre-trans) "a" views
    int const a_start_row = 1;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 1;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

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

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v.data());
      builder.push_back(a2_v.data());
      builder.push_back(a3_v.data());

      return builder;
    }();

    char const trans_b = 't';
    // make 2x3 (pre trans) "b" views
    int const b_start_row = 0;
    int const b_stop_row  = 1;
    int const b_start_col = 0;
    int const b_stop_col  = 2;

    int b_nrows = b_stop_row - b_start_row + 1;

    int b_stride = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v.data());
      builder.push_back(b2_v.data());
      builder.push_back(b3_v.data());

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

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v.data());
      builder.push_back(c2_v.data());
      builder.push_back(c3_v.data());
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
    TestType alpha = 1.0;
    TestType beta  = 0.0;
    int c_stride   = c.stride();
    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_ncols, &b_nrows, &a_nrows, &alpha, &beta,
        &num_batch, resource::host);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: trans a, trans b, alpha = 1.0, beta = 0.0, device")
  {
    char const trans_a = 't';
    // make 3x2 (pre-trans) "a" views
    int const a_start_row = 1;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 1;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const a1_v_d(
        a1_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a2_v_d(
        a2_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a3_v_d(
        a3_d, a_start_row, a_stop_row, a_start_col, a_stop_col);

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

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v_d.data());
      builder.push_back(a2_v_d.data());
      builder.push_back(a3_v_d.data());

      return builder;
    }();

    char const trans_b = 't';
    // make 2x3 (pre trans) "b" views
    int const b_start_row = 0;
    int const b_stop_row  = 1;
    int const b_start_col = 0;
    int const b_stop_col  = 2;
    int b_nrows           = b_stop_row - b_start_row + 1;
    int b_stride          = b1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const b1_v_d(
        b1_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b2_v_d(
        b2_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b3_v_d(
        b3_d, b_start_row, b_stop_row, b_start_col, b_stop_col);

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v_d.data());
      builder.push_back(b2_v_d.data());
      builder.push_back(b3_v_d.data());

      return builder;
    }();

    // make the transposed versions for forming golden value
    fk::matrix<TestType> const b1_t = get_trans(b1_v);
    fk::matrix<TestType> const b2_t = get_trans(b2_v);
    fk::matrix<TestType> const b3_t = get_trans(b3_v);

    // make 2x2 "c" views
    fk::matrix<TestType, mem_type::owner, resource::device> c_d(6, 2);
    fk::matrix<TestType, mem_type::view, resource::device> c1_v_d(c_d, 0, 1, 0,
                                                                  1);
    fk::matrix<TestType, mem_type::view, resource::device> c2_v_d(c_d, 2, 3, 0,
                                                                  1);
    fk::matrix<TestType, mem_type::view, resource::device> c3_v_d(c_d, 4, 5, 0,
                                                                  1);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v_d.data());
      builder.push_back(c2_v_d.data());
      builder.push_back(c3_v_d.data());
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
    TestType alpha = 1.0;
    TestType beta  = 0.0;
    int c_stride   = c_d.stride();
    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_ncols, &b_nrows, &a_nrows, &alpha, &beta,
        &num_batch, resource::device);

    // compare
    fk::matrix<TestType> const c(c_d.clone_onto_host());
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 1.0, beta = 1.0")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v.data());
      builder.push_back(a2_v.data());
      builder.push_back(a3_v.data());

      return builder;
    }();

    char const trans_b = 'n';
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_start_col = 2;
    int const b_stop_col  = 2;
    int b_ncols           = b_stop_col - b_start_col + 1;
    int b_stride          = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v.data());
      builder.push_back(b2_v.data());
      builder.push_back(b3_v.data());

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

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v.data());
      builder.push_back(c2_v.data());
      builder.push_back(c3_v.data());
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
    TestType alpha = 1.0;
    TestType beta  = 1.0;
    int c_stride   = c.stride();
    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_nrows, &b_ncols, &a_ncols, &alpha, &beta,
        &num_batch, resource::host);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 1.0, beta = 1.0, device")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const a1_v_d(
        a1_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a2_v_d(
        a2_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a3_v_d(
        a3_d, a_start_row, a_stop_row, a_start_col, a_stop_col);

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v_d.data());
      builder.push_back(a2_v_d.data());
      builder.push_back(a3_v_d.data());

      return builder;
    }();

    char const trans_b = 'n';
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_start_col = 2;
    int const b_stop_col  = 2;
    int b_ncols           = b_stop_col - b_start_col + 1;
    int b_stride          = b1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const b1_v_d(
        b1_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b2_v_d(
        b2_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b3_v_d(
        b3_d, b_start_row, b_stop_row, b_start_col, b_stop_col);

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v_d.data());
      builder.push_back(b2_v_d.data());
      builder.push_back(b3_v_d.data());

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

    fk::matrix<TestType, mem_type::owner, resource::device> c_d(
        c.clone_onto_device());
    fk::matrix<TestType, mem_type::view, resource::device> c1_v_d(c_d, 0, 1, 0,
                                                                  0);
    fk::matrix<TestType, mem_type::view, resource::device> c2_v_d(c_d, 2, 3, 0,
                                                                  0);
    fk::matrix<TestType, mem_type::view, resource::device> c3_v_d(c_d, 4, 5, 0,
                                                                  0);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v_d.data());
      builder.push_back(c2_v_d.data());
      builder.push_back(c3_v_d.data());
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
    TestType alpha = 1.0;
    TestType beta  = 1.0;
    int c_stride   = c_d.stride();
    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_nrows, &b_ncols, &a_ncols, &alpha, &beta,
        &num_batch, resource::device);

    // compare
    c.transfer_from(c_d);
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 3.0, beta = 0.0")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v.data());
      builder.push_back(a2_v.data());
      builder.push_back(a3_v.data());

      return builder;
    }();

    char const trans_b = 'n';
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_start_col = 2;
    int const b_stop_col  = 2;
    int b_ncols           = b_stop_col - b_start_col + 1;
    int b_stride          = b1.nrows();

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v.data());
      builder.push_back(b2_v.data());
      builder.push_back(b3_v.data());

      return builder;
    }();

    // make 2x1 "c" views
    fk::matrix<TestType> c(6, 1);
    fk::matrix<TestType, mem_type::view> c1_v(c, 0, 1, 0, 0);
    fk::matrix<TestType, mem_type::view> c2_v(c, 2, 3, 0, 0);
    fk::matrix<TestType, mem_type::view> c3_v(c, 4, 5, 0, 0);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v.data());
      builder.push_back(c2_v.data());
      builder.push_back(c3_v.data());
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
    TestType alpha = 3.0;
    TestType beta  = 1.0;
    int c_stride   = c.stride();
    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_nrows, &b_ncols, &a_ncols, &alpha, &beta,
        &num_batch, resource::host);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 3.0, beta = 0.0")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const a1_v_d(
        a1_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a2_v_d(
        a2_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a3_v_d(
        a3_d, a_start_row, a_stop_row, a_start_col, a_stop_col);

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v_d.data());
      builder.push_back(a2_v_d.data());
      builder.push_back(a3_v_d.data());

      return builder;
    }();

    char const trans_b = 'n';
    // make 3x1 "b" views
    int const b_start_row = 1;
    int const b_stop_row  = 3;
    int const b_start_col = 2;
    int const b_stop_col  = 2;
    int b_ncols           = b_stop_col - b_start_col + 1;
    int b_stride          = b1.nrows();

    fk::matrix<TestType, mem_type::view, resource::device> const b1_v_d(
        b1_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b2_v_d(
        b2_d, b_start_row, b_stop_row, b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const b3_v_d(
        b3_d, b_start_row, b_stop_row, b_start_col, b_stop_col);

    fk::matrix<TestType, mem_type::view> const b1_v(b1, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b2_v(b2, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);
    fk::matrix<TestType, mem_type::view> const b3_v(b3, b_start_row, b_stop_row,
                                                    b_start_col, b_stop_col);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v_d.data());
      builder.push_back(b2_v_d.data());
      builder.push_back(b3_v_d.data());

      return builder;
    }();

    // make 2x1 "c" views
    fk::matrix<TestType, mem_type::owner, resource::device> c_d(6, 1);
    fk::matrix<TestType, mem_type::view, resource::device> c1_v_d(c_d, 0, 1, 0,
                                                                  0);
    fk::matrix<TestType, mem_type::view, resource::device> c2_v_d(c_d, 2, 3, 0,
                                                                  0);
    fk::matrix<TestType, mem_type::view, resource::device> c3_v_d(c_d, 4, 5, 0,
                                                                  0);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v_d.data());
      builder.push_back(c2_v_d.data());
      builder.push_back(c3_v_d.data());
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
    TestType alpha = 3.0;
    TestType beta  = 1.0;
    int c_stride   = c_d.stride();
    lib_dispatch::batched_gemm(
        a_vect.data(), &a_stride, &trans_a, b_vect.data(), &b_stride, &trans_b,
        c_vect.data(), &c_stride, &a_nrows, &b_ncols, &a_ncols, &alpha, &beta,
        &num_batch, resource::device);

    // compare
    fk::matrix<TestType> const c(c_d.clone_onto_host());
    REQUIRE(c == gold);
  }
}

TEMPLATE_TEST_CASE("batched gemv", "[batch]", float, double)
{
  int num_batch = 3;
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
         {2},
         {3},
         {4},
         {5},
	 {6}
  };
  // clang-format on

  fk::matrix<TestType, mem_type::owner, resource::device> const a1_d(
      a1.clone_onto_device());
  fk::matrix<TestType, mem_type::owner, resource::device> const a2_d(
      a2.clone_onto_device());
  fk::matrix<TestType, mem_type::owner, resource::device> const a3_d(
      a3.clone_onto_device());

  fk::vector<TestType, mem_type::owner, resource::device> const b1_d(
      b1.clone_onto_device());

  auto const get_trans =
      [](fk::matrix<TestType, mem_type::view> orig) -> fk::matrix<TestType> {
    fk::matrix<TestType> builder(orig);
    return builder.transpose();
  };

  SECTION("batched gemv: no trans, alpha = 1.0, beta = 0.0")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v.data());
      builder.push_back(a2_v.data());
      builder.push_back(a3_v.data());

      return builder;
    }();

    // make 3x1 "b" views
    fk::vector<TestType, mem_type::view> const b1_v(b1, 0, 2);
    fk::vector<TestType, mem_type::view> const b2_v(b1, 1, 3);
    fk::vector<TestType, mem_type::view> const b3_v(b1, 2, 4);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v.data());
      builder.push_back(b2_v.data());
      builder.push_back(b3_v.data());

      return builder;
    }();

    // make 2x1 "c" views
    fk::vector<TestType> c(6);
    fk::vector<TestType, mem_type::view> c1_v(c, 0, 1);
    fk::vector<TestType, mem_type::view> c2_v(c, 2, 3);
    fk::vector<TestType, mem_type::view> c3_v(c, 4, 5);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v.data());
      builder.push_back(c2_v.data());
      builder.push_back(c3_v.data());
      return builder;
    }();

    // do the math to create gold matrix
    fk::vector<TestType> gold(6);
    fk::vector<TestType, mem_type::view> gold1_v(gold, 0, 1);
    fk::vector<TestType, mem_type::view> gold2_v(gold, 2, 3);
    fk::vector<TestType, mem_type::view> gold3_v(gold, 4, 5);
    gold1_v = a1_v * b1_v;
    gold2_v = a2_v * b2_v;
    gold3_v = a3_v * b3_v;

    // call batched gemv
    TestType alpha = 1.0;
    TestType beta  = 0.0;

    lib_dispatch::batched_gemv(a_vect.data(), &a_stride, &trans_a,
                               b_vect.data(), c_vect.data(), &a_nrows, &a_ncols,
                               &alpha, &beta, &num_batch);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemv: no trans, alpha = 1.0, beta = 0.0, device")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    fk::matrix<TestType, mem_type::view, resource::device> const a1_v_d(
        a1_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a2_v_d(
        a2_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a3_v_d(
        a3_d, a_start_row, a_stop_row, a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v_d.data());
      builder.push_back(a2_v_d.data());
      builder.push_back(a3_v_d.data());

      return builder;
    }();

    // make 3x1 "b" views
    fk::vector<TestType, mem_type::view, resource::device> const b1_v_d(b1_d, 0,
                                                                        2);
    fk::vector<TestType, mem_type::view, resource::device> const b2_v_d(b1_d, 1,
                                                                        3);
    fk::vector<TestType, mem_type::view, resource::device> const b3_v_d(b1_d, 2,
                                                                        4);

    fk::vector<TestType, mem_type::view> const b1_v(b1, 0, 2);
    fk::vector<TestType, mem_type::view> const b2_v(b1, 1, 3);
    fk::vector<TestType, mem_type::view> const b3_v(b1, 2, 4);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v_d.data());
      builder.push_back(b2_v_d.data());
      builder.push_back(b3_v_d.data());

      return builder;
    }();

    // make 2x1 "c" views
    fk::vector<TestType, mem_type::owner, resource::device> c_d(6);
    fk::vector<TestType, mem_type::view, resource::device> c1_v_d(c_d, 0, 1);
    fk::vector<TestType, mem_type::view, resource::device> c2_v_d(c_d, 2, 3);
    fk::vector<TestType, mem_type::view, resource::device> c3_v_d(c_d, 4, 5);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v_d.data());
      builder.push_back(c2_v_d.data());
      builder.push_back(c3_v_d.data());
      return builder;
    }();

    // do the math to create gold matrix
    fk::vector<TestType> gold(6);
    fk::vector<TestType, mem_type::view> gold1_v(gold, 0, 1);
    fk::vector<TestType, mem_type::view> gold2_v(gold, 2, 3);
    fk::vector<TestType, mem_type::view> gold3_v(gold, 4, 5);
    gold1_v = a1_v * b1_v;
    gold2_v = a2_v * b2_v;
    gold3_v = a3_v * b3_v;

    // call batched gemv
    TestType alpha = 1.0;
    TestType beta  = 0.0;

    lib_dispatch::batched_gemv(a_vect.data(), &a_stride, &trans_a,
                               b_vect.data(), c_vect.data(), &a_nrows, &a_ncols,
                               &alpha, &beta, &num_batch, resource::device);

    // compare
    fk::vector<TestType> const c(c_d.clone_onto_host());
    REQUIRE(c == gold);
  }

  SECTION("batched gemv: trans, alpha = 1.0, beta = 0.0")
  {
    char const trans_a = 't';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    fk::matrix<TestType> const a1_t = get_trans(a1_v);
    fk::matrix<TestType> const a2_t = get_trans(a2_v);
    fk::matrix<TestType> const a3_t = get_trans(a3_v);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v.data());
      builder.push_back(a2_v.data());
      builder.push_back(a3_v.data());

      return builder;
    }();

    // make 2x1 "b" views
    fk::vector<TestType, mem_type::view> const b1_v(b1, 0, 1);
    fk::vector<TestType, mem_type::view> const b2_v(b1, 1, 2);
    fk::vector<TestType, mem_type::view> const b3_v(b1, 2, 3);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v.data());
      builder.push_back(b2_v.data());
      builder.push_back(b3_v.data());

      return builder;
    }();

    // make 3x1 "c" views
    fk::vector<TestType> c(9);
    fk::vector<TestType, mem_type::view> c1_v(c, 0, 2);
    fk::vector<TestType, mem_type::view> c2_v(c, 3, 5);
    fk::vector<TestType, mem_type::view> c3_v(c, 6, 8);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v.data());
      builder.push_back(c2_v.data());
      builder.push_back(c3_v.data());
      return builder;
    }();

    // do the math to create gold matrix
    fk::vector<TestType> gold(9);
    fk::vector<TestType, mem_type::view> gold1_v(gold, 0, 2);
    fk::vector<TestType, mem_type::view> gold2_v(gold, 3, 5);
    fk::vector<TestType, mem_type::view> gold3_v(gold, 6, 8);
    gold1_v = a1_t * b1_v;
    gold2_v = a2_t * b2_v;
    gold3_v = a3_t * b3_v;

    // call batched gemv
    TestType alpha = 1.0;
    TestType beta  = 0.0;

    lib_dispatch::batched_gemv(a_vect.data(), &a_stride, &trans_a,
                               b_vect.data(), c_vect.data(), &a_nrows, &a_ncols,
                               &alpha, &beta, &num_batch);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemv: trans, alpha = 1.0, beta = 0.0, device")
  {
    char const trans_a = 't';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    fk::matrix<TestType, mem_type::view, resource::device> const a1_v_d(
        a1_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a2_v_d(
        a2_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a3_v_d(
        a3_d, a_start_row, a_stop_row, a_start_col, a_stop_col);

    fk::matrix<TestType> const a1_t = get_trans(a1_v);
    fk::matrix<TestType> const a2_t = get_trans(a2_v);
    fk::matrix<TestType> const a3_t = get_trans(a3_v);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v_d.data());
      builder.push_back(a2_v_d.data());
      builder.push_back(a3_v_d.data());

      return builder;
    }();

    // make 2x1 "b" views
    fk::vector<TestType, mem_type::view, resource::device> const b1_v_d(b1_d, 0,
                                                                        1);
    fk::vector<TestType, mem_type::view, resource::device> const b2_v_d(b1_d, 2,
                                                                        3);
    fk::vector<TestType, mem_type::view, resource::device> const b3_v_d(b1_d, 3,
                                                                        4);

    fk::vector<TestType, mem_type::view> const b1_v(b1, 0, 1);
    fk::vector<TestType, mem_type::view> const b2_v(b1, 2, 3);
    fk::vector<TestType, mem_type::view> const b3_v(b1, 3, 4);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v_d.data());
      builder.push_back(b2_v_d.data());
      builder.push_back(b3_v_d.data());

      return builder;
    }();

    // make 2x1 "c" views
    fk::vector<TestType, mem_type::owner, resource::device> c_d(9);
    fk::vector<TestType, mem_type::view, resource::device> c1_v_d(c_d, 0, 2);
    fk::vector<TestType, mem_type::view, resource::device> c2_v_d(c_d, 3, 5);
    fk::vector<TestType, mem_type::view, resource::device> c3_v_d(c_d, 6, 8);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v_d.data());
      builder.push_back(c2_v_d.data());
      builder.push_back(c3_v_d.data());
      return builder;
    }();

    // do the math to create gold matrix
    fk::vector<TestType> gold(9);
    fk::vector<TestType, mem_type::view> gold1_v(gold, 0, 2);
    fk::vector<TestType, mem_type::view> gold2_v(gold, 3, 5);
    fk::vector<TestType, mem_type::view> gold3_v(gold, 6, 8);
    gold1_v = a1_t * b1_v;
    gold2_v = a2_t * b2_v;
    gold3_v = a3_t * b3_v;

    // call batched gemv
    TestType alpha = 1.0;
    TestType beta  = 0.0;

    lib_dispatch::batched_gemv(a_vect.data(), &a_stride, &trans_a,
                               b_vect.data(), c_vect.data(), &a_nrows, &a_ncols,
                               &alpha, &beta, &num_batch, resource::device);

    // compare
    fk::vector<TestType> const c(c_d.clone_onto_host());
    REQUIRE(c == gold);
  }

  SECTION("batched gemv: no trans, test scaling")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v.data());
      builder.push_back(a2_v.data());
      builder.push_back(a3_v.data());

      return builder;
    }();

    // make 3x1 "b" views
    fk::vector<TestType, mem_type::view> const b1_v(b1, 0, 2);
    fk::vector<TestType, mem_type::view> const b2_v(b1, 1, 3);
    fk::vector<TestType, mem_type::view> const b3_v(b1, 2, 4);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v.data());
      builder.push_back(b2_v.data());
      builder.push_back(b3_v.data());

      return builder;
    }();

    // make 2x1 "c" views
    fk::vector<TestType> c{10, 11, 12, 13, 14, 15};
    fk::vector<TestType, mem_type::view> c1_v(c, 0, 1);
    fk::vector<TestType, mem_type::view> c2_v(c, 2, 3);
    fk::vector<TestType, mem_type::view> c3_v(c, 4, 5);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v.data());
      builder.push_back(c2_v.data());
      builder.push_back(c3_v.data());
      return builder;
    }();

    // do the math to create gold matrix
    fk::vector<TestType> gold(6);
    fk::vector<TestType, mem_type::view> gold1_v(gold, 0, 1);
    fk::vector<TestType, mem_type::view> gold2_v(gold, 2, 3);
    fk::vector<TestType, mem_type::view> gold3_v(gold, 4, 5);

    TestType alpha = 3.0;
    gold1_v        = (a1_v * alpha) * b1_v + c1_v;
    gold2_v        = (a2_v * alpha) * b2_v + c2_v;
    gold3_v        = (a3_v * alpha) * b3_v + c3_v;

    // call batched gemv
    TestType beta = 1.0;
    lib_dispatch::batched_gemv(a_vect.data(), &a_stride, &trans_a,
                               b_vect.data(), c_vect.data(), &a_nrows, &a_ncols,
                               &alpha, &beta, &num_batch);

    // compare
    REQUIRE(c == gold);
  }

  SECTION("batched gemv: no trans, test scaling, device")
  {
    char const trans_a = 'n';
    // make 2x3 "a" views
    int const a_start_row = 2;
    int const a_stop_row  = 3;
    int a_nrows           = a_stop_row - a_start_row + 1;
    int const a_start_col = 0;
    int const a_stop_col  = 2;
    int a_ncols           = a_stop_col - a_start_col + 1;
    int a_stride          = a1.nrows();

    fk::matrix<TestType, mem_type::view> const a1_v(a1, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a2_v(a2, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view> const a3_v(a3, a_start_row, a_stop_row,
                                                    a_start_col, a_stop_col);

    fk::matrix<TestType, mem_type::view, resource::device> const a1_v_d(
        a1_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a2_v_d(
        a2_d, a_start_row, a_stop_row, a_start_col, a_stop_col);
    fk::matrix<TestType, mem_type::view, resource::device> const a3_v_d(
        a3_d, a_start_row, a_stop_row, a_start_col, a_stop_col);

    std::vector<TestType *> a_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(a1_v_d.data());
      builder.push_back(a2_v_d.data());
      builder.push_back(a3_v_d.data());

      return builder;
    }();

    // make 3x1 "b" views
    fk::vector<TestType, mem_type::view, resource::device> const b1_v_d(b1_d, 0,
                                                                        2);
    fk::vector<TestType, mem_type::view, resource::device> const b2_v_d(b1_d, 1,
                                                                        3);
    fk::vector<TestType, mem_type::view, resource::device> const b3_v_d(b1_d, 2,
                                                                        4);

    fk::vector<TestType, mem_type::view> const b1_v(b1, 0, 2);
    fk::vector<TestType, mem_type::view> const b2_v(b1, 1, 3);
    fk::vector<TestType, mem_type::view> const b3_v(b1, 2, 4);

    std::vector<TestType *> b_vect = [&] {
      std::vector<TestType *> builder;

      builder.push_back(b1_v_d.data());
      builder.push_back(b2_v_d.data());
      builder.push_back(b3_v_d.data());

      return builder;
    }();

    // make 2x1 "c" views
    fk::vector<TestType, mem_type::owner, resource::device> c_d{10, 11, 12,
                                                                13, 14, 15};
    fk::vector<TestType, mem_type::view, resource::device> c1_v_d(c_d, 0, 1);
    fk::vector<TestType, mem_type::view, resource::device> c2_v_d(c_d, 2, 3);
    fk::vector<TestType, mem_type::view, resource::device> c3_v_d(c_d, 4, 5);

    fk::vector<TestType> c(c_d.clone_onto_host());
    fk::vector<TestType, mem_type::view> c1_v(c, 0, 1);
    fk::vector<TestType, mem_type::view> c2_v(c, 2, 3);
    fk::vector<TestType, mem_type::view> c3_v(c, 4, 5);

    std::vector<TestType *> c_vect = [&] {
      std::vector<TestType *> builder;
      builder.push_back(c1_v_d.data());
      builder.push_back(c2_v_d.data());
      builder.push_back(c3_v_d.data());
      return builder;
    }();

    // do the math to create gold matrix
    fk::vector<TestType> gold(6);
    fk::vector<TestType, mem_type::view> gold1_v(gold, 0, 1);
    fk::vector<TestType, mem_type::view> gold2_v(gold, 2, 3);
    fk::vector<TestType, mem_type::view> gold3_v(gold, 4, 5);

    TestType alpha = 3.0;

    gold1_v = (a1_v * alpha) * b1_v + c1_v;
    gold2_v = (a2_v * alpha) * b2_v + c2_v;
    gold3_v = (a3_v * alpha) * b3_v + c3_v;

    // call batched gemv
    TestType beta = 1.0;
    lib_dispatch::batched_gemv(a_vect.data(), &a_stride, &trans_a,
                               b_vect.data(), c_vect.data(), &a_nrows, &a_ncols,
                               &alpha, &beta, &num_batch, resource::device);

    // compare
    c.transfer_from(c_d);
    REQUIRE(c == gold);
  }
}

TEMPLATE_TEST_CASE("LU Routines", "[lib_dispatch]", float, double)
{
  fk::matrix<TestType> const A_gold{
      {3.383861628748717e+00, 1.113343240310116e-02, 2.920740795411032e+00},
      {3.210305545769361e+00, 3.412141162288144e+00, 3.934494120167269e+00},
      {1.723479266939425e+00, 1.710451084172946e+00, 4.450671104482062e+00}};

  fk::matrix<TestType> const L_gold{
      {1.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00},
      {9.487106442223131e-01, 1.000000000000000e+00, 0.000000000000000e+00},
      {5.093232099968379e-01, 5.011733347067855e-01, 1.000000000000000e+00}};

  fk::matrix<TestType> const U_gold{
      {3.383861628748717e+00, 1.113343240310116e-02, 2.920740795411032e+00},
      {0.000000000000000e+00, 3.401578756460592e+00, 1.163556238546477e+00},
      {0.000000000000000e+00, 0.000000000000000e+00, 2.379926666803375e+00}};

  fk::matrix<TestType> const I_gold{{1.00000, 0.00000, 0.00000},
                                    {0.00000, 1.00000, 0.00000},
                                    {0.00000, 0.00000, 1.00000}};

  fk::vector<TestType> const B_gold{
      2.084406360034887e-01, 6.444769305362776e-01, 3.687335330031937e-01};
  fk::vector<TestType> const X_gold{
      4.715561567725287e-02, 1.257695999382253e-01, 1.625351700791827e-02};

  fk::vector<TestType> const B1_gold{
      9.789303188021963e-01, 8.085725142873675e-01, 7.370498473207234e-01};
  fk::vector<TestType> const X1_gold{
      1.812300946484165e-01, -7.824949213916167e-02, 1.254969087137521e-01};

  fk::matrix<TestType> LU_gold = L_gold + U_gold - I_gold;

  SECTION("gesv and getrs")
  {
    fk::matrix<TestType> const A_copy = A_gold;
    std::vector<int> ipiv(A_copy.nrows());
    fk::vector<TestType> x = B_gold;

    int rows_A = A_copy.nrows();
    int cols_B = 1;

    int lda = A_copy.stride();
    int ldb = x.size();

    int info;
    lib_dispatch::gesv(&rows_A, &cols_B, A_copy.data(), &lda, ipiv.data(),
                       x.data(), &ldb, &info);

    REQUIRE(info == 0);
    relaxed_comparison(A_copy, LU_gold);
    relaxed_comparison(x, X_gold);

    x          = B1_gold;
    char trans = 'N';
    lib_dispatch::getrs(&trans, &rows_A, &cols_B, A_copy.data(), &lda,
                        ipiv.data(), x.data(), &ldb, &info);

    REQUIRE(info == 0);
    relaxed_comparison(x, X1_gold);
  };
}
