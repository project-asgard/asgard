#include "tests_general.hpp"

using namespace asgard;

// in general, these component tests are only applied to functions
// with non-blas call paths to support integer operations.
//
// direct calls into BLAS are not covered for now
//
// exception: all device code paths are tested
//
TEMPLATE_TEST_CASE("matrix-matrix multiply (lib_dispatch::gemm)",
                   "[lib_dispatch]", test_precs)
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

#ifdef ASGARD_USE_CUDA
  fk::matrix<TestType, mem_type::owner, resource::device> const in1_d(
      in1.clone_onto_device());
  fk::matrix<TestType, mem_type::owner, resource::device> const in2_d(
      in2.clone_onto_device());
#endif

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
    lib_dispatch::gemm(trans_a, trans_b, m, n, k, alpha, in1.data(), lda,
                       in2.data(), ldb, beta, result.data(), ldc);
    REQUIRE(result == ans);
  }
#ifdef ASGARD_USE_CUDA
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

      lib_dispatch::gemm<resource::device>(trans_a, trans_b, m, n, k, alpha,
                                           in1_d.data(), lda, in2_d.data(), ldb,
                                           beta, result_d.data(), ldc);
      fk::matrix<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }
#endif
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
    lib_dispatch::gemm(trans_a, trans_b, m, n, k, alpha, in1_t.data(), lda,
                       in2.data(), ldb, beta, result.data(), ldc);
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
    lib_dispatch::gemm(trans_a, trans_b, m, n, k, alpha, in1.data(), lda,
                       in2_t.data(), ldb, beta, result.data(), ldc);
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
    lib_dispatch::gemm(trans_a, trans_b, m, n, k, alpha, in1_t.data(), lda,
                       in2_t.data(), ldb, beta, result.data(), ldc);
    REQUIRE(result == ans);
  }
#ifdef ASGARD_USE_CUDA
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
      lib_dispatch::gemm<resource::device>(trans_a, trans_b, m, n, k, alpha,
                                           in1_t_d.data(), lda, in2_t_d.data(),
                                           ldb, beta, result_d.data(), ldc);
      fk::matrix<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }
#endif
  SECTION("test scaling")
  {
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    std::fill(result.begin(), result.end(), 1.0);

    fk::matrix<TestType> const gold = [&] {
      fk::matrix<TestType> answer = (in1 * in2) * 2.0;
      std::transform(
          answer.begin(), answer.end(), answer.begin(),
          [](TestType const elem) -> TestType { return elem + 1.0; });

      return answer;
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
    lib_dispatch::gemm(trans_a, trans_b, m, n, k, alpha, in1.data(), lda,
                       in2.data(), ldb, beta, result.data(), ldc);
    REQUIRE(result == gold);
  }
#ifdef ASGARD_USE_CUDA
  SECTION("test scaling, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType> result(in1.nrows(), in2.ncols());
      std::fill(result.begin(), result.end(), 1.0);
      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          result.clone_onto_device());

      fk::matrix<TestType> const gold = [&] {
        fk::matrix<TestType> answer = (in1 * in2) * 2.0;
        std::transform(
            answer.begin(), answer.end(), answer.begin(),
            [](TestType const elem) -> TestType { return elem + 1.0; });

        return answer;
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
      lib_dispatch::gemm<resource::device>(trans_a, trans_b, m, n, k, alpha,
                                           in1_d.data(), lda, in2_d.data(), ldb,
                                           beta, result_d.data(), ldc);
      result.transfer_from(result_d);
      REQUIRE(result == gold);
    }
  }
#endif
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
    lib_dispatch::gemm(trans_a, trans_b, m, n, k, alpha, in1_extended.data(),
                       lda, in2_extended.data(), ldb, beta, result.data(), ldc);
    REQUIRE(result_view == ans);
  }
#ifdef ASGARD_USE_CUDA
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

      lib_dispatch::gemm<resource::device>(
          trans_a, trans_b, m, n, k, alpha, in1_extended_d.data(), lda,
          in2_extended_d.data(), ldb, beta, result_d.data(), ldc);
      fk::matrix<TestType> const result(result_view_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }
#endif
}

TEMPLATE_TEST_CASE("matrix-vector multiply (lib_dispatch::gemv)",
                   "[lib_dispatch]", test_precs)
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
#ifdef ASGARD_USE_CUDA
  fk::matrix<TestType, mem_type::owner, resource::device> const A_d(
      A.clone_onto_device());
  fk::vector<TestType, mem_type::owner, resource::device> const x_d(
      x.clone_onto_device());
#endif

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

    lib_dispatch::gemv(trans_a, m, n, alpha, A.data(), lda, x.data(), inc, beta,
                       result.data(), inc);
    REQUIRE(result == ans);
  }

#ifdef ASGARD_USE_CUDA
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

      lib_dispatch::gemv<resource::device>(trans_a, m, n, alpha, A_d.data(),
                                           lda, x_d.data(), inc, beta,
                                           result_d.data(), inc);

      fk::vector<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }
#endif

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

    lib_dispatch::gemv(trans_a, m, n, alpha, A_trans.data(), lda, x.data(), inc,
                       beta, result.data(), inc);
    REQUIRE(result == ans);
  }
#ifdef ASGARD_USE_CUDA
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

      lib_dispatch::gemv<resource::device>(trans_a, m, n, alpha,
                                           A_trans_d.data(), lda, x_d.data(),
                                           inc, beta, result_d.data(), inc);
      fk::vector<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }
#endif
  SECTION("test scaling")
  {
    fk::vector<TestType> result(ans.size());
    std::fill(result.begin(), result.end(), 1.0);

    fk::vector<TestType> const gold = [&] {
      fk::vector<TestType> answer = (A * x) * 2.0;
      std::transform(
          answer.begin(), answer.end(), answer.begin(),
          [](TestType const elem) -> TestType { return elem + 1.0; });

      return answer;
    }();
    TestType alpha     = 2.0;
    TestType beta      = 1.0;
    int m              = A.nrows();
    int n              = A.ncols();
    int lda            = A.stride();
    int inc            = 1;
    char const trans_a = 'n';

    lib_dispatch::gemv(trans_a, m, n, alpha, A.data(), lda, x.data(), inc, beta,
                       result.data(), inc);
    REQUIRE(result == gold);
  }
#ifdef ASGARD_USE_CUDA
  SECTION("test scaling, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType> result(ans.size());
      std::fill(result.begin(), result.end(), 1.0);
      fk::vector<TestType, mem_type::owner, resource::device> result_d(
          result.clone_onto_device());

      fk::vector<TestType> const gold = [&] {
        fk::vector<TestType> answer = (A * x) * 2.0;
        std::transform(
            answer.begin(), answer.end(), answer.begin(),
            [](TestType const elem) -> TestType { return elem + 1.0; });

        return answer;
      }();

      TestType alpha     = 2.0;
      TestType beta      = 1.0;
      int m              = A.nrows();
      int n              = A.ncols();
      int lda            = A.stride();
      int inc            = 1;
      char const trans_a = 'n';

      lib_dispatch::gemv<resource::device>(trans_a, m, n, alpha, A_d.data(),
                                           lda, x_d.data(), inc, beta,
                                           result_d.data(), inc);
      result.transfer_from(result_d);
      REQUIRE(result == gold);
    }
  }
#endif
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

    lib_dispatch::gemv(trans_a, m, n, alpha, A.data(), lda, x_padded.data(),
                       incx, beta, result.data(), incy);
    REQUIRE(result == gold);
  }

#ifdef ASGARD_USE_CUDA
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

      lib_dispatch::gemv<resource::device>(trans_a, m, n, alpha, A_d.data(),
                                           lda, x_padded_d.data(), incx, beta,
                                           result_d.data(), incy);
      fk::vector<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == gold);
    }
  }
#endif
}

TEMPLATE_TEST_CASE(
    "scale and copy routines (lib_dispatch::scal/lib_dispatch::copy)",
    "[lib_dispatch]", test_precs, int)
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
    lib_dispatch::scal(n, alpha, test.data(), incx);
    REQUIRE(test == x_tripled);
  }

#ifdef ASGARD_USE_CUDA
  SECTION("lib_dispatch::scal - inc = 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> test(
          x.clone_onto_device());
      int n          = x.size();
      TestType alpha = scale;
      int incx       = 1;

      lib_dispatch::scal<resource::device>(n, alpha, test.data(), incx);

      fk::vector<TestType, mem_type::owner, resource::host> const test_h(
          test.clone_onto_host());
      REQUIRE(test_h == x_tripled);
    }
  }
#endif

  SECTION("lib_dispatch::scal - incx =/= 1")
  {
    fk::vector<TestType> test{1, 0, 2, 0, 3, 0, 4, 0, 5};
    fk::vector<TestType> const gold{3, 0, 6, 0, 9, 0, 12, 0, 15};
    int n          = x.size();
    TestType alpha = scale;
    int incx       = 2;

    lib_dispatch::scal(n, alpha, test.data(), incx);

    REQUIRE(test == gold);
  }
#ifdef ASGARD_USE_CUDA
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

      lib_dispatch::scal<resource::device>(n, alpha, test.data(), incx);

      fk::vector<TestType, mem_type::owner, resource::host> const test_h(
          test.clone_onto_host());
      REQUIRE(test_h == gold);
    }
  }
#endif

  SECTION("lib_dispatch::copy - memcpy")
  {
    fk::vector<TestType> const x_test(x);
    fk::vector<TestType> y_test(x.size());
    int n = x.size();
    lib_dispatch::copy(n, x_test.data(), y_test.data());
    REQUIRE(y_test == x);
  }

  SECTION("lib_dispatch::copy - inc = 1")
  {
    fk::vector<TestType> const x_test(x);
    fk::vector<TestType> y_test(x.size());
    int n   = x.size();
    int inc = 1;
    lib_dispatch::copy(n, x_test.data(), inc, y_test.data(), inc);
    REQUIRE(y_test == x);
  }

#ifdef ASGARD_USE_CUDA
  SECTION("lib_dispatch::copy memcpy, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> const x_test(
          x.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> y_test(x.size());
      int n = x.size();

      lib_dispatch::copy<resource::device>(n, x_test.data(), y_test.data());

      fk::vector<TestType, mem_type::owner, resource::host> const y(
          y_test.clone_onto_host());
      REQUIRE(y == x);
    }
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

      lib_dispatch::copy<resource::device>(n, x_test.data(), inc, y_test.data(),
                                           inc);

      fk::vector<TestType, mem_type::owner, resource::host> const y(
          y_test.clone_onto_host());
      REQUIRE(y == x);
    }
  }
#endif

  SECTION("lib_dispatch::copy - inc =/= 1")
  {
    fk::vector<TestType> x_test{1, 0, 2, 0, 3, 0, 4, 0, 5};
    fk::vector<TestType> const gold{1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5};
    fk::vector<TestType> y_test(gold.size());

    int n    = x.size();
    int incx = 2;
    int incy = 3;
    lib_dispatch::copy(n, x_test.data(), incx, y_test.data(), incy);
    REQUIRE(y_test == gold);
  }

#ifdef ASGARD_USE_CUDA
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
      lib_dispatch::copy<resource::device>(n, x_d.data(), incx, y_d.data(),
                                           incy);
      fk::vector<TestType> const y_h(y_d.clone_onto_host());
      REQUIRE(y_h == gold);
    }
  }
#endif
}

TEMPLATE_TEST_CASE("scale/accumulate (lib_dispatch::axpy)", "[lib_dispatch]",
                   test_precs)
{
  fk::vector<TestType> const x = {1, 2, 3, 4, 5};
#ifdef ASGARD_USE_CUDA
  fk::vector<TestType, mem_type::owner, resource::device> const x_d(
      x.clone_onto_device());
#endif

  TestType const scale            = 3;
  fk::vector<TestType> const gold = {4, 7, 10, 13, 16};

  SECTION("lib_dispatch::axpy - inc = 1")
  {
    fk::vector<TestType> y(x.size());
    std::fill(y.begin(), y.end(), 1.0);

    int n          = x.size();
    TestType alpha = scale;
    int inc        = 1;
    lib_dispatch::axpy(n, alpha, x.data(), inc, y.data(), inc);
    REQUIRE(y == gold);
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
    lib_dispatch::axpy(n, alpha, x_extended.data(), incx, y.data(), incy);
    REQUIRE(y == ans);
  }

#ifdef ASGARD_USE_CUDA
  SECTION("lib_dispatch::axpy - inc = 1, device")
  {
    fk::vector<TestType> y(x.size());
    std::fill(y.begin(), y.end(), 1.0);

    int n          = x.size();
    TestType alpha = scale;
    int inc        = 1;
    fk::vector<TestType, mem_type::owner, resource::device> y_d(
        y.clone_onto_device());
    lib_dispatch::axpy<resource::device>(n, alpha, x_d.data(), inc, y_d.data(),
                                         inc);
    y.transfer_from(y_d);
    REQUIRE(y == gold);
  }

  SECTION("lib_dispatch::axpy - inc =/= 1, device")
  {
    fk::vector<TestType, mem_type::owner, resource::device> y_d = {
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    fk::vector<TestType> const ans = {4, 0, 0, 7, 0, 0, 10, 0, 0, 13, 0, 0, 16};
    fk::vector<TestType, mem_type::owner, resource::device> const x_extended = {
        1, 0, 2, 0, 3, 0, 4, 0, 5};
    int n          = x.size();
    TestType alpha = scale;
    int incx       = 2;
    int incy       = 3;
    lib_dispatch::axpy<resource::device>(n, alpha, x_extended.data(), incx,
                                         y_d.data(), incy);
    fk::vector<TestType> const y(y_d.clone_onto_host());
    REQUIRE(y == ans);
  }
#endif
}

TEMPLATE_TEST_CASE("dot product (lib_dispatch::dot)", "[lib_dispatch]",
                   test_precs, int)
{
  fk::vector<TestType> const x = {1, 2, 3, 4, 5};
  fk::vector<TestType> const y = {2, 4, 6, 8, 10};
#ifdef ASGARD_USE_CUDA
  fk::vector<TestType, mem_type::owner, resource::device> const x_d(
      x.clone_onto_device());
  fk::vector<TestType, mem_type::owner, resource::device> const y_d(
      y.clone_onto_device());
#endif
  TestType const gold = 110;

  SECTION("lib_dispatch::dot - inc = 1")
  {
    int n              = x.size();
    int inc            = 1;
    TestType const ans = lib_dispatch::dot(n, x.data(), inc, y.data(), inc);
    REQUIRE(ans == gold);
  }

#ifdef ASGARD_USE_CUDA
  SECTION("lib_dispatch::dot - inc = 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      int n              = x.size();
      int inc            = 1;
      TestType const ans = lib_dispatch::dot<resource::device>(
          n, x_d.data(), inc, y_d.data(), inc);
      REQUIRE(ans == gold);
    }
  }
#endif

  SECTION("lib_dispatch::dot - inc =/= 1")
  {
    fk::vector<TestType> const x_extended = {1, 0, 2, 0, 3, 0, 4, 0, 5};

    fk::vector<TestType> const y_extended = {2, 0, 0, 4, 0, 0, 6,
                                             0, 0, 8, 0, 0, 10};
    int n                                 = x.size();
    int incx                              = 2;
    int incy                              = 3;
    TestType const ans =
        lib_dispatch::dot(n, x_extended.data(), incx, y_extended.data(), incy);
    REQUIRE(ans == gold);
  }

#ifdef ASGARD_USE_CUDA
  SECTION("lib_dispatch::dot - inc =/= 1, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> const
          x_extended_d = {1, 0, 2, 0, 3, 0, 4, 0, 5};

      fk::vector<TestType, mem_type::owner, resource::device> const
          y_extended_d = {2, 0, 0, 4, 0, 0, 6, 0, 0, 8, 0, 0, 10};

      int n              = x.size();
      int incx           = 2;
      int incy           = 3;
      TestType const ans = lib_dispatch::dot<resource::device>(
          n, x_extended_d.data(), incx, y_extended_d.data(), incy);
      REQUIRE(ans == gold);
    }
  }
#endif
}

// this test is cublas specific - out of place inversion
TEMPLATE_TEST_CASE("device inversion test (lib_dispatch::getrf/getri)",
                   "[lib_dispatch]", test_precs)
{
#ifdef ASGARD_USE_CUDA

  fk::matrix<TestType> const test{{0.767135868133925, -0.641484652834663},
                                  {0.641484652834663, 0.767135868133926}};

  fk::matrix<TestType, mem_type::owner, resource::device> test_d(
      test.clone_onto_device());
  fk::vector<int, mem_type::owner, resource::device> ipiv_d(test_d.nrows());

  int m   = test.nrows();
  int n   = test.ncols();
  int lda = test.stride();

  int info  = lib_dispatch::getrf<resource::device>(m, n, test_d.data(), lda,
                                                   ipiv_d.data());
  auto stat = cudaDeviceSynchronize();
  REQUIRE(stat == cudaSuccess);

  REQUIRE(info == 0);

  m = test.nrows();
  n = test.ncols();
  fk::matrix<TestType, mem_type::owner, resource::device> work(2, 2);
  int size = m * n;
  info     = lib_dispatch::getri<resource::device>(
      n, test_d.data(), lda, ipiv_d.data(), work.data(), size);
  stat = cudaDeviceSynchronize();
  REQUIRE(stat == cudaSuccess);

  REQUIRE(info == 0);

  fk::matrix<TestType> const test_copy(work.clone_onto_host());

  // A * inv(A) == I
  REQUIRE((test * test_copy) == eye<TestType>(2));

#endif
}

template<typename P, resource resrc = resource::host>
void test_batched_gemm(int const m, int const n, int const k, int const lda,
                       int const ldb, int const ldc, int const num_batch = 3,
                       bool const trans_a = false, bool const trans_b = false,
                       P const alpha = 1.0, P const beta = 0.0)
{
  expect(m > 0);
  expect(n > 0);
  expect(k > 0);

  int const rows_a = trans_a ? k : m;
  int const cols_a = trans_a ? m : k;
  expect(lda >= rows_a);
  expect(ldc >= m);

  int const rows_b = trans_b ? n : k;
  int const cols_b = trans_b ? k : n;
  expect(ldb >= rows_b);

  std::vector<std::vector<fk::matrix<P, mem_type::owner, resrc>>> matrices =
      [=]() {
        // {a, b, c, gold}
        std::vector<std::vector<fk::matrix<P, mem_type::owner, resrc>>> output(
            4);

        std::random_device rd;
        std::mt19937 mersenne_engine(rd());
        std::uniform_real_distribution<P> dist(-2.0, 2.0);
        auto const gen = [&dist, &mersenne_engine]() {
          return dist(mersenne_engine);
        };

        for (int i = 0; i < num_batch; ++i)
        {
          fk::matrix<P> a(lda, cols_a);
          fk::matrix<P> b(ldb, cols_b);
          fk::matrix<P> c(ldc, n);
          std::generate(a.begin(), a.end(), gen);
          std::generate(b.begin(), b.end(), gen);
          std::generate(c.begin(), c.end(), gen);

          if constexpr (resrc == resource::host)
          {
            output[0].push_back(a);
            output[1].push_back(b);
            output[2].push_back(c);
          }
          else
          {
            output[0].push_back(a.clone_onto_device());
            output[1].push_back(b.clone_onto_device());
            output[2].push_back(c.clone_onto_device());
          }

          fk::matrix<P, mem_type::const_view> const effective_a(
              a, 0, rows_a - 1, 0, cols_a - 1);
          fk::matrix<P, mem_type::const_view> const effective_b(
              b, 0, rows_b - 1, 0, cols_b - 1);
          fk::matrix<P, mem_type::view> effective_c(c, 0, m - 1, 0, n - 1);
          fm::gemm(effective_a, effective_b, effective_c, trans_a, trans_b,
                   alpha, beta);

          if constexpr (resrc == resource::host)
          {
            output[3].push_back(fk::matrix<P>(effective_c));
          }
          else
          {
            output[3].push_back(effective_c.clone_onto_device());
          }
        }
        return output;
      }();

  std::vector<std::vector<P *>> ptrs = [=, &matrices]() {
    std::vector<std::vector<P *>> output(3);
    for (int i = 0; i < num_batch; ++i)
    {
      output[0].push_back(matrices[0][i].data());
      output[1].push_back(matrices[1][i].data());
      output[2].push_back(matrices[2][i].data());
    }
    return output;
  }();

  int lda_          = lda;
  int ldb_          = ldb;
  int ldc_          = ldc;
  int m_            = m;
  int n_            = n;
  int k_            = k;
  char const transa = trans_a ? 't' : 'n';
  char const transb = trans_b ? 't' : 'n';
  P alpha_          = alpha;
  P beta_           = beta;
  int num_batch_    = num_batch;

  lib_dispatch::batched_gemm<resrc>(
      ptrs[0].data(), lda_, transa, ptrs[1].data(), ldb_, transb,
      ptrs[2].data(), ldc_, m_, n_, k_, alpha_, beta_, num_batch_);

  // check results. we only want the effective region of c,
  // i.e. not the padding region that extends to ldc
  auto const effect_c = [m, n](auto const c) {
    return fk::matrix<P, mem_type::const_view>(c, 0, m - 1, 0, n - 1);
  };

  for (int i = 0; i < num_batch; ++i)
  {
    if constexpr (resrc == resource::host)
    {
      REQUIRE(effect_c(matrices[2][i]) == effect_c(matrices[3][i]));
    }
    else
    {
      auto constexpr tol_factor = get_tolerance<P>(10);
      rmse_comparison(effect_c(matrices[2][i].clone_onto_host()),
                      effect_c(matrices[3][i].clone_onto_host()), tol_factor);
    }
  }
}

TEMPLATE_TEST_CASE_SIG("batched gemm", "[lib_dispatch]",
                       ((typename TestType, resource resrc), TestType, resrc),
                       test_precs_host)
{
  SECTION("batched gemm: no trans, no trans, alpha = 1.0, beta = 0.0")
  {
    int const m         = 4;
    int const n         = 4;
    int const k         = 4;
    int const num_batch = 3;
    int const lda       = m;
    int const ldb       = k;
    int const ldc       = m;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch);
  }

  SECTION("batched gemm: trans a, no trans b, alpha = 1.0, beta = 0.0")
  {
    int const m         = 8;
    int const n         = 2;
    int const k         = 3;
    int const num_batch = 2;
    int const lda       = k + 1;
    int const ldb       = k + 2;
    int const ldc       = m;
    bool const trans_a  = true;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a);
  }

  SECTION("batched gemm: no trans a, trans b, alpha = 1.0, beta = 0.0")
  {
    int const m         = 3;
    int const n         = 6;
    int const k         = 5;
    int const num_batch = 4;
    int const lda       = m;
    int const ldb       = n;
    int const ldc       = m + 1;
    bool const trans_a  = false;
    bool const trans_b  = true;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a, trans_b);
  }

  SECTION("batched gemm: trans a, trans b, alpha = 1.0, beta = 0.0")
  {
    int const m         = 9;
    int const n         = 8;
    int const k         = 7;
    int const num_batch = 6;
    int const lda       = k + 1;
    int const ldb       = n + 2;
    int const ldc       = m + 3;
    bool const trans_a  = true;
    bool const trans_b  = true;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a, trans_b);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 3.0, beta = 0.0")
  {
    int const m          = 4;
    int const n          = 4;
    int const k          = 4;
    int const num_batch  = 3;
    int const lda        = m;
    int const ldb        = k;
    int const ldc        = m;
    bool const trans_a   = false;
    bool const trans_b   = false;
    TestType const alpha = 3.0;
    TestType const beta  = 0.0;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a, trans_b, alpha, beta);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 3.0, beta = 2.0")
  {
    int const m          = 4;
    int const n          = 4;
    int const k          = 4;
    int const num_batch  = 3;
    int const lda        = m;
    int const ldb        = k;
    int const ldc        = m;
    bool const trans_a   = false;
    bool const trans_b   = false;
    TestType const alpha = 3.0;
    TestType const beta  = 2.0;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a, trans_b, alpha, beta);
  }
}

#ifdef ASGARD_USE_CUDA
TEMPLATE_TEST_CASE_SIG("batched gemm", "[lib_dispatch]",
                       ((typename TestType, resource resrc), TestType, resrc),
                       test_precs_device)
{
  SECTION("batched gemm: no trans, no trans, alpha = 1.0, beta = 0.0")
  {
    int const m         = 4;
    int const n         = 4;
    int const k         = 4;
    int const num_batch = 3;
    int const lda       = m;
    int const ldb       = k;
    int const ldc       = m;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch);
  }

  SECTION("batched gemm: trans a, no trans b, alpha = 1.0, beta = 0.0")
  {
    int const m         = 8;
    int const n         = 2;
    int const k         = 3;
    int const num_batch = 2;
    int const lda       = k + 1;
    int const ldb       = k + 2;
    int const ldc       = m;
    bool const trans_a  = true;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a);
  }

  SECTION("batched gemm: no trans a, trans b, alpha = 1.0, beta = 0.0")
  {
    int const m         = 3;
    int const n         = 6;
    int const k         = 5;
    int const num_batch = 4;
    int const lda       = m;
    int const ldb       = n;
    int const ldc       = m + 1;
    bool const trans_a  = false;
    bool const trans_b  = true;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a, trans_b);
  }

  SECTION("batched gemm: trans a, trans b, alpha = 1.0, beta = 0.0")
  {
    int const m         = 9;
    int const n         = 8;
    int const k         = 7;
    int const num_batch = 6;
    int const lda       = k + 1;
    int const ldb       = n + 2;
    int const ldc       = m + 3;
    bool const trans_a  = true;
    bool const trans_b  = true;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a, trans_b);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 3.0, beta = 0.0")
  {
    int const m          = 4;
    int const n          = 4;
    int const k          = 4;
    int const num_batch  = 3;
    int const lda        = m;
    int const ldb        = k;
    int const ldc        = m;
    bool const trans_a   = false;
    bool const trans_b   = false;
    TestType const alpha = 3.0;
    TestType const beta  = 0.0;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a, trans_b, alpha, beta);
  }

  SECTION("batched gemm: no trans, no trans, alpha = 3.0, beta = 2.0")
  {
    int const m          = 4;
    int const n          = 4;
    int const k          = 4;
    int const num_batch  = 3;
    int const lda        = m;
    int const ldb        = k;
    int const ldc        = m;
    bool const trans_a   = false;
    bool const trans_b   = false;
    TestType const alpha = 3.0;
    TestType const beta  = 2.0;
    test_batched_gemm<TestType, resrc>(m, n, k, lda, ldb, ldc, num_batch,
                                       trans_a, trans_b, alpha, beta);
  }
}
#endif

TEMPLATE_TEST_CASE("LU Routines", "[lib_dispatch]", test_precs)
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
    fk::matrix<TestType> A_copy = A_gold;
    std::vector<int> ipiv(A_copy.nrows());
    fk::vector<TestType> x = B_gold;

    int rows_A = A_copy.nrows();
    int cols_B = 1;

    int lda = A_copy.stride();
    int ldb = x.size();

    int info = lib_dispatch::gesv(rows_A, cols_B, A_copy.data(), lda,
                                  ipiv.data(), x.data(), ldb);

    auto constexpr tol_factor = get_tolerance<TestType>(10);

    REQUIRE(info == 0);
    rmse_comparison(A_copy, LU_gold, tol_factor);
    rmse_comparison(x, X_gold, tol_factor);

    x          = B1_gold;
    char trans = 'N';
    info       = lib_dispatch::getrs(trans, rows_A, cols_B, A_copy.data(), lda,
                               ipiv.data(), x.data(), ldb);

    REQUIRE(info == 0);
    rmse_comparison(x, X1_gold, tol_factor);
  };
}

TEMPLATE_TEST_CASE("tpsv", "[lib_dispatch]", test_precs)
{
  fk::matrix<TestType> const A_gold{{1., 2., 4.}, {0., 3., 5.}, {0., 0., 6.}};

  fk::vector<TestType> const A_packed_gold{1., 2., 3., 4., 5., 6.};

  fk::vector<TestType> const B_gold{1., 1., 1.};
  fk::vector<TestType> const X_gold{
      0.22222222222222220989, 0.05555555555555557329, 0.16666666666666665741};

  SECTION("gesv and tpsv")
  {
    fk::matrix<TestType> A_copy = A_gold;
    std::vector<int> ipiv(A_copy.nrows());
    fk::vector<TestType> x = B_gold;

    int rows_A = A_copy.nrows();
    int cols_B = 1;

    int lda = A_copy.stride();
    int ldb = x.size();

    int info = lib_dispatch::gesv(rows_A, cols_B, A_copy.data(), lda,
                                  ipiv.data(), x.data(), ldb);

    auto constexpr tol_factor = get_tolerance<TestType>(10);

    REQUIRE(info == 0);
    rmse_comparison(x, X_gold, tol_factor);

    x = B_gold;
    lib_dispatch::tpsv('U', 'N', 'N', ldb, A_packed_gold.data(), x.data(), 1);
    rmse_comparison(x, X_gold, tol_factor);
  };

#ifdef ASGARD_USE_CUDA
  SECTION("tpsv on device")
  {
    fk::vector<TestType, mem_type::owner, resource::device> A_packed_gold_d =
        A_packed_gold.clone_onto_device();
    fk::vector<TestType, mem_type::owner, resource::device> x =
        B_gold.clone_onto_device();
    int ldb = x.size();

    lib_dispatch::tpsv<resource::device>('U', 'N', 'N', ldb,
                                         A_packed_gold_d.data(), x.data(), 1);
    auto constexpr tol_factor = get_tolerance<TestType>(10);
    rmse_comparison(x.clone_onto_host(), X_gold, tol_factor);
  };
#endif
}

TEMPLATE_TEST_CASE(
    "Performs rotation of points in the plane (lib_dispatch::rot)",
    "[lib_dispatch]", test_precs)
{
  fk::vector<TestType> x      = {1., 2., 3.};
  fk::vector<TestType> y      = {2., 4., 6.};
  fk::vector<TestType> gold_x = {1.86602540378443859659, 3.73205080756887719318,
                                 5.59807621135331601181};
  fk::vector<TestType> gold_y = {1.23205080756887741522, 2.46410161513775483044,
                                 3.69615242270663246771};
  TestType const cos          = std::cos(M_PI / 6.);
  TestType const sin          = std::sin(M_PI / 6.);
#ifdef ASGARD_USE_CUDA
  fk::vector<TestType, mem_type::owner, resource::device> x_d(
      x.clone_onto_device());
  fk::vector<TestType, mem_type::owner, resource::device> y_d(
      y.clone_onto_device());
#endif

  SECTION("test lib_dispatch::rot on host")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(10);
    int n                     = x.size();
    lib_dispatch::rot(n, x.data(), 1, y.data(), 1, cos, sin);
    rmse_comparison(x, gold_x, tol_factor);
    rmse_comparison(y, gold_y, tol_factor);
  }

#ifdef ASGARD_USE_CUDA
  SECTION("test lib_dispatch::rot on device")
  {
    auto constexpr tol_factor = get_tolerance<TestType>(10);
    int n                     = x_d.size();
    lib_dispatch::rot<resource::device>(n, x_d.data(), 1, y_d.data(), 1, cos,
                                        sin);
    rmse_comparison(x_d.clone_onto_host(), gold_x, tol_factor);
    rmse_comparison(y_d.clone_onto_host(), gold_y, tol_factor);
  }
#endif
}
