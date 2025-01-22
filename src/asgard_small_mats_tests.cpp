#include "tests_general.hpp"

#include "asgard_small_mats.hpp"

using namespace asgard;

TEST_CASE("small matrix methods", "[small mats]")
{
  SECTION("scal")
  {
    std::vector<double> x = {1, 2, 3};
    smmat::scal(3, 2.0, x.data());
    REQUIRE(fm::rmserr(x, std::vector<double>{2, 4, 6}) < 1.E-15);
    smmat::scal(2, -3.0, x.data());
    REQUIRE(fm::rmserr(x, std::vector<double>{-6, -12, 6}) < 1.E-15);
  }
  SECTION("gemv")
  {
    std::vector<double> y = {1, 2, 5};
    std::vector<double> x = {2, 3};
    std::vector<double> A = {1, 2, 3, 4};
    smmat::gemv(2, 2, A.data(), x.data(), y.data());
    REQUIRE(fm::rmserr(y, std::vector<double>{11, 16, 5}) < 1.E-15);
    smmat::gemv1(2, 2, A.data(), x.data(), y.data());
    REQUIRE(fm::rmserr(y, std::vector<double>{22, 32, 5}) < 1.E-15);
  }
  SECTION("gemm3")
  {
    std::vector<double> A = {1, 3, 2, 4, 3, 5};
    std::vector<double> x = {-1, -2, 4};
    std::vector<double> B = {1, 3, 5, 2, 4, 6};
    std::vector<double> C = {1, 2, 3, 4};
    smmat::gemm3(2, 3, A.data(), x.data(), B.data(), C.data());
    REQUIRE(fm::rmserr(C, std::vector<double>{47, 73, 54, 82}) < 1.E-15);
  }
  SECTION("inv2by2/gemv2by2")
  {
    std::vector<double> A = {1, 2, 3, 4, 5};
    std::vector<double> x = {1, 2, 3};
    smmat::gemv2by2(A.data(), x.data());
    REQUIRE(fm::rmserr(x, std::vector<double>{7, 10, 3}) < 1.E-15);
    smmat::inv2by2(A.data());
    REQUIRE(fm::rmserr(A, std::vector<double>{-2, 1, 1.5, -0.5, 5}) < 1.E-15);
    smmat::gemv2by2(A.data(), x.data());
    REQUIRE(fm::rmserr(x, std::vector<double>{1, 2, 3}) < 1.E-15);
  }
  SECTION("cholesky 2")
  {
    std::vector<double> A = {2, -1, -1, 2};
    std::vector<double> x = {1, 2};
    smmat::gemv2by2(A.data(), x.data());
    REQUIRE(fm::rmserr(x, std::vector<double>{0, 3}) < 1.E-15);
    smmat::potrf(2, A.data());
    REQUIRE(fm::rmserr(A, std::vector<double>{std::sqrt(2.0), -1, -1.0 / std::sqrt(2.0), 3.0 / std::sqrt(6.0)}) < 1.E-15);
    smmat::posv(2, A.data(), x.data());
    REQUIRE(fm::rmserr(x, std::vector<double>{1, 2}) < 1.E-15);
  }
  SECTION("cholesky 4")
  {
    std::vector<double> A = {4, -1, 0, -1, -1, 4, -1, 0, 0, -1, 4, -1, -1, 0, -1, 4};
    std::vector<double> x = {1, 2, 3, 4};
    std::vector<double> y = {1, 2, 3, 4};
    smmat::gemv(4, 4, A.data(), x.data(), y.data());
    REQUIRE(fm::rmserr(y, std::vector<double>{-2, 4, 6, 12}) < 1.E-15);
    smmat::potrf(4, A.data());
    smmat::posv(4, A.data(), y.data());
    REQUIRE(fm::rmserr(x, y) < 1.E-15);
  }
  SECTION("gemm")
  {
    std::vector<double> A = { 1, 2, 3, 4,  5, 6, 7, 8, 9};
    std::vector<double> B = {-1, 1, 2, 3, -4, 5, 1, 2, 3};
    std::vector<double> C = A;
    smmat::gemm(3, A.data(), B.data(), C.data());
    REQUIRE(fm::rmserr(C, std::vector<double>{17, 19, 21, 22, 26, 30, 30, 36, 42}) < 1.E-15);
    smmat::gemm<+1>(3, A.data(), B.data(), C.data());
    REQUIRE(fm::rmserr(C, std::vector<double>{34, 38, 42, 44, 52, 60, 60, 72, 84}) < 1.E-15);
    smmat::gemm<-1>(3, A.data(), B.data(), C.data());
    REQUIRE(fm::rmserr(C, std::vector<double>{17, 19, 21, 22, 26, 30, 30, 36, 42}) < 1.E-15);
    C = std::vector<double>(9, double{-9});
    smmat::gemm<0>(3, A.data(), B.data(), C.data());
    REQUIRE(fm::rmserr(C, std::vector<double>{17, 19, 21, 22, 26, 30, 30, 36, 42}) < 1.E-15);
  }
  SECTION("gemm_tn")
  {
    std::vector<double> A = { 1, 4, 7, 2,  5, 8, 3, 6, 9};
    std::vector<double> B = {-1, 1, 2, 3, -4, 5, 1, 2, 3};
    std::vector<double> C(9, double{0});
    smmat::gemm_tn(3, 3, A.data(), B.data(), C.data());
    REQUIRE(fm::rmserr(C, std::vector<double>{17, 19, 21, 22, 26, 30, 30, 36, 42}) < 1.E-15);
    smmat::gemm_tn<-1>(3, 3, A.data(), B.data(), C.data());
    REQUIRE(fm::rmserr(C, std::vector<double>(9, double{0})) < 1.E-15);
  }
  SECTION("gemm_pair 2")
  {
    std::vector<double> a0 = {1, 2, 3, 4};
    std::vector<double> t0 = {3, 4, 5, 6};
    std::vector<double> a1 = {3, 6, 7, 9};
    std::vector<double> t1 = {-1, 1, 4, -5};
    std::vector<double> C = {1, 2, 3, 4}; // will overwrite
    smmat::gemm_pair(2, a0.data(), t0.data(), a1.data(), t1.data(), C.data());
    REQUIRE(fm::rmserr(C, std::vector<double>{19, 25, 0, 13}) < 1.E-15);
  }
  SECTION("gemm_pairt 2")
  {
    std::vector<double> a0 = {1, 2, 3, 4};
    std::vector<double> t0 = {3, 4, 5, 6};
    std::vector<double> a1 = {3, 6, 7, 9};
    std::vector<double> t1 = {-1, 1, 4, -5};
    std::vector<double> C = {1, 2, 3, 4}; // will overwrite
    smmat::gemm_pairt(2, a0.data(), t0.data(), a1.data(), t1.data(), C.data());
    REQUIRE(fm::rmserr(C, std::vector<double>{43, 56, -10, -7}) < 1.E-15);
  }
  SECTION("kron_block 2")
  {
    std::vector<double> a = {1, 2, 3, 4};
    std::vector<double> b = {2, 3, 4, 5};
    std::vector<double> c = {3, 4, 5, 6};
    std::vector<double> kron2(4 * 4, 1);
    std::vector<double> kron3(8 * 8, 1);
    std::vector<double> ref2 = { 2,  3,  4,  6,
                                 4,  5,  8, 10,
                                 6,  9,  8, 12,
                                12, 15, 16, 20, };
    std::vector<double> ref3 = { 6,  8,  9, 12, 12, 16,  18,  24,
                                10, 12, 15, 18, 20, 24,  30,  36,
                                12, 16, 15, 20, 24, 32,  30,  40,
                                20, 24, 25, 30, 40, 48,  50,  60,
                                18, 24, 27, 36, 24, 32,  36,  48,
                                30, 36, 45, 54, 40, 48,  60,  72,
                                36, 48, 45, 60, 48, 64,  60,  80,
                                60, 72, 75, 90, 80, 96, 100, 120, };

    smmat::kron_block(2, 1, 2, 2, a.data(), kron2.data());
    smmat::kron_block(2, 2, 1, 2, b.data(), kron2.data());

    REQUIRE(fm::rmserr(kron2, ref2) < 1.E-15);

    smmat::kron_block(2, 1, 4, 4, a.data(), kron3.data());
    smmat::kron_block(2, 2, 2, 4, b.data(), kron3.data());
    smmat::kron_block(2, 4, 1, 4, c.data(), kron3.data());

    REQUIRE(fm::rmserr(kron3, ref3) < 1.E-15);
  }
  SECTION("LU factorize")
  {
    std::vector<double> A = {4, 1, 1, 1, 5, 0, 0, 2, 7};
    smmat::getrf(3, A.data());
    std::vector<double> R = {4, 0.25, 0.25, 1, 4.75, -5.263157894736842e-02, 0, 2, 7.105263157894737};
    REQUIRE(fm::rmserr(A, R) < 1.E-15);

    A = {4, 2, 1, 6};
    smmat::getrf(2, A.data());
    R = {4, 0.5, 1, 5.5};
    REQUIRE(fm::rmserr(A, R) < 1.E-15);

    A = {3};
    smmat::getrf(1, A.data());
    REQUIRE(A[0] == 3);
  }
  SECTION("LU apply L and U")
  {
    std::vector<double> A = {4, 1, 1, 1, 5, 0, 0, 2, 7};
    std::vector<double> B = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    smmat::getrf(3, A.data());
    smmat::getrs_l(3, A.data(), B.data());
    std::vector<double> R = {
        1, 1.75, 2.842105263157895e+00,
        4, 4.00, 5.210526315789473e+00,
        7, 6.25, 7.578947368421053e+00,};
    REQUIRE(fm::rmserr(B, R) < 1.E-15);

    B = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    smmat::getrs_u(3, A.data(), B.data());
    R = {1.891812865497076e-01, 2.432748538011696e-01, 4.222222222222222e-01,
         8.257309941520468e-01, 6.970760233918128e-01, 8.444444444444443e-01,
         1.462280701754386e+00, 1.150877192982456e+00, 1.266666666666667e+00};
    REQUIRE(fm::rmserr(B, R) < 1.E-15);

    B = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    smmat::getrs_u_right(3, A.data(), B.data());
    R = {2.500000000000000e-01, 5.000000000000000e-01, 7.500000000000000e-01,
         7.894736842105263e-01, 9.473684210526315e-01, 1.105263157894737e+00,
         7.629629629629628e-01, 8.592592592592592e-01, 9.555555555555555e-01,};
    REQUIRE(fm::rmserr(B, R) < 1.E-15);
  }
}
