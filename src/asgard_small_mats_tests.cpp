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
}
