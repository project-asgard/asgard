#include "blas_free.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("free BLAS", "[blas_free]", float, double, int)
{
  fk::vector<TestType> const gold = {2, 3, 4, 5, 6};
  SECTION("vector scale and accumulate (axpy)")
  {
    TestType const scale = 2.0;

    fk::vector<TestType> test(gold);
    fk::vector<TestType> test_own(gold);
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType> rhs{7, 8, 9, 10, 11};
    fk::vector<TestType> rhs_own(rhs);
    fk::vector<TestType, mem_type::view> rhs_view(rhs_own);

    fk::vector<TestType> const ans = {16, 19, 22, 25, 28};

    REQUIRE(axpy(scale, rhs, test) == ans);
    test = gold;
    REQUIRE(axpy(scale, rhs_view, test) == ans);

    REQUIRE(axpy(scale, rhs, test_view) == ans);
    REQUIRE(test_own == ans);
    test_view = gold;
    REQUIRE(axpy(scale, rhs_view, test_view) == ans);
    REQUIRE(test_own == ans);
  }

  SECTION("vector copy (copy)")
  {
    fk::vector<TestType> test(gold.size());
    fk::vector<TestType> test_own(gold.size());
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType, mem_type::view> const gold_view(gold);

    REQUIRE(copy(gold, test) == gold);
    test.scale(0);
    REQUIRE(copy(gold_view, test) == gold);

    REQUIRE(copy(gold, test_view) == gold);
    REQUIRE(test_own == gold);
    test_own.scale(0);
    REQUIRE(copy(gold_view, test_view) == gold);
    REQUIRE(test_own == gold);
  }

  SECTION("vector scale (scal)")
  {
    TestType const x = 2.0;
    fk::vector<TestType> test(gold);
    fk::vector<TestType> test_own(gold);
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType> const ans = {4, 6, 8, 10, 12};

    REQUIRE(scal(x, test) == ans);
    REQUIRE(scal(x, test_view) == ans);
    REQUIRE(test_own == ans);

    test     = gold;
    test_own = gold;

    TestType const x2 = 0.0;
    fk::vector<TestType> const zeros(gold.size());

    REQUIRE(scal(x2, test) == zeros);
    REQUIRE(scal(x2, test_view) == zeros);
    REQUIRE(test_own == zeros);
  }
}
