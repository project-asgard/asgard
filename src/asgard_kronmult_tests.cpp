
#include "tests_general.hpp"

#include "asgard_kronmult_tests.hpp"

template<typename T>
void test_almost_equal(std::vector<T> const &x, std::vector<T> const &y,
                       int scale = 10)
{
  rmse_comparison<T>(asgard::fk::vector<T>(x), asgard::fk::vector<T>(y),
                     get_tolerance<T>(scale));
}

template<typename T>
void test_kronmult_cpu(int dimensions, int n, int num_batch, int num_matrices,
                       int num_y)
{
  auto data =
      make_kronmult_data<T>(dimensions, n, num_batch, num_matrices, num_y);

  execute_cpu(dimensions, n, data->pA.data(), n, data->pX.data(),
              data->pY.data(), num_batch);

  test_almost_equal(data->output_y, data->reference_y, 100);
}

TEMPLATE_TEST_CASE("testing reference methods", "[kronecker]", float, double)
{
  std::vector<TestType> A    = {1, 2, 3, 4};
  std::vector<TestType> B    = {10, 20, 30, 40};
  auto R                     = kronecker(2, A.data(), 2, B.data());
  std::vector<TestType> gold = {10, 20, 20, 40, 30, 40,  60,  80,
                                30, 60, 40, 80, 90, 120, 120, 160};
  test_almost_equal(R, gold);

  B    = std::vector<TestType>{1, 2, 3, 4, 5, 6, 7, 8, 9};
  R    = kronecker(2, A.data(), 3, B.data());
  gold = std::vector<TestType>{1,  2,  3,  2,  4,  6,  4,  5,  6,  8,  10, 12,
                               7,  8,  9,  14, 16, 18, 3,  6,  9,  4,  8,  12,
                               12, 15, 18, 16, 20, 24, 21, 24, 27, 28, 32, 36};
  test_almost_equal(R, gold);

  std::vector<TestType> x = {10, 20};
  std::vector<TestType> y = {1, 0};
  reference_gemv(2, A.data(), x.data(), y.data());
  gold = std::vector<TestType>{71, 100};
  test_almost_equal(y, gold);
}

TEMPLATE_TEST_CASE("testing kronmult exceptions", "[kronmult]", float, double)
{
  REQUIRE_THROWS_WITH(test_kronmult_cpu<TestType>(1, 5, 1, 1, 1),
                      "kronmult unimplemented n for the cpu");
  REQUIRE_THROWS_WITH(test_kronmult_cpu<TestType>(1, 5, 1, 1, 1),
                      "kronmult unimplemented n for the cpu");
  REQUIRE_THROWS_WITH(
      test_kronmult_cpu<TestType>(10, 2, 1, 1, 1),
      "kronmult unimplemented number of dimensions for the cpu");
}

TEMPLATE_TEST_CASE("testing kronmult cpu", "[execute_cpu]", float, double)
{
  test_kronmult_cpu<TestType>(1, 2, 1, 1, 1);
  test_kronmult_cpu<TestType>(1, 2, 10, 1, 1);
  test_kronmult_cpu<TestType>(1, 2, 10, 5, 1);
  test_kronmult_cpu<TestType>(1, 2, 10, 1, 5);
  test_kronmult_cpu<TestType>(1, 2, 100, 10, 8);

  test_kronmult_cpu<TestType>(1, 2, 70, 9, 7);
  test_kronmult_cpu<TestType>(1, 3, 70, 9, 7);
  test_kronmult_cpu<TestType>(1, 4, 70, 9, 7);

  test_kronmult_cpu<TestType>(2, 2, 70, 9, 7);
  test_kronmult_cpu<TestType>(2, 3, 70, 9, 7);
}

#ifdef ASGARD_USE_CUDA

template<typename T>
void test_kronmult_gpu(int dimensions, int n, int num_batch, int num_matrices,
                       int num_y)
{
  auto data =
      make_kronmult_data<T>(dimensions, n, num_batch, num_matrices, num_y);

  execute_gpu(dimensions, n, data->gpupA.data(), n, data->gpupX.data(),
              data->gpupY.data(), num_batch);

  rmse_comparison<T>(data->gpuy.clone_onto_host(),
                     asgard::fk::vector<T>(data->reference_y),
                     get_tolerance<T>(100));
}

TEMPLATE_TEST_CASE("testing kronmult gpu 1d", "[execute_gpu 1d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult_gpu<TestType>(1, n, 170, 9, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 2d", "[execute_gpu 2d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
  test_kronmult_gpu<TestType>(2, n, 170, 9, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 3d", "[execute_gpu 3d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult_gpu<TestType>(3, n, 170, 9, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 4d", "[execute_gpu 4d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_gpu<TestType>(4, n, 170, 9, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 5d", "[execute_gpu 5d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult_gpu<TestType>(5, n, 170, 9, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 6d", "[execute_gpu 6d]", float, double)
{
  int n = GENERATE(1, 2, 3);
  test_kronmult_gpu<TestType>(6, n, 170, 9, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu general", "[execute_gpu]", float,
                   double)
{
  test_kronmult_gpu<TestType>(1, 2, 1, 1, 1);
  test_kronmult_gpu<TestType>(1, 2, 10, 1, 1);
  test_kronmult_gpu<TestType>(1, 2, 10, 5, 1);
  test_kronmult_gpu<TestType>(1, 2, 10, 1, 5);
  test_kronmult_gpu<TestType>(1, 2, 100, 10, 8);

  test_kronmult_gpu<TestType>(1, 2, 70, 9, 7);
  test_kronmult_gpu<TestType>(1, 3, 70, 9, 7);
  test_kronmult_gpu<TestType>(1, 4, 70, 9, 7);

  test_kronmult_gpu<TestType>(3, 2, 170, 9, 7);
  test_kronmult_gpu<TestType>(3, 3, 170, 9, 7);
  test_kronmult_gpu<TestType>(3, 4, 170, 9, 7);

  test_kronmult_gpu<TestType>(4, 2, 140, 5, 3);
}

#endif
