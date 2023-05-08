
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
void test_kronmult_cpu(int dimensions, int n, int num_y, int output_length,
                       int num_matrices)
{
  auto data =
      make_kronmult_data<T>(dimensions, n, num_y, output_length, num_matrices);

  execute_cpu(dimensions, n, data->pA.data(), n, data->pX.data(),
              data->pY.data(), data->num_batch, data->output_size);

  test_almost_equal(data->output_y, data->reference_y, 100);
}

template<typename T>
void test_kronmult_cpu_v2(int dimensions, int n, int num_y, int output_length,
                          int num_matrices)
{
  constexpr bool precompute = true;
  constexpr bool randomx = false;

  auto data =
      make_kronmult_data<T, precompute, randomx>(dimensions, n, num_y, output_length, num_matrices);

  asgard::fk::vector<T> vA(num_matrices * n * n);
  for (int k=0; k<num_matrices; k++)
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
        vA(k * n * n + i * n + j) = data->matrices[k * n * n + j * n + i];

  asgard::fk::vector<int> iA(data->num_batch * dimensions);
  auto ip = data->pointer_map.begin();
  for (int i = 0; i < data->num_batch; i++)
  {
    ip++;
    for (int j = 0; j < dimensions; j++)
      iA(i*dimensions + j) = n * n * (*ip++);
    ip++;
  }

  asgard::kronmult_matrix<T, asgard::resource::host>
    kmat(dimensions, n, num_y, output_length,
         asgard::fk::vector<int, asgard::mem_type::const_view>(iA),
         asgard::fk::vector<T, asgard::mem_type::const_view>(vA));

  kmat.apply(1.0, data->input_x.data(), 1.0, data->output_y.data());

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
}

#ifndef ASGARD_USE_CUDA // test CPU kronmult only when CUDA is not enabled

TEMPLATE_TEST_CASE("testing kronmult cpu general", "[execute_cpu]", float,
                   double)
{
  test_kronmult_cpu_v2<TestType>(1, 2, 1, 1, 1);
  test_kronmult_cpu_v2<TestType>(1, 2, 1, 1, 5);
  test_kronmult_cpu_v2<TestType>(1, 2, 10, 10, 3);
  test_kronmult_cpu_v2<TestType>(1, 2, 10, 10, 7);

  test_kronmult_cpu<TestType>(1, 2, 1, 1, 1);
  test_kronmult_cpu<TestType>(1, 2, 1, 10, 1);
  test_kronmult_cpu<TestType>(1, 2, 5, 10, 1);
  test_kronmult_cpu<TestType>(1, 2, 10, 1, 5);
  test_kronmult_cpu<TestType>(1, 2, 10, 5, 8);

  test_kronmult_cpu<TestType>(2, 2, 40, 9, 7);
  test_kronmult_cpu<TestType>(2, 3, 40, 9, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 1d", "[execute_cpu 1d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6);
  test_kronmult_cpu<TestType>(1, n, 9, 20, 7);
  if (n <= 4)
    test_kronmult_cpu_v2<TestType>(1, n, 9, 9, 7);
}
TEMPLATE_TEST_CASE("testing kronmult cpu 2d", "[execute_cpu 2d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_cpu<TestType>(2, n, 9, 32, 7);
  if (n <= 4)
    test_kronmult_cpu_v2<TestType>(2, n, 12, 12, 7);
}
TEMPLATE_TEST_CASE("testing kronmult cpu 3d", "[execute_cpu 3d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_cpu<TestType>(3, n, 9, 15, 7);
  if (n <= 4)
    test_kronmult_cpu_v2<TestType>(3, n, 12, 12, 7);
}
TEMPLATE_TEST_CASE("testing kronmult cpu 4d", "[execute_cpu 4d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_cpu<TestType>(4, n, 9, 15, 7);
}
TEMPLATE_TEST_CASE("testing kronmult cpu 5d", "[execute_cpu 5d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_cpu<TestType>(5, n, 9, 12, 7);
}
TEMPLATE_TEST_CASE("testing kronmult cpu 6d", "[execute_cpu 6d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult_cpu<TestType>(6, n, 7, 8, 5);
}
TEMPLATE_TEST_CASE("testing kronmult cpu 6d-general", "[execute_cpu 6d]", float,
                   double)
{
  test_kronmult_cpu<TestType>(6, 5, 2, 2, 2);
}

#endif

#ifdef ASGARD_USE_CUDA

template<typename T>
void test_kronmult_gpu(int dimensions, int n, int num_y, int output_length,
                       int num_matrices)
{
  auto data =
      make_kronmult_data<T>(dimensions, n, num_y, output_length, num_matrices);

  execute_gpu(dimensions, n, data->gpupA.data(), n, data->gpupX.data(),
              data->gpupY.data(), data->num_batch, data->output_size);

  rmse_comparison<T>(data->gpuy.clone_onto_host(),
                     asgard::fk::vector<T>(data->reference_y),
                     get_tolerance<T>(100));
}

TEMPLATE_TEST_CASE("testing kronmult gpu 1d", "[execute_gpu 1d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult_gpu<TestType>(1, n, 9, 20, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 2d", "[execute_gpu 2d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
  test_kronmult_gpu<TestType>(2, n, 9, 20, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 3d", "[execute_gpu 3d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult_gpu<TestType>(3, n, 9, 20, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 4d", "[execute_gpu 4d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_gpu<TestType>(4, n, 9, 20, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 5d", "[execute_gpu 5d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult_gpu<TestType>(5, n, 9, 20, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 6d", "[execute_gpu 6d]", float, double)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult_gpu<TestType>(6, n, 9, 20, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu general", "[execute_gpu]", float,
                   double)
{
  test_kronmult_gpu<TestType>(1, 2, 1, 1, 1);
  test_kronmult_gpu<TestType>(1, 2, 10, 1, 1);
  test_kronmult_gpu<TestType>(1, 2, 10, 5, 1);
  test_kronmult_gpu<TestType>(1, 2, 10, 1, 5);
  test_kronmult_gpu<TestType>(1, 2, 10, 10, 8);

  test_kronmult_gpu<TestType>(1, 2, 9, 10, 7);
  test_kronmult_gpu<TestType>(1, 3, 9, 10, 7);
  test_kronmult_gpu<TestType>(1, 4, 9, 10, 7);

  test_kronmult_gpu<TestType>(3, 2, 9, 20, 7);
  test_kronmult_gpu<TestType>(3, 3, 9, 20, 7);
  test_kronmult_gpu<TestType>(3, 4, 9, 20, 7);

  test_kronmult_gpu<TestType>(4, 2, 5, 30, 3);
}

#endif
