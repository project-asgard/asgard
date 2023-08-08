
#include "tests_general.hpp"

#include "asgard_kronmult_tests.hpp"

template<typename T>
void test_almost_equal(std::vector<T> const &x, std::vector<T> const &y,
                       int scale = 10)
{
  rmse_comparison<T>(asgard::fk::vector<T>(x), asgard::fk::vector<T>(y),
                     get_tolerance<T>(scale));
}

constexpr bool sparse_mode = true;
constexpr bool dense_mode  = false;

template<typename T, bool matrix_mode = sparse_mode,
         asgard::resource rec = asgard::resource::host>
void test_kronmult(int dimensions, int n, int num_rows, int num_terms,
                   int num_matrices)
{
  constexpr bool precompute = true;

  auto data = make_kronmult_data<T, precompute>(dimensions, n, num_rows,
                                                num_terms, num_matrices);

  const int num_batch = num_rows * num_rows;

  asgard::fk::vector<T> vA(num_matrices * n * n);
  std::copy(data->matrices.begin(), data->matrices.end(), vA.begin());

  asgard::fk::vector<int> iA(num_batch * num_terms * dimensions);
  auto ip = data->pointer_map.begin();
  for (int i = 0; i < num_batch * num_terms; i++)
  {
    ip++;
    for (int j = 0; j < dimensions; j++)
      iA(i * dimensions + j) = n * n * (*ip++);
    ip++;
  }

  asgard::kronmult_matrix<T> kmat;

#ifdef ASGARD_USE_CUDA

  if (matrix_mode == sparse_mode)
  {
    int tensor_size =
        asgard::kronmult_matrix<T>::compute_tensor_size(dimensions, n);

    asgard::fk::vector<int> row_indx(num_rows * num_rows);
    asgard::fk::vector<int> col_indx(num_rows * num_rows);

    for (int i = 0; i < num_rows; i++)
    {
      for (int j = 0; j < num_rows; j++)
      {
        row_indx[i * num_rows + j] = i * tensor_size;
        col_indx[i * num_rows + j] = j * tensor_size;
      }
    }
    kmat = asgard::kronmult_matrix<T>(
        dimensions, n, num_rows, num_rows, num_terms,
        row_indx.clone_onto_device(), col_indx.clone_onto_device(),
        iA.clone_onto_device(), vA.clone_onto_device());
  }
  else
  {
    kmat = asgard::kronmult_matrix<T>(dimensions, n, num_rows, num_rows,
                                      num_terms, iA.clone_onto_device(),
                                      vA.clone_onto_device());
  }

  asgard::fk::vector<T, asgard::mem_type::owner, asgard::resource::device> xdev(
      kmat.input_size());
  asgard::fk::vector<T, asgard::mem_type::owner, asgard::resource::device> ydev(
      kmat.output_size());
  kmat.set_workspace(xdev, ydev);

#else

  if (matrix_mode == sparse_mode)
  {
    std::vector<asgard::fk::vector<int>> pntr;
    std::vector<asgard::fk::vector<int>> indx;
    pntr.push_back(asgard::fk::vector<int>(num_rows + 1));
    indx.push_back(asgard::fk::vector<int>(num_rows * num_rows));

    for (int i = 0; i < num_rows; i++)
    {
      pntr[0][i] = i * num_rows;
      for (int j = 0; j < num_rows; j++)
        indx[0][i * num_rows + j] = j;
    }
    pntr[0][num_rows] = indx[0].size();

    std::vector<asgard::fk::vector<int>> list_iA;
    list_iA.push_back(iA);

    kmat = asgard::kronmult_matrix<T>(
        dimensions, n, num_rows, num_rows, num_terms, std::move(pntr),
        std::move(indx), std::move(list_iA), std::move(vA));
  }
  else
  {
    kmat = asgard::kronmult_matrix<T>(dimensions, n, num_rows, num_rows,
                                      num_terms, std::move(iA), std::move(vA));
  }

#endif

#ifdef ASGARD_USE_CUDA
  if constexpr (rec == asgard::resource::device)
  {
    asgard::fk::vector<T, asgard::mem_type::owner, asgard::resource::device> xt(
        kmat.input_size());
    asgard::fk::vector<T, asgard::mem_type::owner, asgard::resource::device> yt(
        kmat.output_size());
    asgard::fk::copy_to_device(xt.data(), data->input_x.data(), xdev.size());
    asgard::fk::copy_to_device(yt.data(), data->output_y.data(), ydev.size());
    kmat.template apply<rec>(1.0, xt.data(), 1.0, yt.data());
    asgard::fk::copy_to_host(data->output_y.data(), yt.data(), yt.size());
  }
  else
  {
    kmat.template apply<rec>(1.0, data->input_x.data(), 1.0,
                             data->output_y.data());
  }
#else
  kmat.apply(1.0, data->input_x.data(), 1.0, data->output_y.data());
#endif

  test_almost_equal(data->output_y, data->reference_y, 100);
}

template<typename P>
void test_kronmult_welem(int dimensions, int n, int num_terms,
                         int num_1d_blocks)
{
  constexpr bool precompute = true;

  auto data = make_kronmult_welem<P, precompute>(dimensions, n, num_terms,
                                                 num_1d_blocks);

  cpu_dense<P>(dimensions, n, data->num_rows(), data->num_rows(), num_terms,
               data->elem.data(), 0, 0, data->get_offsets().data(),
               num_1d_blocks, P{1.0}, data->input_x.data(), P{1.0},
               data->output_y.data());

  asgard::kronmult_matrix<P> kmat(dimensions, n, data->num_rows(), data->num_rows(), num_terms,
                                  std::move(data->coefficients),
                                  asgard::fk::vector<int>(data->elem), 
                                  0, 0, num_1d_blocks);

  test_almost_equal(data->output_y, data->reference_y, 100);
}


TEMPLATE_TEST_CASE("testing reference methods", "[kronecker]", test_precs)
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

TEMPLATE_TEST_CASE("testing kronmult cpu core dense", "[execute_cpu]",
                   test_precs)
{
  test_kronmult<TestType, dense_mode>(1, 2, 1, 1, 1);
  test_kronmult<TestType, dense_mode>(1, 2, 1, 1, 5);
  test_kronmult<TestType, dense_mode>(1, 2, 1, 2, 3);
  test_kronmult<TestType, dense_mode>(1, 2, 10, 2, 7);
  test_kronmult_welem<TestType>(1, 2, 1, 1);
  test_kronmult_welem<TestType>(1, 2, 1, 5);
  test_kronmult_welem<TestType>(1, 2, 2, 5);
  test_kronmult_welem<TestType>(1, 2, 2, 7);
}
TEMPLATE_TEST_CASE("testing kronmult cpu core sparse", "[execute_cpu]",
                   test_precs)
{
  test_kronmult<TestType, sparse_mode>(1, 2, 1, 1, 1);
  test_kronmult<TestType, sparse_mode>(1, 2, 1, 1, 5);
  test_kronmult<TestType, sparse_mode>(1, 2, 1, 2, 3);
  test_kronmult<TestType, sparse_mode>(1, 2, 10, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 1d", "[execute_cpu 1d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6);
  test_kronmult<TestType, dense_mode>(1, n, 11, 2, 7);
  test_kronmult<TestType, sparse_mode>(1, n, 11, 2, 7);
}
TEMPLATE_TEST_CASE("testing kronmult cpu 1d", "[dense_cpu 1d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6);
  //int n = GENERATE(1, 2, 3, 4);
  test_kronmult_welem<TestType>(1, n, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 2d", "[execute_cpu 2d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  //test_kronmult<TestType, dense_mode>(2, n, 12, 3, 7);
  test_kronmult<TestType, sparse_mode>(2, n, 12, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 2d", "[dense_cpu 2d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  //int n = GENERATE(1, 2, 3, 4);
  test_kronmult_welem<TestType>(2, n, 3, 5);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 3d", "[execute_cpu 3d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  //test_kronmult<TestType, dense_mode>(3, n, 12, 2, 7);
  test_kronmult<TestType, sparse_mode>(3, n, 12, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 3d", "[dense_cpu 3d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  //int n = GENERATE(1, 2, 3, 4);
  test_kronmult_welem<TestType>(3, n, 3, 3);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 4d", "[execute_cpu 4d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  //test_kronmult<TestType, dense_mode>(4, n, 9, 2, 7);
  test_kronmult<TestType, sparse_mode>(4, n, 9, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 4d", "[dense_cpu 4d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  //int n = GENERATE(1, 2, 3, 4);
  test_kronmult_welem<TestType>(4, n, 2, 3);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 5d", "[execute_cpu 5d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  //test_kronmult<TestType, dense_mode>(5, n, 8, 2, 7);
  test_kronmult<TestType, sparse_mode>(5, n, 8, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 5d", "[dense_cpu 5d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  //int n = GENERATE(1, 2, 3, 4);
  test_kronmult_welem<TestType>(5, n, 2, 1);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 6d", "[execute_cpu 6d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4);
  //test_kronmult<TestType, dense_mode>(6, n, 6, 2, 7);
  test_kronmult<TestType, sparse_mode>(6, n, 6, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 6d", "[dense_cpu 6d]", test_precs)
{
  int n = GENERATE(1, 2, 3);
  test_kronmult_welem<TestType>(6, n, 2, 2);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 6d (large)", "[dense_cpu 6d]", test_precs)
{
  int n = GENERATE(4, 5);
  test_kronmult_welem<TestType>(6, n, 2, 1);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 6d-general", "[execute_cpu 6d]",
                   test_precs)
{
  // this is technically supported, but it takes too long
  // the Kronecker products actually suffer from the curse of dimensionality
  // and for 6D with n = 5, tensor size is 15,625 flops per product is 468,750,
  // mops per reference Kronecker products is 244,140,625
  // computing a reference solution becomes an issue, so the test is so small
  test_kronmult<TestType, dense_mode>(6, 5, 2, 1, 2);
  test_kronmult<TestType, sparse_mode>(6, 5, 2, 1, 2);
}

#endif

#ifdef ASGARD_USE_CUDA

TEMPLATE_TEST_CASE("testing kronmult gpu 1d", "[execute_gpu 1d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult<TestType, dense_mode>(1, n, 11, 2, 7);
  test_kronmult<TestType, sparse_mode>(1, n, 11, 2, 7);
  test_kronmult<TestType, sparse_mode, asgard::resource::device>(1, n, 11, 2,
                                                                 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 2d", "[execute_gpu 2d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
  test_kronmult<TestType, dense_mode>(2, n, 13, 2, 7);
  test_kronmult<TestType, dense_mode, asgard::resource::device>(2, n, 13, 2, 7);
  test_kronmult<TestType, sparse_mode>(2, n, 13, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 3d", "[execute_gpu 3d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult<TestType, dense_mode>(3, n, 17, 3, 7);
  test_kronmult<TestType, dense_mode, asgard::resource::device>(3, n, 17, 3, 7);
  test_kronmult<TestType, sparse_mode>(3, n, 17, 3, 7);
  test_kronmult<TestType, sparse_mode, asgard::resource::device>(3, n, 17, 3,
                                                                 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 4d", "[execute_gpu 4d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult<TestType, dense_mode>(4, n, 10, 3, 7);
  test_kronmult<TestType, sparse_mode>(4, n, 10, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 5d", "[execute_gpu 5d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult<TestType, dense_mode>(5, n, 10, 2, 7);
  test_kronmult<TestType, sparse_mode>(5, n, 10, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 6d", "[execute_gpu 6d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult<TestType, dense_mode>(6, n, 8, 2, 7);
  test_kronmult<TestType, sparse_mode>(6, n, 8, 2, 7);
}

#endif
