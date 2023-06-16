
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

template<typename T, bool matrix_mode = sparse_mode>
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

  kmat.apply(1.0, data->input_x.data(), 1.0, data->output_y.data());

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

/*

#ifndef ASGARD_USE_CUDA // test CPU kronmult only when CUDA is not enabled

TEMPLATE_TEST_CASE("testing kronmult cpu core dense", "[execute_cpu]",
                   test_precs)
{
  test_kronmult<TestType, dense_mode>(1, 2, 1, 1, 1);
  test_kronmult<TestType, dense_mode>(1, 2, 1, 1, 5);
  test_kronmult<TestType, dense_mode>(1, 2, 1, 2, 3);
  test_kronmult<TestType, dense_mode>(1, 2, 10, 2, 7);
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

TEMPLATE_TEST_CASE("testing kronmult cpu 2d", "[execute_cpu 2d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult<TestType, dense_mode>(2, n, 12, 3, 7);
  test_kronmult<TestType, sparse_mode>(2, n, 12, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 3d", "[execute_cpu 3d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult<TestType, dense_mode>(3, n, 12, 2, 7);
  test_kronmult<TestType, sparse_mode>(3, n, 12, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 4d", "[execute_cpu 4d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult<TestType, dense_mode>(4, n, 9, 2, 7);
  test_kronmult<TestType, sparse_mode>(4, n, 9, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 5d", "[execute_cpu 5d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult<TestType, dense_mode>(5, n, 11, 3, 7);
  test_kronmult<TestType, sparse_mode>(5, n, 11, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 6d", "[execute_cpu 6d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult<TestType, dense_mode>(6, n, 9, 2, 7);
  test_kronmult<TestType, sparse_mode>(6, n, 9, 2, 7);
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
}

TEMPLATE_TEST_CASE("testing kronmult gpu 2d", "[execute_gpu 2d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
  test_kronmult<TestType, dense_mode>(2, n, 13, 2, 7);
  test_kronmult<TestType, sparse_mode>(2, n, 13, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 3d", "[execute_gpu 3d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult<TestType, dense_mode>(3, n, 17, 3, 7);
  test_kronmult<TestType, sparse_mode>(3, n, 17, 3, 7);
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
*/

/*****************************************************************************
 * Testing the ability to split a matrix into multiple calls
 *****************************************************************************/

#include "coefficients.hpp"

using namespace asgard;

template<typename prec>
void test_memory_mode(imex_flag imex)
{
  // make some PDE, no need to be too specific
  fk::vector<int> levels = {5, 5};
  parser parse("two_stream", levels);
  parser_mod::set(parse, parser_mod::degree, 3);

  auto pde = make_PDE<prec>(parse);

  options const opts(parse);

  adapt::distributed_grid grid(*pde, opts);
  basis::wavelet_transform<prec, resource::host> const transformer(opts, *pde);
  generate_dimension_mass_mat(*pde, transformer);
  generate_all_coefficients(*pde, transformer);
  auto const x = grid.get_initial_condition(*pde, transformer, opts);
  generate_dimension_mass_mat(*pde, transformer);

  // one means that all data fits in memory and only one call will be made
  constexpr bool force_sparse = true;

  kron_sparse_cache spcache_null1, spcache_one;
  memory_usage memory_one =
      compute_mem_usage(*pde, grid, opts, imex, spcache_null1);
  auto mat_one              = make_kronmult_matrix(*pde, grid, opts, memory_one,
                                      imex_flag::unspecified, spcache_null1);
  memory_usage spmemory_one = compute_mem_usage(
      *pde, grid, opts, imex, spcache_one, 6, 2147483646, force_sparse);
  auto spmat_one = make_kronmult_matrix(*pde, grid, opts, spmemory_one, imex,
                                        spcache_one, force_sparse);

  kron_sparse_cache spcache_null2, spcache_multi;
  memory_usage memory_multi =
      compute_mem_usage(*pde, grid, opts, imex, spcache_null2, 0, 8000);
  auto mat_multi =
      make_kronmult_matrix(*pde, grid, opts, memory_multi, imex, spcache_null2);
  memory_usage spmemory_multi = compute_mem_usage(
      *pde, grid, opts, imex, spcache_multi, 6, 8000, force_sparse);
  auto spmat_multi = make_kronmult_matrix(*pde, grid, opts, spmemory_multi,
                                          imex, spcache_multi, force_sparse);

  REQUIRE(mat_one.is_onecall());
  REQUIRE(spmat_one.is_onecall());
  REQUIRE(not mat_multi.is_onecall());
  REQUIRE(not spmat_multi.is_onecall());

  fk::vector<prec> y_one(mat_one.output_size());
  fk::vector<prec> y_multi(mat_multi.output_size());
  fk::vector<prec> y_spone(spmat_one.output_size());
  fk::vector<prec> y_spmulti(spmat_multi.output_size());
  REQUIRE(y_one.size() == y_multi.size());
  REQUIRE(y_one.size() == y_spmulti.size());
  REQUIRE(y_one.size() == y_spone.size());

#ifdef ASGARD_USE_CUDA
  fk::vector<prec, mem_type::owner, resource::device> xdev(y_one.size());
  fk::vector<prec, mem_type::owner, resource::device> ydev(y_multi.size());
  mat_one.set_workspace(xdev, ydev);
  mat_multi.set_workspace(xdev, ydev);
  spmat_one.set_workspace(xdev, ydev);
  spmat_multi.set_workspace(xdev, ydev);
#endif
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  // allocate large enough vectors, total size is 24MB
  cudaStream_t load_stream;
  cudaStreamCreate(&load_stream);
  auto worka = fk::vector<int, mem_type::owner, resource::device>(1048576);
  auto workb = fk::vector<int, mem_type::owner, resource::device>(1048576);
  auto irowa = fk::vector<int, mem_type::owner, resource::device>(262144);
  auto irowb = fk::vector<int, mem_type::owner, resource::device>(262144);
  auto icola = fk::vector<int, mem_type::owner, resource::device>(262144);
  auto icolb = fk::vector<int, mem_type::owner, resource::device>(262144);
  mat_multi.set_workspace_ooc(worka, workb, load_stream);
  spmat_multi.set_workspace_ooc(worka, workb, load_stream);
  mat_multi.set_workspace_ooc_sparse(irowa, irowb, icola, icolb);
  spmat_multi.set_workspace_ooc_sparse(irowa, irowb, icola, icolb);
#endif

  mat_one.apply(2.0, x.data(), 0.0, y_one.data());
  mat_multi.apply(2.0, x.data(), 0.0, y_multi.data());
  spmat_one.apply(2.0, x.data(), 0.0, y_spone.data());
  spmat_multi.apply(2.0, x.data(), 0.0, y_spmulti.data());

  rmse_comparison(y_one, y_multi, prec{10});
  rmse_comparison(y_one, y_spone, prec{10});
  rmse_comparison(y_one, y_spmulti, prec{10});

  mat_one.apply(2.5, y_one.data(), 3.0, y_one.data());
  mat_multi.apply(2.5, y_multi.data(), 3.0, y_multi.data());
  spmat_one.apply(2.5, y_spone.data(), 3.0, y_spone.data());
  spmat_multi.apply(2.5, y_spmulti.data(), 3.0, y_spmulti.data());

  rmse_comparison(y_one, y_multi, prec{10});
  rmse_comparison(y_one, y_spone, prec{10});
  rmse_comparison(y_one, y_spmulti, prec{10});

  parameter_manager<prec>::get_instance().reset();
#ifdef ASGARD_USE_GPU_MEM_LIMIT
  cudaStreamDestroy(load_stream);
#endif
}

TEMPLATE_TEST_CASE("testing multi imex unspecified", "unspecified", test_precs)
{
  test_memory_mode<TestType>(imex_flag::unspecified);
}

TEMPLATE_TEST_CASE("testing multi imex implicit", "imex_implicit", test_precs)
{
  test_memory_mode<TestType>(imex_flag::imex_implicit);
}

TEMPLATE_TEST_CASE("testing multi imex explicit", "imex_explicit", test_precs)
{
  test_memory_mode<TestType>(imex_flag::imex_explicit);
}
