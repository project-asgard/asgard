
#include "tests_general.hpp"

#include "asgard_kronmult_tests.hpp"

template<typename T>
void test_almost_equal(std::vector<T> const &x, std::vector<T> const &y,
                       int scale = 10)
{
  rmse_comparison<T>(asgard::fk::vector<T>(x), asgard::fk::vector<T>(y),
                     get_tolerance<T>(scale));
}

#ifndef KRON_MODE_GLOBAL

template<typename T, asgard::resource rec = asgard::resource::host>
void test_kronmult_sparse(int dimensions, int n, int num_rows, int num_terms,
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

  asgard::local_kronmult_matrix<T> kmat;

#ifdef ASGARD_USE_CUDA

  int const tensor_size = asgard::fm::ipow<int>(n, dimensions);

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
  kmat = asgard::local_kronmult_matrix<T>(
      dimensions, n, num_rows, num_rows, num_terms,
      row_indx.clone_onto_device(), col_indx.clone_onto_device(),
      iA.clone_onto_device(), vA.clone_onto_device(),
      std::vector<T>());

  asgard::fk::vector<T, asgard::mem_type::owner, asgard::resource::device> xdev(
      kmat.input_size());
  asgard::fk::vector<T, asgard::mem_type::owner, asgard::resource::device> ydev(
      kmat.output_size());
  kmat.set_workspace(xdev, ydev);

#else

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

  kmat = asgard::local_kronmult_matrix<T>(
      dimensions, n, num_rows, num_rows, num_terms,
      std::move(pntr), std::move(indx), std::move(list_iA), std::move(vA),
      std::vector<T>());

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

template<typename P, asgard::resource rec = asgard::resource::host>
void test_kronmult_dense(int dimensions, int n, int num_terms,
                         int num_1d_blocks)
{
  constexpr bool precompute = true;

  auto data = make_kronmult_welem<P, precompute>(dimensions, n, num_terms,
                                                 num_1d_blocks);

#ifdef ASGARD_USE_CUDA
  std::vector<
      asgard::fk::vector<P, asgard::mem_type::owner, asgard::resource::device>>
      gpu_terms(num_terms);
  asgard::fk::vector<P *> terms_ptr(num_terms);
  for (int t = 0; t < num_terms; t++)
  {
    gpu_terms[t] = data->coefficients[t].clone_onto_device();
    terms_ptr[t] = gpu_terms[t].data();
  }
  auto gpu_terms_ptr = terms_ptr.clone_onto_device();

  asgard::fk::vector<int, asgard::mem_type::owner, asgard::resource::device>
      elem(data->elem.size());
  asgard::fk::copy_to_device(elem.data(), data->elem.data(), elem.size());

  asgard::fk::vector<P, asgard::mem_type::owner, asgard::resource::device> xdev(
      data->input_x.size());
  asgard::fk::vector<P, asgard::mem_type::owner, asgard::resource::device> ydev(
      data->output_y.size());
  asgard::fk::copy_to_device(xdev.data(), data->input_x.data(), xdev.size());
  asgard::fk::copy_to_device(ydev.data(), data->output_y.data(), ydev.size());

  asgard::local_kronmult_matrix<P> kmat(
      dimensions, n, data->num_rows(), data->num_rows(), num_terms,
      std::move(gpu_terms), std::move(elem), 0, 0, num_1d_blocks,
      std::vector<P>());

  kmat.set_workspace(xdev, ydev);

  if constexpr (rec == asgard::resource::device)
  {
    kmat.template apply<asgard::resource::device>(1.0, xdev.data(), 1.0,
                                                  ydev.data());
    asgard::fk::copy_to_host(data->output_y.data(), ydev.data(), ydev.size());
  }
  else
  {
    kmat.apply(1.0, data->input_x.data(), 1.0, data->output_y.data());
  }

#else
  asgard::local_kronmult_matrix<P> kmat(
      dimensions, n, data->num_rows(), data->num_rows(), num_terms,
      std::move(data->coefficients), asgard::fk::vector<int>(data->elem), 0, 0,
      num_1d_blocks, std::vector<P>());

  kmat.apply(1.0, data->input_x.data(), 1.0, data->output_y.data());
#endif

  test_almost_equal(data->output_y, data->reference_y, 100);
}

TEMPLATE_TEST_CASE("testing reference methods", "[kronecker]", test_precs)
{
  std::vector<TestType> A    = {1, 2, 3, 4};
  std::vector<TestType> B    = {10, 20, 30, 40};
  auto R                     = kronecker(2, A.data(), 2, B.data());
  std::vector<TestType> gold = {10, 20, 20, 40, 30, 40, 60, 80,
                                30, 60, 40, 80, 90, 120, 120, 160};
  test_almost_equal(R, gold);

  B    = std::vector<TestType>{1, 2, 3, 4, 5, 6, 7, 8, 9};
  R    = kronecker(2, A.data(), 3, B.data());
  gold = std::vector<TestType>{1, 2, 3, 2, 4, 6, 4, 5, 6, 8, 10, 12,
                               7, 8, 9, 14, 16, 18, 3, 6, 9, 4, 8, 12,
                               12, 15, 18, 16, 20, 24, 21, 24, 27, 28, 32, 36};
  test_almost_equal(R, gold);
}

#ifndef ASGARD_USE_CUDA // test CPU kronmult only when CUDA is not enabled

TEMPLATE_TEST_CASE("testing kronmult cpu core dense", "[cpu_sparse]",
                   test_precs)
{
  test_kronmult_sparse<TestType>(1, 2, 1, 1, 5);
  test_kronmult_dense<TestType>(1, 2, 1, 1);
  test_kronmult_dense<TestType>(1, 2, 1, 5);
  test_kronmult_dense<TestType>(1, 2, 2, 5);
  test_kronmult_dense<TestType>(1, 2, 2, 7);
}
TEMPLATE_TEST_CASE("testing kronmult cpu core sparse", "[cpu_sparse]",
                   test_precs)
{
  test_kronmult_sparse<TestType>(1, 2, 1, 1, 1);
  test_kronmult_sparse<TestType>(1, 2, 1, 1, 5);
  test_kronmult_sparse<TestType>(1, 2, 1, 2, 3);
  test_kronmult_sparse<TestType>(1, 2, 10, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 1d", "[cpu_sparse 1d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6);
  test_kronmult_sparse<TestType>(1, n, 11, 2, 7);
}
TEMPLATE_TEST_CASE("testing kronmult cpu 1d", "[cpu_dense 1d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6);
  test_kronmult_dense<TestType>(1, n, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 2d", "[cpu_sparse 2d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_sparse<TestType>(2, n, 12, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 2d", "[cpu_dense 2d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_dense<TestType>(2, n, 3, 5);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 3d", "[cpu_sparse 3d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_sparse<TestType>(3, n, 12, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 3d", "[cpu_dense 3d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_dense<TestType>(3, n, 3, 3);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 4d", "[cpu_sparse 4d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_sparse<TestType>(4, n, 9, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 4d", "[cpu_dense 4d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_dense<TestType>(4, n, 2, 3);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 5d", "[cpu_sparse 5d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_sparse<TestType>(5, n, 8, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 5d", "[cpu_dense 5d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_dense<TestType>(5, n, 2, 1);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 6d", "[cpu_sparse 6d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult_sparse<TestType>(6, n, 6, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 6d", "[cpu_dense 6d]", test_precs)
{
  int n = GENERATE(1, 2, 3);
  test_kronmult_dense<TestType>(6, n, 2, 2);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 6d (large)", "[cpu_dense 6d]",
                   test_precs)
{
  int n = GENERATE(4, 5);
  test_kronmult_dense<TestType>(6, n, 2, 1);
}

TEMPLATE_TEST_CASE("testing kronmult cpu 6d-general", "[cpu_sparse 6d]",
                   test_precs)
{
  // this is technically supported, but it takes too long
  // the Kronecker products actually suffer from the curse of dimensionality
  // and for 6D with n = 5, tensor size is 15,625 flops per product is 468,750,
  // mops per reference Kronecker products is 244,140,625
  // computing a reference solution becomes an issue, so the test is so small
  test_kronmult_sparse<TestType>(6, 5, 2, 1, 2);
}
#endif

#ifdef ASGARD_USE_CUDA

TEMPLATE_TEST_CASE("testing kronmult gpu 1d", "[gpu_sparse 1d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult_sparse<TestType>(1, n, 11, 2, 7);
  test_kronmult_sparse<TestType, asgard::resource::device>(1, n, 11, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 1d", "[gpu_dense 1d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult_dense<TestType, asgard::resource::host>(1, n, 3, 7);
  test_kronmult_dense<TestType, asgard::resource::device>(1, n, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 2d", "[gpu_sparse 2d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
  test_kronmult_sparse<TestType>(2, n, 13, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 2d", "[gpu_dense 2d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                   18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
  test_kronmult_dense<TestType, asgard::resource::host>(2, n, 3, 7);
  test_kronmult_dense<TestType, asgard::resource::device>(2, n, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 3d", "[gpu_sparse 3d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult_sparse<TestType>(3, n, 17, 3, 7);
  test_kronmult_sparse<TestType, asgard::resource::device>(3, n, 17, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 3d", "[gpu_dense 3d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  test_kronmult_dense<TestType, asgard::resource::host>(3, n, 3, 3);
  test_kronmult_dense<TestType, asgard::resource::device>(3, n, 3, 3);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 4d", "[gpu_sparse 4d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_sparse<TestType>(4, n, 10, 3, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 4d", "[gpu_dense 4d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4, 5);
  test_kronmult_dense<TestType, asgard::resource::host>(4, n, 2, 3);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 5d", "[gpu_sparse 5d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult_sparse<TestType>(5, n, 10, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 5d", "[gpu_dense 5d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult_dense<TestType, asgard::resource::host>(5, n, 2, 1);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 6d", "[gpu_sparse 6d]", test_precs)
{
  int n = GENERATE(1, 2, 3, 4);
  test_kronmult_sparse<TestType>(6, n, 8, 2, 7);
}

TEMPLATE_TEST_CASE("testing kronmult gpu 6d", "[gpu_dense 6d]", test_precs)
{
  int n = GENERATE(1, 2, 3); // TODO: n = 4
  test_kronmult_dense<TestType, asgard::resource::host>(6, n, 2, 1);
}
#endif
#endif // KRON_MODE_GLOBAL

#ifdef KRON_MODE_GLOBAL

TEMPLATE_TEST_CASE("testing simple 1d", "[global kron]", test_precs)
{
  std::minstd_rand park_miller(42);
  std::uniform_real_distribution<TestType> unif(-1.0, 1.0);

  std::vector<int> nindex = {10, 20, 44};
  std::vector<int> levels = {4, 5, 6};

  for (size_t tcase = 0; tcase < nindex.size(); tcase++)
  {
    // very simple test
    asgard::connect_1d conn(levels[tcase], asgard::connect_1d::hierarchy::volume);

    asgard::vector2d<int> ilist(1, nindex[tcase]);
    std::iota(ilist[0], ilist[0] + nindex[tcase], 0);

    // 1d, 1 term, random operator
    std::vector<std::vector<TestType>> vals(1);
    vals[0] = std::vector<TestType>(conn.num_connections());
    for (auto &v : vals[0])
      v = unif(park_miller);

    // random vector
    std::vector<TestType> x(ilist.total_size());
    for (auto &v : x)
      v = unif(park_miller);

    std::vector<TestType> y_ref(x.size(), TestType{0});

    int const num = static_cast<int>(y_ref.size());
    for (int i = 0; i < num; i++)
    {
      for (int j = 0; j < num; j++)
      {
        int const op_index = conn.get_offset(i, j);
        if (op_index > -1) // connected
          y_ref[i] += x[j] * vals[0][op_index];
      }
    }

    asgard::kronmult::permutes perms(1);
    asgard::dimension_sort dsort(ilist);

    std::vector<TestType> y(x.size(), TestType{0});
    std::vector<TestType> w1(y.size(), TestType{0});
    std::vector<TestType> w2(y.size(), TestType{0});
    asgard::kronmult::global_cpu(perms, ilist, dsort, conn, {0}, vals,
                                 TestType{1}, x.data(), y.data(),
                                 w1.data(), w1.data());

    test_almost_equal(y, y_ref);
  }
}

int int_log2(int x)
{
  int l = 0;
  while (x > 0)
  {
    x /= 2;
    l += 1;
  }
  return l;
}

template<typename precision>
void test_global_kron(int num_dimensions, int level)
{
  std::minstd_rand park_miller(42);
  std::uniform_real_distribution<precision> unif(-1.0, 1.0);

  auto indexes = asgard::permutations::generate_lower_index_set(
      num_dimensions,
      [&](std::vector<int> const &index) -> bool {
        int L = 0;
        for (auto const &i : index)
          L += int_log2(i);
        return (L <= level);
      });

  asgard::connect_1d conn(level, asgard::connect_1d::hierarchy::volume);

  asgard::vector2d<int> ilist(num_dimensions, indexes);

  // 1d, 1 term, random operator
  std::vector<std::vector<precision>> vals(num_dimensions);
  for (int d = 0; d < num_dimensions; d++)
  {
    vals[d] = std::vector<precision>(conn.num_connections());
    for (auto &v : vals[d])
      v = unif(park_miller);
  }

  // random vector
  int const num = ilist.num_strips();
  std::vector<precision> x(num);
  for (auto &v : x)
    v = unif(park_miller);

  std::vector<precision> y_ref(x.size(), precision{0});

#pragma omp parallel for
  for (int m = 0; m < num; m++)
  {
    for (int i = 0; i < num; i++)
    {
      precision t = 1;
      for (int d = 0; d < num_dimensions; d++)
      {
        int const op_index = conn.get_offset(ilist[m][d], ilist[i][d]);
        if (op_index == -1)
        {
          t = 0;
          break;
        }
        else
        {
          t *= vals[d][op_index];
        }
      }
      y_ref[m] += x[i] * t;
    }
  }

  asgard::kronmult::permutes perms(num_dimensions);
  asgard::dimension_sort dsort(ilist);

  std::vector<precision> y(y_ref.size(), precision{0});
  std::vector<precision> w1(y.size(), precision{0});
  std::vector<precision> w2(y.size(), precision{0});
  asgard::kronmult::global_cpu(perms, ilist, dsort, conn, {0}, vals,
                               precision{1}, x.data(), y.data(),
                               w1.data(), w2.data());

  test_almost_equal(y, y_ref);
}

TEMPLATE_TEST_CASE("testing global kron 2d, constant basis", "[gkron 2d]", test_precs)
{
  int l = GENERATE(1, 2, 3, 4, 5, 6);
  test_global_kron<TestType>(2, l);
}

TEMPLATE_TEST_CASE("testing global kron 3d, constant basis", "[gkron 3d]", test_precs)
{
  int l = GENERATE(1, 2, 3, 4, 5, 6);
  test_global_kron<TestType>(3, l);
}

TEMPLATE_TEST_CASE("testing global kron 4d, constant basis", "[gkron 4d]", test_precs)
{
  int l = GENERATE(1, 2, 3, 4, 5, 6);
  test_global_kron<TestType>(4, l);
}

TEMPLATE_TEST_CASE("testing global kron 5d, constant basis", "[gkron 5d]", test_precs)
{
  int l = GENERATE(1, 2, 3, 4, 5);
  test_global_kron<TestType>(5, l);
}

#ifdef ASGARD_USE_CUDA
TEMPLATE_TEST_CASE("testing cusparse functionality", "[cusparse]", test_precs)
{
  asgard::gpu::sparse_handle cusparse;

  std::vector<int> pntr      = {0, 2, 5, 8, 10};
  std::vector<int> indx      = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
  std::vector<TestType> vals = {-2.0, 1.0, 1.0, -2.0, 1.0, 1.0, -2.0, 1.0, 1.0, -2.0};

  std::vector<TestType> x = {1.0, 2.0, 3.0, 4.0};
  std::vector<TestType> y_ref(pntr.size() - 1, TestType{0});

  for (size_t r = 0; r < pntr.size() - 1; r++)
    for (int j = pntr[r]; j < pntr[r + 1]; j++)
      y_ref[r] += x[indx[j]] * vals[j];

  asgard::gpu::vector<int> gpntr      = pntr;
  asgard::gpu::vector<int> gindx      = indx;
  asgard::gpu::vector<TestType> gvals = vals;
  asgard::gpu::vector<TestType> gx    = x;

  asgard::gpu::vector<TestType> gy(gx.size());

  asgard::gpu::sparse_matrix<TestType> mat(4, 4, indx.size(), gpntr.data(),
                                           gindx.data(), gvals.data());

  mat.set_vectors(4, TestType{1}, gx.data(), TestType{0}, gy.data());

  size_t work_size = mat.size_workspace(cusparse);
  asgard::gpu::vector<std::byte> work(work_size);

  mat.apply(cusparse, work.data());

  std::vector<TestType> y = gy;
  test_almost_equal(y, y_ref);

  // change the values without changing the matrix object
  vals = {2.0, 0.5, 1.0, 2.0, 1.0, 0.5, -2.0, 1.0, 1.0, -2.0};
  std::fill(y_ref.begin(), y_ref.end(), TestType{0});

  for (size_t r = 0; r < pntr.size() - 1; r++)
    for (int j = pntr[r]; j < pntr[r + 1]; j++)
      y_ref[r] += x[indx[j]] * vals[j]; // recompute y_ref

  asgard::fk::copy_to_device(gvals.data(), vals.data(), vals.size());
  mat.apply(cusparse, work.data());
  y = gy;
  test_almost_equal(y, y_ref);
}
#endif
#endif
