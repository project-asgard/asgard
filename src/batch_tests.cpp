#include "batch.hpp"
#include "chunk.hpp"
#include "coefficients.hpp"
#include "fast_math.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"
#include <numeric>
#include <random>

TEMPLATE_TEST_CASE_SIG("batch", "[batch]",
                       ((typename TestType, resource resrc), TestType, resrc),
                       (double, resource::host), (double, resource::device),
                       (float, resource::host), (float, resource::device))
{
  bool const do_trans = true;

  // clang-format off
  fk::matrix<TestType, mem_type::owner, resrc> const first {
         {12, 22, 32},
         {13, 23, 33},
         {14, 24, 34},
         {15, 25, 35},
         {16, 26, 36},
  };
  fk::matrix<TestType, mem_type::owner, resrc> const second {
         {17, 27, 37},
         {18, 28, 38},
         {19, 29, 39},
         {20, 30, 40},
         {21, 31, 41},
  };
  fk::matrix<TestType, mem_type::owner, resrc> const third {
         {22, 32, 42},
         {23, 33, 43},
         {24, 34, 44},
         {25, 35, 45},
         {26, 36, 46},
  }; // clang-format on

  int const start_row = 0;
  int const stop_row  = 3;
  int const nrows     = stop_row - start_row + 1;
  int const start_col = 1;
  int const stop_col  = 2;
  int const ncols     = stop_col - start_col + 1;
  int const stride    = first.nrows();

  fk::matrix<TestType, mem_type::view, resrc> const first_v(
      first, start_row, stop_row, start_col, stop_col);
  fk::matrix<TestType, mem_type::view, resrc> const second_v(
      second, start_row, stop_row, start_col, stop_col);
  fk::matrix<TestType, mem_type::view, resrc> const third_v(
      third, start_row, stop_row, start_col, stop_col);

  int const num_batch               = 3;
  batch<TestType, resrc> const gold = [&] {
    batch<TestType, resrc> builder(num_batch, nrows, ncols, stride, do_trans);

    builder.assign_entry(first_v, 0);
    builder.assign_entry(second_v, 1);
    builder.assign_entry(third_v, 2);

    return builder;
  }();

  SECTION("batch: constructors, copy/move")
  {
    SECTION("constructor")
    {
      batch<TestType, resrc> const empty(num_batch, nrows, ncols, stride,
                                         do_trans);
      REQUIRE(empty.num_entries() == num_batch);
      REQUIRE(empty.nrows() == nrows);
      REQUIRE(empty.ncols() == ncols);
      REQUIRE(empty.get_stride() == stride);
      REQUIRE(empty.get_trans() == do_trans);

      for (TestType *const ptr : empty)
      {
        REQUIRE(ptr == nullptr);
      }
    }

    SECTION("copy construction")
    {
      batch<TestType, resrc> const gold_copy(gold);
      REQUIRE(gold_copy == gold);
    }

    SECTION("copy assignment")
    {
      batch<TestType, resrc> test(num_batch, nrows, ncols, stride, do_trans);
      test = gold;
      REQUIRE(test == gold);
    }

    SECTION("move construction")
    {
      batch<TestType, resrc> gold_copy(gold);
      batch const test = std::move(gold_copy);
      REQUIRE(test == gold);
    }

    SECTION("move assignment")
    {
      batch<TestType, resrc> test(num_batch, nrows, ncols, stride, do_trans);
      batch<TestType, resrc> gold_copy(gold);
      test = std::move(gold_copy);
      REQUIRE(test == gold);
    }
  }

  SECTION("batch: insert/clear and getters")
  {
    SECTION("insert/getter")
    {
      batch<TestType, resrc> test(num_batch, nrows, ncols, stride, do_trans);
      test.assign_entry(first_v, 0);
      TestType *const *ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == nullptr);
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      test.assign_entry(third_v, 2);
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      test.assign_entry(second_v, 1);
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == second_v.data());
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));
    }

    SECTION("raw pointer insert")
    {
      batch<TestType, resrc> test(num_batch, nrows, ncols, stride, do_trans);
      test.assign_raw(first_v.data(), 0);
      TestType *const *ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == nullptr);
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      test.assign_raw(third_v.data(), 2);
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      test.assign_raw(second_v.data(), 1);
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == first_v.data());
      REQUIRE(ptr_list[1] == second_v.data());
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));
    }

    SECTION("clear")
    {
      batch<TestType, resrc> test(gold);

      // clear should return true when
      // an element was assigned to that index
      REQUIRE(test.clear_entry(0));
      TestType *const *ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == nullptr);
      REQUIRE(ptr_list[1] == second_v.data());
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      // clear should return false when
      // no element was assigned to that index
      REQUIRE(!test.clear_entry(0));

      REQUIRE(test.clear_entry(1));
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == nullptr);
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == third_v.data());
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      REQUIRE(!test.clear_entry(1));

      REQUIRE(test.clear_entry(2));
      ptr_list = test.get_list();
      REQUIRE(ptr_list[0] == nullptr);
      REQUIRE(ptr_list[1] == nullptr);
      REQUIRE(ptr_list[2] == nullptr);
      REQUIRE(ptr_list[0] == test(0));
      REQUIRE(ptr_list[1] == test(1));
      REQUIRE(ptr_list[2] == test(2));

      REQUIRE(!test.clear_entry(2));
    }
  }

  SECTION("batch: utility functions")
  {
    SECTION("is_filled")
    {
      REQUIRE(gold.is_filled());
      batch<TestType, resrc> test(gold);
      test.clear_entry(0);
      REQUIRE(!test.is_filled());
    }

    SECTION("clear_all")
    {
      batch<TestType, resrc> const test = [&] {
        batch<TestType, resrc> gold_copy(gold);
        return gold_copy.clear_all();
      }();

      for (TestType *const ptr : test)
      {
        REQUIRE(ptr == nullptr);
      }
    }

    SECTION("const iterator")
    {
      TestType **const test = new TestType *[num_batch]();
      int counter           = 0;
      for (TestType *const ptr : gold)
      {
        test[counter++] = ptr;
      }

      TestType *const *const gold_list = gold.get_list();

      for (int i = 0; i < num_batch; ++i)
      {
        REQUIRE(test[i] == gold_list[i]);
      }
    }
  }
}

template<typename P, resource resrc = resource::host>
void test_batched_gemm(int const m, int const n, int const k, int const lda,
                       int const ldb, int const ldc, int const num_batch = 3,
                       bool const trans_a = false, bool const trans_b = false,
                       P const alpha = 1.0, P const beta = 0.0)
{
  assert(m > 0);
  assert(n > 0);
  assert(k > 0);

  int const rows_a = trans_a ? k : m;
  int const cols_a = trans_a ? m : k;
  assert(lda >= rows_a);
  assert(ldc >= m);

  int const rows_b = trans_b ? n : k;
  int const cols_b = trans_b ? k : n;
  assert(ldb >= rows_b);

  std::vector<std::vector<fk::matrix<P, mem_type::owner, resrc>>> const
      matrices = [=]() {
        // {a, b, c, gold}
        std::vector<std::vector<fk::matrix<P, mem_type::owner, resrc>>>
            matrices(4);

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
            matrices[0].push_back(a);
            matrices[1].push_back(b);
            matrices[2].push_back(c);
          }
          else
          {
            matrices[0].push_back(a.clone_onto_device());
            matrices[1].push_back(b.clone_onto_device());
            matrices[2].push_back(c.clone_onto_device());
          }

          fk::matrix<P, mem_type::view> const effective_a(a, 0, rows_a - 1, 0,
                                                          cols_a - 1);
          fk::matrix<P, mem_type::view> const effective_b(b, 0, rows_b - 1, 0,
                                                          cols_b - 1);
          fk::matrix<P, mem_type::view> effective_c(c, 0, m - 1, 0, n - 1);
          fm::gemm(effective_a, effective_b, effective_c, trans_a, trans_b,
                   alpha, beta);

          if constexpr (resrc == resource::host)
          {
            matrices[3].push_back(fk::matrix<P>(effective_c));
          }
          else
          {
            matrices[3].push_back(effective_c.clone_onto_device());
          }
        }
        return matrices;
      }();

  auto const batch_build =
      [num_batch](
          std::vector<fk::matrix<P, mem_type::owner, resrc>> const &mats,
          int const nrows, int const ncols, int const stride,
          bool const trans) {
        batch<P, resrc> builder(num_batch, nrows, ncols, stride, trans);
        for (int i = 0; i < num_batch; ++i)
        {
          builder.assign_entry(fk::matrix<P, mem_type::view, resrc>(
                                   mats[i], 0, nrows - 1, 0, ncols - 1),
                               i);
        }
        return builder;
      };

  auto const a_batch = batch_build(matrices[0], rows_a, cols_a, lda, trans_a);
  auto const b_batch = batch_build(matrices[1], rows_b, cols_b, ldb, trans_b);
  auto const c_batch = batch_build(matrices[2], m, n, ldc, false);

  batched_gemm(a_batch, b_batch, c_batch, alpha, beta);

  // check results. we only want the effective region of c,
  // i.e. not the padding region that extends to ldc
  auto const effect_c = [m, n](auto const c) {
    return fk::matrix<P, mem_type::view>(c, 0, m - 1, 0, n - 1);
  };
  for (int i = 0; i < num_batch; ++i)
  {
    if constexpr (resrc == resource::host)
    {
      REQUIRE(effect_c(matrices[2][i]) == effect_c(matrices[3][i]));
    }
    else
    {
      relaxed_comparison(effect_c(matrices[2][i].clone_onto_host()),
                         effect_c(matrices[3][i].clone_onto_host()));
    }
  }
}

TEMPLATE_TEST_CASE_SIG("batched gemm", "[batch]",
                       ((typename TestType, resource resrc), TestType, resrc),
                       (double, resource::host), (double, resource::device),
                       (float, resource::host), (float, resource::device))
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

template<typename P, resource resrc = resource::host>
void test_batched_gemv(int const m, int const n, int const lda,
                       int const num_batch = 3, bool const trans_a = false,
                       P const alpha = 1.0, P const beta = 0.0)
{
  assert(m > 0);
  assert(n > 0);
  assert(lda >= m);

  int const rows_a = trans_a ? n : m;
  int const cols_a = trans_a ? m : n;

  std::random_device rd;
  std::mt19937 mersenne_engine(rd());
  std::uniform_real_distribution<P> dist(-2.0, 2.0);
  auto const gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };

  std::vector<fk::matrix<P, mem_type::owner, resrc>> const a_mats = [=]() {
    std::vector<fk::matrix<P, mem_type::owner, resrc>> a_mats;

    for (int i = 0; i < num_batch; ++i)
    {
      fk::matrix<P> a(lda, n);
      std::generate(a.begin(), a.end(), gen);

      if constexpr (resrc == resource::host)
      {
        a_mats.push_back(a);
      }
      else
      {
        a_mats.push_back(a.clone_onto_device());
      }
    }
    return a_mats;
  }();

  std::vector<std::vector<fk::vector<P, mem_type::owner, resrc>>> const
      vectors = [=, &a_mats]() {
        // {x, y, gold}
        std::vector<std::vector<fk::vector<P, mem_type::owner, resrc>>> vectors(
            3);

        for (int i = 0; i < num_batch; ++i)
        {
          fk::vector<P> x(cols_a);
          fk::vector<P> y(rows_a);
          std::generate(x.begin(), x.end(), gen);
          std::generate(y.begin(), y.end(), gen);

          if constexpr (resrc == resource::host)
          {
            vectors[0].push_back(x);
            vectors[1].push_back(y);
          }
          else
          {
            vectors[0].push_back(x.clone_onto_device());
            vectors[1].push_back(y.clone_onto_device());
          }

          fk::matrix<P, mem_type::view, resrc> const effective_a(
              a_mats[i], 0, m - 1, 0, n - 1);
          fk::vector<P, mem_type::owner, resrc> gold(vectors[1].back());
          fm::gemv(effective_a, vectors[0].back(), gold, trans_a, alpha, beta);
          vectors[2].push_back(gold);
        }
        return vectors;
      }();

  auto const batch_build =
      [num_batch](
          std::vector<fk::matrix<P, mem_type::owner, resrc>> const &mats,
          int const nrows, int const ncols, int const stride,
          bool const trans) {
        batch<P, resrc> builder(num_batch, nrows, ncols, stride, trans);
        for (int i = 0; i < num_batch; ++i)
        {
          builder.assign_entry(fk::matrix<P, mem_type::view, resrc>(
                                   mats[i], 0, nrows - 1, 0, ncols - 1),
                               i);
        }
        return builder;
      };

  // FIXME this is why we need templated lambdas
  // could use auto but lose type info
  auto const batch_build_v =
      [num_batch](
          std::vector<fk::vector<P, mem_type::owner, resrc>> const &vects,
          int const nrows) {
        batch<P, resrc> builder(num_batch, nrows, 1, nrows, false);
        for (int i = 0; i < num_batch; ++i)
        {
          builder.assign_entry(
              fk::matrix<P, mem_type::view, resrc>(vects[i], nrows, 1), i);
        }
        return builder;
      };

  auto const a_batch = batch_build(a_mats, m, n, lda, trans_a);
  auto const x_batch = batch_build_v(vectors[0], cols_a);
  auto const y_batch = batch_build_v(vectors[1], rows_a);

  batched_gemv(a_batch, x_batch, y_batch, alpha, beta);

  for (int i = 0; i < num_batch; ++i)
  {
    if constexpr (resrc == resource::host)
    {
      REQUIRE(vectors[1][i] == vectors[2][i]);
    }
    else
    {
      REQUIRE(vectors[1][i].clone_onto_host() ==
              vectors[2][i].clone_onto_host());
    }
  }
}

TEMPLATE_TEST_CASE_SIG("batched gemv", "[batch]",
                       ((typename TestType, resource resrc), TestType, resrc),
                       (double, resource::host), (double, resource::device),
                       (float, resource::host), (float, resource::device))
{
  SECTION("batched gemv: no trans, alpha = 1.0, beta = 0.0")
  {
    int const m         = 8;
    int const n         = 4;
    int const lda       = m;
    int const num_batch = 4;
    test_batched_gemv<TestType, resrc>(m, n, lda, num_batch);
  }

  SECTION("batched gemv: trans, alpha = 1.0, beta = 0.0")
  {
    int const m         = 8;
    int const n         = 4;
    int const lda       = m + 1;
    int const num_batch = 2;
    bool const trans_a  = true;
    test_batched_gemv<TestType, resrc>(m, n, lda, num_batch, trans_a);
  }

  SECTION("batched gemv: no trans, test scaling")
  {
    int const m          = 12;
    int const n          = 5;
    int const lda        = m + 3;
    int const num_batch  = 5;
    bool const trans_a   = false;
    TestType const alpha = -2.0;
    TestType const beta  = -4.5;
    test_batched_gemv<TestType, resrc>(m, n, lda, num_batch, trans_a, alpha,
                                       beta);
  }
}

TEMPLATE_TEST_CASE("batch allocator", "[batch]", float, double)
{
  SECTION("1d, deg 3")
  {
    int const level     = 2;
    int const degree    = 3;
    int const num_elems = 60;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    int const stride = pde->get_coefficients(0, 0).stride();

    int const gold_size = pde->num_terms * num_elems;

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == 1);
    batch_operands_set<TestType> const batches_dim0 = batches[0];

    int const gold_rows_a   = degree;
    int const gold_cols_a   = degree;
    int const gold_stride_a = stride;
    bool const gold_trans_a = false;
    assert(batches_dim0[0].num_entries() == gold_size);
    assert(batches_dim0[0].nrows() == gold_rows_a);
    assert(batches_dim0[0].ncols() == gold_cols_a);
    assert(batches_dim0[0].get_stride() == gold_stride_a);
    assert(batches_dim0[0].get_trans() == gold_trans_a);

    int const gold_rows_b   = degree;
    int const gold_cols_b   = std::pow(degree, pde->num_dims - 1);
    int const gold_stride_b = degree;
    bool const gold_trans_b = false;
    assert(batches_dim0[1].num_entries() == gold_size);
    assert(batches_dim0[1].nrows() == gold_rows_b);
    assert(batches_dim0[1].ncols() == gold_cols_b);
    assert(batches_dim0[1].get_stride() == gold_stride_b);
    assert(batches_dim0[1].get_trans() == gold_trans_b);

    int const gold_rows_c   = gold_rows_a;
    int const gold_cols_c   = gold_cols_b;
    int const gold_stride_c = gold_rows_a;
    assert(batches_dim0[2].num_entries() == gold_size);
    assert(batches_dim0[2].nrows() == gold_rows_c);
    assert(batches_dim0[2].ncols() == gold_cols_c);
    assert(batches_dim0[2].get_stride() == gold_stride_c);
    assert(batches_dim0[2].get_trans() == false);
  }

  SECTION("1d, deg 6")
  {
    int const level     = 2;
    int const degree    = 6;
    int const num_elems = 400;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    int const stride = pde->get_coefficients(0, 0).stride();

    int const gold_size = pde->num_terms * num_elems;

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == 1);
    batch_operands_set<TestType> const batches_dim0 = batches[0];

    int const gold_rows_a   = degree;
    int const gold_cols_a   = degree;
    int const gold_stride_a = stride;
    bool const gold_trans_a = false;
    assert(batches_dim0[0].num_entries() == gold_size);
    assert(batches_dim0[0].nrows() == gold_rows_a);
    assert(batches_dim0[0].ncols() == gold_cols_a);
    assert(batches_dim0[0].get_stride() == gold_stride_a);
    assert(batches_dim0[0].get_trans() == gold_trans_a);

    int const gold_rows_b   = degree;
    int const gold_cols_b   = std::pow(degree, pde->num_dims - 1);
    int const gold_stride_b = degree;
    bool const gold_trans_b = false;
    assert(batches_dim0[1].num_entries() == gold_size);
    assert(batches_dim0[1].nrows() == gold_rows_b);
    assert(batches_dim0[1].ncols() == gold_cols_b);
    assert(batches_dim0[1].get_stride() == gold_stride_b);
    assert(batches_dim0[1].get_trans() == gold_trans_b);

    int const gold_rows_c   = gold_rows_a;
    int const gold_cols_c   = gold_cols_b;
    int const gold_stride_c = gold_rows_a;
    assert(batches_dim0[2].num_entries() == gold_size);
    assert(batches_dim0[2].nrows() == gold_rows_c);
    assert(batches_dim0[2].ncols() == gold_cols_c);
    assert(batches_dim0[2].get_stride() == gold_stride_c);
    assert(batches_dim0[2].get_trans() == false);
  }

  SECTION("2d, deg 2")
  {
    int const level      = 2;
    int const degree     = 2;
    int const num_elems  = 101;
    int const dimensions = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == dimensions);

    int const gold_size = pde->num_terms * num_elems;
    for (int i = 0; i < dimensions; ++i)
    {
      int const stride = pde->get_coefficients(0, i).stride();
      batch_operands_set<TestType> const batch_dim = batches[i];
      int const gold_rows_a   = i == 0 ? degree : std::pow(degree, i);
      int const gold_cols_a   = degree;
      int const gold_stride_a = i == 0 ? stride : gold_rows_a;
      bool const gold_trans_a = false;
      assert(batch_dim[0].num_entries() == gold_size);
      assert(batch_dim[0].nrows() == gold_rows_a);
      assert(batch_dim[0].ncols() == gold_cols_a);
      assert(batch_dim[0].get_stride() == gold_stride_a);
      assert(batch_dim[0].get_trans() == gold_trans_a);

      int const gold_rows_b = degree;
      int const gold_cols_b =
          i == 0 ? std::pow(degree, pde->num_dims - 1) : degree;
      int const gold_stride_b = i == 0 ? degree : stride;
      bool const gold_trans_b = i == 0 ? false : true;
      assert(batch_dim[1].num_entries() == gold_size);
      assert(batch_dim[1].nrows() == gold_rows_b);
      assert(batch_dim[1].ncols() == gold_cols_b);
      assert(batch_dim[1].get_stride() == gold_stride_b);
      assert(batch_dim[1].get_trans() == gold_trans_b);

      int const gold_rows_c   = gold_rows_a;
      int const gold_cols_c   = gold_cols_b;
      int const gold_stride_c = gold_rows_a;
      assert(batch_dim[2].num_entries() == gold_size);
      assert(batch_dim[2].nrows() == gold_rows_c);
      assert(batch_dim[2].ncols() == gold_cols_c);
      assert(batch_dim[2].get_stride() == gold_stride_c);
      assert(batch_dim[2].get_trans() == false);
    }
  }

  SECTION("2d, deg 5")
  {
    int const level      = 2;
    int const degree     = 5;
    int const num_elems  = 251;
    int const dimensions = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == dimensions);

    int const gold_size = pde->num_terms * num_elems;
    for (int i = 0; i < dimensions; ++i)
    {
      int const stride = pde->get_coefficients(0, i).stride();
      batch_operands_set<TestType> const batch_dim = batches[i];
      int const gold_rows_a   = i == 0 ? degree : std::pow(degree, i);
      int const gold_cols_a   = degree;
      int const gold_stride_a = i == 0 ? stride : gold_rows_a;
      bool const gold_trans_a = false;
      assert(batch_dim[0].num_entries() == gold_size);
      assert(batch_dim[0].nrows() == gold_rows_a);
      assert(batch_dim[0].ncols() == gold_cols_a);
      assert(batch_dim[0].get_stride() == gold_stride_a);
      assert(batch_dim[0].get_trans() == gold_trans_a);

      int const gold_rows_b = degree;
      int const gold_cols_b =
          i == 0 ? std::pow(degree, pde->num_dims - 1) : degree;
      int const gold_stride_b = i == 0 ? degree : stride;
      bool const gold_trans_b = i == 0 ? false : true;
      assert(batch_dim[1].num_entries() == gold_size);
      assert(batch_dim[1].nrows() == gold_rows_b);
      assert(batch_dim[1].ncols() == gold_cols_b);
      assert(batch_dim[1].get_stride() == gold_stride_b);
      assert(batch_dim[1].get_trans() == gold_trans_b);

      int const gold_rows_c   = gold_rows_a;
      int const gold_cols_c   = gold_cols_b;
      int const gold_stride_c = gold_rows_a;
      assert(batch_dim[2].num_entries() == gold_size);
      assert(batch_dim[2].nrows() == gold_rows_c);
      assert(batch_dim[2].ncols() == gold_cols_c);
      assert(batch_dim[2].get_stride() == gold_stride_c);
      assert(batch_dim[2].get_trans() == false);
    }
  }
  SECTION("6d, deg 4")
  {
    int const level      = 3;
    int const degree     = 4;
    int const num_elems  = 100;
    int const dimensions = 6;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    std::vector<batch_operands_set<TestType>> const batches =
        allocate_batches(*pde, num_elems);
    assert(batches.size() == dimensions);

    for (int i = 0; i < dimensions; ++i)
    {
      int const gold_size = [&] {
        if (i == 0 || i == dimensions - 1)
        {
          return pde->num_terms * num_elems;
        }
        return static_cast<int>(std::pow(degree, (dimensions - i - 1))) *
               pde->num_terms * num_elems;
      }();
      int const stride = pde->get_coefficients(0, i).stride();
      batch_operands_set<TestType> const batch_dim = batches[i];
      int const gold_rows_a   = i == 0 ? degree : std::pow(degree, i);
      int const gold_cols_a   = degree;
      int const gold_stride_a = i == 0 ? stride : gold_rows_a;
      bool const gold_trans_a = false;

      assert(batch_dim[0].num_entries() == gold_size);
      assert(batch_dim[0].nrows() == gold_rows_a);
      assert(batch_dim[0].ncols() == gold_cols_a);
      assert(batch_dim[0].get_stride() == gold_stride_a);
      assert(batch_dim[0].get_trans() == gold_trans_a);

      int const gold_rows_b = degree;
      int const gold_cols_b =
          i == 0 ? std::pow(degree, pde->num_dims - 1) : degree;
      int const gold_stride_b = i == 0 ? degree : stride;
      bool const gold_trans_b = i == 0 ? false : true;
      assert(batch_dim[1].num_entries() == gold_size);
      assert(batch_dim[1].nrows() == gold_rows_b);
      assert(batch_dim[1].ncols() == gold_cols_b);
      assert(batch_dim[1].get_stride() == gold_stride_b);
      assert(batch_dim[1].get_trans() == gold_trans_b);

      int const gold_rows_c   = gold_rows_a;
      int const gold_cols_c   = gold_cols_b;
      int const gold_stride_c = gold_rows_a;
      assert(batch_dim[2].num_entries() == gold_size);
      assert(batch_dim[2].nrows() == gold_rows_c);
      assert(batch_dim[2].ncols() == gold_cols_c);
      assert(batch_dim[2].get_stride() == gold_stride_c);
      assert(batch_dim[2].get_trans() == false);
    }
  }
}

TEMPLATE_TEST_CASE("kronmult batching", "[batch]", float, double)
{
  SECTION("1 element, 1d, 1 term")
  {
    int const degree    = 4;
    int const level     = 2;
    int const num_elems = 1;

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    // clang-format off
    fk::matrix<TestType> const A {
        { 2,  3,  4,  5}, 
	{ 6,  7,  8,  9}, 
	{10, 11, 12, 13}, 
	{14, 15, 16, 17}};
    // clang-format on

    auto coeff = pde->get_coefficients(0, 0).clone_onto_host();
    coeff.set_submatrix(0, 0, A);
    fk::matrix<TestType, mem_type::owner, resource::device> const
        coefficient_matrix(coeff.clone_onto_device());

    fk::vector<TestType> const x_h{18, 19, 20, 21};
    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());
    fk::vector<TestType> const gold = A * x_h;

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    fk::matrix<TestType, mem_type::view, resource::device> const coeff_view(
        coefficient_matrix, 0, degree - 1, 0, degree - 1);
    std::vector<fk::matrix<TestType, mem_type::view, resource::device>> const
        As = {coeff_view};
    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(degree);
    fk::vector<TestType, mem_type::view, resource::device> y(y_own);
    std::vector<fk::vector<TestType, mem_type::view, resource::device>>
        work_set           = {};
    int const batch_offset = 0;

    kronmult_to_batch_sets(As, x_view, y, work_set, batches, batch_offset,
                           *pde);

    batch<TestType> const a = batches[0][0];
    batch<TestType> const b = batches[0][1];
    batch<TestType> const c = batches[0][2];

    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a, b, c, alpha, beta);

    fk::vector<TestType, mem_type::owner> const y_h(y.clone_onto_host());
    REQUIRE(gold == y_h);
  }

  SECTION("2 elements, 1d, 1 term")
  {
    int const degree    = 4;
    int const level     = 2;
    int const num_elems = 2;

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    // clang-format off
    fk::matrix<TestType> const A {
        { 2,  3,  4,  5,  6,  7}, 
	{ 8,  9, 10, 11, 12, 13}, 
	{14, 15, 16, 17, 18, 19}, 
	{20, 21, 22, 23, 24, 25}, 
	{26, 27, 28, 29, 30, 31}, 
	{32, 33, 34, 35, 36, 37}};
    // clang-format on

    auto coeff = pde->get_coefficients(0, 0).clone_onto_host();
    coeff.set_submatrix(0, 0, A);
    fk::matrix<TestType, mem_type::owner, resource::device> const
        coefficient_matrix(coeff.clone_onto_device());

    fk::vector<TestType> const x_h{18, 19, 20, 21};
    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());
    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // each element addresses a slightly different part of the underlying
    // coefficients
    fk::matrix<TestType, mem_type::view, resource::device> const A_view_e0(
        coefficient_matrix, 0, degree - 1, 0, degree - 1);
    fk::matrix<TestType, mem_type::view, resource::device> const A_view_e1(
        coefficient_matrix, 2, 2 + degree - 1, 2, 2 + degree - 1);

    fk::matrix<TestType, mem_type::view> const A_view_e0_h(A, 0, degree - 1, 0,
                                                           degree - 1);
    fk::matrix<TestType, mem_type::view> const A_view_e1_h(A, 2, 2 + degree - 1,
                                                           2, 2 + degree - 1);

    fk::vector<TestType> const gold_e0 = A_view_e0_h * x_h;
    fk::vector<TestType> const gold_e1 = A_view_e1_h * x_h;
    fk::vector<TestType> const gold    = gold_e0 + gold_e1;

    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(degree *
                                                                  num_elems);

    // schedule gemms for both elements
    int batch_offset = 0;
    std::vector<fk::matrix<TestType, mem_type::view, resource::device>> const
        As_e0 = {A_view_e0};
    fk::vector<TestType, mem_type::view, resource::device> y_e0(y_own, 0,
                                                                degree - 1);
    std::vector<fk::vector<TestType, mem_type::view, resource::device>>
        work_set_e0 = {};
    kronmult_to_batch_sets(As_e0, x_view, y_e0, work_set_e0, batches,
                           batch_offset, *pde);

    batch_offset = 1;
    std::vector<fk::matrix<TestType, mem_type::view, resource::device>> const
        As_e1 = {A_view_e1};
    fk::vector<TestType, mem_type::view, resource::device> y_e1(
        y_own, degree, y_own.size() - 1);
    std::vector<fk::vector<TestType, mem_type::view, resource::device>>
        work_set_e1 = {};

    kronmult_to_batch_sets(As_e1, x_view, y_e1, work_set_e1, batches,
                           batch_offset, *pde);

    batch<TestType> const a = batches[0][0];
    batch<TestType> const b = batches[0][1];
    batch<TestType> const c = batches[0][2];

    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a, b, c, alpha, beta);

    fk::vector<TestType> const y_e0_h(y_e0.clone_onto_host());
    fk::vector<TestType> const y_e1_h(y_e1.clone_onto_host());

    REQUIRE(gold_e0 == y_e0_h);
    REQUIRE(gold_e1 == y_e1_h);
    REQUIRE(gold == (y_e0_h + y_e1_h));
  }

  SECTION("2 elements, 2d, 2 terms")
  {
    int const degree    = 5;
    int const level     = 2;
    int const num_elems = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    int const num_terms = 2;
    int const num_dims  = 2;
    // first, create example coefficient matrices in the pde
    int const dof = degree * std::pow(2, level);
    std::array<fk::matrix<TestType>, num_terms *num_dims> A_mats_h = {
        fk::matrix<TestType>(dof, dof), fk::matrix<TestType>(dof, dof),
        fk::matrix<TestType>(dof, dof), fk::matrix<TestType>(dof, dof)};

    // create different matrices for each term/dim pairing
    int start = 1;

    std::vector<fk::matrix<TestType, mem_type::owner, resource::device>> A_mats;
    for (fk::matrix<TestType> &mat : A_mats_h)
    {
      std::iota(mat.begin(), mat.end(), start);
      start += dof;
      A_mats.push_back(fk::matrix<TestType, mem_type::owner, resource::device>(
          mat.clone_onto_device()));
    }

    // create input vector
    int const x_size = static_cast<int>(std::pow(degree, pde->num_dims));
    fk::vector<TestType> x_h(x_size);
    std::iota(x_h.begin(), x_h.end(), 1);
    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // create intermediate workspaces
    // and output vectors
    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(
        x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner> gold(x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner, resource::device> work_own(
        x_size * num_elems * num_terms * std::min(num_dims - 1, 2));

    for (int i = 0; i < num_elems; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        // linearize index
        int const kron_index = i * num_terms + j;

        // address y space
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * std::min(num_dims - 1, 2);
        fk::vector<TestType, mem_type::view, resource::device> y_view(
            y_own, y_index, y_index + x_size - 1);
        fk::vector<TestType, mem_type::view> gold_view(gold, y_index,
                                                       y_index + x_size - 1);

        // intermediate workspace
        std::vector<fk::vector<TestType, mem_type::view, resource::device>>
            work_views = {
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index, work_index + x_size - 1)};

        // create A_views
        std::vector<fk::matrix<TestType, mem_type::view, resource::device>>
            A_views;
        std::vector<fk::matrix<TestType, mem_type::view>> A_views_h;
        for (int k = 0; k < pde->num_dims; ++k)
        {
          int const start_row = degree * i;
          int const stop_row  = degree * (i + 1) - 1;
          int const start_col = 0;
          int const stop_col  = degree - 1;
          A_views.push_back(
              fk::matrix<TestType, mem_type::view, resource::device>(
                  A_mats[j * num_dims + k], start_row, stop_row, start_col,
                  stop_col));

          A_views_h.push_back(fk::matrix<TestType, mem_type::view>(
              A_mats_h[j * num_dims + k], start_row, stop_row, start_col,
              stop_col));
        }

        int const batch_offset = kron_index;
        kronmult_to_batch_sets(A_views, x_view, y_view, work_views, batches,
                               batch_offset, *pde);

        gold_view = (A_views_h[1].kron(A_views_h[0])) * x_h;
      }
    }

    for (int k = 0; k < pde->num_dims; ++k)
    {
      batch<TestType> const a = batches[k][0];
      batch<TestType> const b = batches[k][1];
      batch<TestType> const c = batches[k][2];
      TestType const alpha    = 1.0;
      TestType const beta     = 0.0;
      batched_gemm(a, b, c, alpha, beta);
    }

    fk::vector<TestType> const y_h(y_own.clone_onto_host());
    REQUIRE(y_h == gold);
  }

  SECTION("1 element, 3d, 3 terms")
  {
    int const degree    = 5;
    int const level     = 2;
    int const num_elems = 1;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);

    int const num_terms = 3;
    int const num_dims  = 3;
    // first, create example coefficient matrices in the pde
    int const dof = degree * std::pow(2, level);
    std::vector<fk::matrix<TestType>> A_mats_h;
    for (int i = 0; i < num_terms * num_dims; ++i)
    {
      A_mats_h.push_back(fk::matrix<TestType>(dof, dof));
    }

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto const gen = [&dist, &mersenne_engine]() {
      return dist(mersenne_engine);
    };

    // create different matrices for each term/dim pairing
    std::vector<fk::matrix<TestType, mem_type::owner, resource::device>> A_mats;
    for (fk::matrix<TestType> &mat : A_mats_h)
    {
      std::generate(mat.begin(), mat.end(), gen);
      A_mats.push_back(fk::matrix<TestType, mem_type::owner, resource::device>(
          mat.clone_onto_device()));
    }

    // create input vector
    int const x_size = static_cast<int>(std::pow(degree, pde->num_dims));
    fk::vector<TestType> x_h(x_size);
    std::generate(x_h.begin(), x_h.end(), gen);

    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // create intermediate workspaces
    // and output vectors
    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(
        x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner> gold(x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner, resource::device> work_own(
        x_size * num_elems * num_terms * std::min(num_dims - 1, 2));

    for (int i = 0; i < num_elems; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        // linearize index
        int const kron_index = pde->num_terms * i + j;

        // address y space
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * std::min(num_dims - 1, 2);
        fk::vector<TestType, mem_type::view, resource::device> y_view(
            y_own, y_index, y_index + x_size - 1);
        fk::vector<TestType, mem_type::view> gold_view(gold, y_index,
                                                       y_index + x_size - 1);

        // intermediate workspace
        std::vector<fk::vector<TestType, mem_type::view, resource::device>>
            work_views = {
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index, work_index + x_size - 1),
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index + x_size,
                    work_index + x_size * 2 - 1)};

        // create A_views
        std::vector<fk::matrix<TestType, mem_type::view>> A_views_h;
        std::vector<fk::matrix<TestType, mem_type::view, resource::device>>
            A_views;
        for (int k = 0; k < pde->num_dims; ++k)
        {
          int const start_row = degree * i;
          int const stop_row  = degree * (i + 1) - 1;
          int const start_col = 0;
          int const stop_col  = degree - 1;

          A_views.push_back(
              fk::matrix<TestType, mem_type::view, resource::device>(
                  A_mats[j * num_dims + k], start_row, stop_row, start_col,
                  stop_col));

          A_views_h.push_back(fk::matrix<TestType, mem_type::view>(
              A_mats_h[j * num_dims + k], start_row, stop_row, start_col,
              stop_col));
        }

        int const batch_offset = kron_index;
        kronmult_to_batch_sets(A_views, x_view, y_view, work_views, batches,
                               batch_offset, *pde);

        gold_view = (A_views_h[2].kron(A_views_h[1].kron(A_views_h[0]))) * x_h;
      }
    }

    for (int k = 0; k < pde->num_dims; ++k)
    {
      batch<TestType> const a = batches[k][0];
      batch<TestType> const b = batches[k][1];
      batch<TestType> const c = batches[k][2];
      TestType const alpha    = 1.0;
      TestType const beta     = 0.0;
      batched_gemm(a, b, c, alpha, beta);
    }

    // this method of computing "correctness" borrowed from ed's tests:
    //
    // https://code.ornl.gov/lmm/DG-SparseGrid/blob/reference/Kronmult/test1_batch.m
    fk::vector<TestType> const y_h(y_own.clone_onto_host());
    fk::vector<TestType> const diff = gold - y_h;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    TestType const tol = std::numeric_limits<TestType>::epsilon();
    REQUIRE(result <= tol * gold.size());
  }

  SECTION("3 elements, 6d, 6 terms")
  {
    int const degree    = 2;
    int const level     = 2;
    int const num_elems = 3;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    int const num_terms = 6;
    int const num_dims  = 6;
    // first, create example coefficient matrices in the pde
    int const dof = degree * std::pow(2, level);
    std::vector<fk::matrix<TestType>> A_mats_h;
    for (int i = 0; i < num_terms * num_dims; ++i)
    {
      A_mats_h.push_back(fk::matrix<TestType>(dof, dof));
    }

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto const gen = [&dist, &mersenne_engine]() {
      return dist(mersenne_engine);
    };

    // create different matrices for each term/dim pairing
    std::vector<fk::matrix<TestType, mem_type::owner, resource::device>> A_mats;
    for (fk::matrix<TestType> &mat : A_mats_h)
    {
      std::generate(mat.begin(), mat.end(), gen);
      A_mats.push_back(fk::matrix<TestType, mem_type::owner, resource::device>(
          mat.clone_onto_device()));
    }

    // create input vector
    int const x_size = static_cast<int>(std::pow(degree, pde->num_dims));
    fk::vector<TestType> x_h(x_size);
    std::generate(x_h.begin(), x_h.end(), gen);
    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // create intermediate workspaces
    // and output vectors
    fk::vector<TestType, mem_type::view, resource::device> x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(
        x_size * num_elems * num_terms);
    fk::vector<TestType> gold(x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner, resource::device> work_own(
        x_size * num_elems * num_terms * std::min(num_dims - 1, 2));

    for (int i = 0; i < num_elems; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        // linearize index
        int const kron_index = pde->num_terms * i + j;

        // address y space
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * std::min(num_dims - 1, 2);
        fk::vector<TestType, mem_type::view, resource::device> y_view(
            y_own, y_index, y_index + x_size - 1);
        fk::vector<TestType, mem_type::view> gold_view(gold, y_index,
                                                       y_index + x_size - 1);

        // intermediate workspace
        std::vector<fk::vector<TestType, mem_type::view, resource::device>>
            work_views = {
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index, work_index + x_size - 1),
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index + x_size,
                    work_index + x_size * 2 - 1)};

        // create A_views
        std::vector<fk::matrix<TestType, mem_type::view>> A_views_h;
        std::vector<fk::matrix<TestType, mem_type::view, resource::device>>
            A_views;
        for (int k = 0; k < pde->num_dims; ++k)
        {
          int const start_row = degree * i;
          int const stop_row  = degree * (i + 1) - 1;
          int const start_col = 0;
          int const stop_col  = degree - 1;

          A_views.push_back(
              fk::matrix<TestType, mem_type::view, resource::device>(
                  A_mats[j * num_dims + k], start_row, stop_row, start_col,
                  stop_col));

          A_views_h.push_back(fk::matrix<TestType, mem_type::view>(
              A_mats_h[j * num_dims + k], start_row, stop_row, start_col,
              stop_col));
        }

        int const batch_offset = kron_index;
        kronmult_to_batch_sets(A_views, x_view, y_view, work_views, batches,
                               batch_offset, *pde);

        gold_view = A_views_h[5].kron(A_views_h[4].kron(A_views_h[3].kron(
                        A_views_h[2].kron(A_views_h[1].kron(A_views_h[0]))))) *
                    x_h;
      }
    }

    for (int k = 0; k < pde->num_dims; ++k)
    {
      batch<TestType> const a = batches[k][0];
      batch<TestType> const b = batches[k][1];
      batch<TestType> const c = batches[k][2];
      TestType const alpha    = 1.0;
      TestType const beta     = 0.0;
      batched_gemm(a, b, c, alpha, beta);
    }

    // this method of computing "correctness" borrowed from ed's tests:
    //
    // https://
    // code.ornl.gov/lmm/DG-SparseGrid/blob/reference/Kronmult/test1_batch.m
    fk::vector<TestType> const y_h(y_own.clone_onto_host());
    fk::vector<TestType> const diff = gold - y_h;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    TestType const tol = std::numeric_limits<TestType>::epsilon();
    REQUIRE(result <= tol * gold.size());
  }
}

// FIXME - these functions shim safe kronmult batching calls into unsafe ones
// will be cleaned up when test refactor #193 merged in
template<typename P, typename T>
std::vector<P *> views_to_ptrs(std::vector<T> const &views)
{
  if (views.size() == 0)
  {
    return std::vector<P *>();
  }
  using R = typename std::remove_pointer<decltype(views[0].data())>::type;
  static_assert(std::is_same<P, R>::value,
                "view element type must match ptr type");
  std::vector<P *> ptrs;
  for (auto const &view : views)
  {
    ptrs.push_back(view.data());
  }
  return ptrs;
}
template<typename P>
void unsafe_batching(
    std::vector<fk::matrix<P, mem_type::view, resource::device>> const &A,
    fk::vector<P, mem_type::view, resource::device> const &x,
    fk::vector<P, mem_type::view, resource::device> const &y,
    std::vector<fk::vector<P, mem_type::view, resource::device>> const &work,
    std::vector<batch_operands_set<P>> &batches, int const batch_offset,
    PDE<P> const &pde)
{
  unsafe_kronmult_to_batch_sets(views_to_ptrs<P>(A), x.data(), y.data(),
                                views_to_ptrs<P>(work), batches, batch_offset,
                                pde);
}

TEMPLATE_TEST_CASE("kronmult batching - unsafe", "[batch]", float, double)
{
  SECTION("1 element, 1d, 1 term")
  {
    int const degree    = 4;
    int const level     = 2;
    int const num_elems = 1;

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    // clang-format off
    fk::matrix<TestType> const A {
        { 2,  3,  4,  5}, 
	{ 6,  7,  8,  9}, 
	{10, 11, 12, 13}, 
	{14, 15, 16, 17}};
    // clang-format on

    auto coeff = pde->get_coefficients(0, 0).clone_onto_host();
    coeff.set_submatrix(0, 0, A);
    fk::matrix<TestType, mem_type::owner, resource::device> const
        coefficient_matrix(coeff.clone_onto_device());

    fk::vector<TestType> const x_h{18, 19, 20, 21};
    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());
    fk::vector<TestType> const gold = A * x_h;

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    fk::matrix<TestType, mem_type::view, resource::device> const coeff_view(
        coefficient_matrix, 0, degree - 1, 0, degree - 1);
    std::vector<fk::matrix<TestType, mem_type::view, resource::device>> const
        As = {coeff_view};
    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(degree);
    fk::vector<TestType, mem_type::view, resource::device> y(y_own);
    std::vector<fk::vector<TestType, mem_type::view, resource::device>>
        work_set           = {};
    int const batch_offset = 0;

    unsafe_batching(As, x_view, y, work_set, batches, batch_offset, *pde);

    batch<TestType> const a = batches[0][0];
    batch<TestType> const b = batches[0][1];
    batch<TestType> const c = batches[0][2];

    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a, b, c, alpha, beta);

    fk::vector<TestType, mem_type::owner> const y_h(y.clone_onto_host());
    REQUIRE(gold == y_h);
  }

  SECTION("2 elements, 1d, 1 term")
  {
    int const degree    = 4;
    int const level     = 2;
    int const num_elems = 2;

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    // clang-format off
    fk::matrix<TestType> const A {
        { 2,  3,  4,  5,  6,  7}, 
	{ 8,  9, 10, 11, 12, 13}, 
	{14, 15, 16, 17, 18, 19}, 
	{20, 21, 22, 23, 24, 25}, 
	{26, 27, 28, 29, 30, 31}, 
	{32, 33, 34, 35, 36, 37}};
    // clang-format on

    auto coeff = pde->get_coefficients(0, 0).clone_onto_host();
    coeff.set_submatrix(0, 0, A);
    fk::matrix<TestType, mem_type::owner, resource::device> const
        coefficient_matrix(coeff.clone_onto_device());

    fk::vector<TestType> const x_h{18, 19, 20, 21};
    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());
    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // each element addresses a slightly different part of the underlying
    // coefficients
    fk::matrix<TestType, mem_type::view, resource::device> const A_view_e0(
        coefficient_matrix, 0, degree - 1, 0, degree - 1);
    fk::matrix<TestType, mem_type::view, resource::device> const A_view_e1(
        coefficient_matrix, 2, 2 + degree - 1, 2, 2 + degree - 1);

    fk::matrix<TestType, mem_type::view> const A_view_e0_h(A, 0, degree - 1, 0,
                                                           degree - 1);
    fk::matrix<TestType, mem_type::view> const A_view_e1_h(A, 2, 2 + degree - 1,
                                                           2, 2 + degree - 1);

    fk::vector<TestType> const gold_e0 = A_view_e0_h * x_h;
    fk::vector<TestType> const gold_e1 = A_view_e1_h * x_h;
    fk::vector<TestType> const gold    = gold_e0 + gold_e1;

    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(degree *
                                                                  num_elems);

    // schedule gemms for both elements
    int batch_offset = 0;
    std::vector<fk::matrix<TestType, mem_type::view, resource::device>> const
        As_e0 = {A_view_e0};
    fk::vector<TestType, mem_type::view, resource::device> y_e0(y_own, 0,
                                                                degree - 1);
    std::vector<fk::vector<TestType, mem_type::view, resource::device>>
        work_set_e0 = {};
    unsafe_batching(As_e0, x_view, y_e0, work_set_e0, batches, batch_offset,
                    *pde);

    batch_offset = 1;
    std::vector<fk::matrix<TestType, mem_type::view, resource::device>> const
        As_e1 = {A_view_e1};
    fk::vector<TestType, mem_type::view, resource::device> y_e1(
        y_own, degree, y_own.size() - 1);
    std::vector<fk::vector<TestType, mem_type::view, resource::device>>
        work_set_e1 = {};

    unsafe_batching(As_e1, x_view, y_e1, work_set_e1, batches, batch_offset,
                    *pde);

    batch<TestType> const a = batches[0][0];
    batch<TestType> const b = batches[0][1];
    batch<TestType> const c = batches[0][2];

    TestType const alpha = 1.0;
    TestType const beta  = 0.0;
    batched_gemm(a, b, c, alpha, beta);

    fk::vector<TestType> const y_e0_h(y_e0.clone_onto_host());
    fk::vector<TestType> const y_e1_h(y_e1.clone_onto_host());

    REQUIRE(gold_e0 == y_e0_h);
    REQUIRE(gold_e1 == y_e1_h);
    REQUIRE(gold == (y_e0_h + y_e1_h));
  }

  SECTION("2 elements, 2d, 2 terms")
  {
    int const degree    = 5;
    int const level     = 2;
    int const num_elems = 2;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    int const num_terms = 2;
    int const num_dims  = 2;
    // first, create example coefficient matrices in the pde
    int const dof = degree * std::pow(2, level);
    std::array<fk::matrix<TestType>, num_terms *num_dims> A_mats_h = {
        fk::matrix<TestType>(dof, dof), fk::matrix<TestType>(dof, dof),
        fk::matrix<TestType>(dof, dof), fk::matrix<TestType>(dof, dof)};

    // create different matrices for each term/dim pairing
    int start = 1;

    std::vector<fk::matrix<TestType, mem_type::owner, resource::device>> A_mats;
    for (fk::matrix<TestType> &mat : A_mats_h)
    {
      std::iota(mat.begin(), mat.end(), start);
      start += dof;
      A_mats.push_back(fk::matrix<TestType, mem_type::owner, resource::device>(
          mat.clone_onto_device()));
    }

    // create input vector
    int const x_size = static_cast<int>(std::pow(degree, pde->num_dims));
    fk::vector<TestType> x_h(x_size);
    std::iota(x_h.begin(), x_h.end(), 1);
    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // create intermediate workspaces
    // and output vectors
    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(
        x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner> gold(x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner, resource::device> work_own(
        x_size * num_elems * num_terms * std::min(num_dims - 1, 2));

    for (int i = 0; i < num_elems; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        // linearize index
        int const kron_index = i * num_terms + j;

        // address y space
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * std::min(num_dims - 1, 2);
        fk::vector<TestType, mem_type::view, resource::device> y_view(
            y_own, y_index, y_index + x_size - 1);
        fk::vector<TestType, mem_type::view> gold_view(gold, y_index,
                                                       y_index + x_size - 1);

        // intermediate workspace
        std::vector<fk::vector<TestType, mem_type::view, resource::device>>
            work_views = {
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index, work_index + x_size - 1)};

        // create A_views
        std::vector<fk::matrix<TestType, mem_type::view, resource::device>>
            A_views;
        std::vector<fk::matrix<TestType, mem_type::view>> A_views_h;
        for (int k = 0; k < pde->num_dims; ++k)
        {
          int const start_row = degree * i;
          int const stop_row  = degree * (i + 1) - 1;
          int const start_col = 0;
          int const stop_col  = degree - 1;
          A_views.push_back(
              fk::matrix<TestType, mem_type::view, resource::device>(
                  A_mats[j * num_dims + k], start_row, stop_row, start_col,
                  stop_col));

          A_views_h.push_back(fk::matrix<TestType, mem_type::view>(
              A_mats_h[j * num_dims + k], start_row, stop_row, start_col,
              stop_col));
        }

        int const batch_offset = kron_index;
        unsafe_batching(A_views, x_view, y_view, work_views, batches,
                        batch_offset, *pde);

        gold_view = (A_views_h[1].kron(A_views_h[0])) * x_h;
      }
    }

    for (int k = 0; k < pde->num_dims; ++k)
    {
      batch<TestType> const a = batches[k][0];
      batch<TestType> const b = batches[k][1];
      batch<TestType> const c = batches[k][2];
      TestType const alpha    = 1.0;
      TestType const beta     = 0.0;
      batched_gemm(a, b, c, alpha, beta);
    }

    fk::vector<TestType> const y_h(y_own.clone_onto_host());
    REQUIRE(y_h == gold);
  }

  SECTION("1 element, 3d, 3 terms")
  {
    int const degree    = 5;
    int const level     = 2;
    int const num_elems = 1;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);

    int const num_terms = 3;
    int const num_dims  = 3;
    // first, create example coefficient matrices in the pde
    int const dof = degree * std::pow(2, level);
    std::vector<fk::matrix<TestType>> A_mats_h;
    for (int i = 0; i < num_terms * num_dims; ++i)
    {
      A_mats_h.push_back(fk::matrix<TestType>(dof, dof));
    }

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto const gen = [&dist, &mersenne_engine]() {
      return dist(mersenne_engine);
    };

    // create different matrices for each term/dim pairing
    std::vector<fk::matrix<TestType, mem_type::owner, resource::device>> A_mats;
    for (fk::matrix<TestType> &mat : A_mats_h)
    {
      std::generate(mat.begin(), mat.end(), gen);
      A_mats.push_back(fk::matrix<TestType, mem_type::owner, resource::device>(
          mat.clone_onto_device()));
    }

    // create input vector
    int const x_size = static_cast<int>(std::pow(degree, pde->num_dims));
    fk::vector<TestType> x_h(x_size);
    std::generate(x_h.begin(), x_h.end(), gen);

    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // create intermediate workspaces
    // and output vectors
    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(
        x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner> gold(x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner, resource::device> work_own(
        x_size * num_elems * num_terms * std::min(num_dims - 1, 2));

    for (int i = 0; i < num_elems; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        // linearize index
        int const kron_index = pde->num_terms * i + j;

        // address y space
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * std::min(num_dims - 1, 2);
        fk::vector<TestType, mem_type::view, resource::device> y_view(
            y_own, y_index, y_index + x_size - 1);
        fk::vector<TestType, mem_type::view> gold_view(gold, y_index,
                                                       y_index + x_size - 1);

        // intermediate workspace
        std::vector<fk::vector<TestType, mem_type::view, resource::device>>
            work_views = {
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index, work_index + x_size - 1),
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index + x_size,
                    work_index + x_size * 2 - 1)};

        // create A_views
        std::vector<fk::matrix<TestType, mem_type::view>> A_views_h;
        std::vector<fk::matrix<TestType, mem_type::view, resource::device>>
            A_views;
        for (int k = 0; k < pde->num_dims; ++k)
        {
          int const start_row = degree * i;
          int const stop_row  = degree * (i + 1) - 1;
          int const start_col = 0;
          int const stop_col  = degree - 1;

          A_views.push_back(
              fk::matrix<TestType, mem_type::view, resource::device>(
                  A_mats[j * num_dims + k], start_row, stop_row, start_col,
                  stop_col));

          A_views_h.push_back(fk::matrix<TestType, mem_type::view>(
              A_mats_h[j * num_dims + k], start_row, stop_row, start_col,
              stop_col));
        }

        int const batch_offset = kron_index;
        unsafe_batching(A_views, x_view, y_view, work_views, batches,
                        batch_offset, *pde);

        gold_view = (A_views_h[2].kron(A_views_h[1].kron(A_views_h[0]))) * x_h;
      }
    }

    for (int k = 0; k < pde->num_dims; ++k)
    {
      batch<TestType> const a = batches[k][0];
      batch<TestType> const b = batches[k][1];
      batch<TestType> const c = batches[k][2];
      TestType const alpha    = 1.0;
      TestType const beta     = 0.0;
      batched_gemm(a, b, c, alpha, beta);
    }

    // this method of computing "correctness" borrowed from ed's tests:
    //
    // https://code.ornl.gov/lmm/DG-SparseGrid/blob/reference/Kronmult/test1_batch.m
    fk::vector<TestType> const y_h(y_own.clone_onto_host());
    fk::vector<TestType> const diff = gold - y_h;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    TestType const tol = std::numeric_limits<TestType>::epsilon();
    REQUIRE(result <= tol * gold.size());
  }

  SECTION("3 elements, 6d, 6 terms")
  {
    int const degree    = 2;
    int const level     = 2;
    int const num_elems = 3;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    int const num_terms = 6;
    int const num_dims  = 6;
    // first, create example coefficient matrices in the pde
    int const dof = degree * std::pow(2, level);
    std::vector<fk::matrix<TestType>> A_mats_h;
    for (int i = 0; i < num_terms * num_dims; ++i)
    {
      A_mats_h.push_back(fk::matrix<TestType>(dof, dof));
    }

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto const gen = [&dist, &mersenne_engine]() {
      return dist(mersenne_engine);
    };

    // create different matrices for each term/dim pairing
    std::vector<fk::matrix<TestType, mem_type::owner, resource::device>> A_mats;
    for (fk::matrix<TestType> &mat : A_mats_h)
    {
      std::generate(mat.begin(), mat.end(), gen);
      A_mats.push_back(fk::matrix<TestType, mem_type::owner, resource::device>(
          mat.clone_onto_device()));
    }

    // create input vector
    int const x_size = static_cast<int>(std::pow(degree, pde->num_dims));
    fk::vector<TestType> x_h(x_size);
    std::generate(x_h.begin(), x_h.end(), gen);
    fk::vector<TestType, mem_type::owner, resource::device> const x(
        x_h.clone_onto_device());

    std::vector<batch_operands_set<TestType>> batches =
        allocate_batches(*pde, num_elems);

    // create intermediate workspaces
    // and output vectors
    fk::vector<TestType, mem_type::view, resource::device> x_view(x);
    fk::vector<TestType, mem_type::owner, resource::device> y_own(
        x_size * num_elems * num_terms);
    fk::vector<TestType> gold(x_size * num_elems * num_terms);
    fk::vector<TestType, mem_type::owner, resource::device> work_own(
        x_size * num_elems * num_terms * std::min(num_dims - 1, 2));

    for (int i = 0; i < num_elems; ++i)
    {
      for (int j = 0; j < pde->num_terms; ++j)
      {
        // linearize index
        int const kron_index = pde->num_terms * i + j;

        // address y space
        int const y_index    = x_size * kron_index;
        int const work_index = x_size * kron_index * std::min(num_dims - 1, 2);
        fk::vector<TestType, mem_type::view, resource::device> y_view(
            y_own, y_index, y_index + x_size - 1);
        fk::vector<TestType, mem_type::view> gold_view(gold, y_index,
                                                       y_index + x_size - 1);

        // intermediate workspace
        std::vector<fk::vector<TestType, mem_type::view, resource::device>>
            work_views = {
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index, work_index + x_size - 1),
                fk::vector<TestType, mem_type::view, resource::device>(
                    work_own, work_index + x_size,
                    work_index + x_size * 2 - 1)};

        // create A_views
        std::vector<fk::matrix<TestType, mem_type::view>> A_views_h;
        std::vector<fk::matrix<TestType, mem_type::view, resource::device>>
            A_views;
        for (int k = 0; k < pde->num_dims; ++k)
        {
          int const start_row = degree * i;
          int const stop_row  = degree * (i + 1) - 1;
          int const start_col = 0;
          int const stop_col  = degree - 1;

          A_views.push_back(
              fk::matrix<TestType, mem_type::view, resource::device>(
                  A_mats[j * num_dims + k], start_row, stop_row, start_col,
                  stop_col));

          A_views_h.push_back(fk::matrix<TestType, mem_type::view>(
              A_mats_h[j * num_dims + k], start_row, stop_row, start_col,
              stop_col));
        }

        int const batch_offset = kron_index;
        unsafe_batching(A_views, x_view, y_view, work_views, batches,
                        batch_offset, *pde);

        gold_view = A_views_h[5].kron(A_views_h[4].kron(A_views_h[3].kron(
                        A_views_h[2].kron(A_views_h[1].kron(A_views_h[0]))))) *
                    x_h;
      }
    }

    for (int k = 0; k < pde->num_dims; ++k)
    {
      batch<TestType> const a = batches[k][0];
      batch<TestType> const b = batches[k][1];
      batch<TestType> const c = batches[k][2];
      TestType const alpha    = 1.0;
      TestType const beta     = 0.0;
      batched_gemm(a, b, c, alpha, beta);
    }

    // this method of computing "correctness" borrowed from ed's tests:
    //
    // https://
    // code.ornl.gov/lmm/DG-SparseGrid/blob/reference/Kronmult/test1_batch.m
    fk::vector<TestType> const y_h(y_own.clone_onto_host());
    fk::vector<TestType> const diff = gold - y_h;
    auto abs_compare                = [](TestType const a, TestType const b) {
      return (std::abs(a) < std::abs(b));
    };
    TestType const result =
        std::abs(*std::max_element(diff.begin(), diff.end(), abs_compare));
    TestType const tol = std::numeric_limits<TestType>::epsilon();
    REQUIRE(result <= tol * gold.size());
  }
}

template<typename P>
void batch_builder_test(int const degree, int const level, PDE<P> &pde,
                        std::string const &gold_path = {},
                        bool const full_grid         = false)
{
  std::string const grid_str          = full_grid ? "-f" : "";
  std::vector<std::string> const args = {"-l", std::to_string(level), "-d",
                                         std::to_string(degree), grid_str};
  options const o                     = make_options(args);

  element_table const elem_table(o, pde.num_dims);
  int const num_ranks = 1;
  int const my_rank   = 0;
  auto const plan     = get_plan(num_ranks, elem_table);
  auto const subgrid  = plan.at(my_rank);

  generate_all_coefficients(pde);

  host_workspace<P> host_space(pde, subgrid);
  std::fill(host_space.x.begin(), host_space.x.end(), 1.0);

  fk::vector<P> const gold = [&pde, &host_space, &gold_path]() {
    if (pde.num_terms == 1 && pde.num_dims == 1)
    {
      fk::matrix<P> const &coefficient_matrix =
          pde.get_coefficients(0, 0).clone_onto_host();
      return coefficient_matrix * host_space.x;
    }
    return fk::vector<P>(read_vector_from_txt_file(gold_path));
  }();

  auto const chunks = assign_elements(subgrid, get_num_chunks(subgrid, pde));
  rank_workspace<P> rank_space(pde, chunks);

  auto const num_elems = elem_table.size() * elem_table.size();
  auto batches         = allocate_batches(pde, num_elems);
  fm::scal(static_cast<P>(0.0), host_space.fx);

  for (auto const &chunk : chunks)
  {
    // copy in inputs
    copy_chunk_inputs(pde, subgrid, rank_space, host_space, chunk);

    // build batches for this chunk
    build_batches(pde, elem_table, rank_space, chunk, batches);

    // do the gemms
    P const alpha = 1.0;
    P const beta  = 0.0;
    for (int i = 0; i < pde.num_dims; ++i)
    {
      batch<P> const &a = batches[i][0];
      batch<P> const &b = batches[i][1];
      batch<P> const &c = batches[i][2];

      batched_gemm(a, b, c, alpha, beta);
    }

    // do the reduction
    reduce_chunk(pde, rank_space, chunk);

    // copy outputs back
    copy_chunk_outputs(pde, subgrid, rank_space, host_space, chunk);
  }

  // determined emprically 11/19
  auto const tol_scale = 1e4;
  relaxed_comparison(gold, host_space.fx, tol_scale);
}

TEMPLATE_TEST_CASE("batch builder", "[batch]", float, double)
{
  SECTION("1d, 1 term, degree 2, level 2")
  {
    int const degree = 2;
    int const level  = 2;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    batch_builder_test(degree, level, *pde);
  }
  SECTION("1d, 1 term, degree 4, level 3")
  {
    int const degree = 4;
    int const level  = 3;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    batch_builder_test(degree, level, *pde);
  }

  SECTION("2d, 2 terms, level 2, degree 2")
  {
    int const degree = 2;
    int const level  = 2;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    std::string const gold_path =
        "../testing/generated-inputs/batch/continuity2_sg_l2_d2_t1.dat";
    batch_builder_test(degree, level, *pde, gold_path);
  }

  SECTION("2d, 2 terms, level 3, degree 4, full grid")
  {
    int const degree = 4;
    int const level  = 3;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    std::string const gold_path =
        "../testing/generated-inputs/batch/continuity2_fg_l3_d4_t1.dat";
    bool const full_grid = true;
    batch_builder_test(degree, level, *pde, gold_path, full_grid);
  }

  SECTION("3d, 3 terms, level 3, degree 4, sparse grid")
  {
    int const degree = 4;
    int const level  = 3;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    std::string const gold_path =
        "../testing/generated-inputs/batch/continuity3_sg_l3_d4_t1.dat";
    batch_builder_test(degree, level, *pde, gold_path);
  }

  SECTION("6d, 6 terms, level 2, degree 3, sparse grid")
  {
    int const degree = 3;
    int const level  = 2;
    auto pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    std::string const gold_path =
        "../testing/generated-inputs/batch/continuity6_sg_l2_d3_t1.dat";
    batch_builder_test(degree, level, *pde, gold_path);
  }
}
