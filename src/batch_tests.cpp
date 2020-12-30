#include "batch.hpp"
#include "coefficients.hpp"
#include "fast_math.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"
#include "tools.hpp"
#include <numeric>
#include <random>

// FIXME name etc.
template<typename P, resource resrc>
void test_kron(
    std::vector<fk::matrix<P, mem_type::const_view, resrc>> const &matrices,
    fk::vector<P, mem_type::owner, resrc> const &x,
    fk::vector<P, mem_type::owner, resource::host> const &correct)
{
  fk::vector<P, mem_type::const_view, resrc> const x_view(x);

  int const workspace_len = calculate_workspace_length(matrices, x.size());

  fk::vector<P, mem_type::owner, resrc> workspace_0(workspace_len);
  fk::vector<P, mem_type::owner, resrc> workspace_1(workspace_len);
  std::array<fk::vector<P, mem_type::view, resrc>, 2> workspace = {
      fk::vector<P, mem_type::view, resrc>(workspace_0),
      fk::vector<P, mem_type::view, resrc>(workspace_1)};

  fk::vector<P, mem_type::owner, resrc> real_space_owner(correct.size());
  fk::vector<P, mem_type::view, resrc> real_space(real_space_owner);

  batch_chain<P, resrc> chain(matrices, x_view, workspace, real_space);
  chain.execute();

  if constexpr (resrc == resource::device)
  {
    REQUIRE(real_space.clone_onto_host() == correct);
    return;
  }

  else if constexpr (resrc == resource::host)
  {
    REQUIRE(real_space == correct);
    return;
  }
}

TEMPLATE_TEST_CASE("kron", "[kron]", double, float)
{
  SECTION("calculate_workspace_size")
  {
    fk::matrix<TestType> const a(5, 10);
    fk::matrix<TestType> const b(5, 10);
    fk::matrix<TestType> const c(5, 10);
    fk::matrix<TestType> const e(10, 5);
    fk::matrix<TestType> const f(10, 5);

    std::vector<fk::matrix<TestType, mem_type::const_view>> const matrices = {
        fk::matrix<TestType, mem_type::const_view>(a),
        fk::matrix<TestType, mem_type::const_view>(b),
        fk::matrix<TestType, mem_type::const_view>(c),
        fk::matrix<TestType, mem_type::const_view>(e),
        fk::matrix<TestType, mem_type::const_view>(f)};

    int const x_size = std::accumulate(
        matrices.begin(), matrices.end(), 1,
        [](int const i, fk::matrix<TestType, mem_type::const_view> const &m) {
          return i * m.ncols();
        });
    int const correct_size = 1e5;

    REQUIRE(calculate_workspace_length(matrices, x_size) == correct_size);
  }

  SECTION("kron_0_device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> const a = {{2, 3},
                                                                       {4, 5}};

    fk::matrix<TestType, mem_type::owner, resource::device> const b = {{6, 7},
                                                                       {8, 9}};

    std::vector<fk::matrix<TestType, mem_type::const_view,
                           resource::device>> const matrices = {
        fk::matrix<TestType, mem_type::const_view, resource::device>(a),
        fk::matrix<TestType, mem_type::const_view, resource::device>(b)};

    fk::vector<TestType, mem_type::owner, resource::device> const x = {10, 11,
                                                                       12, 13};

    fk::vector<TestType, mem_type::owner, resource::host> const correct = {
        763, 997, 1363, 1781};

    test_kron<TestType, resource::device>(matrices, x, correct);
  }

  SECTION("kron_0_host")
  {
    fk::matrix<TestType, mem_type::owner, resource::host> const a = {{2, 3},
                                                                     {4, 5}};

    fk::matrix<TestType, mem_type::owner, resource::host> const b = {{6, 7},
                                                                     {8, 9}};

    std::vector<fk::matrix<TestType, mem_type::const_view,
                           resource::host>> const matrices = {
        fk::matrix<TestType, mem_type::const_view, resource::host>(a),
        fk::matrix<TestType, mem_type::const_view, resource::host>(b)};

    fk::vector<TestType, mem_type::owner, resource::host> const x = {10, 11, 12,
                                                                     13};

    fk::vector<TestType, mem_type::owner, resource::host> const correct = {
        763, 997, 1363, 1781};

    test_kron<TestType, resource::host>(matrices, x, correct);
  }

  SECTION("kron_1_device")
  {
    auto const matrix_all_twos = [](int const rows, int const cols)
        -> fk::matrix<TestType, mem_type::owner, resource::device> {
      fk::matrix<TestType, mem_type::owner, resource::host> m(rows, cols);

      for (auto &element : m)
        element = 2;

      return m.clone_onto_device();
    };

    auto const m0 = matrix_all_twos(3, 4);
    auto const m1 = matrix_all_twos(5, 3);
    auto const m2 = matrix_all_twos(2, 7);
    auto const m3 = matrix_all_twos(9, 6);
    auto const m4 = matrix_all_twos(12, 3);
    auto const m5 = matrix_all_twos(4, 14);
    auto const m6 = matrix_all_twos(10, 3);
    auto const m7 = matrix_all_twos(6, 5);
    fk::matrix<TestType, mem_type::owner, resource::device> const m8 = {{3, 3},
                                                                        {3, 3}};

    std::vector<fk::matrix<TestType, mem_type::const_view,
                           resource::device>> const matrices = {
        fk::matrix<TestType, mem_type::const_view, resource::device>(m0),
        fk::matrix<TestType, mem_type::const_view, resource::device>(m1),
        fk::matrix<TestType, mem_type::const_view, resource::device>(m2),
        fk::matrix<TestType, mem_type::const_view, resource::device>(m3),
        fk::matrix<TestType, mem_type::const_view, resource::device>(m4),
        fk::matrix<TestType, mem_type::const_view, resource::device>(m5),
        fk::matrix<TestType, mem_type::const_view, resource::device>(m6),
        fk::matrix<TestType, mem_type::const_view, resource::device>(m7),
        fk::matrix<TestType, mem_type::const_view, resource::device>(m8)};

    int const x_size = std::accumulate(
        matrices.begin(), matrices.end(), 1,
        [](int const i,
           fk::matrix<TestType, mem_type::const_view, resource::device> const
               &m) { return i * m.ncols(); });

    fk::vector<TestType, mem_type::owner, resource::host> const x(
        std::vector<TestType>(x_size, 1));

    fk::vector<TestType, mem_type::owner, resource::device> const x_device =
        x.clone_onto_device();

    int const y_size = std::accumulate(
        matrices.begin(), matrices.end(), 1,
        [](int const i,
           fk::matrix<TestType, mem_type::const_view, resource::device> const
               &m) { return i * m.nrows(); });

    fk::vector<TestType> const correct(std::vector<TestType>(
        y_size, x_size * std::pow(2, matrices.size() - 1) * 3));

    test_kron<TestType, resource::device>(matrices, x_device, correct);
  }

  SECTION("kron_1_host")
  {
    auto const matrix_all_twos = [](int const rows, int const cols)
        -> fk::matrix<TestType, mem_type::owner, resource::host> {
      fk::matrix<TestType, mem_type::owner, resource::host> m(rows, cols);
      for (auto &element : m)
        element = 2;
      return m;
    };

    auto const m0 = matrix_all_twos(3, 4);
    auto const m1 = matrix_all_twos(5, 3);
    auto const m2 = matrix_all_twos(2, 7);
    auto const m3 = matrix_all_twos(9, 6);
    auto const m4 = matrix_all_twos(12, 3);
    auto const m5 = matrix_all_twos(4, 14);
    auto const m6 = matrix_all_twos(10, 3);
    auto const m7 = matrix_all_twos(6, 5);
    fk::matrix<TestType, mem_type::owner, resource::host> const m8 = {{3, 3},
                                                                      {3, 3}};

    std::vector<fk::matrix<TestType, mem_type::const_view,
                           resource::host>> const matrices = {
        fk::matrix<TestType, mem_type::const_view, resource::host>(m0),
        fk::matrix<TestType, mem_type::const_view, resource::host>(m1),
        fk::matrix<TestType, mem_type::const_view, resource::host>(m2),
        fk::matrix<TestType, mem_type::const_view, resource::host>(m3),
        fk::matrix<TestType, mem_type::const_view, resource::host>(m4),
        fk::matrix<TestType, mem_type::const_view, resource::host>(m5),
        fk::matrix<TestType, mem_type::const_view, resource::host>(m6),
        fk::matrix<TestType, mem_type::const_view, resource::host>(m7),
        fk::matrix<TestType, mem_type::const_view, resource::host>(m8)};

    int const x_size = std::accumulate(
        matrices.begin(), matrices.end(), 1,
        [](int const i,
           fk::matrix<TestType, mem_type::const_view, resource::host> const
               &m) { return i * m.ncols(); });

    fk::vector<TestType, mem_type::owner, resource::host> const x(
        std::vector<TestType>(x_size, 1));

    int const y_size = std::accumulate(
        matrices.begin(), matrices.end(), 1,
        [](int const i,
           fk::matrix<TestType, mem_type::const_view, resource::host> const
               &m) { return i * m.nrows(); });

    // because the kron matrices consist of all 2s except for the last one,
    //   every value of the output matrix will be the equal:
    fk::vector<TestType> const correct(std::vector<TestType>(
        y_size, x_size * (1 << (matrices.size() - 1)) * 3));

    test_kron<TestType, resource::host>(matrices, x, correct);
  }

  SECTION("kron_2_device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> const a = {
        {1}, {2}, {3}};

    fk::matrix<TestType, mem_type::owner, resource::device> const b = {
        {3, 4, 5}};

    fk::matrix<TestType, mem_type::owner, resource::device> const c = {
        {5}, {6}, {7}};

    fk::matrix<TestType, mem_type::owner, resource::device> const d = {
        {7, 8, 9}};

    std::vector<fk::matrix<TestType, mem_type::const_view,
                           resource::device>> const matrices = {
        fk::matrix<TestType, mem_type::const_view, resource::device>(a),
        fk::matrix<TestType, mem_type::const_view, resource::device>(b),
        fk::matrix<TestType, mem_type::const_view, resource::device>(c),
        fk::matrix<TestType, mem_type::const_view, resource::device>(d)};

    fk::vector<TestType, mem_type::owner, resource::device> const x = {
        1, 1, 1, 1, 1, 1, 1, 1, 1};

    fk::vector<TestType, mem_type::owner, resource::host> const correct = {
        1440, 1728, 2016, 2880, 3456, 4032, 4320, 5184, 6048};

    test_kron<TestType, resource::device>(matrices, x, correct);
  }

  SECTION("kron_2_host")
  {
    fk::matrix<TestType, mem_type::owner, resource::host> const a = {
        {1}, {2}, {3}};

    fk::matrix<TestType, mem_type::owner, resource::host> const b = {{3, 4, 5}};

    fk::matrix<TestType, mem_type::owner, resource::host> const c = {
        {5}, {6}, {7}};

    fk::matrix<TestType, mem_type::owner, resource::host> const d = {{7, 8, 9}};

    std::vector<fk::matrix<TestType, mem_type::const_view,
                           resource::host>> const matrices = {
        fk::matrix<TestType, mem_type::const_view, resource::host>(a),
        fk::matrix<TestType, mem_type::const_view, resource::host>(b),
        fk::matrix<TestType, mem_type::const_view, resource::host>(c),
        fk::matrix<TestType, mem_type::const_view, resource::host>(d)};

    fk::vector<TestType, mem_type::owner, resource::host> const x = {
        1, 1, 1, 1, 1, 1, 1, 1, 1};

    fk::vector<TestType, mem_type::owner, resource::host> const correct = {
        1440, 1728, 2016, 2880, 3456, 4032, 4320, 5184, 6048};

    test_kron<TestType, resource::host>(matrices, x, correct);
  }
}

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

  fk::matrix<TestType, mem_type::const_view, resrc> const first_v(
      first, start_row, stop_row, start_col, stop_col);
  fk::matrix<TestType, mem_type::const_view, resrc> const second_v(
      second, start_row, stop_row, start_col, stop_col);
  fk::matrix<TestType, mem_type::const_view, resrc> const third_v(
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
  tools::expect(m > 0);
  tools::expect(n > 0);
  tools::expect(k > 0);

  int const rows_a = trans_a ? k : m;
  int const cols_a = trans_a ? m : k;
  tools::expect(lda >= rows_a);
  tools::expect(ldc >= m);

  int const rows_b = trans_b ? n : k;
  int const cols_b = trans_b ? k : n;
  tools::expect(ldb >= rows_b);

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
          builder.assign_entry(fk::matrix<P, mem_type::const_view, resrc>(
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
    return fk::matrix<P, mem_type::const_view>(c, 0, m - 1, 0, n - 1);
  };

  P const tol_factor = std::is_same<P, double>::value ? 1e-15 : 1e-6;

  for (int i = 0; i < num_batch; ++i)
  {
    if constexpr (resrc == resource::host)
    {
      rmse_comparison(effect_c(matrices[2][i]), effect_c(matrices[3][i]),
                      tol_factor);
    }

    else
    {
      rmse_comparison(effect_c(matrices[2][i].clone_onto_host()),
                      effect_c(matrices[3][i].clone_onto_host()), tol_factor);
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
  tools::expect(m > 0);
  tools::expect(n > 0);
  tools::expect(lda >= m);

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

          fk::matrix<P, mem_type::const_view, resrc> const effective_a(
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
          builder.assign_entry(fk::matrix<P, mem_type::const_view, resrc>(
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
              fk::matrix<P, mem_type::const_view, resrc>(vects[i], nrows, 1),
              i);
        }
        return builder;
      };

  auto const a_batch = batch_build(a_mats, m, n, lda, trans_a);
  auto const x_batch = batch_build_v(vectors[0], cols_a);
  auto const y_batch = batch_build_v(vectors[1], rows_a);

  batched_gemv(a_batch, x_batch, y_batch, alpha, beta);

  P const tol_factor = 1e-17;
  for (int i = 0; i < num_batch; ++i)
  {
    if constexpr (resrc == resource::host)
    {
      rmse_comparison(vectors[1][i], vectors[2][i], tol_factor);
    }

    else
    {
      rmse_comparison(vectors[1][i].clone_onto_host(),
                      vectors[2][i].clone_onto_host(), tol_factor);
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
