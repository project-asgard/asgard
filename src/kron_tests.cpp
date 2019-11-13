#include "kron.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("kron", "[kron]", double, float)
{
  SECTION("calculate_workspace_size")
  {
    fk::matrix<TestType> const a(5, 10);
    fk::matrix<TestType> const b(5, 10);
    fk::matrix<TestType> const c(5, 10);
    fk::matrix<TestType> const e(10, 5);
    fk::matrix<TestType> const f(10, 5);

    std::vector<fk::matrix<TestType, mem_type::view>> const matrix = {
        fk::matrix<TestType, mem_type::view>(a),
        fk::matrix<TestType, mem_type::view>(b),
        fk::matrix<TestType, mem_type::view>(c),
        fk::matrix<TestType, mem_type::view>(e),
        fk::matrix<TestType, mem_type::view>(f)};

    int const x_size =
        std::accumulate(matrix.begin(), matrix.end(), 1,
                        [](int const i, fk::matrix<TestType, mem_type::view> const &m) {
                          return i * m.ncols();
                        });
    int const correct_size = 1e5;

    REQUIRE(calculate_workspace_len(matrix, x_size) == correct_size);
  }

  SECTION("kron_0_device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> const a = {{2, 3},
                                                                 {4, 5}};

    fk::matrix<TestType, mem_type::owner, resource::device> const b = {{6, 7},
                                                                 {8, 9}};

    std::vector<fk::matrix<TestType, mem_type::view, resource::device>>
    const matrix = {fk::matrix<TestType, mem_type::view, resource::device>(a),
                  fk::matrix<TestType, mem_type::view, resource::device>(b)};

    fk::vector<TestType, mem_type::owner, resource::device> const x = {10, 11,
                                                                       12, 13};

    fk::vector<TestType, mem_type::owner, resource::host> const correct = {
        763, 997, 1363, 1781};

    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);

    batch_job<TestType, resource::device> job = kron_batch(matrix, x_view);

    fk::vector<TestType, mem_type::owner, resource::device> const r =
        execute_batch_job(job);

    REQUIRE(r.clone_onto_host() == correct);
  }

  SECTION("kron_0_host")
  {
    fk::matrix<TestType, mem_type::owner, resource::host > const a = {{2, 3},
                                                                 {4, 5}};

    fk::matrix<TestType, mem_type::owner, resource::host > const b = {{6, 7},
                                                                 {8, 9}};

    std::vector<fk::matrix<TestType, mem_type::view, resource::host >>
    const matrix = {fk::matrix<TestType, mem_type::view, resource::host >(a),
                  fk::matrix<TestType, mem_type::view, resource::host >(b)};

    fk::vector<TestType, mem_type::owner, resource::host> const x = {10, 11,
                                                                       12, 13};

    fk::vector<TestType, mem_type::owner, resource::host> const correct = {
        763, 997, 1363, 1781};

    fk::vector<TestType, mem_type::view, resource::host> const x_view(x);

    batch_job<TestType, resource::host> job = kron_batch(matrix, x_view);

    fk::vector<TestType, mem_type::owner, resource::host> const r =
        execute_batch_job(job);

    REQUIRE(r == correct);
  }

  SECTION("kron_1_device")
  {
    auto matrix_all_twos = [](int const rows, int const cols)
        -> fk::matrix<TestType, mem_type::owner, resource::device> {

      fk::matrix<TestType, mem_type::owner, resource::host> m(rows, cols);

      for( auto &element : m ) element = 2;

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

    std::vector<fk::matrix<TestType, mem_type::view, resource::device>> const matrix =
        {fk::matrix<TestType, mem_type::view, resource::device>(m0),
         fk::matrix<TestType, mem_type::view, resource::device>(m1),
         fk::matrix<TestType, mem_type::view, resource::device>(m2),
         fk::matrix<TestType, mem_type::view, resource::device>(m3),
         fk::matrix<TestType, mem_type::view, resource::device>(m4),
         fk::matrix<TestType, mem_type::view, resource::device>(m5),
         fk::matrix<TestType, mem_type::view, resource::device>(m6),
         fk::matrix<TestType, mem_type::view, resource::device>(m7),
         fk::matrix<TestType, mem_type::view, resource::device>(m8)};

    int x_size = std::accumulate(
        matrix.begin(), matrix.end(), 1,
        [](int i, fk::matrix<TestType, mem_type::view, resource::device> const &m) {
          return i * m.ncols();
        });

    fk::vector<TestType, mem_type::owner, resource::host> const x(
        std::vector<TestType>(x_size, 1));

    int y_size = std::accumulate(
        matrix.begin(), matrix.end(), 1,
        [](int i, fk::matrix<TestType, mem_type::view, resource::device> const &m) {
          return i * m.nrows();
        });

    fk::vector<TestType> const correct(
        std::vector<TestType>(y_size, x_size * (1 << (matrix.size() - 1)) * 3));

    batch_job<TestType, resource::device> job = kron_batch(
        matrix, fk::vector<TestType, mem_type::view, resource::device>(
                    x.clone_onto_device()));

    fk::vector<TestType, mem_type::owner, resource::device> const r =
        execute_batch_job(job);

    REQUIRE(r.clone_onto_host() == correct);
  }

  SECTION("kron_1_host")
  {
    auto matrix_all_twos = [](int const rows, int const cols)
        -> fk::matrix<TestType, mem_type::owner, resource::host> {

      fk::matrix<TestType, mem_type::owner, resource::host> m(rows, cols);

      for (int i = 0; i < rows; ++i)
      {
        for (int j = 0; j < cols; ++j)
        {
          m(i, j) = 2;
        }
      }

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

    std::vector<fk::matrix<TestType, mem_type::view, resource::host>> const matrix =
        {fk::matrix<TestType, mem_type::view, resource::host>(m0),
         fk::matrix<TestType, mem_type::view, resource::host>(m1),
         fk::matrix<TestType, mem_type::view, resource::host>(m2),
         fk::matrix<TestType, mem_type::view, resource::host>(m3),
         fk::matrix<TestType, mem_type::view, resource::host>(m4),
         fk::matrix<TestType, mem_type::view, resource::host>(m5),
         fk::matrix<TestType, mem_type::view, resource::host>(m6),
         fk::matrix<TestType, mem_type::view, resource::host>(m7),
         fk::matrix<TestType, mem_type::view, resource::host>(m8)};

    int x_size = std::accumulate(
        matrix.begin(), matrix.end(), 1,
        [](int i, fk::matrix<TestType, mem_type::view, resource::host> const &m) {
          return i * m.ncols();
        });

    fk::vector<TestType, mem_type::owner, resource::host> const x(
        std::vector<TestType>(x_size, 1));

    int y_size = std::accumulate(
        matrix.begin(), matrix.end(), 1,
        [](int i, fk::matrix<TestType, mem_type::view, resource::host> const &m) {
          return i * m.nrows();
        });

    fk::vector<TestType> const correct(
        std::vector<TestType>(y_size, x_size * (1 << (matrix.size() - 1)) * 3));

    batch_job<TestType, resource::host> job = kron_batch(
        matrix, fk::vector<TestType, mem_type::view, resource::host>(x));

    fk::vector<TestType, mem_type::owner, resource::host> const r =
        execute_batch_job(job);

    REQUIRE(r == correct);
  }

  SECTION("kron_2_device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> const a = {{1}, {2}, {3}};

    fk::matrix<TestType, mem_type::owner, resource::device> const b = {{3, 4, 5}};

    fk::matrix<TestType, mem_type::owner, resource::device> const c = {{5}, {6}, {7}};

    fk::matrix<TestType, mem_type::owner, resource::device> const d = {{7, 8, 9}};

    std::vector<fk::matrix<TestType, mem_type::view, resource::device>>
    const matrix = {fk::matrix<TestType, mem_type::view, resource::device>(a),
                  fk::matrix<TestType, mem_type::view, resource::device>(b),
                  fk::matrix<TestType, mem_type::view, resource::device>(c),
                  fk::matrix<TestType, mem_type::view, resource::device>(d)};

    fk::vector<TestType, mem_type::owner, resource::device> const x = {
        1, 1, 1, 1, 1, 1, 1, 1, 1};

    fk::vector<TestType, mem_type::owner, resource::host> const correct = {
        1440, 1728, 2016, 2880, 3456, 4032, 4320, 5184, 6048};

    fk::vector<TestType, mem_type::view, resource::device> const x_view(x);

    batch_job<TestType, resource::device> bj = kron_batch(matrix, x_view);

    fk::vector<TestType, mem_type::owner, resource::device> const r =
        execute_batch_job(bj);

    REQUIRE(r.clone_onto_host() == correct);
  }

  SECTION("kron_2_host")
  {
    fk::matrix<TestType, mem_type::owner, resource::host> const a = {{1}, {2}, {3}};

    fk::matrix<TestType, mem_type::owner, resource::host> const b = {{3, 4, 5}};

    fk::matrix<TestType, mem_type::owner, resource::host> const c = {{5}, {6}, {7}};

    fk::matrix<TestType, mem_type::owner, resource::host> const d = {{7, 8, 9}};

    std::vector<fk::matrix<TestType, mem_type::view, resource::host>>
    const matrix = {fk::matrix<TestType, mem_type::view, resource::host>(a),
                  fk::matrix<TestType, mem_type::view, resource::host>(b),
                  fk::matrix<TestType, mem_type::view, resource::host>(c),
                  fk::matrix<TestType, mem_type::view, resource::host>(d)};

    fk::vector<TestType, mem_type::owner, resource::host> const x = {
        1, 1, 1, 1, 1, 1, 1, 1, 1};

    fk::vector<TestType, mem_type::owner, resource::host> const correct = {
        1440, 1728, 2016, 2880, 3456, 4032, 4320, 5184, 6048};

    fk::vector<TestType, mem_type::view, resource::host> const x_view(x);

    batch_job<TestType, resource::host> bj = kron_batch(matrix, x_view);

    fk::vector<TestType, mem_type::owner, resource::host > const r =
        execute_batch_job(bj);

    REQUIRE(r == correct);
  }
}
