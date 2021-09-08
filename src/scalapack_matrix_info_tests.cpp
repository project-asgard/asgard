#include "cblacs_grid.hpp"
#include "distribution.hpp"
#include "scalapack_matrix_info.hpp"
#include "tests_general.hpp"

#include <array>

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEST_CASE("Generating scalapack matrix info serial", "[scalapack_matrix_info]")
{
  int rows{4}, cols{8};
  fk::scalapack_matrix_info info(rows, cols);
  REQUIRE(info.local_rows() == rows);
  REQUIRE(info.local_cols() == cols);
  int *desc                   = info.get_desc();
  std::array<int, 9> ref_desc = {{1, 0, rows, cols, rows, cols, 0, 0, rows}};

  for (int i = 0; i < 9; ++i)
  {
    REQUIRE(desc[i] == ref_desc[i]);
  }

  rows = 2;
  cols = 4;
  info.resize(rows, cols);

  REQUIRE(info.local_rows() == rows);
  REQUIRE(info.local_cols() == cols);
  desc     = info.get_desc();
  ref_desc = {{1, 0, rows, cols, rows, cols, 0, 0, rows}};

  for (int i = 0; i < 9; ++i)
  {
    REQUIRE(desc[i] == ref_desc[i]);
  }
}

TEST_CASE("Generating scalapack matrix info parallel",
          "[scalapack_matrix_info]")
{
  auto grid = std::make_shared<cblacs_grid>();
  int rows{4}, cols{4};
  int mb{2}, nb{2};
  fk::scalapack_matrix_info info(rows, cols, mb, nb, grid);
  if (get_num_ranks() == 1)
  {
    REQUIRE(info.local_rows() == rows);
    REQUIRE(info.local_cols() == cols);
  }
  else
  {
    REQUIRE(info.local_rows() == rows / mb);
    REQUIRE(info.local_cols() == cols / nb);
  }
  int *desc = info.get_desc();
  std::array<int, 9> ref_desc;
  if (get_num_ranks() == 1)
  {
    ref_desc = {{1, 0, rows, cols, mb, nb, 0, 0, rows}};
  }
  else
  {
    ref_desc = {{1, 0, rows, cols, mb, nb, 0, 0, rows / mb}};
  }
  for (int i = 0; i < 9; ++i)
  {
    REQUIRE(desc[i] == ref_desc[i]);
  }

  rows = 8;
  cols = 8;
  info.resize(rows, cols);
  if (get_num_ranks() == 1)
  {
    REQUIRE(info.local_rows() == rows);
    REQUIRE(info.local_cols() == cols);
  }
  else
  {
    REQUIRE(info.local_rows() == rows / mb);
    REQUIRE(info.local_cols() == cols / nb);
  }
  desc = info.get_desc();
  if (get_num_ranks() == 1)
  {
    ref_desc = {{1, 0, rows, cols, mb, nb, 0, 0, rows}};
  }
  else
  {
    ref_desc = {{1, 0, rows, cols, mb, nb, 0, 0, rows / mb}};
  }
  for (int i = 0; i < 9; ++i)
  {
    REQUIRE(desc[i] == ref_desc[i]);
  }
}
