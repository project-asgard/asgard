#include "cblacs_grid.hpp"
#include "distribution.hpp"
#include "scalapack_vector_info.hpp"
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

TEST_CASE("Generating scalapack vector info serial", "[scalapack_vector_info]")
{
  int size{4};
  fk::scalapack_vector_info info(size);
  REQUIRE(info.local_size() == size);
  int *desc                   = info.get_desc();
  std::array<int, 9> ref_desc = {{1, 0, size, 1, size, 1, 0, 0, size}};

  for (int i = 0; i < 9; ++i)
  {
    REQUIRE(desc[i] == ref_desc[i]);
  }

  size = 2;
  info.resize(size);
  REQUIRE(info.local_size() == size);
  desc     = info.get_desc();
  ref_desc = {{1, 0, size, 1, size, 1, 0, 0, size}};

  for (int i = 0; i < 9; ++i)
  {
    REQUIRE(desc[i] == ref_desc[i]);
  }
}

TEST_CASE("Generating scalapack vector info parallel",
          "[scalapack_vector_info]")
{
  auto grid = std::make_shared<cblacs_grid>();
  int size{4};
  int mb{2};
  fk::scalapack_vector_info info(size, mb, grid);
  if (get_num_ranks() == 1)
  {
    REQUIRE(info.local_size() == size);
  }
  else
  {
    REQUIRE(info.local_size() == size / mb);
  }
  int *desc = info.get_desc();
  std::array<int, 9> ref_desc;
  if (get_num_ranks() == 1)
  {
    ref_desc = {{1, 0, size, 1, mb, 1, 0, 0, size}};
  }
  else
  {
    ref_desc = {{1, 0, size, 1, mb, 1, 0, 0, size / mb}};
  }
  for (int i = 0; i < 9; ++i)
  {
    REQUIRE(desc[i] == ref_desc[i]);
  }

  size = 8;
  info.resize(size);
  if (get_num_ranks() == 1)
  {
    REQUIRE(info.local_size() == size);
  }
  else
  {
    REQUIRE(info.local_size() == size / mb);
  }
  desc = info.get_desc();
  if (get_num_ranks() == 1)
  {
    ref_desc = {{1, 0, size, 1, mb, 1, 0, 0, size}};
  }
  else
  {
    ref_desc = {{1, 0, size, 1, mb, 1, 0, 0, size / mb}};
  }
  for (int i = 0; i < 9; ++i)
  {
    REQUIRE(desc[i] == ref_desc[i]);
  }
}
