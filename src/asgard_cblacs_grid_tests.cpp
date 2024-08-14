#include "asgard_cblacs_grid.hpp"
#include "tests_general.hpp"

using namespace asgard;

int main(int argc, char *argv[])
{
  initialize_distribution();

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

TEST_CASE("Generating a cblacs grid.", "[cblacs_grid]")
{
  if (!is_active())
  {
    return;
  }

  int myrank    = get_rank();
  int num_ranks = get_num_ranks();
  int nprow     = std::sqrt(num_ranks);
  auto grid     = get_grid();
  int myrow     = grid->get_myrow();
  int mycol     = grid->get_mycol();
  if (get_num_ranks() != 2 && get_num_ranks() != 3)
  {
    REQUIRE(myrank / nprow == myrow);
    REQUIRE(myrank % nprow == mycol);
  }

  int local_rows = grid->local_rows(4, 1);
  int local_cols = grid->local_cols(4, 1);
  if (num_ranks == 4)
  {
    // 4 elements on each process
    REQUIRE(local_rows * local_cols == 4);
  }
  else if (get_num_ranks() != 2 && get_num_ranks() != 3)
  {
    // 16 elements on one process
    REQUIRE(local_rows * local_cols == 16);
  }

  local_rows = grid->local_rows(4, 256);
  local_cols = grid->local_cols(4, 256);
  if (myrank == 0)
  {
    // 16 elements on zeroth proces
    REQUIRE(local_rows * local_cols == 16);
  }
  else
  {
    // None on other processes
    REQUIRE(local_rows * local_cols == 0);
  }
}
