#include "distribution.hpp"
#include "tests_general.hpp"

TEST_CASE("subgrid struct", "[distribution]")
{
  int const row_start = 0;
  int const row_stop  = 4;
  int const col_start = 1;
  int const col_stop  = 3;
  element_subgrid const e(row_start, row_stop, col_start, col_stop);

  SECTION("construct/copy construct")
  {
    REQUIRE(e.row_start == row_start);
    REQUIRE(e.row_stop == row_stop);
    REQUIRE(e.col_start == col_start);
    REQUIRE(e.col_stop == col_stop);

    element_subgrid const e2(e);
    REQUIRE(e == e2);

    REQUIRE(e2.row_start == row_start);
    REQUIRE(e2.row_stop == row_stop);
    REQUIRE(e2.col_start == col_start);
    REQUIRE(e2.col_stop == col_stop);
  }

  SECTION("dimensions functions")
  {
    REQUIRE(e.nrows() == 5);
    REQUIRE(e.ncols() == 3);
    REQUIRE(e.size() == 15);
  }

  SECTION("translation functions")
  {
    REQUIRE(e.to_global_row(3) == 3);
    REQUIRE(e.to_global_col(2) == 3);
    REQUIRE(e.to_local_row(0) == 0);
    REQUIRE(e.to_local_col(1) == 0);

    int const row_start = 2;
    int const row_stop  = 5;
    int const col_start = 0;
    int const col_stop  = 4;
    element_subgrid const e2(row_start, row_stop, col_start, col_stop);

    REQUIRE(e2.to_global_row(3) == 5);
    REQUIRE(e2.to_global_col(0) == 0);
    REQUIRE(e2.to_local_row(3) == 1);
    REQUIRE(e2.to_local_col(2) == 2);
  }
}

TEST_CASE("rank subgrid function", "[distribution]")
{
  SECTION("1 rank, whole problem") {}

  SECTION("2 ranks") {}

  SECTION("3 ranks - odd/nonsquare") {}

  SECTION("9 ranks - odd/square") {}
}

TEST_CASE("distribution plan function", "[distribution]") {}
