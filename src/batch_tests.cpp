#include "tests_general.hpp"

#include "batch.hpp"

TEMPLATE_TEST_CASE("batch_list: constructors, copy/move", "[batch]", float,
                   double)
{
  REQUIRE(true);
}

TEMPLATE_TEST_CASE("batch_list: insert into batch", "[batch]", float, double)
{
  REQUIRE(true);
}

TEMPLATE_TEST_CASE("batch_list: execute gemm", "[batch]", float, double)
{
  REQUIRE(true);
}
