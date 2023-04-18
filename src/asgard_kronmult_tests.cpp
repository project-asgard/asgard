
#include "./device/asgard_kronmult.hpp"

#include <iostream>

#include "tests_general.hpp"

using namespace asgard::kronmult;

TEMPLATE_TEST_CASE("testing kronmult 1D",
                   "[kronmult1D]", float, double)
{
    std::vector<double> x(1);
    REQUIRE(x.size() == 1);
}
