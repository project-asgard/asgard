
//-----------------------------------------------------------------------------
//
// This file generates the main() for the Catch2 tests.
//
// We compile it here separately to cut down on recompilation times for the main
// software components.
//
//-----------------------------------------------------------------------------

#define CATCH_CONFIG_MAIN
#include "tests_general.hpp"
#include "catch.hpp"

// Someday I should come up with a more elegant solution here
// https://github.com/catchorg/Catch2/blob/master/docs/assertions.md
// https://github.com/catchorg/Catch2/blob/master/docs/matchers.md
//
void compareVectors(std::vector<double> a, std::vector<double> b) {
  for (size_t i = 0; i < a.size(); i++)
    if (a[i] !=
        Approx(b[i]).epsilon(std::numeric_limits<double>::epsilon() * 2))
      FAIL("" << a[i] << " != " << b[i]);
}

void compare2dVectors(std::vector<std::vector<double>> a,
                      std::vector<std::vector<double>> b) {
  for (size_t i = 0; i < a.size(); i++)
    compareVectors(a[i], b[i]);
}
