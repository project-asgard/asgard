//-----------------------------------------------------------------------------
//
// some utilities possibly useful/needed across various component tests
//
//-----------------------------------------------------------------------------

#ifndef _tests_general_h_
#define _tests_general_h_

#include "catch.hpp"
#include <vector>

// Someday I should come up with a more elegant solution here
// https://github.com/catchorg/Catch2/blob/master/docs/assertions.md
// https://github.com/catchorg/Catch2/blob/master/docs/matchers.md
//
template<typename P>
void compareVectors(std::vector<P> a, std::vector<P> b)
{
  if constexpr (std::is_floating_point<P>::value)
  {
    for (size_t i = 0; i < a.size(); i++)
      if (a[i] != Approx(b[i]).epsilon(std::numeric_limits<P>::epsilon() * 2))
        FAIL("" << a[i] << " != " << b[i]);
  }
  else
  {
    for (size_t i = 0; i < a.size(); i++)
    {
      if (a[i] != b[i]) { FAIL("" << a[i] << " != " << b[i]); }
    }
  }
}

template<typename P>
void compare2dVectors(std::vector<std::vector<P>> a,
                      std::vector<std::vector<P>> b)
{
  for (size_t i = 0; i < a.size(); i++)
    compareVectors(a[i], b[i]);
}

#endif
