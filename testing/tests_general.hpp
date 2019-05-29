//-----------------------------------------------------------------------------
//
// some utilities possibly useful/needed across various component tests
//
//-----------------------------------------------------------------------------

#ifndef _tests_general_h_
#define _tests_general_h_

#include "../src/pde.hpp"
#include "../src/program_options.hpp"
#include "catch.hpp"
#include <string>
#include <vector>

// Someday I should come up with a more elegant solution here
// https://github.com/catchorg/Catch2/blob/master/docs/assertions.md
// https://github.com/catchorg/Catch2/blob/master/docs/matchers.md
// FIXME we hardly use std::vect...do we still need this?
template<typename P>
void compare_vectors(std::vector<P> a, std::vector<P> b)
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
      if (a[i] != b[i])
      {
        FAIL("" << a[i] << " != " << b[i]);
      }
    }
  }
}

template<typename P>
void compare_2d_vectors(std::vector<std::vector<P>> a,
                        std::vector<std::vector<P>> b)
{
  for (size_t i = 0; i < a.size(); i++)
    compare_vectors(a[i], b[i]);
}

template<typename P>
fk::vector<P> default_initial_condition(fk::vector<P> const vect)
{
  return vect;
}

template<typename P>
dimension<P> make_dummy_dim(
    int const level = 0, int const degree = 0, P const domain_min = 0.0,
    P const domain_max                     = 0.0,
    boundary_condition const left          = boundary_condition::periodic,
    boundary_condition const right         = boundary_condition::periodic,
    vector_func<P> const initial_condition = default_initial_condition<P>,
    std::string const name                 = "")
{
  return dimension<P>(left, right, domain_min, domain_max, level, degree,
                      initial_condition, name);
}

options make_options(std::vector<std::string> const arguments);

#endif
