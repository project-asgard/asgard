//-----------------------------------------------------------------------------
//
// some utilities possibly useful/needed across various component tests
//
//-----------------------------------------------------------------------------

#pragma once

#include "src/fast_math.hpp"
#include "src/pde.hpp"
#include "src/program_options.hpp"
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <catch2/catch_all.hpp>

template<typename P>
constexpr P get_tolerance(int ulp)
{
  return std::numeric_limits<P>::epsilon()*ulp;
}

/* These functions implement: norm( v0 - v1 ) < tolerance * max( norm(v0),
 * norm(v1) )*/
template<typename P, mem_type mem, mem_type omem>
void rmse_comparison(fk::vector<P, mem> const &v0,
                     fk::vector<P, omem> const &v1, P const tolerance)
{
  auto const diff_norm = fm::nrm2(v0 - v1);

  auto const abs_compare = [](auto const a, auto const b) {
    return (std::abs(a) < std::abs(b));
  };

  auto const max = std::max(
      static_cast<P>(1.0),
      std::max(std::abs(*std::max_element(v0.begin(), v0.end(), abs_compare)),
               std::abs(*std::max_element(v1.begin(), v1.end(), abs_compare))));
  Catch::StringMaker<P>::precision = 20;
  REQUIRE((diff_norm / max) < (tolerance * std::sqrt(v0.size())));
}

template<typename P, mem_type mem, mem_type omem>
void rmse_comparison(fk::matrix<P, mem> const &m0,
                     fk::matrix<P, omem> const &m1, P const tolerance)
{
  auto const diff_norm = fm::frobenius(m0 - m1);

  auto const abs_compare = [](auto const a, auto const b) {
    return (std::abs(a) < std::abs(b));
  };
  auto const max = std::max(
      static_cast<P>(1.0),
      std::max(std::abs(*std::max_element(m0.begin(), m0.end(), abs_compare)),
               std::abs(*std::max_element(m1.begin(), m1.end(), abs_compare))));

  REQUIRE((diff_norm / max) < (tolerance * std::sqrt(m0.size())));
}

// Someday I should come up with a more elegant solution here
// https://github.com/catchorg/Catch2/blob/master/docs/assertions.md
// https://github.com/catchorg/Catch2/blob/master/docs/matchers.md
// FIXME we hardly use std::vect...do we still need this?
template<typename P>
void compare_vectors(std::vector<P> const &a, std::vector<P> const &b)
{
  if constexpr (std::is_floating_point<P>::value)
  {
    for (size_t i = 0; i < a.size(); i++)
      if (a[i] != Catch::Approx(b[i]).epsilon(std::numeric_limits<P>::epsilon() * 2))
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

// WARNING for tests only!
// features rely on options, parser, and PDE constructed w/ same arguments
options make_options(std::vector<std::string> const arguments);
parser make_parser(std::vector<std::string> const arguments);

template<typename T>
std::string to_string_with_precision(T const a_value, int const precision = 6)
{
  std::ostringstream out;
  out.precision(precision);
  out << std::fixed << a_value;
  return out.str();
}

template<typename P>
using enable_for_fp = std::enable_if_t<std::is_floating_point_v<P>>;

// compare two fp types with some tolerance
template<typename P, typename P_ = P, typename = enable_for_fp<P_>>
void relaxed_fp_comparison(P const first, P const second,
                           double const tol_fac = 1e1)
{
  auto const tol       = std::numeric_limits<P>::epsilon() * tol_fac;
  auto const scale_fac = std::max(
      std::max(static_cast<P>(1.0), std::abs(first)), std::abs(second));
  REQUIRE_THAT(first, Catch::Matchers::WithinAbs(second, tol * scale_fac));
}

// compare two tensor types
// (specifically: templated containers with .data() access to backing container)
// scaled for value magnitude
// tol_fac can be used to adjust tolerance; this number is multipled by epsilon
// to form the tolerance
template<typename comparable_1, typename comparable_2>
void relaxed_comparison(comparable_1 const &first, comparable_2 const &second,
                        double const tol_fac = 1e1)
{
  REQUIRE(first.size() == second.size());
  // retrieving comparable's contained datatype
  // was using template template parameters, but clang complained
  using P = typename std::remove_pointer<decltype(first.data())>::type;
  using R = typename std::remove_pointer<decltype(second.data())>::type;
  static_assert(std::is_same<P, R>::value, "containers must hold same type");
  Catch::StringMaker<P>::precision = 15;
  auto first_it                    = first.begin();
  std::for_each(
      second.begin(), second.end(),
      [&first_it, tol_fac](auto const &second_elem) {
        auto const tol = std::numeric_limits<P>::epsilon() * tol_fac;
        auto const scale_fac =
            std::max(std::max(static_cast<P>(1.0), std::abs(*first_it)),
                     std::abs(second_elem));
        REQUIRE_THAT(*first_it++,
                     Catch::Matchers::WithinAbs(second_elem, tol * scale_fac));
      });
}
