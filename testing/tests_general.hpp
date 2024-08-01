//-----------------------------------------------------------------------------
//
// some utilities possibly useful/needed across various component tests
//
//-----------------------------------------------------------------------------

#pragma once

#include "asgard.hpp"
#include <catch2/catch_all.hpp>

/*!
 * \defgroup AsgardTesting Miscellaneous testing utilities
 *
 * Helper functions to facilitate testing.
 */

/*!
 * \brief Macro that expands to the compiled precisions
 */
#ifdef ASGARD_ENABLE_DOUBLE
#ifdef ASGARD_ENABLE_FLOAT
#define test_precs double, float
#else
#define test_precs double
#endif
#elif defined(ASGARD_ENABLE_FLOAT)
#define test_precs float
#endif

#ifdef ASGARD_ENABLE_DOUBLE
#ifdef ASGARD_ENABLE_FLOAT
#define test_precs_host (double, resource::host), (float, resource::host)
#else
#define test_precs_host (double, resource::host)
#endif
#elif defined(ASGARD_ENABLE_FLOAT)
#define test_precs_host (float, resource::host)
#endif

#ifdef ASGARD_USE_CUDA
#ifdef ASGARD_ENABLE_DOUBLE
#ifdef ASGARD_ENABLE_FLOAT
#define test_precs_device (double, resource::device), (float, resource::device)
#else
#define test_precs_device (double, resource::device)
#endif
#elif defined(ASGARD_ENABLE_FLOAT)
#define test_precs_device (float, resource::device)
#endif
#endif

static inline const std::filesystem::path gold_base_dir{ASGARD_GOLD_BASE_DIR};

template<typename P>
constexpr P get_tolerance(int ulp)
{
  return std::numeric_limits<P>::epsilon() * ulp;
}

/* These functions implement: norm( v0 - v1 ) < tolerance * max( norm(v0),
 * norm(v1) )*/
template<typename P, asgard::mem_type mem, asgard::mem_type omem>
void rmse_comparison(asgard::fk::vector<P, mem> const &v0,
                     asgard::fk::vector<P, omem> const &v1, P const tolerance)
{
  auto const diff_norm = asgard::fm::nrm2(v0 - v1);

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

template<typename P, asgard::mem_type mem, asgard::mem_type omem>
void rmse_comparison(asgard::fk::matrix<P, mem> const &m0,
                     asgard::fk::matrix<P, omem> const &m1, P const tolerance)
{
  auto const diff_norm = asgard::fm::frobenius(m0 - m1);

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
  if constexpr (std::is_floating_point_v<P>)
  {
    for (size_t i = 0; i < a.size(); i++)
      if (a[i] !=
          Catch::Approx(b[i]).epsilon(std::numeric_limits<P>::epsilon() * 2))
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
asgard::fk::vector<P>
default_initial_condition(asgard::fk::vector<P> const vect)
{
  return vect;
}

template<typename P>
asgard::dimension<P>
make_dummy_dim(int const level = 0, int const degree = 0,
               P const domain_min = 0.0, P const domain_max = 0.0,
               asgard::boundary_condition const left =
                   asgard::boundary_condition::periodic,
               asgard::boundary_condition const right =
                   asgard::boundary_condition::periodic,
               asgard::vector_func<P> const initial_condition =
                   default_initial_condition<P>,
               std::string const name = "")
{
  return asgard::dimension<P>(left, right, domain_min, domain_max, level,
                              degree, initial_condition, name);
}

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
  static_assert(std::is_same_v<P, R>, "containers must hold same type");
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

/**
 * \brief Calculates the integral of a vector over the given dimension
 */
template<typename P>
P calculate_integral(asgard::fk::vector<P> const &input,
                     asgard::dimension<P> const &dim)
{
  int const degree           = dim.get_degree();
  auto const legendre_values = asgard::legendre_weights<P>(
      degree, -1.0, 1.0, asgard::quadrature_mode::use_fixed);
  int const num_quad   = legendre_values[0].size();
  int const num_cells  = input.size() / num_quad;
  P const grid_spacing = (dim.domain_max - dim.domain_min) / num_cells;

  asgard::fk::matrix<P> coefficients(num_cells, num_quad);
  for (int elem = 0; elem < num_cells; elem++)
  {
    for (int d = 0; d < num_quad; d++)
    {
      coefficients(elem, d) = 0.5 * input[elem * num_quad + d];
    }
  }

  asgard::fk::matrix<P> w(num_quad, num_quad);
  w.update_col(0, legendre_values[1]);

  asgard::fk::matrix<P> input_weighted(num_cells, num_quad);
  asgard::fm::gemm(coefficients, w, input_weighted);
  asgard::fm::scal(P{0.5}, input_weighted);

  P sum = 0.0;
  for (int elem = 0; elem < num_cells; elem++)
  {
    for (int d = 0; d < num_quad; d++)
    {
      sum += input_weighted(elem, d);
    }
  }

  P integral = grid_spacing * sum;
  return integral;
}

/**
 * \brief Calculate the discrete l-2 norm of the difference between two vectors.
 */
template<typename P>
P nrm2_dist(asgard::fk::vector<P> const &x, asgard::fk::vector<P> const &y)
{
  expect(x.size() == y.size());

  P l2 = 0.0;
  for (int i = 0; i < x.size(); i++)
  {
    P const diff = x[i] - y[i];
    l2 += diff * diff;
  }
  return std::sqrt(l2);
}
