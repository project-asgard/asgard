#pragma once
#include "tests_general.hpp"

namespace asgard
{
/*!
 * Hand crafted function designed to test
 * different aspects of the interpolatoin framework.
 */
template<typename precision>
struct test_functions
{
  // pre-computed constant
  static precision constexpr s3 = 1.73205080756887729; // sqrt(3.0)
  // first basis function, constant one
  static precision one(precision) { return 1.0; }
  // second basis funciton, linear Lagendre polynomial order 1
  static precision lag1(precision x) { return 2 * s3 * x - s3; }
  // general linear function (i.e., combination of one and lag1)
  static precision lin(precision x) { return x + 3.0; }
  // Linear functions with discontinuity in value or derivative,
  // those align to grids with resolution 2^k (for some k)
  // and thus will yield non-trivial but exact projections on the basis.
  // The number correspnds to the min-level required for exactness.
  static precision lin1(precision x) { return std::abs(x - 0.5); }
  static precision lin2(precision x) { return std::abs(lin1(x) - 0.25); }
  static precision lin3(precision x) { return (x < 0.5) ? lin2(2 * x) : -lin2(2 * x - 1); }

  // interpolation basis functions
  static precision ibasis0(precision x) { return -3 * x + 2; }
  static precision ibasis1(precision x) { return 3 * x - 1; }
  static precision ibasis2(precision x) { return (x < 0.5) ? (-6 * x + 2) : 0; }
  static precision ibasis3(precision x) { return (x < 0.5) ? 0 : (6 * x - 4); }
};

} // namespace asgard
