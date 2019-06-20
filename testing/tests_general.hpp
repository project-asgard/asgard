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
#include <numeric>
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

// A function to construct a closure that will iterate over two iterators and
// compare them using a provided lambda.
// This is used to build functions like `relaxed_comparison` below.
//
// It runs the test function for each pair of objects in first and second iter,
// and returns false if at any point the comparison is false
template<class F>
inline auto const cons_relaxed_comparison(F const test)
{
  return [test](auto const &first_iter, auto const &second_iter) {
    auto first_ptr = first_iter.begin();
    for (auto second : second_iter)
    {
      // For some reason, you cannot use < operator,
      // which I would prefer. However, this is just as safe,
      // it just looks quite a bit scarier.
      assert(first_ptr != first_iter.end());
      if (!test(*first_ptr, second))
      {
        return false;
      }
      first_ptr++;
    }
    return true;
  };
}

// the relaxed comparison is due to:
// 1) difference in precision in calculations
// (c++ float/double vs matlab always double)
// 2) the reordered operations make very subtle differences
// requiring relaxed comparison for certain inputs
//
// This works by iterating over each element in the first and second iterators,
// and confirming that their values are each equal within a given precision
template<typename T>
auto const relaxed_comparison =
    [](auto const &first, auto const &second, auto const precision) {
      return cons_relaxed_comparison(
          [precision](auto const &first, auto const &second) {
            // Test that the pair of values of each iter in first and second are
            // within a certain precision of each other
            return Approx(first).epsilon(std::numeric_limits<T>::epsilon() *
                                         precision) == second;
          })(first, second);
    };

// This is used for creating a function that reduces two iterators into an
// accumulated value.
// This is used to implement diff comparison
template<class F>
auto const cons_reduce_comparison(F const transform)
{
  // A closure that iterates over the two iterators
  //  and returns the accumulated value
  return [transform](auto const &first_iter, auto const &second_iter,
                     auto accumulator_init) {
    // Create the iterators
    auto first_ptr   = first_iter.begin();
    auto second_ptr  = second_iter.begin();
    auto accumulator = accumulator_init;
    // Confirm that both first_ptr and second_ptr are within
    // the bounds of their respective iterators
    while (first_ptr < first_iter.end())
    {
      accumulator = transform(*first_ptr, *second_ptr, accumulator);
      first_ptr++;
      second_ptr++;
    }
    return accumulator;
  };
}

// Confirm the difference between two iterators is below a tolerance
template<typename T>
auto const
    diff_comparison = [](auto const &first_iter, auto const &second_iter) {
      // The difference between the first objects in each iter
      const auto initial_diff = *first_iter.begin() - *second_iter.begin();

      // Create a function that will accumulate the greatest difference
      // for each pair in the zipped iterators first_iter and second_iter
      T const result = std::abs(cons_reduce_comparison(
          [](auto const &first, auto const &second, auto const accumulator) {
            auto const diff = std::abs(first - second);
            if (diff > accumulator)
            {
              return diff;
            }
            return accumulator;
          })(first_iter, second_iter, initial_diff));

      // Return whether or not the difference is greater than the tolerance
      if constexpr (std::is_same<T, double>::value)
      {
        T const tol = std::numeric_limits<T>::epsilon() * 1e5;
        return result <= tol;
      }
      else
      {
        T const tol = std::numeric_limits<T>::epsilon() * 1e3;
        return result <= tol;
      }
    };
#endif
