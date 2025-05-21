#pragma once
// one place for all std headers
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "asgard_build_info.hpp"

// optional includes
#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

#ifdef ASGARD_USE_ROCM
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#endif

#ifdef ASGARD_USE_MPI
#include "mpi.h"
#endif

#ifndef NDEBUG
namespace asgard::debug {
  //! debug tools, write a range-like object
  template<typename range_like>
  void dump(range_like const &x) {
    for (auto const &v : x)
      std::cout << v << '\n';
  }
  //! debug tools, write the first n entries of a range-like object
  template<typename range_like>
  void dump(int n, range_like const &x) {
    for (int i = 0; i < n; i++)
      std::cout << x[i] << '\n';
  }
}
#endif

namespace asgard::tools
{
#ifndef NDEBUG
#define expect(cond) assert(cond)
#else
#define expect(cond) ((void)(cond))
#endif
// simple layer over assert to prevent unused variable warnings when
// expects disabled

/*!
 * \brief Simple profiling tool, allows us to time different sections of code
 *
 * The timer is not thread safe and should be used at a coarse level,
 * e.g., time formation of coefficients and kronmult as opposed to individual
 * small linear algebra operations.
 *
 * The timer can start and stop events using human readable strings as keys
 * and prints human readable report in the end.
 *
 * - only one event with a given key can be running at a time
 *   but different events can be nested
 * - nested events should labeled as such, otherwise the percentages in
 *   the report will be skewed
 */
class simple_timer
{
public:
  //! single instance of time
  using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

  //! internal use, stores data for a given event-key
  struct events_list {
    //! if set, the event is currently running and srated at started.value()
    std::optional<time_point> started;
    //! each duration for this event
    std::vector<double> intervals;
    //! if doing a kronmult event, report the Gflops/s
    std::vector<double> gflops;
    //! indicates whether to include in % of total time
    bool is_nested = false;
    //! during reporting, will be set to he sum of the intervals
    double sum = 0;
  };

  //! called at the start of the program
  simple_timer() : start_(current_time())
  {}

  //! start an event for the given id
  std::string const &start(std::string const &id)
  {
    expect(!id.empty());

    events_.try_emplace(id, events_list());
    expect(not events_[id].started);

    events_[id].started = current_time();

    return id;
  }

  //! stop the event and record the duration and flops (if present)
  double stop(std::string const &id, double const flops = -1)
  {
#ifdef ASGARD_USE_CUDA
#ifndef NDEBUG
    cudaDeviceSynchronize(); // needed for accurate kronmult timing
#endif
#endif

    events_list &event = events_[id];
    expect(event.started.has_value());

    event.intervals.push_back(duration_since(event.started));

    event.started.reset();

    if (flops != -1) {
      expect(flops >= 0);
      // flops -> Gflops has factor 1.E-9, ms -> seconds has factor 1.E-3
      // flops / ms -> Gflops / second has factor 1.E-9 / 1.E-3 = 1.E-6
      event.gflops.push_back(1.E-6 * flops / event.intervals.back());
    }

    return event.intervals.back();
  }

  //! indicates the times is active
  static bool enabled() {
    return true;
  }

  //! get the performance report for recorded events
  std::string report();

  //! returns the current time
  static time_point current_time() {
    return std::chrono::high_resolution_clock::now();
  }

  //! compute the time elapsed from start to the current_time()
  static double duration_since(time_point const &start) {
    return std::chrono::duration<double, std::milli>(current_time() - start).count();
  }
  //! compute the time elapsed from start to the current_time(), overload
  static double duration_since(std::optional<time_point> const &start) {
    return duration_since(start.value());
  }

private:
  //! kepps track of the start of the simulation
  time_point start_;
  //! for each event key, stores a list of durations
  std::map<std::string, events_list> events_;
};

/*!
 * \brief Used in place of simple_timer to disable timing events
 */
class null_timer
{
public:
  //! no-op start null-timer
  std::string const &start(std::string const &id) {
    return id;
  }
  //! no-op stop null-timer
  double stop(std::string const &, double const = -1) {
    return 0;
  }
  //! indicates the times is not active
  static bool enabled() {
    return false;
  }
  //! reports that the times is disabled
  std::string report() {
    return "<builtin timer disabled>\n";
  }
};

#ifdef ASGARD_USE_TIMER
inline simple_timer timer;
#else
inline null_timer timer;
#endif

/*!
 * Allows for RAII style of timing for blocks of code.
 * The constructor will initiate the timer for the given event,
 * the destructor will stop the timer.
 */
struct time_event
{
  //! \brief Constructor, start timing.
  time_event(std::string const &event_name)
      : event_name_(timer.start(event_name)), flops(-1)
  {}
  //! \brief Constructor, start timing for flop count.
  time_event(std::string const &event_name, double op_flops)
      : event_name_(timer.start(event_name)), flops(op_flops)
  {}
  //! \brief Destructor, stop timing.
  ~time_event() { timer.stop(event_name_, flops); }

  //! \brief Name of the event being timed.
  std::string const event_name_;
  //! \brief FLOPs, for the case when we are timing linear algebra.
  double flops;
};

//! null time event
struct null_time_event {
  null_time_event() = default;
  ~null_time_event() = default;
};
#ifdef ASGARD_USE_TIMER
//! initialize a timing session
inline time_event time_session(std::string const &name) {
  return time_event(name);
}
#else
//! skip timing when the timer has been disabled
inline null_time_event time_session(std::string const &) {
  return null_time_event();
}
#endif

//! converts a number to a string with format 1,234,567
inline std::string split_style(int64_t num) {
  if (num == 0)
    return "0";
  int64_t bound = 1000000000; // one billion
  std::string s = "";
  for (int i = 0; i < 4; i++) {
    if (s != "")
      s += ',';
    int64_t x = (bound > 0) ? num / bound : num;
    if (x >= 100) {
      s += std::to_string(x);
    } else if (x >= 10) {
      if (s != "")
        s += '0';
      s += std::to_string(x);
    } else if (x > 0) {
      if (s != "")
        s += "00";
      s += std::to_string(x);
    } else {
      if (s != "")
        s += "000";
    }
    num %= bound;
    bound /= 1000;
  }
  return s;
};

} // namespace asgard::tools

//! shortcuts for mpi commands
namespace asgard::mpi
{
#ifdef ASGARD_USE_MPI
//! returns the rank in the current comm
inline int comm_rank(MPI_Comm const comm) {
    int me;
    MPI_Comm_rank(comm, &me);
    return me;
}
//! (debug) returns the rank the world rank
inline int world_rank() { return comm_rank(MPI_COMM_WORLD); }
//! (debug) returns true if the world rank matches
inline bool is_world_rank(int rank) { return (world_rank() == rank); }

//! returns the size of the comm
inline int comm_size(MPI_Comm const comm){
  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  return nprocs;
}
//! return the size of the world comm
inline int world_size() { return comm_size(MPI_COMM_WORLD); }

//! given a C++ type T, return the corresponding MPI data type
template<typename T>
inline constexpr MPI_Datatype datatype() {
  using Q = std::remove_cv_t<T>;
  if constexpr (std::is_same_v<double, Q>)
    return MPI_DOUBLE;
  else if constexpr (std::is_same_v<float, Q>)
    return MPI_FLOAT;
  else if constexpr (std::is_same_v<int, Q>)
    return MPI_INT;
  else
    static_assert(std::is_same_v<double, Q>, "unknown MPI data-type");
}

#else
inline constexpr bool is_world_rank(int) { return true; }
inline constexpr int world_rank() { return 0; }
inline constexpr int world_size() { return 1; }
#endif

} // namespace asgard::mpi

namespace asgard
{
/*!
 * \brief Runtime assert, throw runtime error with the file, line, and info.
 *
 * Similar to cassert but is not disabled in Release mode.
 * Used to sanitize the user input.
 */
#define rassert(_result_, _info_)                                                                                   \
  if (!(_result_))                                                                                                  \
  {                                                                                                                 \
    throw std::runtime_error(std::string((_info_)) + " @file: " + __FILE__ + " line: " + std::to_string(__LINE__)); \
  }

/*!
 * \brief Suppressed warnings about unused variables
 *
 * An expressive way to indicate that a variable is intentionally left unused.
 */
auto const ignore = [](auto ignored) { (void)ignored; };

/*!
 * \brief Iterator/generator for a sequence of integers
 *
 * This is needed for the indexof template
 *
 * Technically satisfies the requirements for legacy iterator
 * but do not use directly, will be used internally in indexof
 */
template<typename idx_type = int64_t>
struct index_iterator
{
  using iterator_category = std::random_access_iterator_tag;

  using value_type      = idx_type;
  using difference_type = idx_type;
  using reference       = idx_type &;
  using pointer         = idx_type *;

  idx_type &operator*() { return value_; }
  idx_type const &operator*() const { return value_; }
  bool operator!=(index_iterator const &other) const { return value_ != other.value_; }
  index_iterator &operator++()
  {
    ++value_;
    return *this;
  }
  index_iterator &operator++(int) { return index_iterator{value_++}; }
  index_iterator &operator--()
  {
    --value_;
    return *this;
  }
  index_iterator &operator--(int) { return index_iterator{value_--}; }

  idx_type value_;
};

/*!
 * \brief Allows for range for-loops but using indexes
 *
 * There is a repeated pattern in coding when cross-referencing entries
 * between different vectors:
 * \code
 *   for (size_t i = 0; i < u.size(); i++)
 *     u[i] = std::sqrt(x[i]);
 * \endcode
 * The operation can be done with a std::transform but it leads to a messy
 * lambda capture and potential shadow. The index can be used to cross
 * reference more complex structures where iterators would be messy and
 * non-trivial, e.g., rows/columns of a matrix, sparse grid indexes, or
 * entries in a vector2d. The index also helps keep a more expressive
 * mathematical notation.
 *
 * On the other hand, the pattern is tedious to write over and over.
 *
 * This template provides an alternative and allows for syntax like:
 * \code
 *   for (auto i : indexof(u)) // i is int64_t
 *     u[i] = std::sqrt(x[i]);
 *
 *   for (auto i : indexof<int>(u)) // i is int
 *     u[i] = std::sqrt(x[i]);
 *
 *   for (auto i : indexof<size_t>(1, num_dimensions)) // i is size_t
 *     u[i] = std::sqrt(x[i]);
 * \endcode
 *
 * At -O3 Godbolt compiler profile yields the same code as for the constructs
 * for-indexof and the regular for-loop.
 *
 * Not sure how this relates to vectorization for small-matrix operations
 * and the construct is incompatible with OpenMP for and simd pragmas.
 */
template<typename idx_type = int64_t>
struct indexof
{
  template<typename vector_type>
  indexof(vector_type const &f)
      : beg_(0), end_(static_cast<idx_type>(f.size()))
  {}
  indexof(int num)
      : beg_(0), end_(static_cast<idx_type>(num))
  {}
  indexof(int64_t num)
      : beg_(0), end_(static_cast<idx_type>(num))
  {}
  indexof(size_t num)
      : beg_(0), end_(static_cast<idx_type>(num))
  {}
  template<typename cidx_type>
  indexof(cidx_type b, cidx_type e)
      : beg_(b), end_(e)
  {}

  index_iterator<idx_type> begin() const { return index_iterator<idx_type>{beg_}; }
  index_iterator<idx_type> end() const { return index_iterator<idx_type>{end_}; }

  idx_type beg_;
  idx_type end_;
};

//! Alias for the regular int-case, saves on typing when using smaller ranges
using iindexof = indexof<int>;


/*!
 * \brief Allows for range for-loops but using indexes
 *
 * This construct is similar to asgard::indexof but it focuses on working
 * with slices of vectors (as opposed to entire vectors).
 * The index-range defaults to int although it can use larger indexes.
 */
template<typename idx_type = int>
struct indexrange
{
  indexrange() : beg_(0), end_(0) {}

  template<typename range_type>
  indexrange(range_type const &r)
      : beg_(r.begin()), end_(r.end())
  {}
  indexrange(idx_type b, idx_type e)
      : beg_(b), end_(e)
  {}
  indexrange(int e)
      : beg_(0), end_(e)
  {}
  template<typename T>
  indexrange(std::vector<T> const &vec)
      : beg_(0), end_(static_cast<idx_type>(vec.size()))
  {}

  index_iterator<idx_type> begin() const { return index_iterator<idx_type>{beg_}; }
  index_iterator<idx_type> end() const { return index_iterator<idx_type>{end_}; }

  idx_type ibegin() const { return beg_; }
  idx_type iend() const { return end_; }

  bool empty() const { return (beg_ == end_); }
  bool contains(idx_type a) const { return (beg_ <= a and a < end_); }

  idx_type beg_;
  idx_type end_;
};

/*!
 * \brief Indicates a group of indexes, marked by the begin_ and end_ that is one after the last index
 *
 * The asgard::indexrange structure is used in the ranged-for-loop,
 * the asgard::irange struct is used to hold and manipulate the indexes,
 * e.g., create, assign after assignment, etc
 */
class irange {
public:
  //! create a new range with the given begin and end
  irange(int b, int e)
    : begin_(b), end_(e)
  {
    expect(e >= b);
  }
  //! create a new range from zero to the given end
  irange(int e)
    : begin_(0), end_(e)
  {
    expect(e >= 0);
  }

  //! returns the number of indexes
  int size() const { return (end_ - begin_); }
  //! returns the begin index
  int begin() const { return begin_; }
  //! returns the end index
  int end() const { return end_; }
  //! returns true if the range is empty
  bool empty() const { return (end_ <= begin_); }

private:
  //! first index of the range
  int begin_ = 0;
  //! one after the last index of the range, when begin_ == end_ we have an empty range
  int end_   = 0;
};

//! helper struct that add layer of indirection to the type
template<typename T>
struct no_deduce_struct {
  //! defines the type it's given
  using type = T;
};

/*!
 * \brief Helper method that guides template type deduction
 *
 * Example, consider the operation of multiplying a vector by a scalar and look
 * at the template signature:
 * \code
 *   template<typename P> scal(P alpha, std::vector<P> &x) { ... }
 * \endcode
 * Clearly, the vector type is more important than the scalar, since the precision
 * of the output is determined by the bits in x and not alpha.
 * However, consider the usage:
 * \code
 *   std::vector<double> x64 = ....;
 *   std::vector<double> x32 = ....;
 *
 *   scal(0.5, x64);  // OK, works fine since both x and 0.5 use 64-bits, P = double
 *   scal(0.5f, x32); // OK, both use 32-bits, P = float
 *
 *   scal(0.5, x32); // Error, P is deduced different for alpha and x
 *   scal(2, x64);   // Error, for alpha P is int but x uses doubles
 * \endcode
 *
 * Using the no_deduce tag will stop the compiler form inferring the type from alpha,
 * the type will be inferred from x and alpha will be converted.
 * \code
 *   template<typename P> scal(no_deduce<P> alpha, std::vector<P> &x) { ... }
 *
 *   scal(0.5, x32); // works fine, alpha = 0.5 is converted to 32-bits
 *   scal(2, x64);  // works fine, alpha = 2 is converted to 64-bits
 * \endcode
 */
template<typename T>
using no_deduce = typename no_deduce_struct<T>::type;

/*!
 * \brief Easy syntax check is a type is double
 *
 * Usage:
 * \code
 *   if constexpr (is_double<P>) {
 *     std::cout << " using 64-bit double precision\n";
 *     dgetrf(...); // call BLAS double precision
 *     cusolverDnDgetrf(...); // call cuSolver double
 *   }
 * \endcode
 */
template<typename T>
constexpr bool is_double = std::is_same_v<double, T>;

/*!
 * \brief Easy syntax check is a type is float
 *
 * Usage:
 * \code
 *   if constexpr (is_float<P>) {
 *     std::cout << " using 32-bit single precision\n";
 *     sgetrf(...); // call BLAS single precision
 *     cusolverDnSgetrf(...); // call cuSolver single
 *   }
 * \endcode
 */
template<typename T>
constexpr bool is_float = std::is_same_v<float, T>;

} // namespace asgard
