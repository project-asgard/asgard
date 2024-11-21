#pragma once
// one place for all std headers
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
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
#include <vector>

#include "asgard_build_info.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
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

} // namespace asgard::tools

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

} // namespace asgard
