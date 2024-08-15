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

// simple profiling object
// this is NOT thread safe
namespace asgard::tools
{
#ifndef NDEBUG
#define expect(cond) assert(cond)
#else
#define expect(cond) ((void)(cond))
#endif
// simple layer over assert to prevent unused variable warnings when
// expects disabled

struct timing_stats
{
  double avg;
  double min;
  double max;
  double med;
  double gflops;
  size_t ncalls;
};

class simple_timer
{
public:
  std::string const start(std::string const &identifier)
  {
    expect(!identifier.empty());
    id_to_start_[identifier] = std::chrono::high_resolution_clock::now();
    return identifier;
  }

  void stop(std::string const &identifier, double const flops = -1)
  {
#ifdef ASGARD_USE_CUDA
#ifndef NDEBUG
    cudaDeviceSynchronize(); // needed for accurate kronmult timing
#endif
#endif
    expect(!identifier.empty());
    expect(id_to_start_.count(identifier) == 1);
    auto const beg = id_to_start_[identifier];
    auto const end = std::chrono::high_resolution_clock::now();
    double const dur =
        std::chrono::duration<double, std::milli>(end - beg).count();

    id_to_start_.erase(identifier);
    insert(id_to_times_, identifier, dur);

    if (flops != -1)
    {
      expect(flops >= 0);
      auto const gflops = flops / 1e9;
      expect(dur >= 0.0);

      auto const gflops_per_sec = gflops / (dur * 1e-3); // to seconds

      insert(id_to_flops_, identifier, gflops_per_sec);
      expect(id_to_times_.count(identifier) == id_to_flops_.count(identifier));
    }
  }

  // get performance report for recorded functions
  std::string report();

  // get times for some key, mostly for testing for now
  std::vector<double> const &get_times(std::string const &id)
  {
    expect(id_to_times_.count(id) == 1);
    return id_to_times_[id];
  }

  // uses the map of timings to calculate avg, min, max, med, calls, for each
  // key similar to what is displayed in the report() function, but returns a
  // vector for use elsewhere
  void get_timing_stats(std::map<std::string, timing_stats> &stat_map);
  void get_timing_stats(std::map<std::string, std::vector<double>> &stat_map);

private:
  // little helper for creating a new list if no values exist for key
  void insert(std::map<std::string, std::vector<double>> &mapping,
              std::string const &key, double const time)
  {
    mapping.try_emplace(key, std::vector<double>());
    mapping[key].push_back(time);
  }

  timing_stats
  calculate_timing_stats(std::string const &&id, std::vector<double> &&times);

  // stores function identifier -> list of times recorded
  std::map<std::string, std::vector<double>> id_to_times_;

  // stores function identifier -> list of flops recorded
  std::map<std::string, std::vector<double>> id_to_flops_;

  std::map<std::string,
           std::chrono::time_point<std::chrono::high_resolution_clock>>
      id_to_start_;
};

inline simple_timer timer;

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
