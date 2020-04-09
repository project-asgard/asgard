#pragma once
#include <cassert>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

// simple profiling object
// this is NOT thread safe for now - only one thread should be calling class
// funcs at a time, if we need this, just need to wrap map access with locks
namespace timer
{
class recorder
{
public:
  // wrap function call and record time spent, value return version
  template<typename F, typename... Args>
  auto operator()(F &&f, std::string const identifier, Args &&... args) ->
      typename std::enable_if<
          !std::is_same_v<decltype(f(std::forward<Args>(args)...)), void>,
          decltype(f(std::forward<Args>(args)...))>::type
  {
    assert(!identifier.empty());
    assert(id_to_start_.count(identifier) == 0);
    auto const beg = std::chrono::high_resolution_clock::now();
    auto const ret = std::forward<F>(f)(std::forward<Args>(args)...);
    auto const end = std::chrono::high_resolution_clock::now();
    auto const dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - beg)
            .count();
    insert(identifier, dur);
    return ret;
  }

  template<typename F, typename... Args>
  auto operator()(F &&f, std::string const identifier, Args &&... args) ->
      typename std::enable_if<
          std::is_same_v<decltype(f(std::forward<Args>(args)...)), void>,
          void>::type
  {
    assert(!identifier.empty());
    assert(id_to_start_.count(identifier) == 0);
    auto const beg = std::chrono::high_resolution_clock::now();
    std::forward<F>(f)(std::forward<Args>(args)...);
    auto const end = std::chrono::high_resolution_clock::now();
    auto const dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - beg)
            .count();
    insert(identifier, dur);
  }

  std::string const &start(std::string const &identifier)
  {
    assert(!identifier.empty());
    assert(id_to_start_.count(identifier) == 0);
    id_to_start_[identifier] = std::chrono::high_resolution_clock::now();
    return identifier;
  }

  void stop(std::string const &identifier)
  {
    assert(!identifier.empty());
    assert(id_to_start_.count(identifier) == 1);
    auto const beg = id_to_start_[identifier];
    auto const end = std::chrono::high_resolution_clock::now();
    auto const dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - beg)
            .count();
    id_to_start_.erase(identifier);
    insert(identifier, dur);
  }

  // get performance report for recorded functions
  std::string report();

  // get times for some key, mostly for testing for now
  std::vector<double> const &get_times(std::string const &id)
  {
    assert(id_to_times_.count(id) == 1);
    return id_to_times_[id];
  }

private:
  // little helper for creating a new list if no values exist for key
  void insert(std::string const &key, double const time)
  {
    id_to_times_.try_emplace(key, std::vector<double>());
    id_to_times_[key].push_back(time);
  }

  // stores function identifier -> list of operator()times recorded
  std::map<std::string, std::vector<double>> id_to_times_;

  std::map<std::string,
           std::chrono::time_point<std::chrono::high_resolution_clock>>
      id_to_start_;
};
extern recorder record;
} // namespace timer
