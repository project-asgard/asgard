#pragma once

#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace timer
{
class recorder
{
public:
  // wrap function call and record time spent, value return version
  template<typename F, typename... Args>
  auto run(F &&f, std::string const identifier, Args &&... args) ->
      typename std::enable_if<
          !std::is_same_v<decltype(f(std::forward<Args>(args)...)), void>,
          decltype(f(std::forward<Args>(args)...))>::type;

  // same, void return version
  template<typename F, typename... Args>
  auto run(F &&f, std::string const identifier, Args &&... args) ->
      typename std::enable_if<
          std::is_same_v<decltype(f(std::forward<Args>(args)...)), void>,
          void>::type;

  // get performance report for recorded functions
  std::string report() const;

private:
  // little helper for creating a new list if no values exist for key
  void insert(std::string const &key, double const time)
  {
    id_to_times.try_emplace(key, std::vector<double>());
    id_to_times[key].push_back(time);
  };

  // stores function identifier -> list of runtimes recorded
  std::map<std::string const, std::vector<double>> id_to_times;
};

} // namespace timer
