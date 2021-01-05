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
namespace tools
{
// simple layer over assert to prevent unused variable warnings when
// tools::expects disabled
inline void expect(bool const condition)
{
  auto const ignore = [](auto ignored) { (void)ignored; };
  ignore(condition);
  assert(condition);
}

class simple_timer
{
public:
  std::string const start(std::string const &identifier)
  {
    tools::expect(!identifier.empty());
    tools::expect(id_to_start_.count(identifier) == 0);
    id_to_start_[identifier] = std::chrono::high_resolution_clock::now();
    return identifier;
  }

  void stop(std::string const &identifier, double const flops = -1)
  {
    tools::expect(!identifier.empty());
    tools::expect(id_to_start_.count(identifier) == 1);
    auto const beg = id_to_start_[identifier];
    auto const end = std::chrono::high_resolution_clock::now();
    auto const dur =
        std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
            .count();

    id_to_start_.erase(identifier);
    insert(id_to_times_, identifier, dur * 1e-3); // to ms

    if (flops != -1)
    {
      tools::expect(flops > 0);
      auto const gflops = flops / 1e9;
      tools::expect(dur >= 0.0);

      auto const gflops_per_sec =
          dur == 0.0 ? std::numeric_limits<double>::infinity()
                     : gflops / (static_cast<double>(dur) * 1e-6); // to seconds

      insert(id_to_flops_, identifier, gflops_per_sec);
      tools::expect(id_to_times_.count(identifier) ==
                    id_to_flops_.count(identifier));
    }
  }

  // get performance report for recorded functions
  std::string report();

  // get times for some key, mostly for testing for now
  std::vector<double> const &get_times(std::string const &id)
  {
    tools::expect(id_to_times_.count(id) == 1);
    return id_to_times_[id];
  }

private:
  // little helper for creating a new list if no values exist for key
  void insert(std::map<std::string, std::vector<double>> &mapping,
              std::string const &key, double const time)
  {
    mapping.try_emplace(key, std::vector<double>());
    mapping[key].push_back(time);
  }

  // stores function identifier -> list of times recorded
  std::map<std::string, std::vector<double>> id_to_times_;

  // stores function identifier -> list of flops recorded
  std::map<std::string, std::vector<double>> id_to_flops_;

  std::map<std::string,
           std::chrono::time_point<std::chrono::high_resolution_clock>>
      id_to_start_;
};

extern simple_timer timer;

} // namespace tools
