#include "tools.hpp"
#include <algorithm>
// FIXME use string format after C++20
// #include <format>
#include <cstdio>
#include <math.h>
#include <numeric>
#include <sstream>

namespace asgard::tools
{
std::string simple_timer::report()
{
  std::ostringstream report;
  report << "\nperformance report, all times in ms...\n\n";
  char const *fmt =
      "%s - avg: %.4f min: %.3f max: %.3f med: %.4f %s calls: %d \n";
  for (auto [id, times] : id_to_times_)
  {
    auto const avg =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    auto const min = *std::min_element(times.begin(), times.end());
    auto const max = *std::max_element(times.begin(), times.end());

    // calculate median
    auto const middle_it = times.begin() + times.size() / 2;
    std::nth_element(times.begin(), middle_it, times.end());
    auto const med =
        times.size() % 2 == 0
            ? (*std::max_element(times.begin(), middle_it) + *middle_it) / 2
            : *middle_it;

    auto const avg_flops = [this, id = id]() {
      if (id_to_flops_.count(id) > 0)
      {
        auto const flops = id_to_flops_[id];
        auto const sum   = std::accumulate(flops.begin(), flops.end(), 0.0);

        if (isinf(sum))
        {
          return std::string(" avg gflops: inf");
        }
        auto const average = sum / flops.size();
        return std::string(" avg gflops: ") + std::to_string(average);
      }
      return std::string("");
    }();
    auto size = snprintf(nullptr, 0, fmt, id.c_str(), avg, min, max, med,
                         avg_flops.c_str(), times.size());
    std::string out(size + 1, ' ');
    snprintf(out.data(), size + 1, fmt, id.c_str(), avg, min, max, med,
             avg_flops.c_str(), times.size());
    report << out;
  }
  return report.str();
}
simple_timer timer;

} // namespace asgard::tools
