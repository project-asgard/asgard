#include "timer.hpp"
#include <algorithm>
// FIXME use string format after C++20
// #include <format>
#include <numeric>
#include <sstream>

namespace timer
{
std::string recorder::report()
{
  std::ostringstream report;
  report << "\nperformance report, all times in ms...\n\n";

  for (auto [id, times] : id_to_times)
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

    report << id << " - avg: " << avg << " min: " << min << " max: " << max
           << " med: " << med << " calls: " << times.size() << '\n';
  }
  return report.str();
}

recorder record;

} // namespace timer
