#include "tools.hpp"
#include <algorithm>
// FIXME use string format after C++20
// #include <format>
#include <iomanip>
#include <math.h>
#include <numeric>
#include <sstream>
#include <cstdio>

namespace tools
{
std::string simple_timer::report()
{
  std::ostringstream report;
  report << "\nperformance report, all times in ms...\n\n";
  const char * fmt1 = " - avg: %f min: %f max: %f med: %f";
  const char * fmt2 = " calls: %d \n";
  std::cout << id_to_times_.size();
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
        auto const avg = sum / flops.size();
        return std::string(" avg gflops: ") + std::to_string(avg);
      }
      return std::string("");
    }();
    auto size1 = snprintf(NULL, 0, fmt1, (double)avg, (double)min, (double)max, (double)med);
    auto size2 = snprintf(NULL, 0, fmt2, times.size());
    std::string out1(size1+1, ' ');
    std::string out2(size2+1, ' ');
    snprintf(out1.data(), size1 + 1, fmt1, avg, min, max, med);
    snprintf(out2.data(), size2 + 1, fmt2, times.size());
    report << id << out1 << avg_flops << out2;
    //std::cout << id << out1 << avg_flops << out2;
    //report << id << " - avg: " << avg << " min: " << min << " max: " << max
    //       << " med: " << med << avg_flops << " calls: " << times.size()
    //       << '\n';
  }
  //std::cout << "report:\n" << report.str();
  return report.str();
}
simple_timer timer;

} // namespace tools
