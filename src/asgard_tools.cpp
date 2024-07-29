#include "asgard_tools.hpp"

namespace asgard::tools
{
std::string simple_timer::report()
{
  std::ostringstream report;
  report << "\nperformance report, all times in ms...\n\n";
  char const *fmt =
      "%s - avg: %.7f min: %.7f max: %.7f med: %.7f %s calls: %d \n";
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

// Helper function to calculate the avg, min, max, med, gflops, and ncalls for a
// given key
timing_stats simple_timer::calculate_timing_stats(std::string const &&id,
                                                  std::vector<double> &&times)
{
  double const avg =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  double const min = *std::min_element(times.begin(), times.end());
  double const max = *std::max_element(times.begin(), times.end());

  // calculate median
  auto const middle_it = times.begin() + times.size() / 2;
  std::nth_element(times.begin(), middle_it, times.end());
  double const med =
      times.size() % 2 == 0
          ? (*std::max_element(times.begin(), middle_it) + *middle_it) / 2
          : *middle_it;

  double const avg_flops = [this, id = id]() -> double {
    if (id_to_flops_.count(id) > 0)
    {
      auto const flops = id_to_flops_[id];
      auto const sum   = std::accumulate(flops.begin(), flops.end(), 0.0);

      if (isinf(sum))
      {
        return -1.0;
      }
      auto const average = sum / flops.size();
      return average;
    }
    return -1.0;
  }();

  return timing_stats{avg, min, max, med, avg_flops, times.size()};
}

void simple_timer::get_timing_stats(
    std::map<std::string, timing_stats> &stat_map)
{
  stat_map = std::map<std::string, timing_stats>();
  for (auto [id, times] : id_to_times_)
  {
    stat_map[id] = calculate_timing_stats(std::move(id), std::move(times));
  }
}

void simple_timer::get_timing_stats(
    std::map<std::string, std::vector<double>> &stat_map)
{
  stat_map = std::map<std::string, std::vector<double>>();
  for (auto [id, times] : id_to_times_)
  {
    auto stats   = calculate_timing_stats(std::move(id), std::move(times));
    stat_map[id] = std::vector<double>{
        stats.avg, stats.min, stats.max,
        stats.med, stats.gflops, static_cast<double>(stats.ncalls)};
  }
}

} // namespace asgard::tools
