#include "asgard_tools.hpp"

namespace asgard::tools
{

// formats the string, e.g., 3.00  3.10  3.00
std::string pad_string(double x)
{
  std::ostringstream os;
  os.precision(3);
  os << x;

  std::string res = os.str();
  
  std::string::size_type dot = res.find(".");
  if (dot < res.size()) {
    if (res.size() - dot < 4) {
      std::string::size_type rem = 4 + dot - res.size();

      if (rem > 0) 
        while(--rem) res += '0';
    } else {
      res = res.substr(0, dot + 2);
      while (res.size() < 4)
        res += '0';
    }
  } else {
    res += ".00";
  }

  std::string pre = "";
  std::string::size_type rem = 15 - res.size();
  
  if (rem > 0)
    while (--rem) pre += ' ';

  return pre + res;
}

std::string pad_string(size_t x)
{
  std::string res = std::to_string(x);

  if (res.size() < 12) {
    std::string::size_type rem = 12 - res.size();
    std::string pad = "";
    while (--rem)
      pad += ' ';
    return pad + res;
  }
  return res;
}

std::string simple_timer::report()
{
  std::ostringstream report;

  report << "\nperformance report\n";
  report << "  - all times in ms\n\n";

  std::string::size_type max_key = 0;
  double total = 0.0;
  for (auto [id, times] : id_to_times_) {
    max_key = std::max(id.size(), max_key);
    total   = std::accumulate(times.begin(), times.end(), total);
  }

  std::string::size_type rem = max_key + 1;
  while (--rem) report << ' ';

  report << "         total";
  report << "     % of total";
  report << "      count";
  report << "       average";
  report << "           min";
  report << "           max\n";

  for (auto [id, times] : id_to_times_) {
    double const sum = std::accumulate(times.begin(), times.end(), 0.0);
    double const avg = sum / static_cast<double>(times.size());
    double const min = *std::min_element(times.begin(), times.end());
    double const max = *std::max_element(times.begin(), times.end());

    rem = max_key - id.size() + 1;
    report << id;
    while (--rem) report << ' ';

    report << pad_string(sum);
    report << pad_string(100.0 * sum / total) << "%";
    report << pad_string(times.size());
    report << pad_string(avg);
    report << pad_string(min);
    report << pad_string(max) << '\n';
  }

  report << "\n";

  rem = max_key + 1;
  while (--rem) report << ' ';
  report << "      Gflops/s\n";

  for (auto [id, times] : id_to_times_) {
    if (id_to_flops_.count(id) > 0) {
      auto const &flops = id_to_flops_[id];
      double const fsum = std::accumulate(flops.begin(), flops.end(), 0.0);

      rem = max_key - id.size() + 1;
      report << id;
      while (--rem) report << ' ';

      report << pad_string(fsum / flops.size());
    }
  }

  return report.str();

  report << "\n\n";

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

        if (std::isinf(sum))
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
    // the last char in snprintf is null-terminator, if written to the sstream
    // and redirected into a file, the encoding is misinterpreted
    if (out.size() > 0)
      report << out.substr(0, out.size() - 1);
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

      if (std::isinf(sum))
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
