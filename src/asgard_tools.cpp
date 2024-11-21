#include "asgard_tools.hpp"

namespace asgard::tools
{

std::string::size_type constexpr double_block = 14;
std::string::size_type constexpr int_block = 11;
std::string::size_type constexpr percent_reduce = 4;

// formats the string, e.g., 3.00  3.10  3.00
template<std::string::size_type size>
std::string pad_left(std::string const &s) {
  if (s.size() < size)
    return std::string(size - s.size(), ' ') + s;
  else
    return s;
}
std::string pad_left(std::string::size_type size, std::string const &s) {
  if (s.size() < size)
    return std::string(size - s.size(), ' ') + s;
  else
    return s;
}

std::string pad_string(double x)
{
  std::ostringstream os;
  os.precision(2);
  os << std::fixed << x;

  std::string res = os.str();

  std::string::size_type dot = res.find(".");
  if (dot < res.size()) {
    if (res.size() - dot < 4) {
      std::string::size_type rem = 3 + dot - res.size();

      if (rem > 0)
        res += std::string(rem, '0');
    } else {
      res = res.substr(0, dot + 2);
      if (res.size() < 4)
        res += std::string(4 - res.size(), '0');
    }
  } else {
    res += ".00";
  }

  return pad_left<double_block>(res);
}

std::string pad_string(size_t x)
{
  std::string res = std::to_string(x);
  return pad_left<int_block>(res);
}

std::string pad_string_percent(double x, double total) {
  std::string res = pad_string(100.0 * x / total);
  res = res.substr(percent_reduce, res.size() - percent_reduce - 1);
  return res + '%';
}

std::string simple_timer::report()
{
  // time since the timer was initialized (program started)
  double const total_time = duration_since(start_);

  std::ostringstream report;

  report << "\nperformance report, total time: " << pad_string(total_time) << "ms\n";
  report << "  - all times in ms, 1000ms = 1 second\n\n";

  std::string const ev =  "-- events --  ";
  std::string::size_type max_key = ev.size();
  for (auto [id, event] : events_)
    max_key = std::max(id.size(), max_key);

  report << pad_left(max_key, ev);

  report << pad_left<double_block>("-- time");
  report << pad_left<double_block - percent_reduce>("-- % time");
  report << pad_left<int_block>("-- count");
  report << pad_left<double_block>("-- average");
  report << pad_left<double_block>("-- min");
  report << pad_left<double_block>("-- max") << '\n';

  for (auto [id, event] : events_) {
    auto &times = event.intervals;

    if (event.started) // currently running timer
      times.push_back(duration_since(event.started));

    double const sum = std::accumulate(times.begin(), times.end(), 0.0);
    double const avg = sum / static_cast<double>(times.size());
    double const min = *std::min_element(times.begin(), times.end());
    double const max = *std::max_element(times.begin(), times.end());

    report << pad_left(max_key, id);

    report << pad_string(sum);
    report << pad_string_percent(sum, total_time);
    report << pad_string(times.size());
    report << pad_string(avg);
    report << pad_string(min);
    report << pad_string(max) << '\n';

    if (event.started)
      times.pop_back();
  }

  report << "\n";

  std::string const gf =  "-- Gflops/s --  ";
  max_key = std::max(max_key, gf.size());

  report << pad_left(max_key, gf) << pad_left<double_block>("-- average")
         << pad_left<double_block>("-- min") << pad_left<double_block>("-- max") << "\n";

  for (auto [id, event] : events_) {
    if (not event.gflops.empty()) {
      auto const &gflops = event.gflops;
      double const fsum = std::accumulate(gflops.begin(), gflops.end(), 0.0);
      double const min = *std::min_element(gflops.begin(), gflops.end());
      double const max = *std::max_element(gflops.begin(), gflops.end());

      report << pad_left(max_key, id);

      report << pad_string(fsum / gflops.size()) << pad_string(min) << pad_string(max) << '\n';
    }
  }

  return report.str();
}

} // namespace asgard::tools
