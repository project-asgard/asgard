#include "timer.hpp"
#include <algorithm>
// FIXME use string format after C++20
// #include <format>
#include <numeric>
#include <sstream>

namespace timer
{
template<typename F, typename... Args>
auto recorder::run(F &&f, std::string const identifier, Args &&... args) ->
    typename std::enable_if<
        !std::is_same_v<decltype(f(std::forward<Args>(args)...)), void>,
        decltype(f(std::forward<Args>(args)...))>::type
{
  auto const beg = std::chrono::high_resolution_clock::now();
  auto const ret = std::forward<F>(f)(std::forward<Args>(args)...);
  auto const end = std::chrono::high_resolution_clock::now();
  auto const dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
  insert(identifier, dur);
  return ret;
}

template<typename F, typename... Args>
auto recorder::run(F &&f, std::string const identifier, Args &&... args) ->
    typename std::enable_if<
        std::is_same_v<decltype(f(std::forward<Args>(args)...)), void>,
        void>::type
{
  auto const beg = std::chrono::high_resolution_clock::now();
  std::forward<F>(f)(std::forward<Args>(args)...);
  auto const end = std::chrono::high_resolution_clock::now();
  auto const dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
  insert(identifier, dur);
}

std::string recorder::report() const
{
  std::ostringstream report;
  for (auto const [id, times] : id_to_times)
  {
    auto const avg =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    auto const min = *std::min(times.begin(), times.end());
    auto const max = *std::max(times.begin(), times.end());
    // calculate median
    auto const middle_it = times.begin() + times.size() / 2;
    std::nth_element(times.begin(), middle_it, times.end());
    auto const med =
        times.size() % 2 == 0
            ? (*std::max_element(times.begin(), middle_it) + *middle_it) / 2
            : *middle_it;

    report << id << "- avg: " << avg << " min: " << min << " max: " << max
           << " med: " << med << " calls: " << times.size() << '\n';
  }
  return report.str();
}

} // namespace timer
