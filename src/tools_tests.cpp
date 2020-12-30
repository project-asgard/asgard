#include "tests_general.hpp"
#include "tools.hpp"
#include <algorithm>
#include <random>
#include <sstream>

// function that does nothing productive, but takes some time...
// for testing the timer
double shuffle_random(int const num_items)
{
  tools::expect(num_items > 0);
  std::random_device rd;
  std::mt19937 mersenne_engine(rd());
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  auto const gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };

  std::vector<double> items(num_items);
  std::generate(items.begin(), items.end(), gen);
  std::shuffle(items.begin(), items.end(), mersenne_engine);
  return items[0];
}

static auto constexpr tol = 1e3;
TEST_CASE("test timer")
{
  tools::simple_timer timer;
  int const items_to_gen       = 100000;
  int const iterations         = 10;
  std::string const identifier = "waste_time";
  for (int i = 0; i < iterations; ++i)
  {
    timer.start(identifier);
    double const val = shuffle_random(items_to_gen);
    timer.stop(identifier);
    tools::expect(val > 0.0); // to avoid comp. warnings
  }
  std::string const report = timer.report();

  std::stringstream s1(report.substr(report.find("avg: ")));
  std::string s;
  s1 >> s;
  double avg;
  s1 >> avg;

  std::stringstream s2(report.substr(report.find("min: ")));
  s2 >> s;
  double min;
  s2 >> min;

  std::stringstream s3(report.substr(report.find("max: ")));
  s3 >> s;
  double max;
  s3 >> max;

  std::stringstream s4(report.substr(report.find("med: ")));
  s4 >> s;
  double med;
  s4 >> med;

  std::stringstream s5(report.substr(report.find("calls: ")));
  s5 >> s;
  int calls;
  s5 >> calls;

  auto const &times = timer.get_times(identifier);

  SECTION("avg")
  {
    double sum = 0.0;
    for (double const &time : times)
    {
      sum += time;
    }
    double const gold_average = sum / times.size();
    relaxed_fp_comparison(avg, gold_average, tol);
  }

  SECTION("min/max")
  {
    double gold_min = std::numeric_limits<double>::max();
    double gold_max = std::numeric_limits<double>::min();

    for (double const &time : times)
    {
      gold_min = time < gold_min ? time : gold_min;
      gold_max = time > gold_max ? time : gold_max;
    }

    relaxed_fp_comparison(max, gold_max, tol);
    relaxed_fp_comparison(min, gold_min, tol);
  }

  SECTION("med")
  {
    std::vector<double> time_copy(times);
    std::sort(time_copy.begin(), time_copy.end());
    auto const mid        = time_copy.size() / 2;
    double const gold_med = (time_copy.size() % 2 == 0)
                                ? (time_copy[mid] + time_copy[mid - 1]) / 2
                                : time_copy[mid];
    relaxed_fp_comparison(med, gold_med, tol);
  }

  SECTION("count") { REQUIRE(calls == static_cast<int>(times.size())); }
}
