#include "asgard_tools.hpp"

#include "tests_general.hpp"

using namespace asgard;

// function that does nothing productive, but takes some time...
// for testing the timer
double shuffle_random(int const num_items)
{
  expect(num_items > 0);
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
struct time_event_tag
{};
struct simple_timer_tag
{};
TEMPLATE_TEST_CASE("test timer", "[timing test]", time_event_tag,
                   simple_timer_tag)
{
  tools::timer = tools::simple_timer();

  int const items_to_gen       = 100000;
  int const iterations         = 10;
  std::string const identifier = "waste_time";
  for (int i = 0; i < iterations; ++i)
  {
    if constexpr (std::is_same_v<TestType, simple_timer_tag>)
    {
      // testing direct calls to timer
      tools::timer.start(identifier);
      double const val = shuffle_random(items_to_gen);
      tools::timer.stop(identifier);
      expect(val > 0.0); // to avoid comp. warnings
    }
    else
    {
      // testing the sue of the
      tools::time_event timing(identifier);
      double const val = shuffle_random(items_to_gen);
      expect(val > 0.0); // to avoid comp. warnings
    }
  }
  std::string const report = tools::timer.report();

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

  auto const &times = tools::timer.get_times(identifier);

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

TEST_CASE("for-indexof testing", "[indexing testing]")
{
  std::vector<double> x(10);
  std::vector<int64_t> r;
  r.reserve(x.size());

  for (auto i : indexof(x))
  {
    static_assert(std::is_same_v<decltype(i), int64_t>);
    r.push_back(i);
  }

  REQUIRE(r.size() == x.size());
  for (int64_t i = 0; i < 10; i++)
    REQUIRE(r[i] == i);

  std::vector<int> ir;
  ir.reserve(8);

  for (auto i : indexof<int>(1, 6))
  {
    static_assert(std::is_same_v<decltype(i), int>);
    ir.push_back(i);
  }

  for (int i = 1; i < 6; i++)
    REQUIRE(ir[i - 1] == i);

  size_t s = 0;
  for (auto i : indexof<size_t>(x.size()))
  {
    static_assert(std::is_same_v<decltype(i), size_t>);
    s += i;
  }
  REQUIRE(s == 45);
}
