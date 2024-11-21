#include "asgard_tools.hpp"

#include "tests_general.hpp"

#include <thread>

using namespace asgard;

TEST_CASE("test timer", "[timing test]")
{
  // the timer is intended to produce human-readable output
  // real testing is done by reading the output
  // check here that there are no crashes

  tools::timer.start("testing");

  auto start = tools::simple_timer::current_time();

  {
    auto session = tools::time_session("regulat session");
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    {
      auto session = tools::time_session("nested session");
      std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
  }

  double dur = tools::simple_timer::duration_since(start);
  REQUIRE(dur >= 7.0); // must have waited above, keep this loose

  auto const ttime = tools::timer.stop("testing");
  ignore(ttime);
#ifdef ASGARD_USE_TIMER
  REQUIRE(ttime >= 7.0); // must have waited above, keep this loose

  auto report = tools::timer.report();
  REQUIRE(report.find("testing") < report.size());
  REQUIRE(report.find("regulat session") < report.size());
  REQUIRE(report.find("nested session") < report.size());
  REQUIRE(report.find("100%") >= report.size());
#endif
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
