#include "asgard_test_macros.hpp"

#include <thread> // needed for "sleep"

using namespace asgard;

void test_indexof()
{
  current_test name_("for (auto i : indexof(vec))");

  std::vector<double> x(10);
  std::vector<int64_t> r;
  r.reserve(x.size());

  for (auto i : indexof(x))
  {
    static_assert(std::is_same_v<decltype(i), int64_t>);
    r.push_back(i);
  }

  tassert(r.size() == x.size());
  for (int64_t i = 0; i < 10; i++)
    tassert(r[i] == i);

  std::vector<int> ir;
  ir.reserve(8);

  for (auto i : indexof<int>(1, 6))
  {
    static_assert(std::is_same_v<decltype(i), int>);
    ir.push_back(i);
  }

  for (int i = 1; i < 6; i++)
    tassert(ir[i - 1] == i);

  size_t s = 0;
  for (auto i : indexof<size_t>(x.size()))
  {
    static_assert(std::is_same_v<decltype(i), size_t>);
    s += i;
  }
  tassert(s == 45);
}

void test_timer()
{
  current_test name_("timer testing");
  // the timer is intended to produce human-readable output
  // real testing is done by reading the output
  // check here that there are no crashes

  tools::timer.start("testing");

  auto start = tools::simple_timer::current_time();

  {
    auto session1 = tools::time_session("regulat session");
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    {
      auto session2 = tools::time_session("nested session");
      std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
  }

  double dur = tools::simple_timer::duration_since(start);
  tassert(dur >= 7.0); // must have waited above, keep this loose

  auto const ttime = tools::timer.stop("testing");
  ignore(ttime);
#ifdef ASGARD_USE_TIMER
  tassert(ttime >= 7.0); // must have waited above, keep this loose

  auto report = tools::timer.report();
  tassert(report.find("testing") < report.size());
  tassert(report.find("regulat session") < report.size());
  tassert(report.find("nested session") < report.size());
  tassert(report.find("100%") >= report.size());
#endif
}

int main(int argc, char **argv)
{
  libasgard_runtime running_(argc, argv);

  all_tests global_("misc tools");

  test_indexof();
  test_timer();

  return 0;
}
