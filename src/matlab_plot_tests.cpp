#include "matlab_plot.hpp"

#include "tests_general.hpp"
#include <string_view>

const static std::string ml_plot_tag("[matlab_plot]");

TEST_CASE("create matlab session", ml_plot_tag)
{
  matlab_plot ml;

  SECTION("using sync connect") { REQUIRE_NOTHROW(ml.connect()); }

  SECTION("starting new with no cmd line opts") { REQUIRE_NOTHROW(ml.start()); }

  REQUIRE(ml.is_open());

  ml.close();
}

// TODO: bad global.. need to fix this, but don't want to restart matlab for
// each test moving this to a singleton class might be slightly better
static matlab_plot *ml = nullptr;

matlab_plot *get_session(matlab_plot *instance)
{
  if (!instance)
  {
    instance = new matlab_plot();
  }
  if (!instance->is_open())
  {
    instance->start();
  }
  return instance;
}

TEST_CASE("creating scalar params", ml_plot_tag)
{
  ml = get_session(ml);

  SECTION("string")
  {
    std::string test("test string");
    REQUIRE_NOTHROW(ml->add_param(test));
  }

  SECTION("integer")
  {
    int test = 25;
    REQUIRE_NOTHROW(ml->add_param(test));
  }

  ml->reset_params();
}

TEMPLATE_TEST_CASE("creating vector params", ml_plot_tag, float, double)
{
  SECTION("fk::vector") {}

  SECTION("fk::matrix") {}
}

// TODO: The testing data for the following two test cases needs to be moved to
// a .dat file
TEMPLATE_TEST_CASE("generate plotting nodes", ml_plot_tag, float, double)
{
  SECTION("continuity2d, lev = 2") {}

  SECTION("continuity2d, lev = 2, full grid") {}

  SECTION("continuity2d, lev = 4") {}

  SECTION("continuity2d, lev = 4, full grid") {}

  SECTION("diffusion2d") {}
}

TEMPLATE_TEST_CASE("generate element coords for plotting", ml_plot_tag, float,
                   double)
{
  SECTION("continuity1d") {}

  SECTION("continuity2d") {}

  SECTION("diffusion2d") {}
}

// This might be a problem if test ordering is not guaranteed..
TEST_CASE("close session")
{
  ml = get_session(ml);

  REQUIRE(ml->is_open());

  REQUIRE_NOTHROW(ml->close());
}