#include "tests_general.hpp"

static auto const boundary_conditions_base_dir =
    gold_base_dir / "boundary_conditions";

using namespace asgard;
using namespace asgard::boundary_conditions;

int main(int argc, char *argv[])
{
  initialize_distribution();

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

TEMPLATE_TEST_CASE("no-test", "[none]", test_precs)
{
  REQUIRE(true);
}
