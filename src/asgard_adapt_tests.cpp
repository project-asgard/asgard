
#include "tests_general.hpp"

static auto constexpr adapt_threshold = 1e-4;

static auto const adapt_base_dir = gold_base_dir / "adapt";

using namespace asgard;

struct distribution_test_init
{
  void set_my_rank(const int rank) { my_rank = rank; }
  void set_num_ranks(const int size) { num_ranks = size; }
  int get_my_rank() const { return my_rank; }
  int get_num_ranks() const { return num_ranks; }

private:
  int my_rank;
  int num_ranks;
};
static distribution_test_init distrib_test_info;

int main(int argc, char *argv[])
{
  auto const [rank, total_ranks] = initialize_distribution();
  distrib_test_info.set_my_rank(rank);
  distrib_test_info.set_num_ranks(total_ranks);

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

TEMPLATE_TEST_CASE("no-test", "[none]", test_precs)
{
  REQUIRE(true);
}
