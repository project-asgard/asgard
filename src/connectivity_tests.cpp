#include "connectivity.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <string>
#include <vector>

TEST_CASE("one-dimensional connectivity", "[connectivity]")
{
  SECTION("simple test for indexing function")
  {
    // test some vals calc'ed by hand :)
    std::vector<int> const golds{0, 0, 524388};
    std::vector<int> const levels{0, 0, 20};
    std::vector<int> const cells{0, 500, 100};

    for (size_t i = 0; i < golds.size(); ++i)
    {
      REQUIRE(get_1d_index(levels[i], cells[i]) == golds[i]);
    }
  }
  SECTION("one-dimensional connectivity function, levels = 1")
  {
    int const num_levels = 1;
    fk::matrix<int> gold = {{1, 1}, {1, 1}};
    REQUIRE(connect_1d(num_levels) == gold);
  }
  SECTION("one-dimensional connectivity function, levels = 3")
  {
    int const num_levels = 3;
    //clang-format off
    fk::matrix<int> gold = {{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1},
                            {1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1},
                            {1, 1, 1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 1, 1, 0},
                            {1, 1, 1, 1, 0, 1, 1, 1}, {1, 1, 1, 1, 1, 0, 1, 1}};
    //clang-format on
    REQUIRE(connect_1d(num_levels) == gold);
  }
}
