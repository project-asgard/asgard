#include "connectivity.hpp"

#include "element_table.hpp"
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
    REQUIRE(make_1d_connectivity(num_levels) == gold);
  }
  SECTION("one-dimensional connectivity function, levels = 3")
  {
    int const num_levels = 3;
    // clang-format off
    fk::matrix<int> gold = {{1, 1, 1, 1, 1, 1, 1, 1}, 
	    		    {1, 1, 1, 1, 1, 1, 1, 1},
                            {1, 1, 1, 1, 1, 1, 1, 1}, 
			    {1, 1, 1, 1, 1, 1, 1, 1},
                            {1, 1, 1, 1, 1, 1, 0, 1}, 
			    {1, 1, 1, 1, 1, 1, 1, 0},
                            {1, 1, 1, 1, 0, 1, 1, 1}, 
			    {1, 1, 1, 1, 1, 0, 1, 1}};
    // clang-format on
    REQUIRE(make_1d_connectivity(num_levels) == gold);
  }
}

TEST_CASE("n-dimensional connectivity", "[connectivity]")
{
  SECTION("2 dimensions, level 2, sparse grid")
  {
    list_set gold = list_set(16, {0, 1, 2, 3, 4, 5, 6, 7});

    int const levels = 2;
    int const dims   = 2;
    Options o        = make_options({"-l", std::to_string(levels)});
    element_table t(o, dims);

    list_set connectivity = make_connectivity(t, dims, levels, levels);
    REQUIRE(connectivity == gold);
  }
  SECTION("2 dimensions, level 3, sparse grid")
  {
     
    // clang-format off
    list_set gold = lis	  

    // clang-format off
    int const levels = 3;
    int const dims   = 2;
    Options o        = make_options({"-l", std::to_string(levels), "-f"});
    element_table t(o, dims);
  }
  SECTION("3 dimensions, level 2, sparse grid")
  {
    int const levels = 2;
    int const dims   = 2;
    Options o        = make_options({"-l", std::to_string(levels)});
    element_table t(o, dims);
  }
}
