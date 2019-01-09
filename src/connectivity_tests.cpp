#include "connectivity.hpp"

#include "element_table.hpp"
#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <numeric>
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
    list_set gold = list_set(8, {0, 1, 2, 3, 4, 5, 6, 7});

    int const levels = 2;
    int const dims   = 2;
    Options o        = make_options({"-l", std::to_string(levels)});
    element_table t(o, dims);

    list_set connectivity = make_connectivity(t, dims, levels, levels);
    REQUIRE(connectivity == gold);
  }
  SECTION("2 dimensions, level 2, full grid")
  {
    list_set gold =
        list_set(16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    int const levels = 2;
    int const dims   = 2;
    Options o        = make_options({"-l", std::to_string(levels), "-f"});
    element_table t(o, dims);

    list_set connectivity = make_connectivity(t, dims, levels * dims, levels);
    REQUIRE(connectivity == gold);
  }
  SECTION("3 dimensions, level 3, sparse grid")
  {
    int const num_elements = 38;
    std::vector<int> fully_connected(num_elements);
    std::iota(fully_connected.begin(), fully_connected.end(), 0);
    list_set gold;
    for (auto i = 0; i < num_elements; ++i)
    {
      std::vector<int> element_i = fully_connected;
      auto start                 = element_i.begin();
      if (i == 4)
      {
        element_i.erase(start + 6);
      }
      else if (i == 5)
      {
        element_i.erase(start + 7);
      }
      else if (i == 6)
      {
        element_i.erase(start + 4);
      }
      else if (i == 7)
      {
        element_i.erase(start + 5);
      }
      else if (i == 16)
      {
        element_i.erase(start + 18);
      }
      else if (i == 17)
      {
        element_i.erase(start + 19);
      }
      else if (i == 18)
      {
        element_i.erase(start + 16);
      }
      else if (i == 19)
      {
        element_i.erase(start + 17);
      }
      else if (i == 34)
      {
        element_i.erase(start + 36);
      }
      else if (i == 35)
      {
        element_i.erase(start + 37);
      }
      else if (i == 36)
      {
        element_i.erase(start + 34);
      }
      else if (i == 37)
      {
        element_i.erase(start + 35);
      }
      gold.push_back(element_i);
    }
    int const levels = 3;
    int const dims   = 3;
    Options o        = make_options({"-l", std::to_string(levels)});
    element_table t(o, dims);
  }
}
