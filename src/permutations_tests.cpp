
#include "tests_general.hpp"

static auto const permutations_base_dir = gold_base_dir / "permutations";

using namespace asgard;

TEST_CASE("Permutations builders", "[permutations]")
{
  std::string const zero = "0";
  std::string const one  = "1";
  std::vector<int> const dims{1, 2, 4, 6};
  std::vector<int> const ns{1, 4, 6, 8};
  std::vector<bool> const ord_by_ns{false, true, false, true};

  SECTION("permutations leq")
  {
    std::string const out_base = "perm_leq_";
    for (int i = 0; i < static_cast<int>(dims.size()); ++i)
    {
      std::string const file_base = out_base + std::to_string(dims[i]) + "_" +
                                    std::to_string(ns[i]) + "_" +
                                    (ord_by_ns[i] ? one : zero);
      std::string const file_path  = file_base + ".dat";
      std::string const count_path = file_base + "_count.dat";
      fk::matrix<int> const gold =
          read_matrix_from_txt_file<int>(permutations_base_dir / file_path);
      REQUIRE(permutations::get_lequal(dims[i], ns[i], ord_by_ns[i]) == gold);
    }
  }

  SECTION("permutations max")
  {
    std::string const out_base = "perm_max_";
    for (int i = 0; i < static_cast<int>(dims.size()); ++i)
    {
      std::string const file_base = out_base + std::to_string(dims[i]) + "_" +
                                    std::to_string(ns[i]) + "_" +
                                    (ord_by_ns[i] ? one : zero);
      std::string const file_path  = file_base + ".dat";
      std::string const count_path = file_base + "_count.dat";
      fk::matrix<int> const gold =
          read_matrix_from_txt_file<int>(permutations_base_dir / file_path);
      REQUIRE(permutations::get_max(dims[i], ns[i], ord_by_ns[i]) == gold);
    }
  }
  SECTION("index leq max - small manually computed example")
  {
    // hand-computed example
    permutations::list_set const lists{{0, 1}, {0, 3}, {0, 1}, {2, 5}};
    int const max_sum = 5;
    int const max_val = 3;
    // clang-format off
    fk::matrix<int> const gold =
                 {{0, 0, 0, 0},
	    		  {1, 0, 0, 0},
				  {0, 1, 0, 0},
				  {0, 0, 1, 0},
				  {1, 0, 1, 0}};
    // clang-format on

    REQUIRE(permutations::get_leq_max_indices(lists, lists.size(), max_sum,
                                              max_val) == gold);
    REQUIRE(permutations::count_leq_max_indices(lists, lists.size(), max_sum,
                                                max_val) == gold.nrows());
  }

  SECTION("index leq max - matlab computed example")
  {
    std::string const gold_path  = "index_leq_max_4d_10s_4m.dat";
    std::string const count_path = "index_leq_max_4d_10s_4m_count.dat";

    fk::matrix<int> const gold = [=] {
      fk::matrix<int> indices =
          read_matrix_from_txt_file<int>(permutations_base_dir / gold_path);

      // output values are indices; must adjust for matlab 1-indexing
      std::transform(indices.begin(), indices.end(), indices.begin(),
                     [](int &elem) { return elem - 1; });
      return indices;
    }();
    int const count_gold = static_cast<int>(
        read_scalar_from_txt_file(permutations_base_dir / count_path));

    // clang-format off
    permutations::list_set const lists{{2, 3},
	    	 {0, 1, 2, 3, 4},
		   	 {0, 1, 2, 3},
		   	 {1, 2, 3, 4, 5}};
    // clang-format on
    int const max_sum = 10;
    int const max_val = 4;

    REQUIRE(permutations::get_leq_max_indices(lists, lists.size(), max_sum,
                                              max_val) == gold);
    REQUIRE(permutations::count_leq_max_indices(lists, lists.size(), max_sum,
                                                max_val) == count_gold);
  }
}

TEST_CASE("Non-uniform level permutations builders", "[permutations]")
{
  std::string const zero = "0";
  std::string const one  = "1";

  // clang-format off
  std::vector<fk::vector<int>> const test_levels = {
  {3, 3},
  {1, 4},
  {1, 5, 8},
  {10, 6, 9, 10},
  {2, 10, 1, 5, 4, 7}
  };
  // clang-format on

  SECTION("permutations leq")
  {
    std::string const out_base = "perm_leq_d_";

    for (int i = 0; i < static_cast<int>(test_levels.size()); ++i)
    {
      auto const sort             = (i + 1) % 2;
      std::string const file_base = out_base +
                                    std::to_string(test_levels[i].size()) +
                                    "_" + (sort ? one : zero);
      std::string const file_path  = file_base + ".dat";
      std::string const count_path = file_base + "_count.dat";

      auto const gold =
          read_matrix_from_txt_file<int>(permutations_base_dir / file_path);

      auto const max_level =
          *std::max_element(test_levels[i].begin(), test_levels[i].end());

      REQUIRE(permutations::get_lequal_multi(test_levels[i],
                                             test_levels[i].size(), max_level,
                                             sort) == gold);
    }
  }

  SECTION("permutations max")
  {
    std::string const out_base = "perm_max_d_";

    for (int i = 0; i < static_cast<int>(test_levels.size()); ++i)
    {
      auto const sort             = (i + 1) % 2;
      std::string const file_path = out_base +
                                    std::to_string(test_levels[i].size()) +
                                    "_" + (sort ? one : zero) + ".dat";

      auto const gold =
          read_matrix_from_txt_file<int>(permutations_base_dir / file_path);

      REQUIRE(permutations::get_max_multi(test_levels[i], test_levels[i].size(),
                                          sort) == gold);
    }
  }
}
