#include "matlab_utilities.hpp"
#include "permutations.hpp"
#include "tests_general.hpp"
#include <vector>

TEST_CASE("Permutations builders", "[permutations]")
{
  std::string const zero = "0";
  std::string const one  = "1";
  std::vector<int> const dims{1, 2, 4, 6};
  std::vector<int> const ns{1, 4, 6, 8};
  std::vector<bool> const ord_by_ns{false, true, false, true};

  SECTION("permutations eq")
  {
    std::string const out_base =
        "../testing/generated-inputs/permutations/perm_eq_";
    for (int i = 0; i < static_cast<int>(dims.size()); ++i)
    {
      std::string const file_base = out_base + std::to_string(dims[i]) + "_" +
                                    std::to_string(ns[i]) + "_" +
                                    (ord_by_ns[i] ? one : zero);
      std::string const file_path  = file_base + ".dat";
      std::string const count_path = file_base + "_count.dat";

      // when i == 0, these arguments yield a scalar
      fk::matrix<int> const gold = [=] {
        if (i == 0)
        {
          int const gold_val =
              static_cast<int>(read_scalar_from_txt_file(file_path));
          return fk::matrix<int>{{gold_val}};
        }
        return fk::matrix<int>(read_matrix_from_txt_file(file_path));
      }();

      int const count_gold =
          static_cast<int>(read_scalar_from_txt_file(count_path));
      REQUIRE(get_eq_permutations(dims[i], ns[i], ord_by_ns[i]) == gold);
      REQUIRE(count_eq_permutations(dims[i], ns[i]) == count_gold);
    }
  }

  SECTION("permutations leq")
  {
    std::string const out_base =
        "../testing/generated-inputs/permutations/perm_leq_";
    for (int i = 0; i < static_cast<int>(dims.size()); ++i)
    {
      std::string const file_base = out_base + std::to_string(dims[i]) + "_" +
                                    std::to_string(ns[i]) + "_" +
                                    (ord_by_ns[i] ? one : zero);
      std::string const file_path  = file_base + ".dat";
      std::string const count_path = file_base + "_count.dat";
      fk::matrix<int> const gold =
          fk::matrix<int>(read_matrix_from_txt_file(file_path));
      int const count_gold =
          static_cast<int>(read_scalar_from_txt_file(count_path));
      REQUIRE(get_leq_permutations(dims[i], ns[i], ord_by_ns[i]) == gold);
      REQUIRE(count_leq_permutations(dims[i], ns[i]) == count_gold);
    }
  }

  SECTION("permutations max")
  {
    std::string const out_base =
        "../testing/generated-inputs/permutations/perm_max_";
    for (int i = 0; i < static_cast<int>(dims.size()); ++i)
    {
      std::string const file_base = out_base + std::to_string(dims[i]) + "_" +
                                    std::to_string(ns[i]) + "_" +
                                    (ord_by_ns[i] ? one : zero);
      std::string const file_path  = file_base + ".dat";
      std::string const count_path = file_base + "_count.dat";
      fk::matrix<int> const gold =
          fk::matrix<int>(read_matrix_from_txt_file(file_path));
      int const count_gold =
          static_cast<int>(read_scalar_from_txt_file(count_path));
      REQUIRE(get_max_permutations(dims[i], ns[i], ord_by_ns[i]) == gold);
      REQUIRE(count_max_permutations(dims[i], ns[i]) == count_gold);
    }
  }
  SECTION("index leq max - small manually computed example")
  {
    // hand-computed example
    list_set const lists{{0, 1}, {0, 3}, {0, 1}, {2, 5}};
    int const max_sum = 5;
    int const max_val = 3;
    // clang-format off
    fk::matrix<int> const gold = {{0, 0, 0, 0}, 
	    		          {1, 0, 0, 0}, 
				  {0, 1, 0, 0}, 
				  {0, 0, 1, 0}, 
				  {1, 0, 1, 0}};
    // clang-format on
    REQUIRE(get_leq_max_indices(lists, lists.size(), max_sum, max_val) == gold);
    REQUIRE(count_leq_max_indices(lists, lists.size(), max_sum, max_val) ==
            gold.nrows());
  }

  SECTION("index leq max - matlab computed example")
  {
    std::string const gold_path =
        "../testing/generated-inputs/permutations/index_leq_max_4d_10s_4m.dat";
    std::string const count_path = "../testing/generated-inputs/permutations/"
                                   "index_leq_max_4d_10s_4m_count.dat";

    fk::matrix<int> const gold = [=] {
      fk::matrix<int> indices =
          fk::matrix<int>(read_matrix_from_txt_file(gold_path));

      // output values are indices; must adjust for matlab 1-indexing
      std::transform(indices.begin(), indices.end(), indices.begin(),
                     [](int &elem) { return elem - 1; });
      return indices;
    }();
    int const count_gold =
        static_cast<int>(read_scalar_from_txt_file(count_path));

    // clang-format off
    list_set const lists{{2, 3}, 
	    	   	 {0, 1, 2, 3, 4}, 
		   	 {0, 1, 2, 3}, 
		   	 {1, 2, 3, 4, 5}};
    // clang-format on
    int const max_sum = 10;
    int const max_val = 4;

    REQUIRE(get_leq_max_indices(lists, lists.size(), max_sum, max_val) == gold);
    REQUIRE(count_leq_max_indices(lists, lists.size(), max_sum, max_val) ==
            count_gold);
  }
}
