#include "element_table.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <string>

TEST_CASE("element table constructor/accessors/size", "[element_table]")
{
  std::string out_base =
      "../testing/generated-inputs/element_table/element_table";

  SECTION("1D element table")
  {
    int const levels = 1;
    int const dims   = 1;
    options o        = make_options({"-l", std::to_string(levels)});
    element_table<int> t(o, dims);
    std::string test_base = out_base + "_1_1_SG";
    std::string file_path = test_base + ".dat";
    auto gold = fk::matrix<int>(read_matrix_from_txt_file(file_path));
    for (auto i = 0; i < t.size(); ++i)
    {
      fk::vector<int> gold_coords =
          gold.extract_submatrix(i, 0, 1, gold.ncols());

      REQUIRE(t.get_coords(i) == gold_coords);
      REQUIRE(t.get_index(gold_coords) == i);
    }
    REQUIRE(t.size() == 2);
  }

  SECTION("2D element table")
  {
    int const levels_2 = 3;
    int const dims_2   = 2;
    options o_2        = make_options({"-l", std::to_string(levels_2)});
    element_table<int> t_2(o_2, dims_2);
    //for(auto it = t_2.forward_table.cbegin(); it != t_2.forward_table.cend(); ++it)
    //{
    //  fk::vector<int> first = it->first;
    //  first.print("first");
    //  std::cout << it->second  << "\n";
    //}
    //for (std::vector<fk::vector<int>>::iterator it = t_2.reverse_table.begin() ; it != t_2.reverse_table.end(); ++it)
    //{
    //  fk::vector<int> thisvec =  *it;
    //  thisvec.print("thisvec");
    //}
    std::string test_base = out_base + "_2_3_SG";
    std::string file_path = test_base + ".dat";
    auto gold = fk::matrix<int>(read_matrix_from_txt_file(file_path));
    
    std::string test_base_ind = out_base + "_index_2_3_SG";
    std::string file_path_ind = test_base_ind + ".dat";
    //std::cout << file_path_ind << std::endl;
    auto gold_ind = fk::matrix<int>(read_matrix_from_txt_file(file_path_ind));
    //gold.print("gold");
    //gold_ind.print("gold_ind");
    for (auto i = 0; i < t_2.size(); ++i)
    {
      fk::vector<int> gold_coords =
          gold.extract_submatrix(i, 0, 1, gold.ncols());
      //std::cout << "t2.get_coods and gold_coords " <<  std::endl;
      fk::vector<int> t2coords= t_2.get_coords(i);
      //t2coords.print("t2coords");
      //gold_coords.print("gold_coords");
      REQUIRE(t_2.get_coords(i) == gold_coords);
      int t2getinds= t_2.get_index(gold_coords);
      fk::vector<int> indices = gold_ind.extract_submatrix(0,i,1,1);
      int this_ind = indices(0);
      //std::cout << "t2.get_index and i "<< t2getinds << " "  << this_ind << std::endl;
      REQUIRE(t_2.get_index(gold_coords) == this_ind);
    }
    //for (auto i = 0; i < t_2.size(); ++i)
    //{
    //  fk::vector<int> t_3 = t_2.get_coords(i);
    //  t_3.print("t3");
    //}
    REQUIRE(t_2.size() == 20);
  }

  SECTION("3D element table")
  {
    int const levels_3 = 4;
    int const dims_3   = 3;
    // test full grid
    options o_3 = make_options({"-l", std::to_string(levels_3), "-f"});
    element_table<int> t_3(o_3, dims_3);
    std::string test_base = out_base + "_3_4_FG";
    std::string file_path = test_base + ".dat";
    
    std::string test_base_ind = out_base + "_index_3_4_FG";
    std::string file_path_ind = test_base_ind + ".dat";
    
    auto gold = fk::matrix<int>(read_matrix_from_txt_file(file_path));
    auto gold_ind = fk::matrix<int>(read_matrix_from_txt_file(file_path_ind));
    
    for (auto i = 0; i < t_3.size(); ++i)
    {
      fk::vector<int> gold_coords =
          gold.extract_submatrix(i, 0, 1, gold.ncols());
      
      fk::vector<int> indices = gold_ind.extract_submatrix(0,i,1,1);
      int this_ind = indices(0);

      REQUIRE(t_3.get_coords(i) == gold_coords);
      REQUIRE(t_3.get_index(gold_coords) == this_ind);
    }
    REQUIRE(t_3.size() == 4096);
  }
}

TEST_CASE("Static helper - cell builder", "[element_table]")
{
  element_table<int> t(
      make_options({"-l", std::to_string(3), "-d", std::to_string(2)}), 1);

  SECTION("cell index set builder")
  {
    std::vector<fk::vector<int>> levels_set = {{1}, {1, 2}, {2, 1}, {2, 3}};

    std::vector<fk::matrix<int>> gold_set = {
        {{0}},
        {{0, 0}, {0, 1}},
        {{0, 0}, {1, 0}},
        {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}}};

    for (size_t i = 0; i < gold_set.size(); ++i)
    {
      REQUIRE(t.get_cell_index_set(levels_set[i]) == gold_set[i]);
    }
  }
}
