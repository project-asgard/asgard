#include "io.hpp"

#include "tensors.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("highfive interface to HDF5", "[io]", double, float, int)
{
  SECTION("check it writes the correct file")
  {
    std::string const output_file_name("asgard_test.h5");
    std::string const output_dataset_name("asgard_test");

    // the golden values
    fk::vector<TestType> const gold_vec1{1, 2, 3, 4, 5};
    fk::vector<TestType> const gold_vec2{5, 6, 7, 8, 9};

    // setup output file and write initial condition
    auto output_dataset = initialize_output_file(gold_vec1);

    // write output to file
    update_output_file(output_dataset, gold_vec2);

    // now read back what we wrote out
    std::vector<std::vector<TestType>> read_data;
    output_dataset.read(read_data);

    auto const dataset_size = output_dataset.getDimensions();
    auto const vec1         = read_data[0];
    auto const vec2         = read_data[1];
    REQUIRE(vec1.size() == gold_vec1.size());
    REQUIRE(vec2.size() == gold_vec2.size());

    for (int i = 0; i < static_cast<int>(vec.size()); i++)
    {
      REQUIRE(vec1[i] == gold_vec1[i]);
      REQUIRE(vec2[i] == gold_vec2[i]);
    }
  }
}
