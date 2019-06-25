#include "io.hpp"

#include "tensors.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("highfive interface to HDF5", "[io]", double, float, int)
{
  SECTION("check it writes the correct file")
  {
    // One problem with this test is that it will overwrite any existing
    // "arsgard.h5" file in the present directory.

    std::string const output_file_name("asgard.h5");
    std::string const output_dataset_name("asgard");

    // the golden values

    std::vector<TestType> gold_svec1{1, 2, 3, 4, 5};
    std::vector<TestType> gold_svec2{5, 6, 7, 8, 9};

    fk::vector<TestType> gold_vec1 = gold_svec1;
    fk::vector<TestType> gold_vec2 = gold_svec2;

    // setup output file and write initial condition

    auto output_dataset = initialize_output_file(gold_vec1);

    // write output to file

    update_output_file(output_dataset, gold_vec2);

    // now read back what we wrote out

    std::vector<std::vector<TestType>> read_data;
    output_dataset.read(read_data);

    auto dataset_size = output_dataset.getDimensions();
    auto vec1         = read_data[0];
    auto vec2         = read_data[1];
    for (int i = 0; i <= 4; i++)
    {
      REQUIRE(vec1[i] == gold_svec1[i]);
      REQUIRE(vec2[i] == gold_svec2[i]);
    }
  }
}
