#include "tests_general.hpp"

// workaround for missing include issue with highfive
#include <highfive/H5Easy.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5DataSpace.hpp>

using namespace asgard;

template<typename P>
HighFive::DataSet
initialize_output_file(fk::vector<P> const &vec,
                       std::string const output_dataset_name = "asgard")
{
  std::string const output_file_name = output_dataset_name + ".h5";
  unsigned int vec_size              = (unsigned int)vec.size();

  // Open file object
  HighFive::File file(output_file_name, HighFive::File::ReadWrite |
                                            HighFive::File::Create |
                                            HighFive::File::Truncate);

  // Create dataspace
  HighFive::DataSpace dataspace = HighFive::DataSpace(
      {1, vec_size}, {HighFive::DataSpace::UNLIMITED, vec_size});

  // Use chunking
  HighFive::DataSetCreateProps props;
  props.add(HighFive::Chunking(std::vector<hsize_t>{2, 2}));

  // Create dataset
  HighFive::DataSet dataset = file.createDataSet(
      output_dataset_name, dataspace, HighFive::AtomicType<P>(), props);

  // Write initial contion to t=0 slice of output file
  dataset.select({0, 0}, {1, vec_size}).write(vec.to_std());

  return dataset;
}

template<typename P>
void update_output_file(HighFive::DataSet &dataset, fk::vector<P> const &vec,
                        std::string const output_dataset_name = "asgard")
{
  std::string const output_file_name = output_dataset_name + ".h5";
  unsigned int vec_size              = (unsigned int)vec.size();

  // Get the size of the existing dataset
  auto dataset_size = dataset.getDimensions();
  // Resize in the time dimension by 1
  dataset.resize({dataset_size[0] + 1, dataset_size[1]});
  // Write the latest vec into the new row
  dataset.select({dataset_size[0], 0}, {1, vec_size}).write(vec.to_std());
}

TEMPLATE_TEST_CASE("highfive interface to HDF5", "[io]", test_precs, int)
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
    REQUIRE(static_cast<int>(vec1.size()) == gold_vec1.size());
    REQUIRE(static_cast<int>(vec2.size()) == gold_vec2.size());

    for (int i = 0; i < static_cast<int>(vec1.size()); i++)
    {
      REQUIRE(vec1[i] == gold_vec1(i));
      REQUIRE(vec2[i] == gold_vec2(i));
    }
  }
}
