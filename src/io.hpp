#pragma once

#include "tensors.hpp"
#include <highfive/H5Easy.hpp>

std::string const output_file_name("asgard.h5");
std::string const output_dataset_name("asgard");

template<typename P>
HighFive::DataSet initialize_output_file(fk::vector<P> const &vec)
{
  // Cast vec.size() to a long unsigned int to make the warnings happy.
  // Why is fk::vector.size() returning a signed it anyway?

  unsigned int vec_size = (unsigned int)vec.size();

  // Open file object

  HighFive::File file(output_file_name, HighFive::File::ReadWrite |
                                            HighFive::File::Create |
                                            HighFive::File::Truncate);

  // Create dataspace

  HighFive::DataSpace dataspace = HighFive::DataSpace(
      {1, vec_size}, {HighFive::DataSpace::UNLIMITED, vec_size});

  // Use chunk

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
void update_output_file(HighFive::DataSet &dataset, fk::vector<P> &vec)
{
  // Cast vec.size() to a long unsigned int to make the warnings happy.
  // Why is fk::vector.size() returning a signed it anyway?

  unsigned int vec_size = (unsigned int)vec.size();

  // Get the size of the existing dataset

  auto dataset_size = dataset.getDimensions();

  // Resize in the time dimension by 1

  dataset.resize({dataset_size[0] + 1, dataset_size[1]});

  // Write the latest vec into the new row

  dataset.select({dataset_size[0], 0}, {1, vec_size}).write(vec.to_std());
}
