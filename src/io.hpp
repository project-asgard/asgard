#pragma once

#include "pde.hpp"
#include "tensors.hpp"
#include "transformations.hpp"

// workaround for missing include issue with highfive
// clang-format off
#include <numeric>
#include <highfive/H5Easy.hpp>
// clang-format on
namespace asgard
{
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

template<typename P>
void write_output(PDE<P> const &pde, fk::vector<P> const &vec, P const time,
                  int const file_index,
                  std::string const output_dataset_name = "asgard")
{
  std::string const output_file_name =
      output_dataset_name + "_" + std::to_string(file_index) + ".h5";
  unsigned int vec_size = (unsigned int)vec.size();

  auto const dims = pde.get_dimensions();
  // TODO: FIX - assumes levels are const across dims
  static auto const nodes =
      gen_realspace_nodes(dims[0].get_degree(), dims[0].get_level(),
                          dims[0].domain_min, dims[0].domain_max);

  // TODO: Rewrite this entirely!
  HighFive::File file(output_file_name, HighFive::File::ReadWrite |
                                            HighFive::File::Create |
                                            HighFive::File::Truncate);

  H5Easy::DumpOptions opts;
  opts.setChunkSize(std::vector<hsize_t>{2});

  H5Easy::dump(file, "time", time);
  H5Easy::dump(file, "nodes", nodes.to_std());
  H5Easy::dump(file, "soln", vec.to_std(), opts);
}

} // namespace asgard
