#pragma once

#include "tensors.hpp"

// workaround for missing include issue with highfive
// clang-format off
#if defined(ASGARD_IO_HIGHFIVE) 
#include <numeric>
#include <highfive/H5Easy.hpp>
#endif

// clang-format on

#if defined(ASGARD_IO_MATLAB_DIR)
#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"

template<typename P>
class matlab_engine
{
public:
  matlab_engine();

  ~matlab_engine();

  void send_to_matlab(fk::vector<P> const &v);

private:
  std::string matlab_solution_name;
  std::string matlab_tmp_var_name;

  matlab::data::ArrayFactory array_factory;

  std::unique_ptr<matlab::engine::MATLABEngine> engine;

  int time_step_index;
};
#endif

#if defined(ASGARD_IO_HIGHFIVE)
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
#endif
