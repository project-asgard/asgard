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
void generate_initial_moments(
    PDE<P> &pde, options const &program_opts,
    adapt::distributed_grid<P> const &adaptive_grid,
    asgard::basis::wavelet_transform<P, resource::host> const &transformer,
    fk::vector<P> const &initial_condition)
{
  // create 1D version of PDE and element table for wavelet->realspace
  // mappings
  PDE pde_1d = PDE(pde, PDE<P>::extract_dim0);
  adapt::distributed_grid adaptive_grid_1d(pde_1d, program_opts);

  // Create workspace for wavelet transform
  auto const dense_size = dense_space_size(pde_1d);
  fk::vector<P, mem_type::owner, resource::host> workspace(dense_size * 2);
  std::array<fk::vector<P, mem_type::view, resource::host>, 2> tmp_workspace = {
      fk::vector<P, mem_type::view, resource::host>(workspace, 0,
                                                    dense_size - 1),
      fk::vector<P, mem_type::view, resource::host>(workspace, dense_size,
                                                    dense_size * 2 - 1)};

  for (size_t i = 0; i < pde.moments.size(); ++i)
  {
    fk::vector<P> moment_vec(dense_size);
    pde.moments[i].createMomentReducedMatrix(pde, adaptive_grid.get_table());
    fm::gemv(pde.moments[i].get_moment_matrix(), initial_condition, moment_vec);
    pde.moments[i].create_realspace_moment(pde_1d, moment_vec,
                                           adaptive_grid_1d.get_table(),
                                           transformer, tmp_workspace);
  }
}

template<typename P>
void write_output(PDE<P> const &pde, parser const &cli_input,
                  fk::vector<P> const &vec, P const time, int const file_index,
                  std::string const output_dataset_name = "asgard")
{
  std::string const output_file_name =
      output_dataset_name + "_" + std::to_string(file_index) + ".h5";

  // TODO: Rewrite this entirely!
  HighFive::File file(output_file_name, HighFive::File::ReadWrite |
                                            HighFive::File::Create |
                                            HighFive::File::Truncate);

  H5Easy::DumpOptions opts;
  opts.setChunkSize(std::vector<hsize_t>{2});

  H5Easy::dump(file, "pde", cli_input.get_pde_string());
  H5Easy::dump(file, "degree", cli_input.get_degree());
  H5Easy::dump(file, "dt", cli_input.get_dt());
  H5Easy::dump(file, "time", time);
  H5Easy::dump(file, "ndims", pde.num_dims);
  H5Easy::dump(file, "max_level", pde.max_level);
  auto const dims = pde.get_dimensions();
  for (size_t dim = 0; dim < dims.size(); ++dim)
  {
    auto const nodes =
        gen_realspace_nodes(dims[dim].get_degree(), dims[dim].get_level(),
                            dims[dim].domain_min, dims[dim].domain_max);
    H5Easy::dump(file, "nodes" + std::to_string(dim), nodes.to_std());
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_level",
                 dims[dim].get_level());
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_min",
                 dims[dim].domain_min);
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_max",
                 dims[dim].domain_max);
  }

  H5Easy::dump(file, "soln", vec.to_std(), opts);

  // save E field
  H5Easy::dump(file, "Efield", pde.E_field.to_std(), opts);

  if (pde.moments.size() > 0)
  {
    // save realspace moments
    H5Easy::dump(file, "nmoments", pde.moments.size());
    for (size_t i = 0; i < pde.moments.size(); ++i)
    {
      H5Easy::dump(file, "moment" + std::to_string(i),
                   pde.moments[i].get_realspace_moment().to_std(), opts);
    }
  }

  // save gmres error and iteration counts
  for (size_t i = 0; i < pde.gmres_outputs.size(); ++i)
  {
    H5Easy::dump(file, "gmres" + std::to_string(i) + "_err",
                 pde.gmres_outputs[i].error, opts);
    H5Easy::dump(file, "gmres" + std::to_string(i) + "_num_outer",
                 pde.gmres_outputs[i].outer_iter, opts);
    H5Easy::dump(file, "gmres" + std::to_string(i) + "_num_inner",
                 pde.gmres_outputs[i].inner_iter, opts);
  }
}

} // namespace asgard
