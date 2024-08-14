#pragma once
#include "moment.hpp"
#include "solver.hpp"

// workaround for missing include issue with highfive
// clang-format off
#include <highfive/H5Easy.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5DataSpace.hpp>
// clang-format on

namespace asgard
{
template<typename P>
HighFive::CompoundType create_timing_stats()
{
  return {{"avg", HighFive::create_datatype<double>()},
          {"min", HighFive::create_datatype<double>()},
          {"max", HighFive::create_datatype<double>()},
          {"med", HighFive::create_datatype<double>()},
          {"gflops", HighFive::create_datatype<double>()},
          {"ncalls", HighFive::create_datatype<size_t>()}};
}
} // namespace asgard

HIGHFIVE_REGISTER_TYPE(asgard::tools::timing_stats,
                       asgard::create_timing_stats<double>)

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
    PDE<P> &pde, std::vector<moment<P>> &moments,
    adapt::distributed_grid<P> const &adaptive_grid,
    asgard::basis::wavelet_transform<P, resource::host> const &transformer,
    fk::vector<P> const &initial_condition)
{
  // create 1D version of PDE and element table for wavelet->realspace
  // mappings
  PDE pde_1d = PDE(pde, PDE<P>::extract_dim0);
  adapt::distributed_grid adaptive_grid_1d(pde_1d);

  // Create workspace for wavelet transform
  int const dense_size = dense_space_size(pde_1d);
  int quad_dense_size  = 1;
  auto const &dims     = pde_1d.get_dimensions();
  for (size_t i = 0; i < dims.size(); i++)
  {
    quad_dense_size *=
        asgard::dense_dim_size(ASGARD_NUM_QUADRATURE, dims[i].get_level());
  }

  fk::vector<P, mem_type::owner, resource::host> workspace(quad_dense_size * 2);
  std::array<fk::vector<P, mem_type::view, resource::host>, 2> tmp_workspace = {
      fk::vector<P, mem_type::view, resource::host>(workspace, 0,
                                                    quad_dense_size - 1),
      fk::vector<P, mem_type::view, resource::host>(workspace, quad_dense_size,
                                                    quad_dense_size * 2 - 1)};

#ifdef ASGARD_USE_CUDA
  fk::vector<P, mem_type::owner, resource::device> initial_condition_d =
      initial_condition.clone_onto_device();
#endif
  for (auto i : indexof(moments))
  {
    moments[i].createMomentReducedMatrix(pde, adaptive_grid.get_table());
#ifdef ASGARD_USE_CUDA
    fk::vector<P, mem_type::owner, resource::device> moment_vec(dense_size);

    fm::sparse_gemv(moments[i].get_moment_matrix_dev(), initial_condition_d,
                    moment_vec);
#else
    fk::vector<P, mem_type::owner, resource::host> moment_vec(dense_size);

    fm::sparse_gemv(moments[i].get_moment_matrix_dev(), initial_condition,
                    moment_vec);
#endif
    moments[i].create_realspace_moment(pde_1d, moment_vec,
                                       adaptive_grid_1d.get_table(),
                                       transformer, tmp_workspace);
  }
}

// the method expects either root or fixed name, one must be empty and one not
// the root is appended with step-number and .h5 extension
// the fixed filename is used "as-is" without any changes
template<typename P>
void write_output(PDE<P> const &pde, std::vector<moment<P>> const &moments,
                  fk::vector<P> const &vec, P const time, int const file_index,
                  int const dof, elements::table const &hash_table,
                  std::string const output_dataset_root  = "asgard",
                  std::string const output_dataset_fixed = "")
{
  tools::timer.start("write_output");

  expect(not output_dataset_root.empty() or not output_dataset_fixed.empty());

  std::string const output_file_name = [&]()
      -> std::string {
    if (output_dataset_root.empty())
    {
      expect(not output_dataset_fixed.empty());
      return output_dataset_fixed;
    }
    else
    {
      expect(output_dataset_fixed.empty());
      return output_dataset_root + "_" + std::to_string(file_index) + ".h5";
    }
  }();

  // TODO: Rewrite this entirely!
  HighFive::File file(output_file_name, HighFive::File::ReadWrite |
                                            HighFive::File::Create |
                                            HighFive::File::Truncate);

  H5Easy::DumpOptions opts;
  opts.setChunkSize(std::vector<hsize_t>{2});

  // TODO: needs to be checked further based on problem sizes
  HighFive::DataSetCreateProps plist;
  // just a temporary hack
  if (hash_table.get_active_table().size() <= 32)
    plist.add(HighFive::Chunking(hsize_t{4}));
  else if (hash_table.get_active_table().size() <= 64)
    plist.add(HighFive::Chunking(hsize_t{32}));
  else
    plist.add(HighFive::Chunking(hsize_t{64}));
  plist.add(HighFive::Deflate(9));

  auto const &options = pde.options();

  H5Easy::dump(file, "title", options.title);
  H5Easy::dump(file, "subtitle", options.subtitle);

  auto const dims = pde.get_dimensions();
  H5Easy::dump(file, "pde", options.pde_choice ? static_cast<int>(options.pde_choice.value()) : -1);
  H5Easy::dump(file, "degree", dims[0].get_degree());
  H5Easy::dump(file, "dt", pde.get_dt());
  H5Easy::dump(file, "time", time);
  H5Easy::dump(file, "ndims", pde.num_dims());
  H5Easy::dump(file, "max_levels", options.max_levels);
  H5Easy::dump(file, "dof", dof);
  // H5Easy::dump(file, "cli", cli_input.cli_opts); // seems too much
  for (size_t dim = 0; dim < dims.size(); ++dim)
  {
    auto const nodes =
        gen_realspace_nodes(dims[dim].get_degree(), dims[dim].get_level(),
                            dims[dim].domain_min, dims[dim].domain_max);
    file.createDataSet<P>(
            "nodes" + std::to_string(dim),
            HighFive::DataSpace({static_cast<size_t>(nodes.size())}))
        .write_raw(nodes.data());
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_level",
                 dims[dim].get_level());
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_min",
                 dims[dim].domain_min);
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_max",
                 dims[dim].domain_max);
  }

  auto &elements = hash_table.get_active_table();
  file.createDataSet<int>(
          "elements",
          HighFive::DataSpace({static_cast<size_t>(elements.size())}), plist)
      .write_raw(elements.data());

  file.createDataSet<P>(
          "soln", HighFive::DataSpace({static_cast<size_t>(vec.size())}), plist)
      .write_raw(vec.data());

  // save E field
  H5Easy::dump(file, "Efield", pde.E_field.to_std(), opts);
  H5Easy::dump(file, "Esource", pde.E_source.to_std(), opts);
  H5Easy::dump(file, "phi", pde.phi.to_std(), opts);

  if (moments.size() > 0)
  {
    // save realspace moments
    H5Easy::dump(file, "nmoments", moments.size());
    for (auto i : indexof(moments))
    {
      file.createDataSet<P>("moment" + std::to_string(i),
                            HighFive::DataSpace({static_cast<size_t>(
                                moments[i].get_realspace_moment().size())}))
          .write_raw(moments[i].get_realspace_moment().data());
    }
  }

  // save gmres error and iteration counts
  for (size_t i = 0; i < pde.gmres_outputs.size(); ++i)
  {
    H5Easy::dump(file, "gmres" + std::to_string(i) + "_err",
                 pde.gmres_outputs[i].error, opts);
    H5Easy::dump(file, "gmres" + std::to_string(i) + "_num_total",
                 pde.gmres_outputs[i].iterations, opts);
  }

  bool const do_adapt = !!options.adapt_threshold;
  H5Easy::dump(file, "do_adapt", do_adapt);
  H5Easy::dump(file, "grid_type", static_cast<int>(options.grid.value()));
  H5Easy::dump(file, "starting_levels", options.start_levels);

  if (do_adapt)
  {
    H5Easy::dump(file, "adapt_thresh", options.adapt_threshold.value());

    // if using adaptivity, save some stats about DOF coarsening/refining and
    // GMRES stats for each adapt step
    H5Easy::dump(file, "adapt_initial_dof", pde.adapt_info.initial_dof);
    H5Easy::dump(file, "adapt_coarsen_dof", pde.adapt_info.coarsen_dof);
    H5Easy::dump(file, "adapt_num_refines", pde.adapt_info.refine_dofs.size());
    H5Easy::dump(file, "adapt_refine_dofs", pde.adapt_info.refine_dofs);

    // Transform GMRES stats for each adaptive step into arrays to reduce number
    // of H5 datasets and make it easier to process later.
    // TODO: this needs to be refactored into its own dataset within the H5
    // file.
    size_t num_gmres_calls = pde.gmres_outputs.size();
    size_t num_adapt_steps = pde.adapt_info.gmres_stats.size();
    std::vector<std::vector<P>> step_errors(num_gmres_calls);
    std::vector<std::vector<int>> step_num_total(num_gmres_calls);

    for (size_t gmres = 0; gmres < num_gmres_calls; gmres++)
    {
      step_errors[gmres].resize(num_adapt_steps);
      step_num_total[gmres].resize(num_adapt_steps);
      // Combine stats for all steps into a single array
      for (size_t step = 0; step < num_adapt_steps; step++)
      {
        step_errors[gmres][step] =
            pde.adapt_info.gmres_stats[step][gmres].error;
        step_num_total[gmres][step] =
            pde.adapt_info.gmres_stats[step][gmres].iterations;
      }

      std::string const prefix = "adapt_gmres" + std::to_string(gmres);
      H5Easy::dump(file, prefix + "_err", step_errors[gmres]);
      H5Easy::dump(file, prefix + "_num_total", step_num_total[gmres]);
    }
  }

  H5Easy::dump(file, "isolver_tolerance", options.isolver_tolerance.value());
  H5Easy::dump(file, "isolver_iterations", options.isolver_iterations.value());
  H5Easy::dump(file, "isolver_outer_iterations", options.isolver_outer_iterations.value());

  // save some basic build info
  H5Easy::dump(file, "GIT_BRANCH", std::string(GIT_BRANCH));
  H5Easy::dump(file, "GIT_COMMIT_HASH", std::string(GIT_COMMIT_HASH));
  H5Easy::dump(file, "GIT_COMMIT_SUMMARY", std::string(GIT_COMMIT_SUMMARY));
  H5Easy::dump(file, "BUILD_TIME", std::string(BUILD_TIME));
#if defined(ASGARD_USE_CUDA)
  bool constexpr using_gpu = true;
#else
  bool constexpr using_gpu = false;
#endif
  H5Easy::dump(file, "USING_GPU", using_gpu);

  // save performance timers to the /timings/ group
  auto timing_stat_type = create_timing_stats<double>();
  timing_stat_type.commit(file, "timing_stats");

  std::map<std::string, tools::timing_stats> timings;
  tools::timer.get_timing_stats(timings);
  auto timing_group = file.createGroup("timings");
  for (auto [id, times] : timings)
  {
    timing_group
        .createDataSet(
            id,
            HighFive::DataSpace(
                HighFive::DataSpace::DataspaceType::dataspace_scalar),
            timing_stat_type)
        .write(times);
  }

  file.flush();
  tools::timer.stop("write_output");
}

template<typename P>
void read_restart_metadata(prog_opts &options, std::string const &restart_file)
{
  std::cout << "--- Reading metadata from restart file '" << restart_file
            << "' ---\n";

  if (!std::filesystem::exists(restart_file))
  {
    throw std::runtime_error("Could not open restart file: " + restart_file);
  }

  HighFive::File file(restart_file, HighFive::File::ReadOnly);

  options.title    = H5Easy::load<std::string>(file, "title");
  options.subtitle = H5Easy::load<std::string>(file, "subtitle");

  int const pde_choice = H5Easy::load<int>(file, "pde");

  int const degree = H5Easy::load<int>(file, "degree");
  P const dt       = H5Easy::load<P>(file, "dt");
  P const time     = H5Easy::load<P>(file, "time");

  int const ndims = H5Easy::load<int>(file, "ndims");
  std::string levels;
  for (int dim = 0; dim < ndims; ++dim)
  {
    levels += std::to_string(H5Easy::load<int>(
        file, std::string("dim" + std::to_string(dim) + "_level")));
    levels += " ";
  }

  int const dof   = H5Easy::load<int>(file, "dof");
  int const gridt = H5Easy::load<int>(file, "grid_type");

  options.grid = static_cast<grid_type>(gridt);

  // TODO: this will be used for validation in the future
  ignore(dof);

  if (pde_choice < 0 and options.pde_choice)
    options.pde_choice.reset();
  else
    options.pde_choice = static_cast<PDE_opts>(pde_choice);

  options.degree = degree;
  options.dt     = dt;

  // check if the restart file was run with adaptivity
  bool const restart_used_adapt =
      H5Easy::load<bool>(file, std::string("do_adapt"));

  options.max_levels =
      H5Easy::load<std::vector<int>>(file, std::string("max_levels"));

  assert(options.max_levels.size() == static_cast<size_t>(ndims));

  // restore the max adaptivity levels if set in the file
  std::string max_adapt_str;
  if (restart_used_adapt)
  {
    for (size_t lev = 0; lev < options.max_levels.size(); lev++)
    {
      max_adapt_str += std::to_string(options.max_levels[lev]);
      if (lev < options.max_levels.size() - 1)
        max_adapt_str += " ";
    }
  }

  if (get_local_rank() == 0)
    std::cout << options;
}

template<typename P>
struct restart_data
{
  fk::vector<P> solution;
  P const time;
  int step_index;
  std::vector<int64_t> active_table;
  int max_level;
};

template<typename P>
restart_data<P> read_output(PDE<P> &pde, elements::table const &hash_table,
                            std::vector<moment<P>> &moments,
                            std::string const &restart_file)
{
  tools::timer.start("read_output");

  std::cout << "--- Loading from restart file '" << restart_file << "' ---\n";

  if (!std::filesystem::exists(restart_file))
  {
    throw std::runtime_error("Could not open restart file: " + restart_file);
  }

  HighFive::File file(restart_file, HighFive::File::ReadOnly);

  int const max_level = H5Easy::load<int>(file, std::string("max_level"));
  P const dt          = H5Easy::load<P>(file, std::string("dt"));
  P const time        = H5Easy::load<P>(file, std::string("time"));

  std::vector<int64_t> active_table =
      H5Easy::load<std::vector<int64_t>>(file, std::string("elements"));

  fk::vector<P> solution =
      fk::vector<P>(H5Easy::load<std::vector<P>>(file, std::string("soln")));

  // load E field
  pde.E_field = std::move(
      fk::vector<P>(H5Easy::load<std::vector<P>>(file, std::string("Efield"))));

  for (int dim = 0; dim < pde.num_dims(); ++dim)
  {
    int level = H5Easy::load<int>(
        file, std::string("dim" + std::to_string(dim) + "_level"));
    pde.get_dimensions()[dim].set_level(level);
    pde.update_dimension(dim, level);
    pde.rechain_dimension(dim);
  }

  // load realspace moments
  int const num_moments = H5Easy::load<int>(file, std::string("nmoments"));
  expect(static_cast<int>(moments.size()) == num_moments);
  for (auto i : indexof(num_moments))
  {
    moments[i].createMomentReducedMatrix(pde, hash_table);
    moments[i].set_realspace_moment(
        fk::vector<P>(H5Easy::load<std::vector<P>>(
            file, std::string("moment" + std::to_string(i)))));
  }

  int step_index = static_cast<int>(time / dt);

  std::cout << " Setting time step index as = " << step_index << "\n";

  tools::timer.stop("read_output");

  return restart_data<P>{solution, time, step_index, active_table, max_level};
}

} // namespace asgard
