#include "asgard_io.hpp"

#include <highfive/H5Easy.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5DataSpace.hpp>

namespace asgard
{
template<typename P>
void generate_initial_moments(
    PDE<P> &pde, std::vector<moment<P>> &moments,
    adapt::distributed_grid<P> const &adaptive_grid,
    asgard::basis::wavelet_transform<P, resource::host> const &transformer,
    fk::vector<P> const &initial_condition)
{
  // create 1D version of PDE and element table for wavelet->realspace
  // mappings
  PDE<P> pde_1d = PDE<P>(pde, PDE<P>::extract_dim0);
  adapt::distributed_grid<P> adaptive_grid_1d(pde_1d);

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

template<typename P>
void write_output(PDE<P> const &pde, std::vector<moment<P>> const &moments,
                  fk::vector<P> const &vec, P const time, int const file_index,
                  int const dof, elements::table const &hash_table,
                  std::string const &output_dataset_root,
                  std::string const &output_dataset_fixed)
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
    H5Easy::dump(file, "dim" + std::to_string(dim) + "_name",
                 dims[dim].name);
  }

  auto &elements = hash_table.get_active_table();
  file.createDataSet<int>(
          "elements",
          HighFive::DataSpace({static_cast<size_t>(elements.size())}), plist)
      .write_raw(elements.data());

  file.createDataSet<P>(
          "state", HighFive::DataSpace({static_cast<size_t>(vec.size())}), plist)
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

  H5Easy::dump(file, "timer_report", tools::timer.report());

  file.flush();
  tools::timer.stop("write_output");
}

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
      fk::vector<P>(H5Easy::load<std::vector<P>>(file, std::string("state")));

  // load E field
  pde.E_field = std::move(
      fk::vector<P>(H5Easy::load<std::vector<P>>(file, std::string("Efield"))));

  for (int dim = 0; dim < pde.num_dims(); ++dim)
  {
    int level = H5Easy::load<int>(
        file, std::string("dim" + std::to_string(dim) + "_level"));
    pde.get_dimensions()[dim].set_level(level);
    pde.update_dimension(dim, level);
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

#ifdef ASGARD_ENABLE_DOUBLE
template void generate_initial_moments<double>(
    PDE<double> &, std::vector<moment<double>> &,
    adapt::distributed_grid<double> const &,
    asgard::basis::wavelet_transform<double, resource::host> const &,
    fk::vector<double> const &);
template void write_output<double>(
    PDE<double> const &, std::vector<moment<double>> const &,
    fk::vector<double> const &, double const, int const,
    int const, elements::table const &, std::string const &, std::string const &);
template restart_data<double> read_output<double>(
    PDE<double> &, elements::table const &, std::vector<moment<double>> &,
    std::string const &);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void generate_initial_moments(
    PDE<float> &, std::vector<moment<float>> &,
    adapt::distributed_grid<float> const &,
    asgard::basis::wavelet_transform<float, resource::host> const &,
    fk::vector<float> const &);
template void write_output<float>(
    PDE<float> const &, std::vector<moment<float>> const &,
    fk::vector<float> const &, float const, int const,
    int const, elements::table const &, std::string const &, std::string const &);
template restart_data<float> read_output<float>(
    PDE<float> &, elements::table const &, std::vector<moment<float>> &,
    std::string const &);
#endif

} // namespace asgard
