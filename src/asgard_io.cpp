#include "asgard_io.hpp"

#include <highfive/H5Easy.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5DataSpace.hpp>

namespace asgard
{

template<typename P>
void write_output(PDE<P> const &pde, // std::vector<moment<P>> const &moments,
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
  for (size_t dim = 0; dim < dims.size(); ++dim)
  {
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
  H5Easy::dump(file, "isolver_inner_iterations", options.isolver_inner_iterations.value());

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
restart_data<P> read_output(PDE<P> &pde, std::string const &restart_file)
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

  for (int dim = 0; dim < pde.num_dims(); ++dim)
  {
    int level = H5Easy::load<int>(
        file, std::string("dim" + std::to_string(dim) + "_level"));
    pde.get_dimensions()[dim].set_level(level);
    pde.update_dimension(dim, level);
  }

  int step_index = static_cast<int>(time / dt);

  tools::timer.stop("read_output");

  return restart_data<P>{solution, time, step_index, active_table, max_level};
}

template<typename P>
void h5manager<P>::write(PDEv2<P> const &pde, int degree, sparse_grid const &grid,
                         time_data<P> const &dtime, std::vector<P> const &state,
                         std::string const &filename)
{
  tools::time_event writing("write output");

  expect(not filename.empty());

  HighFive::File file(filename, HighFive::File::ReadWrite |
                                  HighFive::File::Create |
                                    HighFive::File::Truncate);

  HighFive::DataSetCreateProps vopts; // opts for larger data sets
  if (grid.num_indexes() >= 128)
    vopts.add(HighFive::Chunking(hsize_t{64}));
  else
    vopts.add(HighFive::Chunking(hsize_t{1}));
  vopts.add(HighFive::Deflate(9));

  // sanity check file version string here, detects whether this is an asgard file
  // and which version was used, bump version with time, sync with save style/data
  // as opposed to release version, try to keep it backwards compatible
  H5Easy::dump(file, "asgard file version", asgard_file_version);

  int const num_dims  = pde.num_dims();
  auto const &options = pde.options();
  auto const &domain  = pde.domain();

  H5Easy::dump(file, "title", options.title);
  H5Easy::dump(file, "subtitle", options.subtitle);
  H5Easy::dump(file, "default_plotter_view", options.default_plotter_view);

  H5Easy::dump(file, "num_dims", domain.num_dims_);
  H5Easy::dump(file, "max_level", pde.max_level_);
  H5Easy::dump(file, "degree", degree);

  { // domain section
    std::vector<P> drng(2 * num_dims);
    for (int d : iindexof(num_dims)) {
      drng[2 * d]     = domain.xleft(d);
      drng[2 * d + 1] = domain.xright(d);
      H5Easy::dump(file, "dim" + std::to_string(d) + "_name", domain.dnames_[d]);
    }
    H5Easy::dump(file, "domain_range", drng);
    H5Easy::dump(file, "num_pos", domain.num_pos_);
    H5Easy::dump(file, "num_vel", domain.num_vel_);
  }

  { // sparse_grid section
    H5Easy::dump(file, "grid_generation", grid.generation_);
    H5Easy::dump(file, "grid_mgroup", grid.mgroup);
    H5Easy::dump(file, "grid_num_indexes", grid.num_indexes());
    std::vector<int> lvl(grid.level_.data(), grid.level_.data() + num_dims);
    H5Easy::dump(file, "grid_level", lvl);
    std::copy_n(grid.max_index_.data(), num_dims, lvl.data());
    H5Easy::dump(file, "grid_max_index", lvl);

    std::vector<int> const &indexes = grid.iset_.indexes_;
    file.createDataSet<int>("grid_indexes", HighFive::DataSpace(indexes.size()), vopts)
        .write_raw(indexes.data());

    double const adapt = options.adapt_threshold.value_or(-1);
    H5Easy::dump(file, "grid_adapt_threshold", adapt);
  }

  file.createDataSet<P>("state", HighFive::DataSpace(state.size()), vopts)
      .write_raw(state.data());

  { // time stepping data section
    H5Easy::dump(file, "dtime_smethod", static_cast<int>(dtime.smethod_));
    H5Easy::dump(file, "dtime_dt", dtime.dt_);
    H5Easy::dump(file, "dtime_stop", dtime.stop_time_);
    H5Easy::dump(file, "dtime_time", dtime.time_);
    H5Easy::dump(file, "dtime_step", dtime.step_);
    H5Easy::dump(file, "dtime_remaining", dtime.num_remain_);
  }

  { // solver data section
    H5Easy::dump(file, "solver_method", static_cast<int>(options.solver.value_or(solver_method::direct)));
    H5Easy::dump(file, "solver_itol", options.isolver_tolerance.value_or(-1));
    H5Easy::dump(file, "solver_iter", options.isolver_iterations.value_or(-1));
    H5Easy::dump(file, "solver_inner", options.isolver_inner_iterations.value_or(-1));
  }

  H5Easy::dump(file, "timer_report", tools::timer.report());
}

template<typename P>
void h5manager<P>::read(std::string const &filename, bool silent, PDEv2<P> &pde,
                        sparse_grid &grid, time_data<P> &dtime, std::vector<P> &state)
{
  HighFive::File file(filename, HighFive::File::ReadOnly);

  {
    int fversion = 0;
    try {
      fversion = H5Easy::load<int>(file, "asgard file version");
    } catch (HighFive::Exception &) {
      std::cerr << "exception encountered when using H5Easy::load() on file '" + filename + "'\n"
                   " - Failed to read the 'asgard file version'\n"
                   " - Is this an ASGarD file?\n";
      throw;
    }

    if (fversion != asgard_file_version)
      throw std::runtime_error("wrong file version, is this an asgard file?");
  }

  int const num_dims = H5Easy::load<int>(file, "num_dims");

  { // sanity checking
    int num_pos  = H5Easy::load<int>(file, "num_pos");
    int num_vel  = H5Easy::load<int>(file, "num_vel");

    if (num_dims != pde.num_dims())
      throw std::runtime_error("Mismatch in the number of dimensions, "
                               "pde is set for '" + std::to_string(pde.num_dims()) +
                               "' but the file contains data for '" + std::to_string(num_dims) +
                               "'. The restart file must match the dimensions.");

    if (num_pos != pde.domain().num_pos())
      throw std::runtime_error("Mismatch in the number of position dimensions, "
                               "pde is set for '" + std::to_string(pde.domain().num_pos()) +
                               "' but the file contains data for '" + std::to_string(num_pos) +
                               "'. The restart file must match the dimensions.");

    if (num_vel != pde.domain().num_vel())
      throw std::runtime_error("Mismatch in the number of velocity dimensions, "
                               "pde is set for '" + std::to_string(pde.domain().num_vel()) +
                               "' but the file contains data for '" + std::to_string(num_vel) +
                               "'. The restart file must match the dimensions.");

    std::vector<P> drng = H5Easy::load<std::vector<P>>(file, "domain_range");

    if (drng.size() != static_cast<size_t>(2 * num_dims))
      throw std::runtime_error("File corruption detected, mismatch in the provided number of domain ranges");

    if (not silent) {
      P constexpr tol = (std::is_same_v<P, double>) ? 1.E-14 : 1.E-6;
      for (int d : iindexof(num_dims)) {
        P val = std::max( std::abs(drng[2 * d]), std::abs(drng[2 * d + 1]) );
        P err = std::max( std::abs(pde.domain().xleft(d) - drng[2 * d]),
                          std::abs(pde.domain().xright(d) - drng[2 * d + 1]) );
        if (val > 1) // if large, switch to relative error
          err /= val;
        if (err > tol) { // should probably be an error, but hard to judge on what is "significant mismatch"
          std::cout << " -- ASGarD WARNING: dimension " << d << " has mismatch in the end-points.\n";
          std::cout << std::scientific;
          std::cout.precision((std::is_same_v<P, double>) ? 16 : 8);
          std::cout << "  expected:      " << std::setw(25) << pde.domain().xleft(d)
                    << std::setw(25) << pde.domain().xright(d) << '\n';
          std::cout << "  found in file: " << std::setw(25) << drng[2 * d]
                    << std::setw(25) << drng[2 * d + 1] << '\n';
        }
      }
    }

    std::string title = H5Easy::load<std::string>(file, "title");
    if (pde.options_.title.empty()) {
      pde.options_.title = title;
    } else if (title != pde.options_.title) {
      std::cout << " -- ASGarD WARNING: mismatch in the problem title, possibly using the wrong restart file.\n";
      std::cout << "  expected:      " << pde.options().title << '\n';
      std::cout << "  found in file: " << title << '\n';
    }

  }

  std::string subtitle = H5Easy::load<std::string>(file, "subtitle");
  if (pde.options_.subtitle.empty()) // if user has new subtitle, keep it, else set from file
    pde.options_.subtitle = H5Easy::load<std::string>(file, "subtitle");
  pde.options_.default_plotter_view = H5Easy::load<std::string>(file, "default_plotter_view");

  pde.options_.degree = H5Easy::load<int>(file, "degree");

  { // reading time parameters
    time_method sm = pde.options_.step_method.value_or(
        static_cast<time_method>(H5Easy::load<int>(file, std::string("dtime_smethod"))));

    P const stop    = pde.options_.stop_time.value_or(-1);
    P const dt      = pde.options_.dt.value_or(-1);
    int64_t const n = pde.options_.num_time_steps.value_or(-1);

    P fstop                 = H5Easy::load<P>(file, "dtime_stop");
    P const curr_time       = H5Easy::load<P>(file, "dtime_time");
    int64_t const curr_step = H5Easy::load<int64_t>(file, "dtime_step");

    if (stop >= 0 and dt >= 0 and n >= 0)
      throw std::runtime_error("cannot simultaneously specify -dt, -num-steps, and -time");

    if (stop >= 0 and stop <= curr_time)
      throw std::runtime_error("cannot reset the final time to an instance before the current time");

    // replacing defaults ... whenever it makes sense
    // the basic logic is to prioritize stop-time and dt and infer the remaining number of steps
    if (dt >= 0) { // overriding dt
      if (stop >= 0) { // and the stop time
        dtime = time_data<P>(sm,
                             typename time_data<P>::input_dt{dt},
                             typename time_data<P>::input_stop_time{stop - curr_time});
      } else if (n >= 0) {
        dtime = time_data<P>(sm, typename time_data<P>::input_dt{dt}, n);
        fstop += dtime.stop_time_;
      } else {
        dtime = time_data<P>(sm,
                             typename time_data<P>::input_dt{dt},
                             typename time_data<P>::input_stop_time{fstop - curr_time});
      }
    } else if (stop >= 0) { // overriding the stop time, dt is not set
      if (n >= 0) {
        dtime = time_data<P>(sm, n, typename time_data<P>::input_stop_time{stop - curr_time});
      } else {
        P const fdt = H5Easy::load<P>(file, "dtime_dt");
        dtime = time_data<P>(sm,
                             typename time_data<P>::input_dt{fdt},
                             typename time_data<P>::input_stop_time{stop - curr_time});
      }
    } else if (n >= 0) { // only n is specified
      dtime = time_data<P>(
            sm, n, typename time_data<P>::input_stop_time{fstop - curr_time});
    } else {
      P const fdt   = H5Easy::load<P>(file, "dtime_dt");
      dtime = time_data<P>(sm,
                           typename time_data<P>::input_dt{fdt},
                           typename time_data<P>::input_stop_time{fstop - curr_time});
    }

    // the setup above mostly focuses on the number of steps and the final time used
    dtime.stop_time_ = (stop >= 0) ? stop : fstop;
    dtime.time_      = curr_time;
    dtime.step_      = curr_step;
  }

  { // reading the grid
    int64_t num_indexes = H5Easy::load<int64_t>(file, "grid_num_indexes");

    grid.generation_ = H5Easy::load<int>(file, "grid_generation");
    grid.mgroup      = H5Easy::load<int>(file, "grid_mgroup");

    std::vector<int> lvl = H5Easy::load<std::vector<int>>(file, "grid_level");
    for (int d : iindexof(num_dims))
      grid.level_[d] = lvl[d];

    lvl = H5Easy::load<std::vector<int>>(file, "grid_max_index");
    for (int d : iindexof(num_dims))
      grid.max_index_[d] = lvl[d];

    grid.iset_.num_dimensions_ = num_dims;
    grid.iset_.num_indexes_    = num_indexes;

    grid.iset_.indexes_ = H5Easy::load<std::vector<int>>(file, "grid_indexes");

    if (grid.iset_.indexes_.size() != static_cast<size_t>(num_dims * num_indexes))
      throw std::runtime_error("file corruption detected: wrong number of sparse grid "
                               "indexes found in the file");

    grid.dsort_ = dimension_sort(grid.iset_);

    // checking the max levels, we can reset the max level for the simulation
    // first we follow the same logic for specifying either all dims or a single int
    // then we do not allow the max level to be reduced below the current level
    // to do this, we will have to delete indexes, which is complicated (maybe do later)
    int max_level = H5Easy::load<int>(file, "max_level");
    // TODO: figure out the max-level logic
    if (pde.options_.max_levels.empty()) { // reusing the max levels
      pde.max_level_ = max_level;
    } else {
      std::vector<int> &max_levels = pde.options_.max_levels;
      if (max_levels.size() == 1 and num_dims > 1)
        max_levels.resize(num_dims, pde.options_.max_levels.front());

      if (max_levels.size() != static_cast<size_t>(num_dims))
        throw std::runtime_error("the max levels must include either a single entry"
                                 "indicating uniform max or one entry per dimension");

      for (int d : iindexof(num_dims)) {
        if (grid.level_[d] > max_levels[d])
          throw std::runtime_error("cannot set new max level below the current level "
                                   "of the grid");
      }

      pde.max_level_ = *std::max_element(max_levels.begin(), max_levels.end());

      // overriding the loaded max-indexes
      for (int d : iindexof(num_dims))
        grid.max_index_[d] = (max_levels[d] == 0) ? 1 : fm::ipow2(max_levels[d]);
    }

    if (not pde.options_.adapt_threshold) { // no adapt is loaded
      if (not pde.options_.set_no_adapt) { // adaptivity wasn't explicitly canceled
        double const adapt = H5Easy::load<double>(file, "grid_adapt_threshold");
        if (adapt > 0) // if negative, then adaptivity was never set to begin with
          pde.options_.adapt_threshold = adapt;
      }
    }

    pde.max_level_ = max_level;
  }

  { // solver data section
    if (not pde.options_.solver)
      pde.options_.solver = static_cast<solver_method>(H5Easy::load<int>(file, "solver_method"));
    if (not pde.options_.isolver_tolerance) {
      pde.options_.isolver_tolerance = H5Easy::load<double>(file, "solver_itol");
      if (pde.options_.isolver_tolerance.value() < 0)
        pde.options_.isolver_tolerance = pde.options_.default_isolver_tolerance;
    }
    if (not pde.options_.isolver_iterations) {
      pde.options_.isolver_iterations = H5Easy::load<int>(file, "solver_iter");
      if (pde.options_.isolver_iterations.value() < 0)
        pde.options_.isolver_iterations = pde.options_.default_isolver_iterations;
    }
    if (not pde.options_.isolver_inner_iterations) {
      pde.options_.isolver_inner_iterations = H5Easy::load<int>(file, "solver_inner");
      if (pde.options_.isolver_inner_iterations.value() < 0)
        pde.options_.isolver_inner_iterations = pde.options_.default_isolver_inner_iterations;
    }
  }

  state = H5Easy::load<std::vector<P>>(file, "state");

  int64_t const size = grid.num_indexes() * fm::ipow(pde.options_.degree.value() + 1, num_dims);

  if (state.size() != static_cast<size_t>(size))
    throw std::runtime_error("file corruption detected: wrong number of state coefficients "
                             "found in the file");
}

#ifdef ASGARD_ENABLE_DOUBLE
template void write_output<double>(
    PDE<double> const &, // std::vector<moment<double>> const &,
    fk::vector<double> const &, double const, int const,
    int const, elements::table const &, std::string const &, std::string const &);
template restart_data<double> read_output<double>(
    PDE<double> &, std::string const &);

template class h5manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void write_output<float>(
    PDE<float> const &, // std::vector<moment<float>> const &,
    fk::vector<float> const &, float const, int const,
    int const, elements::table const &, std::string const &, std::string const &);
template restart_data<float> read_output<float>(
    PDE<float> &, std::string const &);

template class h5manager<float>;
#endif

} // namespace asgard
