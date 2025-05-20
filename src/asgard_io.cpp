#include "asgard_io.hpp"

#include <highfive/H5Easy.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5DataSpace.hpp>

namespace asgard
{
template<typename P>
void h5manager<P>::write(prog_opts const &options, pde_domain<P> const &domain,
                         int degree, sparse_grid const &grid,
                         time_data const &dtime, std::vector<P> const &state,
                         std::vector<aux_field_entry<P>> const &aux_fields,
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

  int const num_dims  = domain.num_dims();

  H5Easy::dump(file, "title", options.title);
  H5Easy::dump(file, "subtitle", options.subtitle);
  H5Easy::dump(file, "default_plotter_view", options.default_plotter_view);

  H5Easy::dump(file, "num_dims", domain.num_dims_);
  H5Easy::dump(file, "max_level", options.max_level());
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
    double const adapt_rel = options.adapt_threshold.value_or(-1);
    H5Easy::dump(file, "grid_adapt_relative", adapt_rel);
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

  { // aux fields section
    H5Easy::dump(file, "num_aux_fields", static_cast<int>(aux_fields.size()));
    for (int i : iindexof(aux_fields)) {
      H5Easy::dump(file, "aux_field_" + std::to_string(i) + "_name", aux_fields[i].name);
      file.createDataSet<P>(
          "aux_field_" + std::to_string(i) + "_data",
          HighFive::DataSpace(aux_fields[i].data.size()), vopts).write_raw(aux_fields[i].data.data());
      file.createDataSet<int>(
          "aux_field_" + std::to_string(i) + "_grid",
          HighFive::DataSpace(aux_fields[i].grid.size()), vopts).write_raw(aux_fields[i].grid.data());
    }
  }
}

template<typename P>
void h5manager<P>::read(std::string const &filename, bool silent,
                        prog_opts &options, pde_domain<P> &domain,
                        sparse_grid &grid, time_data &dtime,
                        std::vector<aux_field_entry<P>> &aux_fields, std::vector<P> &state)
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
    int const num_pos  = H5Easy::load<int>(file, "num_pos");
    int const num_vel  = H5Easy::load<int>(file, "num_vel");

    if (num_dims != domain.num_dims())
      throw std::runtime_error("Mismatch in the number of dimensions, "
                               "pde is set for '" + std::to_string(domain.num_dims()) +
                               "' but the file contains data for '" + std::to_string(num_dims) +
                               "'. The restart file must match the dimensions.");

    if (num_pos != domain.num_pos())
      throw std::runtime_error("Mismatch in the number of position dimensions, "
                               "pde is set for '" + std::to_string(domain.num_pos()) +
                               "' but the file contains data for '" + std::to_string(num_pos) +
                               "'. The restart file must match the dimensions.");

    if (num_vel != domain.num_vel())
      throw std::runtime_error("Mismatch in the number of velocity dimensions, "
                               "pde is set for '" + std::to_string(domain.num_vel()) +
                               "' but the file contains data for '" + std::to_string(num_vel) +
                               "'. The restart file must match the dimensions.");

    std::vector<P> drng = H5Easy::load<std::vector<P>>(file, "domain_range");

    if (drng.size() != static_cast<size_t>(2 * num_dims))
      throw std::runtime_error("File corruption detected, mismatch in the provided number of domain ranges");

    if (not silent) {
      P constexpr tol = (std::is_same_v<P, double>) ? 1.E-14 : 1.E-6;
      for (int d : iindexof(num_dims)) {
        P val = std::max( std::abs(drng[2 * d]), std::abs(drng[2 * d + 1]) );
        P err = std::max( std::abs(domain.xleft(d) - drng[2 * d]),
                          std::abs(domain.xright(d) - drng[2 * d + 1]) );
        if (val > 1) // if large, switch to relative error
          err /= val;
        if (err > tol) { // should probably be an error, but hard to judge on what is "significant mismatch"
          std::cout << " -- ASGarD WARNING: dimension " << d << " has mismatch in the end-points.\n";
          std::cout << std::scientific;
          std::cout.precision((std::is_same_v<P, double>) ? 16 : 8);
          std::cout << "  expected:      " << std::setw(25) << domain.xleft(d)
                    << std::setw(25) << domain.xright(d) << '\n';
          std::cout << "  found in file: " << std::setw(25) << drng[2 * d]
                    << std::setw(25) << drng[2 * d + 1] << '\n';
        }
      }
    }

    std::string title = H5Easy::load<std::string>(file, "title");
    if (options.title.empty()) {
      options.title = title;
    } else if (title != options.title) {
      std::cout << " -- ASGarD WARNING: mismatch in the problem title, possibly using the wrong restart file.\n";
      std::cout << "  expected:      " << options.title << '\n';
      std::cout << "  found in file: " << title << '\n';
    }

  }

  std::string subtitle = H5Easy::load<std::string>(file, "subtitle");
  if (options.subtitle.empty()) // if user has new subtitle, keep it, else set from file
    options.subtitle = H5Easy::load<std::string>(file, "subtitle");
  options.default_plotter_view = H5Easy::load<std::string>(file, "default_plotter_view");

  options.degree = H5Easy::load<int>(file, "degree");

  { // reading time parameters
    time_method sm = options.step_method.value_or(
        static_cast<time_method>(H5Easy::load<int>(file, std::string("dtime_smethod"))));

    P const stop    = options.stop_time.value_or(-1);
    P const dt      = options.dt.value_or(-1);
    int64_t const n = options.num_time_steps.value_or(-1);

    P fstop                 = H5Easy::load<P>(file, "dtime_stop");
    P const curr_time       = H5Easy::load<P>(file, "dtime_time");
    int64_t const curr_step = H5Easy::load<int64_t>(file, "dtime_step");

    rassert(not (stop >= 0 and dt >= 0 and n >= 0),
            "cannot simultaneously specify -dt, -num-steps, and -time");

    rassert(not (stop >= 0 and stop <= curr_time),
            "cannot reset the final time to an instance before the current time");

    // replacing defaults ... whenever it makes sense
    // the basic logic is to prioritize stop-time and dt and infer the remaining number of steps
    if (n == 0) { // special case, read data but stop time stepping
      rassert(stop < 0 or stop <= curr_time,
              "cannot jump to a new stop time with zero time steps");
      dtime = time_data(sm);
      fstop = curr_time;
    } else if (dt >= 0) { // overriding dt
      if (stop >= 0) { // and the stop time
        dtime = time_data(sm,
                          typename time_data::input_dt{dt},
                          typename time_data::input_stop_time{stop - curr_time});
      } else if (n >= 0) {
        dtime = time_data(sm, typename time_data::input_dt{dt}, n);
        fstop += dtime.stop_time_;
      } else {
        dtime = time_data(sm,
                          typename time_data::input_dt{dt},
                          typename time_data::input_stop_time{fstop - curr_time});
      }
    } else if (stop >= 0) { // overriding the stop time, dt is not set
      if (n >= 0) {
        dtime = time_data(sm, n, typename time_data::input_stop_time{stop - curr_time});
      } else {
        P const fdt = H5Easy::load<P>(file, "dtime_dt");
        dtime = time_data(sm,
                          typename time_data::input_dt{fdt},
                          typename time_data::input_stop_time{stop - curr_time});
      }
    } else if (n >= 0) { // only n is specified
      dtime = time_data(
            sm, n, typename time_data::input_stop_time{fstop - curr_time});
    } else {
      P const fdt   = H5Easy::load<P>(file, "dtime_dt");
      dtime = time_data(sm,
                        typename time_data::input_dt{fdt},
                        typename time_data::input_stop_time{fstop - curr_time});
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
    options.loaded_max_level_ = H5Easy::load<int>(file, "max_level");
    // TODO: figure out the max-level logic
    if (not options.max_levels.empty()) { // reusing the max levels
      std::vector<int> &max_levels = options.max_levels;
      if (max_levels.size() == 1 and num_dims > 1)
        max_levels.resize(num_dims, options.max_levels.front());

      if (max_levels.size() != static_cast<size_t>(num_dims))
        throw std::runtime_error("the max levels must include either a single entry"
                                 "indicating uniform max or one entry per dimension");

      for (int d : iindexof(num_dims)) {
        if (grid.level_[d] > max_levels[d])
          throw std::runtime_error("cannot set new max level below the current level "
                                   "of the grid");
      }

      // overriding the loaded max-indexes
      for (int d : iindexof(num_dims))
        grid.max_index_[d] = (max_levels[d] == 0) ? 1 : fm::ipow2(max_levels[d]);
    }

    if (not options.adapt_threshold) { // no adapt is loaded
      if (not options.set_no_adapt) { // adaptivity wasn't explicitly canceled
        double const adapt = H5Easy::load<double>(file, "grid_adapt_threshold");
        if (adapt > 0) // if negative, then adaptivity was never set to begin with
          options.adapt_threshold = adapt;
        double const adapt_rel = H5Easy::load<double>(file, "grid_adapt_relative");
        if (adapt_rel > 0) // if negative, then adaptivity was never set to begin with
          options.adapt_ralative = adapt_rel;
      }
    }
  }

  { // solver data section
    if (not options.solver)
      options.solver = static_cast<solver_method>(H5Easy::load<int>(file, "solver_method"));
    if (not options.isolver_tolerance) {
      options.isolver_tolerance = H5Easy::load<double>(file, "solver_itol");
      if (options.isolver_tolerance.value() < 0)
        options.isolver_tolerance = options.default_isolver_tolerance;
    }
    if (not options.isolver_iterations) {
      options.isolver_iterations = H5Easy::load<int>(file, "solver_iter");
      if (options.isolver_iterations.value() < 0)
        options.isolver_iterations = options.default_isolver_iterations;
    }
    if (not options.isolver_inner_iterations) {
      options.isolver_inner_iterations = H5Easy::load<int>(file, "solver_inner");
      if (options.isolver_inner_iterations.value() < 0)
        options.isolver_inner_iterations = options.default_isolver_inner_iterations;
    }
  }

  state = H5Easy::load<std::vector<P>>(file, "state");

  int64_t const size = grid.num_indexes() * fm::ipow(options.degree.value() + 1, num_dims);

  if (state.size() != static_cast<size_t>(size))
    throw std::runtime_error("file corruption detected: wrong number of state coefficients "
                             "found in the file");

  { // reading aux fields
    int const num_aux = H5Easy::load<int>(file, "num_aux_fields");
    aux_fields.resize(num_aux);
    for (int i : iindexof(num_aux)) {
      aux_fields[i].name = H5Easy::load<std::string>(file, "aux_field_" + std::to_string(i) + "_name");
      aux_fields[i].data = H5Easy::load<std::vector<P>>(file, "aux_field_" + std::to_string(i) + "_data");
      aux_fields[i].grid = H5Easy::load<std::vector<int>>(file, "aux_field_" + std::to_string(i) + "_grid");
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template class h5manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class h5manager<float>;
#endif

} // namespace asgard
