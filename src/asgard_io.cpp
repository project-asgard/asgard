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
                         moment_manager<P> const &moms,
                         std::vector<aux_field_entry<P>> const &aux_fields,
                         std::string const &filename)
{
  tools::time_event writing("write output");

  expect(not filename.empty());

  HighFive::File file(filename, HighFive::File::ReadWrite |
                                  HighFive::File::Create |
                                    HighFive::File::Truncate);

  HighFive::DataSetCreateProps vopts; // opts for larger data sets
  vopts.add(HighFive::Chunking(hsize_t{64}));
  vopts.add(HighFive::Deflate(9));

  HighFive::DataSetCreateProps sopts; // opts for small data sets
  sopts.add(HighFive::Chunking(hsize_t{1}));
  sopts.add(HighFive::Deflate(9));

  auto write_vector = [&](std::string const &name, auto const &data)
      -> void
    {
      using dtype = typename std::remove_reference_t<decltype(data)>::value_type;
      if (data.size() >= 128) { // use large options
        file.createDataSet<dtype>(name, HighFive::DataSpace(data.size()), vopts)
            .write_raw(data.data());
      } else { // use small options
        file.createDataSet<dtype>(name, HighFive::DataSpace(data.size()), sopts)
            .write_raw(data.data());
      }
    };

  // sanity check file version string here, detects whether this is an asgard file
  // and which version was used, bump version with time, sync with save style/data
  // as opposed to release version, try to keep it backwards compatible
  H5Easy::dump(file, "asgard file version", asgard_file_version);

  int const num_dims  = domain.num_dims();

  H5Easy::dump(file, "title", options.title);
  H5Easy::dump(file, "subtitle", options.subtitle);
  H5Easy::dump(file, "default_plotter_view", options.default_plotter_view);

  H5Easy::dump(file, "num_dims", domain.num_dims_);
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

    write_vector("grid_indexes", grid.iset_.indexes_);

    double const adapt = options.adapt_threshold.value_or(-1);
    H5Easy::dump(file, "grid_adapt_threshold", adapt);
    double const adapt_rel = options.adapt_threshold.value_or(-1);
    H5Easy::dump(file, "grid_adapt_relative", adapt_rel);
  }

  write_vector("state", state);

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
    int const num_aux = static_cast<int>(aux_fields.size()) + ((!!moms) ? moms.num_moments() : 0);
    H5Easy::dump(file, "num_aux_fields", num_aux);
    for (int i : iindexof(aux_fields)) {
      H5Easy::dump(file, "aux_field_" + std::to_string(i) + "_name", aux_fields[i].name);
      write_vector("aux_field_" + std::to_string(i) + "_data", aux_fields[i].data);
      write_vector("aux_field_" + std::to_string(i) + "_grid", aux_fields[i].grid);
      H5Easy::dump(file, "aux_field_" + std::to_string(i) + "_dims", aux_fields[i].num_dimensions);
    }
  }

  if (moms) { // saving moments as additional aux-fields
    std::vector<P> vals;
    for (int i : iindexof(moms.num_moments()))
    {
      int const auxid = static_cast<int>(aux_fields.size()) + i;
      moms.compute(grid, moment_id{i}, state, vals);
      H5Easy::dump(file, "aux_field_" + std::to_string(auxid) + "_name",
                   std::string("__moment_") + moms.get_by_id(moment_id{i}).to_string());
      write_vector("aux_field_" + std::to_string(auxid) + "_data", vals);
      write_vector("aux_field_" + std::to_string(auxid) + "_grid", moms.get_grid_indexes());
      H5Easy::dump(file, "aux_field_" + std::to_string(auxid) + "_dims", domain.num_pos());
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
  } // end of sanity check

  std::string subtitle = H5Easy::load<std::string>(file, "subtitle");
  if (options.subtitle.empty()) // if user has new subtitle, keep it, else set from file
    options.subtitle = H5Easy::load<std::string>(file, "subtitle");
  options.default_plotter_view = H5Easy::load<std::string>(file, "default_plotter_view");

  options.degree = H5Easy::load<int>(file, "degree");

  { // reading time parameters
    // Here, we can have potentially 3 different values with different priorities
    // 1. Currently provided in options, from cli, input file, etc.
    // 2. Values in the restart file
    // 3. Default values in the options
    // However, should avoid using the default values, since those may be set for a different grid.
    //
    // We must check what we have use the values with highest priorities,
    // but have to be careful with some exceptions:
    // - if the current step-method is "steady" and we are switching to something else,
    //   then the stop time, dt and num-steps in the file are not-valid.
    // - if the current file was set for zero steps (or end time 0), no time steps were taken
    //   therefore the dt in the file is not valid.

    time_method const file_sm
        = static_cast<time_method>(H5Easy::load<int>(file, std::string("dtime_smethod")));

    time_method const sm = options.step_method.value_or(file_sm);

    // potentially new values from the options
    auto const &new_stop  = options.stop_time;
    auto const &new_dt    = options.dt;
    auto const &new_steps = options.num_time_steps;

    // reads a valid stop time from the file or returns empty
    auto get_file_stop = [&]() -> std::optional<double>
      {
        if (file_sm == time_method::steady and sm != time_method::steady) {
          // switching away from steady-state, current stop time is meaningless
          return {};
        } else {
          return H5Easy::load<double>(file, "dtime_stop");
        }
      };
    auto get_file_dt = [&](int64_t current_step, int64_t file_remain_steps)
        -> std::optional<double>
      {
        if (file_sm == time_method::steady
            or (current_step == 0 and file_remain_steps == 0))
        {
          // steady state has no time-step, also if no time-stepping was set
          // (i.e., 0 steps were done) the dt in the current file is invalid
          return {};
        } else {
          return H5Easy::load<double>(file, "dtime_dt");
        }
      };

    double const curr_time  = H5Easy::load<double>(file, "dtime_time");
    int64_t const curr_step = H5Easy::load<int64_t>(file, "dtime_step");
    std::optional<double> stop_time; // effective new stop time

    rassert(not new_stop or not new_dt or not new_steps,
            "cannot simultaneously specify -dt, -num-steps, and -time");

    rassert(new_stop.value_or(curr_time) >= curr_time,
            "cannot reset the final time to an instance before the current time");

    if (new_steps and new_steps.value() == 0) {
      // special case, read the data but don't do any time-stepping
      dtime     = time_data(sm);
      stop_time = curr_time;
    } else if (new_dt) {
      // setting new time-step dt, check what else is new
      time_data::input_dt dt{new_dt.value()};
      if (new_stop) {
        dtime = time_data(sm, dt, time_data::input_stop_time{new_stop.value() - curr_time});
        stop_time = new_stop;
      } else if (new_steps) {
        dtime = time_data(sm, dt, new_steps.value());
        stop_time = curr_time + dtime.stop_time_;
      } else {
        // new time-step, reusing the time from the file
        auto const file_stop = get_file_stop();
        rassert(file_stop, "new dt is provided but -num-steps or -time must also be provided");
        double const end_time = file_stop.value();
        dtime = time_data(sm, dt, time_data::input_stop_time{end_time - curr_time});
        stop_time = end_time;
      }
    } else if (new_stop) {
      // new stop time provided, but not dt
      if (new_steps) {
        dtime = time_data(sm, new_steps.value(),
                          time_data::input_stop_time{new_stop.value() - curr_time});
      } else {
        // new stop time provided, try to reuse the dt from the file
        int64_t const file_remain_steps = H5Easy::load<int64_t>(file, "dtime_remaining");
        auto const file_dt = get_file_dt(curr_step, file_remain_steps);
        rassert(file_dt, "new -time is provided but -dt or -num-steps must also be provided");
        double const dt = file_dt.value();
        dtime = time_data(sm, time_data::input_dt(dt),
                          time_data::input_stop_time{new_stop.value() - curr_time});
      }
      stop_time = new_stop;
    } else if (new_steps) {
      int const n = new_steps.value();
      // new number of steps, check for a valid dt or stop-time
      int64_t const file_remain_steps = H5Easy::load<int64_t>(file, "dtime_remaining");
      auto const file_dt = get_file_dt(curr_step, file_remain_steps);
      if (file_dt) {
        // reuse the time-step and count the new number of steps
        dtime = time_data(sm, time_data::input_dt{file_dt.value()}, n);
        stop_time = curr_time + dtime.stop_time_;
      } else {
        // file did not contain a valid time-step, check if it has a valid final time
        auto const file_stop = get_file_stop();
        rassert(file_stop, "file loaded with new number of time steps, -time or -dt is also required");
        dtime = time_data(sm, n, time_data::input_stop_time{file_stop.value() - curr_time});
        stop_time = file_stop;
      }
    } else {
      // no new values, reusing from file or defaults
      int64_t const file_n = H5Easy::load<int64_t>(file, "dtime_remaining");
      if (file_n == 0 or sm == time_method::steady) {
        // the end has been reached, or no time-stepping
        dtime = time_data(sm);
        // the file may still contain a valid dt from a previous run, keep it
        dtime.dt_ = get_file_dt(curr_step, file_n).value_or(0);
        stop_time = curr_time;
      } else {
        auto const file_dt   = get_file_dt(curr_step, file_n);
        auto const file_stop = get_file_stop();

        rassert(file_dt and file_stop,
                "attempting to change the time-stepping method but the file does not contain valid parameters, "
                "-time and -dt are needed");

        dtime = time_data(sm, time_data::input_dt{file_dt.value()},
                          time_data::input_stop_time{file_stop.value() - curr_time});
        stop_time = file_stop;
      }
    }

    expect(stop_time); // should always happen

    // the setup above mostly focuses on the number of steps and the final time used
    dtime.stop_time_ = stop_time.value();
    dtime.time_      = curr_time;
    dtime.step_      = curr_step;

    if (not options.step_method)
      options.step_method = sm;
  }

  { // reading the grid
    int64_t num_indexes = H5Easy::load<int64_t>(file, "grid_num_indexes");

    grid.generation_ = H5Easy::load<int>(file, "grid_generation");
    grid.mgroup      = H5Easy::load<int>(file, "grid_mgroup");

    std::vector<int> lvl = H5Easy::load<std::vector<int>>(file, "grid_level");
    std::copy_n(lvl.begin(), num_dims, grid.level_.begin());

    if (options.max_levels.empty()) {
      // reusing the existing max-level/max-index
      lvl = H5Easy::load<std::vector<int>>(file, "grid_max_index");
      std::copy_n(lvl.begin(), num_dims, grid.max_index_.begin());
      options.max_levels.resize(num_dims, 0);
      for (int d : iindexof(num_dims))
        options.max_levels[d] = fm::intlog2(lvl[d]);
    } else {
      // updating the max, ignore the old and make sure the new is not less than the current
      if (num_dims > 1) {
        int const l = options.max_levels.front(); // uniform max
        options.max_levels.resize(num_dims, l);
      }
      for (int d : iindexof(num_dims)) {
        options.max_levels[d] = std::max(options.max_levels[d], grid.level_[d]);
        grid.max_index_[d]    = fm::ipow2(options.max_levels[d]);
      }
    }

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
          options.adapt_relative = adapt_rel;
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
    aux_fields.resize(0);
    aux_fields.reserve(num_aux);
    for (int i : iindexof(num_aux)) {
      std::string const name = H5Easy::load<std::string>(file, "aux_field_" + std::to_string(i) + "_name");
      if (name.rfind("__moment_", 0) == 0)
        continue;
      aux_fields.emplace_back();
      aux_fields.back().name = name;
      aux_fields.back().num_dimensions = H5Easy::load<int>(file, "aux_field_" + std::to_string(i) + "_dims");
      aux_fields.back().data = H5Easy::load<std::vector<P>>(file, "aux_field_" + std::to_string(i) + "_data");
      aux_fields.back().grid = H5Easy::load<std::vector<int>>(file, "aux_field_" + std::to_string(i) + "_grid");
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
