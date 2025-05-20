#include "asgard_discretization.hpp"

namespace asgard
{

template<typename precision>
void discretization_manager<precision>::start_cold(pde_scheme<precision> &pde)
{
  int const degree_ = options_.degree.value();

  if (high_verbosity()) {
    std::cout << '\n';
    #ifdef ASGARD_HAS_GITINFO
    std::cout << "ASGarD: git-branch '" << ASGARD_GIT_BRANCH << "'\n";
    std::cout << "  " << ASGARD_GIT_COMMIT_HASH << ASGARD_GIT_COMMIT_SUMMARY << '\n';
    std::cout << " -- discretization options --\n";
    #else
    std::cout << " -- ASGarD release " << ASGARD_RELEASE_INFO << '\n';
    #endif
  } else {
    if (not stop_verbosity())
      std::cout << "\n -- ASGarD discretization options --\n";
  }

  grid = sparse_grid(options_);

  if (not stop_verbosity()) {
    if (not options_.title.empty())
      std::cout << "    title: " << options_.title << '\n';
    if (not options_.subtitle.empty())
      std::cout << "           " << options_.subtitle << '\n';

    std::cout << "basis degree: " << degree_;
    switch (degree_) {
      case 0:
        std::cout << " (constant)";
        break;
      case 1:
        std::cout << " (linear)";
        break;
      case 2:
        std::cout << " (quadratic)";
        break;
      case 3:
        std::cout << " (cubic)";
        break;
      default:
        break;
    };
    std::cout << '\n';

    std::cout << grid;
    if (options_.adapt_threshold)
      std::cout << "  adaptive tolerance: " << options_.adapt_threshold.value() << '\n';
    if (options_.adapt_ralative)
      std::cout << "  relative tolerance: " << options_.adapt_ralative.value() << '\n';
    if (not options_.adapt_threshold and not options_.adapt_ralative)
      std::cout << "  non-adaptive\n";
  }

  { // setting up the time-step approach
    // if no method is set, defaulting to explicit time-stepping
    time_method sm = options_.step_method.value_or(time_method::rk3);

    time_data dtime; // initialize below

    precision stop = options_.stop_time.value_or(-1);
    precision dt   = options_.dt.value_or(-1);
    int64_t n      = options_.num_time_steps.value_or(-1);

    if (sm == time_method::steady) {
      stop  = options_.stop_time.value_or(options_.default_stop_time.value_or(0));
      dtime = time_data(stop);
    } else {
      rassert(not (stop >= 0 and dt >= 0 and n >= 0),
        "Must provide exactly two of the three time-stepping parameters: -dt, -num-steps, -time");

      // replace options with defaults, when appropriate
      if (n == 0 or stop == 0) { // initial conditions only, no time stepping
        n = 0;
        dt = 0;
        stop = -1; // ignore stop below
      } else if (n > 0) {
        if (stop < 0 and dt < 0) {
          dt = options_.default_dt.value_or(-1);
          if (dt < 0) {
            stop = options_.default_stop_time.value_or(-1);
            if (stop < 0)
              throw std::runtime_error("number of steps provided, but no dt or stop-time");
          }
        }
      } else if (stop >= 0) { // no num-steps, but dt may be provided or have a default
        if (dt < 0) {
          dt = options_.default_dt.value_or(-1);
          if (dt < 0)
            throw std::runtime_error("stop-time provided but no time-step or number of steps");
        }
      } else if (dt >= 0) { // both n and stop are unspecified
        stop = options_.default_stop_time.value_or(-1);
        if (stop < 0)
          throw std::runtime_error("dt provided, but no stop-time or number of steps");
      } else { // nothing provided, look for defaults
        dt   = options_.default_dt.value_or(-1);
        stop = options_.default_stop_time.value_or(-1);
        if (dt < 0 or stop < 0)
          throw std::runtime_error("need at least two time parameters: -dt, -num-steps, -time");
      }

      if (n >= 0 and stop >= 0 and dt < 0)
        dtime = time_data(sm, n, time_data::input_stop_time{stop});
      else if (dt >= 0 and stop >= 0 and n < 0)
        dtime = time_data(sm, time_data::input_dt{dt},
                          time_data::input_stop_time{stop});
      else if (dt >= 0 and n >= 0 and stop < 0)
        dtime = time_data(sm, time_data::input_dt{dt}, n);
      else
        throw std::runtime_error("how did this happen?");
    }

    if (is_imex(sm)) {
      stepper = time_advance_manager<precision>(dtime, options_, pde.imex_im(), pde.imex_ex());
    } else {
      stepper = time_advance_manager<precision>(dtime, options_);
    }
  }

  if (not stop_verbosity())
    std::cout << stepper;

  if (stepper.needs_solver() and not options_.solver)
    throw std::runtime_error("the selected time-stepping method requires a solver, "
                             "or a default solver set in the pde specification");

  hier = hierarchy_manipulator(degree_, domain_);

  // first we must initialize the terms, which will also initialize the kron
  // operations and the interpolation engine
  terms = term_manager<precision>(options_, domain_, pde, grid, hier, conn);

  set_initial_condition();

  start_moments(); // grid may have changes above, wait to start the moments

  if (not stop_verbosity()) {
    int64_t const dof = grid.num_indexes() * hier.block_size();
    std::cout << "initial degrees of freedom: " << tools::split_style(dof) << "\n\n";
  }

  if (stepper.needed_precon() == precon_method::adi) {
    terms.build_matrices(grid, conn, hier, precon_method::adi,
                         0.5 * stepper.data.dt());
  } else
    terms.build_matrices(grid, conn, hier);

  if (high_verbosity())
    progress_report();
}

template<typename precision>
void discretization_manager<precision>::restart_from_file(pde_scheme<precision> &pde)
{
#ifdef ASGARD_USE_HIGHFIVE
  tools::time_event timing_("restart from file");

  time_data dtime;
  h5manager<precision>::read(options_.restart_file, high_verbosity(),
                             options_, domain_, grid,
                             dtime, aux_fields, state);

  conn = connection_patterns(options_.max_level());

  hier = hierarchy_manipulator(options_.degree.value(), domain_);

  if (is_imex(dtime.step_method())) {
    stepper = time_advance_manager<precision>(dtime, options_, pde.imex_im(), pde.imex_ex());
  } else {
    stepper = time_advance_manager<precision>(dtime, options_);
  }

  stepper = time_advance_manager<precision>(dtime, options_);

  terms = term_manager<precision>(options_, domain_, pde, grid, hier, conn);

  start_moments();

  if (stepper.needed_precon() == precon_method::adi) {
    precision const substep
        = (options_.step_method.value() == time_method::cn) ? 0.5 : 1;
    terms.build_matrices(grid, conn, hier, precon_method::adi,
                         substep * stepper.data.dt());
  } else
    terms.build_matrices(grid, conn, hier);

  if (not stop_verbosity()) {
    if (not options_.title.empty())
      std::cout << "  title: " << options_.title << '\n';
    if (not options_.subtitle.empty())
      std::cout << "subtitle: " << options_.subtitle << '\n';
    std::cout << grid;
    if (options_.adapt_threshold)
      std::cout << "  adaptive tolerance: " << options_.adapt_threshold.value() << '\n';
    if (options_.adapt_ralative)
      std::cout << "  relative tolerance: " << options_.adapt_ralative.value() << '\n';
    if (not options_.adapt_threshold and not options_.adapt_ralative)
      std::cout << "  non-adaptive\n";
    std::cout << stepper;
    if (high_verbosity())
      progress_report();
  }

#else
  throw std::runtime_error("restarting from a file requires CMake option "
                           "-DASGARD_USE_HIGHFIVE=ON");
#endif
}

template<typename precision>
void discretization_manager<precision>::start_moments() {
  // process the moments, can compute moments based on the initial conditions
  if (terms.deps().poisson or terms.deps().num_moments > 0) {
    // the poisson solver needs 1 moment
    int const num      = std::max(terms.deps().num_moments, 1);
    int const pos_size = fm::ipow2(grid.current_level(0));
    int const mom_size = pos_size * (degree() + 1);
    moms1d = moments1d(num, degree(), options_.max_level(), domain_);
    if (terms.deps().poisson) {
      poisson = solvers::poisson(degree(), domain_.xleft(0), domain_.xright(0),
                                 grid.current_level(0));

      // skip the first solve, putting in dummy data for the term construction
      // the electric_field is pw-constant, does not have degrees + 1 entries
      terms.cdata.electric_field.resize(pos_size);
    }
    terms.cdata.moments.resize(num * mom_size);
  }
}

template<typename precision>
void discretization_manager<precision>::save_snapshot(std::filesystem::path const &filename) const {
#ifdef ASGARD_USE_HIGHFIVE
  #ifdef ASGARD_USE_MPI
  if (not is_leader())
    return;
  #endif
  h5manager<precision>::write(options_, domain_, degree(), grid, stepper.data,
                              state, aux_fields, filename);
#else
  ignore(filename);
  throw std::runtime_error("saving to a file requires CMake option -DASGARD_USE_HIGHFIVE=ON");
#endif
}

template<typename precision>
void discretization_manager<precision>::set_initial_condition()
{
  precision const atol = options_.adapt_threshold.value_or(0);
  precision const rtol = options_.adapt_ralative.value_or(0);

  #ifdef ASGARD_USE_MPI
  if (not is_leader()) {
    this->grid_sync();
    state.resize(grid.num_indexes() * hier.block_size());
    terms.prapare_workspace(grid);
    return;
  }
  #endif

  bool keep_refining = true;

  constexpr precision time = 0;

  int iterations = 0;
  while (keep_refining)
  {
    state.resize(grid.num_indexes() * hier.block_size());

    if (initial_md_)
      terms.interp(grid, conn, time, 1, initial_md_, 0, state, terms.kwork, terms.it1);
    else
      std::fill(state.begin(), state.end(), precision{0});

    for (int i : iindexof(initial_sep_)) {
      expect(initial_sep_[i].num_dims() == num_dims());

      terms.rebuild_mass_matrices(grid);

      std::array<block_diag_matrix<precision>, max_num_dimensions> mock;

      hier.template project_separable<data_mode::increment>
            (initial_sep_[i], domain_, grid, terms.lmass, time, 1, state.data());
    }

    if (atol > 0 or rtol > 0) {
      // on the first iteration, do both refine and coarsen with a full-adapt
      // on follow-on iteration, only add more nodes for stability and to avoid stagnation
      sparse_grid::strategy mode = (iterations == 0) ? sparse_grid::strategy::adapt
                                                     : sparse_grid::strategy::refine;
      int const gid = grid.generation();
      grid.refine(atol, rtol, hier.block_size(), conn[connect_1d::hierarchy::volume], mode, state);

      // if the grid remained the same, there's nothing to do
      keep_refining = (gid != grid.generation());

      if (keep_refining) // should only do this if using interpolation, otherwise just do at the end
        terms.prapare_workspace(grid);

    } else { // no refinement set, use the grid as-is
      keep_refining = false;
    }

    iterations++;
  }

  this->grid_sync();
}

template<typename precision> void
discretization_manager<precision>::project_function(
    std::vector<separable_func<precision>> const &sep,
    md_func<precision> const &, std::vector<precision> &out) const
{
  tools::time_event performance_("project functions");

  if (out.empty())
    out.resize(state.size());
  else {
    out.resize(state.size());
    std::fill(out.begin(), out.end(), 0);
  }

  precision time = stepper.data.time();

  terms.rebuild_mass_matrices(grid);
  for (int i : iindexof(sep)) {
    hier.template project_separable<data_mode::increment>
          (sep[i], domain_, grid, terms.lmass, time, 1, out.data());
  }
}

template<typename precision> void
discretization_manager<precision>::do_poisson_update(std::vector<precision> const &field) const {
  expect(field.size() == static_cast<size_t>(grid.num_indexes() * fm::ipow(degree() + 1, grid.num_dims())));

  std::vector<precision> moment0;
  moms1d->project_moment(0, grid, field, moment0);

  int const level = grid.current_level(0);
  hier.reconstruct1d(1, level, span2d<precision>(degree() + 1, fm::ipow2(level), moment0.data()));

  poisson.solve_periodic(moment0, terms.cdata.electric_field);
}

template<typename precision>
void discretization_manager<precision>::print_mats() const {
  int const num_dims = terms.num_dims;
  for (auto tid : iindexof(terms.terms)) {
    for (int d : iindexof(num_dims)) {
      std::cout << " term = " << tid << "  dim = " << d << '\n';
      if (terms.terms[tid].coeffs[d].empty()) {
        std::cout << "identity\n";
      } else {
        terms.terms[tid].coeffs[d].to_full(conn).print(std::cout);
      }
      std::cout << '\n';
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template class discretization_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class discretization_manager<float>;
#endif

} // namespace asgard
