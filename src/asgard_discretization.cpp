#include "asgard_discretization.hpp"

namespace asgard
{

template<typename precision>
discretization_manager<precision>::discretization_manager(
    pde_scheme<precision> pde_in, verbosity_level verbosity)
  : verb(pde_in.options().verbosity.value_or(verbosity)),
    pde2(std::move(pde_in)), conn(pde2.max_level())
{
  init_compute();

  if (pde2.num_dims() == 0)
    throw std::runtime_error("cannot discretize an empty pde");

  if (pde2.options().restarting())
    restart_from_file();
  else
    start_cold();
}

template<typename precision>
void discretization_manager<precision>::start_cold()
{
  auto const &options = pde2.options();

  degree_ = options.degree.value();

  if (high_verbosity()) {
    std::cout << "Branch: " << GIT_BRANCH << '\n';
    std::cout << "Commit Summary: " << GIT_COMMIT_HASH
                    << GIT_COMMIT_SUMMARY << '\n';
    std::cout << "The library was built on " << BUILD_TIME << '\n';
  }

  if (not stop_verbosity())
    std::cout << "\n -- ASGarD discretization options --\n";

  sgrid = sparse_grid(options);

  if (not stop_verbosity()) {
    if (not options.title.empty())
      std::cout << "       title: " << options.title << '\n';
    if (not options.subtitle.empty())
      std::cout << "    subtitle: " << options.subtitle << '\n';

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

    std::cout << sgrid;
    if (options.adapt_threshold)
      std::cout << "  adaptive tolerance: " << options.adapt_threshold.value() << '\n';
    if (options.adapt_ralative)
      std::cout << "  relative tolerance: " << options.adapt_ralative.value() << '\n';
    if (not options.adapt_threshold and not options.adapt_ralative)
      std::cout << "  non-adaptive\n";
  }

  { // setting up the time-step approach
    // if no method is set, defaulting to explicit time-stepping
    time_method sm = options.step_method.value_or(time_method::rk3);

    time_data<precision> dtime; // initialize below

    precision stop = options.stop_time.value_or(-1);
    precision dt   = options.dt.value_or(-1);
    int64_t n      = options.num_time_steps.value_or(-1);

    if (sm == time_method::steady) {
      stop  = options.stop_time.value_or(options.default_stop_time.value_or(0));
      dtime = time_data<precision>(stop);
    } else {
      if (stop >= 0 and dt >= 0 and n >= 0)
        throw std::runtime_error("Must provide exactly two of the three time-stepping parameters: "
                                "-dt, -num-steps, -time");

      // replace options with defaults, when appropriate
      if (n >= 0) {
        if (stop < 0 and dt < 0) {
          dt = options.default_dt.value_or(-1);
          if (dt < 0) {
            stop = options.default_stop_time.value_or(-1);
            if (stop < 0)
              throw std::runtime_error("number of steps provided, but no dt or stop-time");
          }
        }
      } else if (stop >= 0) { // no num-steps, but dt may be provided or have a default
        if (dt < 0) {
          dt = options.default_dt.value_or(-1);
          if (dt < 0)
            throw std::runtime_error("stop-time provided but no time-step or number of steps");
        }
      } else if (dt >= 0) { // both n and stop are unspecified
        stop = options.default_stop_time.value_or(-1);
        if (stop < 0)
          throw std::runtime_error("dt provided, but no stop-time or number of steps");
      } else { // nothing provided, look for defaults
        dt   = options.default_dt.value_or(-1);
        stop = options.default_stop_time.value_or(-1);
        if (dt < 0 or stop < 0)
          throw std::runtime_error("need at least two time parameters: -dt, -num-steps, -time");
      }

      if (n >= 0 and stop >= 0 and dt < 0)
        dtime = time_data<precision>(
            sm, n, typename time_data<precision>::input_stop_time{stop});
      else if (dt >= 0 and stop >= 0 and n < 0)
        dtime = time_data<precision>(sm,
                                    typename time_data<precision>::input_dt{dt},
                                    typename time_data<precision>::input_stop_time{stop});
      else if (dt >= 0 and n >= 0 and stop < 0)
        dtime = time_data<precision>(sm, typename time_data<precision>::input_dt{dt}, n);
      else
        throw std::runtime_error("how did this happen?");
    }

    if (is_imex(sm)) {
      stepper = time_advance_manager<precision>(dtime, options, pde2.imex_im(), pde2.imex_ex());
    } else {
      stepper = time_advance_manager<precision>(dtime, options);
    }
  }

  if (not stop_verbosity())
    std::cout << stepper;

  if (stepper.needs_solver() and not options.solver)
    throw std::runtime_error("the selected time-stepping method requires a solver, "
                             "or a default solver set in the pde specification");

  hier = hierarchy_manipulator(degree_, pde2.domain());

  // first we must initialize the terms, which will also initialize the kron
  // operations and the interpolation engine
  terms = term_manager<precision>(pde2, sgrid, hier, conn);

  start_moments();

  set_initial_condition();

  if (not stop_verbosity()) {
    int64_t const dof = sgrid.num_indexes() * hier.block_size();
    std::cout << "initial degrees of freedom: " << tools::split_style(dof) << "\n\n";
  }

  if (stepper.needed_precon() == precon_method::adi) {
    terms.build_matrices(sgrid, conn, hier, precon_method::adi,
                         0.5 * stepper.data.dt());
  } else
    terms.build_matrices(sgrid, conn, hier);

  if (high_verbosity())
    progress_report();
}

template<typename precision>
void discretization_manager<precision>::restart_from_file()
{
#ifdef ASGARD_USE_HIGHFIVE
  tools::time_event timing_("restart from file");

  time_data<precision> dtime;
  h5manager<precision>::read(pde2.options().restart_file, high_verbosity(), pde2, sgrid,
                             dtime, aux_fields, state);

  conn = connection_patterns(pde2.max_level());

  auto const &options = pde2.options();

  degree_ = options.degree.value();

  hier = hierarchy_manipulator(degree_, pde2.domain());

  if (is_imex(dtime.step_method())) {
    stepper = time_advance_manager<precision>(dtime, options, pde2.imex_im(), pde2.imex_ex());
  } else {
    stepper = time_advance_manager<precision>(dtime, options);
  }

  stepper = time_advance_manager<precision>(dtime, options);

  terms = term_manager<precision>(pde2, sgrid, hier, conn);

  start_moments();

  if (stepper.needed_precon() == precon_method::adi) {
    precision const substep
        = (options.step_method.value() == time_method::cn) ? 0.5 : 1;
    terms.build_matrices(sgrid, conn, hier, precon_method::adi,
                         substep * stepper.data.dt());
  } else
    terms.build_matrices(sgrid, conn, hier);

  if (not stop_verbosity()) {
    if (not options.title.empty())
      std::cout << "  title: " << options.title << '\n';
    if (not options.subtitle.empty())
      std::cout << "subtitle: " << options.subtitle << '\n';
    std::cout << sgrid;
    if (options.adapt_threshold)
      std::cout << "  adaptive tolerance: " << options.adapt_threshold.value() << '\n';
    if (options.adapt_ralative)
      std::cout << "  relative tolerance: " << options.adapt_ralative.value() << '\n';
    if (not options.adapt_threshold and not options.adapt_ralative)
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
    int const mom_size = fm::ipow2(sgrid.current_level(0)) * (degree_ + 1);
    moms1d = moments1d(num, degree_, pde2.max_level(), pde2.domain());
    if (terms.deps().poisson) {
      poisson = solvers::poisson(degree_, pde2.domain().xleft(0), pde2.domain().xright(0),
                                 sgrid.current_level(0));

      // skip the first solve, putting in dummy data for the term construction
      terms.cdata.electric_field.resize(mom_size);
    }
    terms.cdata.moments.resize(num * mom_size);
  }
}

template<typename precision>
void discretization_manager<precision>::save_snapshot(std::filesystem::path const &filename) const {
#ifdef ASGARD_USE_HIGHFIVE
  h5manager<precision>::write(pde2, degree_, sgrid, stepper.data, state, aux_fields, filename);
#else
  ignore(filename);
  throw std::runtime_error("saving to a file requires CMake option -DASGARD_USE_HIGHFIVE=ON");
#endif
}

template<typename precision>
void discretization_manager<precision>::set_initial_condition()
{
  auto const &options = pde2.options();
  std::vector<separable_func<precision>> const &sep = pde2.ic_sep();

  precision const atol = options.adapt_threshold.value_or(0);
  precision const rtol = options.adapt_ralative.value_or(0);

  bool keep_refining = true;

  constexpr precision time = 0;

  int iterations = 0;
  while (keep_refining)
  {
    state.resize(sgrid.num_indexes() * hier.block_size());

    if (pde2.ic_md())
      terms.interp(sgrid, conn, time, 1, pde2.ic_md(), 0, state, terms.kwork, terms.it1);
    else
      std::fill(state.begin(), state.end(), precision{0});

    for (int i : iindexof(sep)) {
      expect(sep[i].num_dims() == pde2.num_dims());

      terms.rebuild_mass_matrices(sgrid);

      std::array<block_diag_matrix<precision>, max_num_dimensions> mock;

      hier.template project_separable<data_mode::increment>
            (sep[i], pde2.domain(), sgrid, terms.lmass, time, 1, state.data());
    }

    if (atol > 0 or rtol > 0) {
      // on the first iteration, do both refine and coarsen with a full-adapt
      // on followon iteration, only add more nodes for stability and to avoid stagnation
      sparse_grid::strategy mode = (iterations == 0) ? sparse_grid::strategy::adapt
                                                     : sparse_grid::strategy::refine;
      int const gid = sgrid.generation();
      sgrid.refine(atol, rtol, hier.block_size(), conn[connect_1d::hierarchy::volume], mode, state);

      // if the grid remained the same, there's nothing to do
      keep_refining = (gid != sgrid.generation());

      if (keep_refining) // should only do this if using interpolation, otherwise just do at the end
        terms.prapare_workspace(sgrid);

    } else { // no refinement set, use the grid as-is
      keep_refining = false;
    }

    iterations++;
  }
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

  terms.rebuild_mass_matrices(sgrid);
  for (int i : iindexof(sep)) {
    hier.template project_separable<data_mode::increment>
          (sep[i], pde2.domain(), sgrid, terms.lmass, time, 1, out.data());
  }
}

template<typename precision> void
discretization_manager<precision>::do_poisson_update(std::vector<precision> const &field) const {
  if (not poisson)
    return; // nothing to update, no term has Poisson dependence

  expect(field.size() == static_cast<size_t>(sgrid.num_indexes() * fm::ipow(degree_ + 1, sgrid.num_dims())));

  std::vector<precision> moment0;
  moms1d->project_moment(0, sgrid, field, moment0);

  int const level = sgrid.current_level(0);
  hier.reconstruct1d(1, level, span2d<precision>(degree_ + 1, fm::ipow2(level), moment0.data()));

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
