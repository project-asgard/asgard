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
void discretization_manager<precision>::set_initial_condition_v1()
{
  auto const &options = pde->options();

  auto &dims   = pde->get_dimensions();
  size_t num_initial = dims.front().initial_condition.size();
  for (auto const &dim : dims)
    rassert(dim.initial_condition.size() == num_initial,
            "each dimension must define equal number of initial conditions");

  // some PDEs incorrectly implement the initial conditions in terms of
  // strictly spatially dependent functions times a separable time component
  // the time-component comes from the exact solution and this is incorrect
  // since the actual functions set as initial conditions are not
  precision const tmult = pde->has_exact_time() ? pde->exact_time(0.0) : 1;

  std::vector<vector_func<precision>> fx;
  fx.reserve(dims.size());

  std::vector<precision> icn, ic; // interpolation parameters

  bool keep_working  = true; // make at least once cycle and refine if refining
  bool final_coarsen = false; // set to true after the final iteration

  // we need to make at least one iteration to generate the initial conditions
  // if refining, we want to keep working until there is no more refinement
  // after that, we want another iteration to coarsen the solution
  // (refining and coarsening should be one step ... but priorities)
  while (keep_working)
  {
    auto const &subgrid = grid.get_subgrid(get_rank());
    state.resize(subgrid.ncols() * hier.block_size());
    std::fill(state.begin(), state.end(), precision{0});

    for (auto i : indexof(num_initial))
    {
      fx.clear();
      for (auto const &dim : dims)
        fx.push_back(dim.initial_condition[i]);

      hier.template project_separable<data_mode::increment>
          (state.data(), dims, fx, matrices.dim_dv, matrices.dim_mass, grid, 0,
           tmult, subgrid.col_start, subgrid.col_stop);
    }
#ifdef KRON_MODE_GLOBAL
    if (pde->interp_initial())
    {
      kronops.make(imex_flag::unspecified, *pde, matrices, grid);
      vector2d<precision> const &nodes = kronops.get_inodes();
      icn.resize(state.size());
      pde->interp_initial()(nodes, icn);
      kronops.get_project(icn.data(), ic);

      for (auto i : indexof(ic))
        state[i] += ic[i];

      kronops.clear();
    }
#endif

    if (options.adapt_threshold) // adapting
    {
      if (final_coarsen)
        keep_working = false;
      else
      {
        auto refined = grid.refine(state, options);

        if (static_cast<size_t>(refined.size()) == state.size())
        {
          auto coarse = grid.coarsen(refined, options);
          final_coarsen = true;
        }
        auto const new_levels = adapt::get_levels(grid.get_table(), dims.size());
        for (int d : indexof<int>(dims))
          dims[d].set_level(new_levels[d]);
      }
    }
    else
      keep_working = false;
  }
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

template<typename precision>
void discretization_manager<precision>::ode_rhs(
    imex_flag imflag, precision t, std::vector<precision> const &x,
    std::vector<precision> &R) const
{
  R.resize(x.size());

#ifdef ASGARD_USE_MPI
  element_subgrid const &subgrid = grid.get_subgrid(get_rank());

  distribution_plan const &plan = grid.get_distrib_plan();

  int64_t const row_size = hier.block_size() * subgrid.nrows();

  // MPI-mode has two extra steps, reduce the result across the rows
  // then redistribute across the columns
  // two work-vectors are needed
  static std::vector<precision> local_row;
  static std::vector<precision> reduced_row;

  local_row.resize(row_size);
  reduced_row.resize(row_size);
#else
  std::vector<precision> &local_row   = R;
  std::vector<precision> &reduced_row = R;
#endif

  kronops.make(imflag, *pde, matrices, grid);
  {
    tools::time_event performance("kronmult", kronops.flops(imflag));
    kronops.apply(imflag, t, 1.0, x.data(), 0.0, local_row.data());
  }

#ifdef ASGARD_USE_MPI
  reduce_results(local_row, reduced_row, plan, get_rank());
#endif

  {
    tools::time_event performance("computing sources");
    for (auto const &source : pde->sources())
      hier.template project_separable<data_mode::increment>
          (reduced_row.data(), pde->get_dimensions(), source.source_funcs(),
          matrices.dim_dv, matrices.dim_mass, grid, t, source.time_func()(t));
  }{
    tools::time_event performance("computing boundary conditions");

    boundary_conditions::generate_scaled_bc(fixed_bc[0], fixed_bc[1], *pde, t, reduced_row);
  }

#ifdef ASGARD_USE_MPI
  exchange_results(reduced_row, R, hier.block_size(), plan, get_rank());
#endif
}

template<typename precision>
void discretization_manager<precision>::ode_irhs(
    precision t, std::vector<precision> const &x, std::vector<precision> &R) const
{
  if (R.empty())
    R.resize(x.size());
  else {
    R.resize(x.size());
    std::fill(R.begin(), R.end(), 0);
  }

  if (not pde->sources().empty())
  {
    tools::time_event performance("computing sources");

    auto const &src = pde->sources();
    hier.template project_separable<data_mode::replace>
          (R.data(), pde->get_dimensions(), src[0].source_funcs(),
          matrices.dim_dv, matrices.dim_mass, grid, t, src[0].time_func()(t));

    for (size_t i = 1; i < src.size(); i++)
      hier.template project_separable<data_mode::increment>
          (R.data(), pde->get_dimensions(), src[i].source_funcs(),
          matrices.dim_dv, matrices.dim_mass, grid, t, src[i].time_func()(t));
  }

  {
    tools::time_event performance("computing boundary conditions");

    boundary_conditions::generate_scaled_bc(fixed_bc[0], fixed_bc[1], *pde, t, R);

    precision const dt = pde->get_dt();
    for (auto i : indexof(R))
      R[i] = x[i] + dt * R[i];
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

  if (pde) { // version 1
    auto const &table = grid.get_table();
    expect(field.size() == static_cast<size_t>(table.size() * fm::ipow(degree_ + 1, pde->num_dims())));

    int const level = pde->get_dimensions()[0].get_level();
    std::vector<precision> moment0;
    moms1d->project_moment(0, level, field, table, moment0);

    hier.reconstruct1d(1, level, span2d<precision>(degree_ + 1, fm::ipow2(level), moment0.data()));

    poisson.solve_periodic(moment0, matrices.edata.electric_field);

    if (matrices.edata.electric_field_infnrm)
    {
      precision emax = 0;
      for (auto e : matrices.edata.electric_field)
        emax = std::max(emax, std::abs(e));
      matrices.edata.electric_field_infnrm = emax;
    }
  } else {
    expect(field.size() == static_cast<size_t>(sgrid.num_indexes() * fm::ipow(degree_ + 1, sgrid.num_dims())));

    std::vector<precision> moment0;
    moms1d->project_moment(0, sgrid, field, moment0);

    int const level = sgrid.current_level(0);
    hier.reconstruct1d(1, level, span2d<precision>(degree_ + 1, fm::ipow2(level), moment0.data()));

    poisson.solve_periodic(moment0, terms.cdata.electric_field);
  }
}

template<typename precision>
fk::vector<precision>
discretization_manager<precision>::current_mpistate() const
{
  auto const s = element_segment_size(*pde);

  // gather results from all ranks. not currently writing the result anywhere
  // yet, but rank 0 holds the complete result after this call
  int my_rank = 0;
#ifdef ASGARD_USE_MPI
  int status = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  expect(status == 0);
#endif

  return gather_results<precision>(state, grid.get_distrib_plan(), my_rank, s);
}

template<typename precision>
void discretization_manager<precision>::print_mats() const {
  if (pde) { // version 1
    int const num_dims = pde->num_dims();
    for (auto tid : iindexof(pde->num_terms())) {
      for (int d : iindexof(num_dims)) {
        std::cout << " term = " << tid << "  dim = " << d << '\n';
        if (matrices.term_coeffs[tid * num_dims + d].empty()) {
          std::cout << "identity\n";
        } else {
          matrices.term_coeffs[tid * num_dims + d].to_full(conn).print(std::cout);
        }
        std::cout << '\n';
      }
    }
  } else {
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
}

#ifdef ASGARD_ENABLE_DOUBLE
template class discretization_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class discretization_manager<float>;
#endif

} // namespace asgard
