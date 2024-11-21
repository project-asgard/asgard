#include "asgard_discretization.hpp"

namespace asgard
{

template<typename precision>
discretization_manager<precision>::discretization_manager(
    std::unique_ptr<PDE<precision>> &&pde_in, verbosity_level verbosity)
    : verb(verbosity), pde(std::move(pde_in)), grid(*pde), conn(pde->max_level()),
      transformer(*pde, verb), degree_(0), dt_(0), time_(0), time_step_(0),
      final_time_step_(0), matrices(*pde), kronops(&conn, verbosity)
{
  rassert(!!pde, "invalid pde object");

  auto const &options = pde->options();

  if (high_verbosity())
  {
    node_out() << "Branch: " << GIT_BRANCH << '\n';
    node_out() << "Commit Summary: " << GIT_COMMIT_HASH
                    << GIT_COMMIT_SUMMARY << '\n';
    node_out() << "This executable was built on " << BUILD_TIME << '\n';

#ifdef ASGARD_USE_HIGHFIVE
    if (not options.restart_file.empty())
    {
        node_out() << "--- restarting from a file ---\n";
        node_out() << "  filename: " << options.restart_file << '\n';
    }
    else if (get_local_rank() == 0)
        std::cout << options;
#else
    if (get_local_rank() == 0)
        std::cout << options;
#endif

    node_out() << "--- begin setup ---" << '\n';
  }

  // initialize the discretization variables
  degree_ = options.degree.value();

  hier = hierarchy_manipulator(degree_, pde->get_dimensions());

  dt_ = pde->get_dt();

  final_time_step_ = options.num_time_steps.value();

  if (high_verbosity())
  {
    node_out() << "  degrees of freedom: " << degrees_of_freedom() << '\n';
    node_out() << "  generating: initial conditions..." << '\n';
  }

  set_initial_condition();

  if (high_verbosity())
  {
    node_out() << "  degrees of freedom (post initial adapt): "
               << degrees_of_freedom() << '\n';
    node_out() << "  generating: dimension mass matrices..." << '\n';
  }

  if (high_verbosity())
    node_out() << "  generating: coefficient matrices..." << '\n';

  this->compute_coefficients();

  auto const msg = grid.get_subgrid(get_rank());
  fixed_bc = boundary_conditions::make_unscaled_bc_parts(
        *pde, grid.get_table(), transformer, hier, matrices, conn, msg.row_start, msg.row_stop);

  if (high_verbosity())
    node_out() << "  generating: moment vectors..." << '\n';

  if (not pde->initial_moments.empty())
  {
    moments.reserve(pde->initial_moments.size());
    for (auto &minit : pde->initial_moments)
      moments.emplace_back(minit);

    for (auto &m : moments)
    {
      m.createFlist(*pde);
      expect(m.get_fList().size() > 0);

      m.createMomentVector(*pde, grid.get_table());
      expect(m.get_vector().size() > 0);
    }
  }

  if (options.step_method.value() == time_advance::method::imex)
    reset_moments();

  // -- setup output file and write initial condition
#ifdef ASGARD_USE_HIGHFIVE
  if (not options.restart_file.empty())
  {
    restart_data<precision> data = read_output(
        *pde, grid.get_table(), moments, options.restart_file);
    state      = data.solution.to_std();
    time_step_ = data.step_index;

    grid.recreate_table(data.active_table);
  }
  else
  {
    // compute the realspace moments for the initial file write
    generate_initial_moments<precision>(*pde, moments, grid, transformer, state);
  }
  if (options.wavelet_output_freq and options.wavelet_output_freq.value() > 0)
  {
    write_output<precision>(*pde, moments, state, precision{0.0}, 0, state.size(),
                            grid.get_table(), "asgard_wavelet");
  }
#endif
}

template<typename precision>
void discretization_manager<precision>::set_initial_condition()
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
void discretization_manager<precision>::ode_rhs(
    imex_flag imflag, precision t, std::vector<precision> const &x,
    std::vector<precision> &R) const
{
  R.resize(x.size());

  element_subgrid const &subgrid = grid.get_subgrid(get_rank());

#ifdef ASGARD_USE_MPI
  distribution_plan const &plan  = grid.get_distrib_plan();

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

  {
    tools::time_event performance("kronmult-make/set");
    kronops.make(imflag, *pde, matrices, grid);
  }{
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

    auto const bc0 = boundary_conditions::generate_scaled_bc(
        fixed_bc[0], fixed_bc[1], *pde, subgrid.row_start, subgrid.row_stop, t);

    for (auto i : indexof(bc0))
      reduced_row[i] += bc0[i];
  }

#ifdef ASGARD_USE_MPI
  exchange_results(reduced_row, R, hier.block_size(), plan, get_rank());
#endif
}

template<typename precision>
void discretization_manager<precision>::ode_irhs(
    precision t, std::vector<precision> const &x, std::vector<precision> &R) const
{
  R.resize(x.size());

  if (not pde->sources().empty())
  {
    tools::time_event performance("computing sources");

    auto const &sources = pde->sources();
    hier.template project_separable<data_mode::replace>
          (R.data(), pde->get_dimensions(), sources[0].source_funcs(),
          matrices.dim_dv, matrices.dim_mass, grid, t, sources[0].time_func()(t));

    for (size_t i = 1; i < sources.size(); i++)
      hier.template project_separable<data_mode::increment>
          (R.data(), pde->get_dimensions(), sources[i].source_funcs(),
          matrices.dim_dv, matrices.dim_mass, grid, t, sources[i].time_func()(t));
  }

  {
    tools::time_event performance("computing boundary conditions");

    element_subgrid const &subgrid = grid.get_subgrid(get_rank());
    auto const bc0 = boundary_conditions::generate_scaled_bc(
        fixed_bc[0], fixed_bc[1], *pde, subgrid.row_start, subgrid.row_stop, t);

    expect(static_cast<size_t>(bc0.size()) == R.size());

    precision const dt = pde->get_dt();
    if (pde->sources().empty())
      for (auto i : indexof(bc0))
        R[i] = x[i] + dt * bc0[i];
    else
      for (auto i : indexof(bc0))
        R[i] = x[i] + dt * (bc0[i] + R[i]);
  }
}
template<typename precision>
void discretization_manager<precision>::ode_sv(imex_flag imflag,
                                               std::vector<precision> &x) const
{
  auto const &options     = pde->options();
  solve_opts const solver = options.solver.value();

  static fk::vector<precision> sol; // used by the iterative solvers

  switch (solver)
  {
  case solve_opts::gmres:
  case solve_opts::bicgstab: {
      kronops.make(imflag, *pde, matrices, grid);
      precision const tolerance = *options.isolver_tolerance;
      int const restart         = *options.isolver_iterations;
      int const max_iter        = *options.isolver_outer_iterations;
      sol.resize(static_cast<int>(x.size()));
      std::copy(x.begin(), x.end(), sol.begin());
      if (solver == solve_opts::gmres)
        solver::simple_gmres_euler<precision, resource::host>(
            pde->get_dt(), imflag, kronops, sol, x, restart, max_iter, tolerance);
      else
        solver::bicgstab_euler<precision, resource::host>(
          pde->get_dt(), imflag, kronops, sol, x, max_iter, tolerance);

      std::copy(sol.begin(), sol.end(), x.begin());
    }
    break;
  default: // case solve_opts::direct:
    rassert(!!op_matrix, "must specify the operator matrix first");
    fm::getrs(op_matrix->A, x, op_matrix->ipiv);
    break;
  };
}

template<typename precision>
void discretization_manager<precision>::save_snapshot(std::filesystem::path const &filename) const
{
#ifdef ASGARD_USE_HIGHFIVE
  fk::vector<precision> fstate(state);
  write_output(*pde, moments, fstate, time_, time_step_, fstate.size(),
               grid.get_table(), "", filename);
#else
  ignore(filename);
  throw std::runtime_error("save_snapshot() requires CMake option -DASGARD_USE_HIGHFIVE=ON");
#endif
}

template<typename precision>
void discretization_manager<precision>::checkpoint() const
{
#ifdef ASGARD_USE_HIGHFIVE
  if (pde->is_output_step(time_step_))
  {
    if (high_verbosity())
      node_out() << "  checkpointing at step = " << time_step_
                  << " (time = " << time_ << ")\n";

    write_output<precision>(*pde, moments, state, time_, time_step_,
                            state.size(), grid.get_table(), "asgard_wavelet");
  }
#endif
}

template<typename precision>
std::optional<std::vector<precision>>
discretization_manager<precision>::get_exact_solution() const
{
  if (not pde->has_analytic_soln() and not pde->interp_exact())
    return {};

  tools::time_event performance("computing exact solution");

  if (pde->has_analytic_soln())
  {
    std::vector<precision> u_exact(state.size());

    auto const &exact_funcs = pde->exact_vector_funcs();
    for (auto const &func_batch : exact_funcs)
    {
      bool tfunc = (static_cast<int>(func_batch.size()) == pde->num_dims() + 1);

      precision const tmult = tfunc
                             ? func_batch.back()(fk::vector<precision>(), time_)[0]
                             : precision{1};

      hier.template project_separable<data_mode::replace>
          (u_exact.data(), pde->get_dimensions(), func_batch, matrices.dim_dv,
           matrices.dim_mass, grid, time_, tmult);
    }

    return u_exact;
  }
  else
  {
    vector2d<precision> const &inodes = kronops.get_inodes();
    std::vector<precision> u_exact(inodes.num_strips());
    pde->interp_exact()(time_, inodes, u_exact);
    return u_exact;
  }
}

template<typename precision>
std::optional<std::array<std::vector<precision>, 2>>
discretization_manager<precision>::rmse_exact_sol() const
{
  if (not pde->has_analytic_soln() and not pde->interp_exact())
    return {};

  tools::time_event performance("computing exact solution");

  if (pde->has_analytic_soln())
  {
    static std::vector<precision> solution;
    solution.resize(state.size());

    auto const &exact_funcs = pde->exact_vector_funcs();
    for (auto const &func_batch : exact_funcs)
    {
      bool tfunc = (static_cast<int>(func_batch.size()) == pde->num_dims() + 1);

      precision const tmult = tfunc
                             ? func_batch.back()(fk::vector<precision>(), time_)[0]
                             : precision{1};

      hier.template project_separable<data_mode::replace>
          (solution.data(), pde->get_dimensions(), func_batch, matrices.dim_dv,
           matrices.dim_mass, grid, time_, tmult);
    }

    // calculate root mean squared error
    auto const RMSE           = fm::rmserr(state, solution);
    auto const relative_error = 100 * RMSE  / fm::nrminf(solution);
    return gather_errors<precision>(RMSE, relative_error);
  }
  else
  {
#ifdef KRON_MODE_GLOBAL
    vector2d<precision> const &inodes = kronops.get_inodes();
    static std::vector<precision> u_exact;
    u_exact.resize(inodes.num_strips());
    pde->interp_exact()(time_, inodes, u_exact);

    std::vector<precision> u_comp = kronops.get_nodals(state.data());

    auto const RMSE           = fm::rmserr(u_comp, u_exact);
    auto const relative_error = 100 * RMSE  / fm::nrminf(u_exact);
    return gather_errors<precision>(RMSE, relative_error);
#endif
    return {};
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

#ifdef ASGARD_ENABLE_DOUBLE
template class discretization_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class discretization_manager<float>;
#endif

} // namespace asgard
